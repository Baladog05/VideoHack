"""Gap-filling agent: the search -> re-ingest -> search again loop from the brief, run on its own.

1. Find the weakest category in the dataset (fewest confirmed edge cases).
2. Search for it and record the "before" state (best match, how vague the current captions are).
3. Re-ingest the most promising videos through VSS with a prompt focused on that category.
4. When VSS finishes, search again, judge the re-described clips and add confirmed edge cases.
State lives in data/gapfill.json; the server polls running jobs in a background thread."""
import json, os, time, threading, uuid
from collections import Counter
from vss import api
from taxonomy import CAMERAS, CATEGORIES
from miner import judge, HERE
from ask import load, score, add

STATE = os.path.join(HERE, "data", "gapfill.json")
LOCK = threading.Lock()
CHUNKS = int(os.environ.get("GAPFILL_CHUNKS", 2))
BASE = ("Dashcam/street footage for self-driving training data. Describe every pedestrian, cyclist and vehicle, how close "
        "they get to the camera vehicle (far, near, very close) and whether they enter its path. ")

def _load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": []}

def _save(s):
    tmp = STATE + ".tmp"; json.dump(s, open(tmp, "w"), indent=1); os.replace(tmp, STATE)

def coverage():
    d = load()
    n = Counter(c["category"] for c in d["clips"] if c.get("is_edge_case"))
    return {k: n.get(k, 0) for k in CATEGORIES}

def weakest():
    cov = coverage()
    return min(CATEGORIES, key=lambda k: cov[k])

def _search(cat, top_k=15):
    rows, chunks = [], []
    for q in CATEGORIES[cat]["queries"]:
        r = api("/api/v1/search", {"query": q, "top_k": top_k, "llm_top_n": 1, "min_similarity": 0.1})
        rows += [x for x in r.get("results", []) if x.get("camera_id") in CAMERAS]
        chunks += [x for x in r.get("chunk_results", []) if x.get("camera_id") in CAMERAS]
    return rows, chunks

def start(cat=None):
    cat = cat or weakest()
    s = _load()
    used = {t["original_video"] for r in s["runs"] for t in r["targets"]}
    rows, chunks = _search(cat)
    best = {}
    for c in chunks:
        if c["original_video"] not in used and c["similarity_score"] > best.get(c["original_video"], {}).get("similarity_score", -1):
            best[c["original_video"]] = c
    picks = sorted(best.values(), key=lambda c: -c["similarity_score"])[:CHUNKS]
    if not picks: raise RuntimeError("no videos left to re-ingest for this category")
    prompt = (BASE + "Focus especially on: " + CATEGORIES[cat]["definition"] + " Say explicitly when this happens.")[:800]
    run = {"id": uuid.uuid4().hex[:8], "category": cat, "label": CATEGORIES[cat]["label"], "started": time.time(),
           "status": "reingesting", "prompt": prompt,
           "before": {"needles": coverage()[cat], "best_similarity": round(max([r["similarity_score"] for r in rows] or [0]), 3),
                      "captions": [p.get("reasoning_content", "")[:160] for p in picks]},
           "targets": [], "after": None, "found": []}
    for p in picks:
        job = api("/api/v1/dashboard/reingest", {"original_video": p["original_video"], "chunk_count": 1, "custom_prompt": prompt})
        run["targets"].append({"original_video": p["original_video"], "filename": p.get("filename"), "camera_id": p.get("camera_id"),
                               "job_id": job.get("job_id"), "sources": job.get("destination_sources", []),
                               "indexed": 0, "total": job.get("copied_segments", 6), "status": "running"})
    with LOCK:
        s = _load(); s["runs"].insert(0, run); _save(s)
    return run

def _finish(run):
    """Search again and judge the re-described clips."""
    cat = run["category"]
    new_src = {src for t in run["targets"] for src in t["sources"]}
    rows, _ = _search(cat, top_k=40)
    hits = {r["source"]: r for r in rows if r["source"] in new_src}
    for t in run["targets"]:   # clips that did not rank still get judged via their metadata
        for src in t["sources"]:
            if src not in hits:
                try:
                    m = api("/api/v1/videos/metadata?source=" + __import__("urllib.parse").parse.quote(src))
                    m = m.get("row", m)
                    if m.get("reasoning_content"): hits[src] = {**m, "source": src, "similarity_score": 0}
                except Exception:
                    pass
    d, found = load(), []
    for src, r in hits.items():
        try:
            c = score(judge({**r, "_queries": CATEGORIES[cat]["queries"], "_cats": {"gapfill"}}), d)
        except Exception:
            continue
        if c["is_edge_case"]:
            c["added_by"] = "gap-filler"; add(c); found.append({"id": c["id"], "title": c["title"], "category": c["category"], "value": c["value"]})
    run["after"] = {"needles": coverage()[cat], "best_similarity": round(max([r["similarity_score"] for r in rows] or [0]), 3),
                    "ranked": sum(1 for r in rows if r["source"] in new_src), "judged": len(hits),
                    "captions": [h.get("reasoning_content", "")[:160] for h in list(hits.values())[:2]]}
    run["found"] = found
    run["status"] = "done"

def poll_once():
    with LOCK: s = _load()
    changed = False
    for run in s["runs"]:
        if run["status"] != "reingesting": continue
        for t in run["targets"]:
            if t["status"] in ("completed", "failed"): continue
            try:
                j = api("/api/v1/dashboard/reingest/" + t["job_id"])
            except Exception:
                continue   # status endpoint flaps between replicas; try again next poll
            t["indexed"], t["status"] = j.get("indexed_segments") or 0, j.get("status") or t["status"]
            changed = True
        if all(t["status"] in ("completed", "failed", "error") for t in run["targets"]):
            run["status"] = "searching again"
            with LOCK:
                cur = _load(); cur["runs"] = [run if r["id"] == run["id"] else r for r in cur["runs"]]; _save(cur)
            try: _finish(run)
            except Exception as e: run["status"], run["error"] = "failed", str(e)[:200]
            changed = True
    if changed:
        with LOCK:
            cur = _load()
            by = {r["id"]: r for r in s["runs"]}
            cur["runs"] = [by.get(r["id"], r) for r in cur["runs"]]
            _save(cur)

def poller():
    while True:
        try: poll_once()
        except Exception: pass
        time.sleep(15)

def status():
    return {"coverage": coverage(), "weakest": weakest(), "labels": {k: v["label"] for k, v in CATEGORIES.items()}, "runs": _load()["runs"]}
