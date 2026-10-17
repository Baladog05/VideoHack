"""Live agent: an autonomous explorer over the footage index.

Each round it picks a few hunts, digs deeper into each one (results it has not judged yet),
judges the new clips in parallel, and auto-adds confirmed edge cases (edge_score >= WATCH_MIN) to the
dataset. Every few rounds it asks the LLM to invent new hunts based on what it has found so far, so it
keeps exploring instead of re-running the same searches. Used by GET /api/watch (SSE)."""
import os, time, random
from concurrent.futures import ThreadPoolExecutor
from vss import api, llm_json
from taxonomy import CAMERAS, CATEGORIES
from miner import judge
from ask import load, score, add

INTERVAL = int(os.environ.get("WATCH_INTERVAL", 8))
MIN_EDGE = float(os.environ.get("WATCH_MIN", 6.5))
PER_ROUND = 3        # hunts per round
NEW_PER_HUNT = 3     # unseen clips judged per hunt
MAX_DEPTH = 100

def _invent(found_titles, existing):
    sys_p = ("You help mine rare edge cases from dashcam footage for self-driving training data. "
             "Reply as JSON {\"queries\": [..]} with 3 short, concrete video search queries for rare, "
             "safety-critical driving situations that are NOT already in the list given.")
    user = "Already searched:\n- " + "\n- ".join(existing[-25:]) + "\n\nRecent finds:\n- " + "\n- ".join(found_titles[-10:] or ["(none yet)"])
    try:
        qs = llm_json(sys_p, user).get("queries", [])
        return [q.strip() for q in qs if isinstance(q, str) and q.strip() and q.strip() not in existing][:3]
    except Exception:
        return []

def watch(stop):
    """Generator of events; `stop()` returns True when the client has gone."""
    seen = {c["source"] for c in load()["clips"]}
    found_titles = [c["title"] for c in load()["clips"] if c.get("is_edge_case")]
    hunts = [q for v in CATEGORIES.values() for q in v["queries"]]
    random.shuffle(hunts)
    depth = {q: 0 for q in hunts}
    stats = {"judged": 0, "found": 0, "skipped": 0}
    pool = ThreadPoolExecutor(4)
    rnd = 0
    yield {"type": "start", "known": len(seen), "hunts": len(hunts)}
    while not stop():
        rnd += 1
        if rnd % 2 == 0:
            yield {"type": "think", "message": "Inventing new hunts from what it has found"}
            for q in _invent(found_titles, hunts):
                hunts.append(q); depth[q] = 0
                yield {"type": "new_hunt", "query": q}
        open_hunts = [q for q in hunts if depth[q] < MAX_DEPTH] or hunts
        batch = sorted(open_hunts, key=lambda q: depth[q])[:PER_ROUND]
        for q in batch:
            if stop(): return
            yield {"type": "scan", "round": rnd, "query": q, "depth": depth[q], "queries": [q]}
            try:
                fresh = []
                while not fresh and depth[q] < MAX_DEPTH:   # keep digging this hunt until something unseen turns up
                    k = min(depth[q] + 20, MAX_DEPTH)
                    rows = api("/api/v1/search", {"query": q, "top_k": k, "llm_top_n": 1, "min_similarity": 0.1}).get("results", [])
                    depth[q] = k if len(rows) >= k else MAX_DEPTH
                    for r in rows:
                        if r.get("camera_id") not in CAMERAS: continue
                        if r["source"] in seen: stats["skipped"] += 1; continue
                        seen.add(r["source"]); fresh.append(r)
                        if len(fresh) >= NEW_PER_HUNT: break
                if not fresh:
                    yield {"type": "dry", "query": q, "stats": stats}
                    continue
                d = load()
                futs = [pool.submit(judge, {**r, "_queries": [q], "_cats": {"watch"}}) for r in fresh]
                for f in futs:
                    if stop(): return
                    c = score(f.result(), d)
                    stats["judged"] += 1
                    hit = c["is_edge_case"] and (c["edge_score"] or 0) >= MIN_EDGE
                    yield {"type": "judged", "title": c["title"], "edge_score": c["edge_score"], "hit": bool(hit), "stats": stats}
                    if hit:
                        c["added_by"] = "live-agent"; add(c); stats["found"] += 1
                        found_titles.append(c["title"])
                        yield {"type": "found", "clip": c, "stats": stats}
            except Exception as e:
                yield {"type": "error", "message": str(e)[:200]}
        for s in range(INTERVAL):
            if stop(): return
            time.sleep(1)
            yield {"type": "ping"} if s % 5 == 4 else None
