"""Haystack miner: search every road camera for rare driving scenarios, judge each candidate with an LLM
(using the clips before/after) plus detector motion, score edge-case value, and write data/edge_cases.json.
Run: python3 miner.py            (env: MINER_TOP_K per search, default 10)"""
import hashlib, json, os, re, sys, time, urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from vss import api, llm_json, LLM_MODEL
from motion import detections, analyse
from taxonomy import CAMERAS, CATEGORIES

HERE = os.path.dirname(os.path.abspath(__file__))
TOP_K = int(os.environ.get("MINER_TOP_K", 10))
CAT_TEXT = "\n".join(f'- {k}: {v["definition"]}' for k, v in CATEGORIES.items())
SYSTEM = f"""You curate training data for self-driving cars. Routine driving (cars in lanes, people on sidewalks,
waiting at lights, normal crossings at a crosswalk) is NOT useful. Rare, hard scenarios ARE.
Edge-case categories:
{CAT_TEXT}
You get the description of a 5-second target clip, the clips before and after it, and object-detector measurements.
Judge only the TARGET clip, using only the evidence given. Score:
- edge_score 1-10: 1-2 routine; 3-4 mildly unusual; 5-6 a genuine but common edge case; 7-8 rare and valuable; 9-10 exceptional.
- difficulty 1-10: how hard this would be for an autonomous driving stack to handle correctly.
Reply with JSON only:
{{"is_edge_case": bool, "category": one of [{", ".join(CATEGORIES)}, "routine"],
 "edge_score": int, "difficulty": int, "title": "<=8 word headline",
 "explanation": "1-2 sentences: what happens and why it is hard for a self-driving car",
 "actors": ["road users involved"], "conditions": ["weather/lighting/road conditions"],
 "label_tags": ["3-6 short snake_case tags for a dataset"]}}"""

def search(cat, q):
    t0 = time.time()
    rows = api("/api/v1/search", {"query": q, "top_k": TOP_K, "llm_top_n": 1, "min_similarity": 0.2}).get("results", [])
    return cat, q, [r for r in rows if r.get("camera_id") in CAMERAS], time.time() - t0

def neighbour(source, delta):
    m = re.search(r"segment_(\d+)_of_(\d+)", source)
    if not m: return ""
    n, tot = int(m.group(1)) + delta, int(m.group(2))
    if not 1 <= n <= tot: return "(none)"
    try:
        r = api("/api/v1/videos/metadata?source=" + urllib.parse.quote(source.replace(m.group(0), f"segment_{n:03d}_of_{tot:03d}")))
        r = r.get("row", r) if isinstance(r, dict) else {}
        return (r.get("reasoning_content") or "")[:700]
    except Exception:
        return "(unavailable)"

def judge(row):
    src = row["source"]
    motion, boxes = analyse(detections(src))
    bid = hashlib.md5(src.encode()).hexdigest()[:12]
    json.dump(boxes, open(os.path.join(HERE, "data", "boxes", bid + ".json"), "w"))
    user = (f"Camera: {row.get('camera_id')} ({row.get('location')})\n"
            f"BEFORE: {neighbour(src, -1)}\n\nTARGET: {row.get('reasoning_content') or ''}\n\nAFTER: {neighbour(src, +1)}\n\n"
            f"Detector peak counts: {json.dumps(row.get('object_counts') or {})}\n"
            f"Detector measurements (road users in the vehicle's path): " + "; ".join(motion["reasons"]))
    for attempt in range(2):
        try:
            s = llm_json(SYSTEM, user); break
        except Exception as e:
            s = {"is_edge_case": False, "category": "routine", "edge_score": 1, "difficulty": 1, "title": "Unscored",
                 "explanation": f"judge failed: {e}"[:200], "actors": [], "conditions": [], "label_tags": []}
    m = re.search(r"segment_(\d+)_of", src)
    return {
        "id": bid, "source": src, "original_video": row.get("original_video"), "boxes_id": bid,
        "camera_id": row.get("camera_id"), "location": row.get("location"),
        "segment_start_sec": row.get("segment_start_sec"), "segment_end_sec": row.get("segment_end_sec"),
        "segment": f'{row.get("segment_number")}/{row.get("total_segments")}',
        "similarity": round(row.get("similarity_score") or 0, 3), "object_counts": row.get("object_counts") or {},
        "description": row.get("reasoning_content") or "", "matched_queries": row["_queries"], "search_categories": sorted(row["_cats"]),
        "motion_raw": motion["motion_raw"], "motion_reasons": motion["reasons"],
        **{k: s.get(k) for k in ("is_edge_case", "category", "edge_score", "difficulty", "title", "explanation",
                                 "actors", "conditions", "label_tags")},
    }

def main():
    t0 = time.time()
    jobs = [(c, q) for c, v in CATEGORIES.items() for q in v["queries"]]
    cands = {}
    with ThreadPoolExecutor(6) as ex:
        for cat, q, rows, dt in ex.map(lambda a: search(*a), jobs):
            for r in rows:
                c = cands.setdefault(r["source"], {**r, "_queries": [], "_cats": set()})
                c["_queries"].append(q); c["_cats"].add(cat)
            print(f"[{cat}] {q}: {len(rows)} road hits  (candidates: {len(cands)})", flush=True)
    try: stats = api("/api/v1/dashboard/stats?scope=all")
    except Exception: stats = {}
    print(f"judging {len(cands)} candidates with {LLM_MODEL} ...", flush=True)
    with ThreadPoolExecutor(8) as ex:
        clips = list(ex.map(judge, cands.values()))
    # motion -> percentile 0-10 across candidates
    raws = sorted(c["motion_raw"] for c in clips)
    for c in clips:
        c["motion_score"] = round(10 * sum(r <= c["motion_raw"] for r in raws) / len(raws)) if c["motion_raw"] > 0 else 0
    # category rarity: categories with fewer confirmed examples are worth more (gap-filling)
    for c in clips:  # final edge-case rule first, so rarity counts only confirmed clips
        c["is_edge_case"] = bool(c["is_edge_case"]) and c["category"] != "routine" and (c["edge_score"] or 0) >= 5
    confirmed = Counter(c["category"] for c in clips if c["is_edge_case"])
    mx = max(confirmed.values() or [1])
    for c in clips:
        rarity = 10 * (1 - confirmed.get(c["category"], 0) / (mx + 1)) if c["is_edge_case"] else 0
        c["rarity_score"] = round(rarity, 1)
        c["value"] = round(min(10, 0.55 * (c["edge_score"] or 1) + 0.2 * (c["difficulty"] or 1)
                                   + 0.15 * c["motion_score"] + 0.1 * rarity), 1)
    clips.sort(key=lambda c: (c["is_edge_case"], c["value"]), reverse=True)
    total_segments = (stats or {}).get("overview", {}).get("indexed_clips")
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": LLM_MODEL, "cameras": CAMERAS,
           "categories": {k: v["label"] for k, v in CATEGORIES.items()},
           "funnel": {"indexed_segments": total_segments, "searched_queries": len(jobs), "candidates": len(clips),
                      "edge_cases": sum(c["is_edge_case"] for c in clips)},
           "runtime_sec": round(time.time() - t0, 1), "clips": clips}
    json.dump(out, open(os.path.join(HERE, "data", "edge_cases.json"), "w"), indent=1)
    keep = {c["boxes_id"] for c in clips}  # drop overlay files from older runs
    for f in os.listdir(os.path.join(HERE, "data", "boxes")):
        if f[:-5] not in keep: os.remove(os.path.join(HERE, "data", "boxes", f))
    print(json.dumps(out["funnel"]), f"in {out['runtime_sec']}s")
    print("by category:", dict(Counter(c["category"] for c in clips if c["is_edge_case"])))
    for c in clips[:12]:
        print(f'  {c["value"]:>4} [{c["category"]}] {c["title"]} :: {c["camera_id"]} {c["source"].rsplit("/",1)[-1]}')

if __name__ == "__main__":
    main()
