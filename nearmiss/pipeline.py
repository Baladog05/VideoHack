"""Find candidate near-miss moments in the dashcam footage, score each with an LLM,
and write data/incidents.json for the web app.  Run: python3 pipeline.py"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from vss import api, llm_json, LLM_MODEL

CAMERA = os.environ.get("NEARMISS_CAMERA", "pie_cam-3")
QUERIES = {
    "pedestrian": ["pedestrian stepping into the road in front of the car",
                   "person crossing close in front of the moving vehicle",
                   "pedestrian crossing outside the crosswalk"],
    "cyclist":    ["cyclist cutting across in front of the vehicle",
                   "bicycle very close to the car"],
    "vehicle":    ["car braking hard to avoid a collision",
                   "vehicle pulling out suddenly in front",
                   "car turning across the path of the vehicle",
                   "vehicle running a stop sign or red light"],
}
SYSTEM = """You are a road-safety analyst reviewing dashcam footage descriptions from the ego (camera) vehicle.
Decide whether the segment shows a NEAR MISS: a moment where a collision with a pedestrian, cyclist or vehicle
was plausibly avoided or the gap was dangerously small. Ordinary safe interactions (waiting at a stop sign while
someone crosses well ahead) are NOT near misses. Be strict and only use what the description says.
Reply with JSON only:
{"near_miss": bool, "risk": integer 1-10, "type": "pedestrian"|"cyclist"|"vehicle"|"none",
 "title": "<=8 word headline", "explanation": "1-2 sentences: who, what happened, how close",
 "factors": ["short risk factors"]}"""

def search(q):
    res = api("/api/v1/search", {"query": q, "top_k": 12, "llm_top_n": 1, "min_similarity": 0.2,
                                 "metadata_filters": {"camera_id": CAMERA}})
    return res.get("results", [])

def score(row):
    desc = row.get("reasoning_content") or ""
    user = (f"Segment description:\n{desc}\n\nDetected objects (peak count per class): "
            f"{json.dumps(row.get('object_counts') or {})}")
    try:
        s = llm_json(SYSTEM, user)
    except Exception as e:
        print("  score failed:", e, file=sys.stderr)
        s = {"near_miss": False, "risk": 1, "type": "none", "title": "Unscored", "explanation": str(e)[:200], "factors": []}
    return {
        "source": row["source"], "original_video": row.get("original_video"),
        "drive": next((p for p in row["source"].split("_") if p.startswith("set")), ""),
        "start_sec": row.get("segment_start_sec"), "end_sec": row.get("segment_end_sec"),
        "segment": f'{row.get("segment_number")}/{row.get("total_segments")}',
        "similarity": round(row.get("similarity_score") or 0, 3),
        "object_counts": row.get("object_counts") or {},
        "description": desc, "matched_queries": row["_queries"], "search_category": row["_cat"],
        **{k: s.get(k) for k in ("near_miss", "risk", "type", "title", "explanation", "factors")},
    }

def main():
    cands = {}
    for cat, qs in QUERIES.items():
        for q in qs:
            for r in search(q):
                c = cands.setdefault(r["source"], {**r, "_queries": [], "_cat": cat})
                c["_queries"].append(q)
                c["similarity_score"] = max(c.get("similarity_score") or 0, r.get("similarity_score") or 0)
            print(f"searched: {q}  (candidates so far: {len(cands)})")
    print(f"scoring {len(cands)} candidates with {LLM_MODEL} ...")
    with ThreadPoolExecutor(8) as ex:
        incidents = list(ex.map(score, cands.values()))
    incidents.sort(key=lambda x: (bool(x["near_miss"]), x["risk"] or 0, x["similarity"]), reverse=True)
    out = os.path.join(os.path.dirname(__file__), "data", "incidents.json")
    json.dump({"camera": CAMERA, "model": LLM_MODEL, "incidents": incidents}, open(out, "w"), indent=1)
    nm = [i for i in incidents if i["near_miss"]]
    print(f"done: {len(incidents)} scored, {len(nm)} near misses -> {out}")
    for i in nm[:10]:
        print(f'  risk {i["risk"]} [{i["type"]}] {i["title"]} :: {i["drive"]} {i["source"].rsplit("/",1)[-1]}')

if __name__ == "__main__":
    main()
