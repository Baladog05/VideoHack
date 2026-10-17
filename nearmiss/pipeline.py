"""Find candidate near-miss moments in the dashcam footage, score each by
(1) an LLM reading the scene description plus the clips before and after, and
(2) motion of YOLO boxes in the vehicle's path. Writes data/incidents.json.  Run: python3 pipeline.py"""
import hashlib, json, os, re, sys, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from vss import api, llm_json, LLM_MODEL
from motion import detections, analyse

HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA = os.environ.get("NEARMISS_CAMERA", "pie_cam-3")
QUERIES = {
    "pedestrian": ["pedestrian stepping into the road in front of the car", "person crossing close in front of the moving vehicle",
                   "pedestrian crossing outside the crosswalk", "pedestrian very close to the ego vehicle"],
    "cyclist":    ["cyclist cutting across in front of the vehicle", "bicycle very close to the car"],
    "vehicle":    ["car braking hard to avoid a collision", "vehicle pulling out suddenly in front",
                   "car turning across the path of the vehicle", "vehicle running a stop sign or red light",
                   "gap to the vehicle ahead closes fast"],
}
SYSTEM = """You are a road-safety analyst reviewing dashcam footage from the ego (camera) vehicle.
You get the description of the target 5-second segment, the segments just before and after it, and
motion measurements from an object detector (how much of the frame road users fill in the vehicle's path).
Decide whether the target segment shows a NEAR MISS: a collision with a pedestrian, cyclist or vehicle was
plausibly avoided, or the gap became dangerously small, or someone violated right of way close to the vehicle.
Calibrate risk 1-10: 1-2 routine traffic; 3-4 close but controlled interaction; 5-6 a hazard needing the driver
to react; 7-8 clear near miss; 9-10 extremely close call. Use only the evidence given.
Reply with JSON only:
{"near_miss": bool, "risk": int, "type": "pedestrian"|"cyclist"|"vehicle"|"none",
 "title": "<=8 word headline", "explanation": "1-2 sentences: who, what happened, how close",
 "factors": ["short risk factors"]}"""

def neighbour(source, delta):
    m = re.search(r"segment_(\d+)_of_(\d+)", source)
    if not m: return ""
    n, tot = int(m.group(1)) + delta, int(m.group(2))
    if not 1 <= n <= tot: return "(none)"
    src = source.replace(m.group(0), f"segment_{n:03d}_of_{tot:03d}")
    try:
        r = api("/api/v1/videos/metadata?source=" + urllib.parse.quote(src))
        r = r.get("row", r) if isinstance(r, dict) else {}
        return (r.get("reasoning_content") or "")[:700]
    except Exception:
        return "(unavailable)"

def search(q):
    return api("/api/v1/search", {"query": q, "top_k": 12, "llm_top_n": 1, "min_similarity": 0.2,
                                  "metadata_filters": {"camera_id": CAMERA}}).get("results", [])

def score(row):
    src = row["source"]
    motion, boxes = analyse(detections(src))
    bid = hashlib.md5(src.encode()).hexdigest()[:12]
    json.dump(boxes, open(os.path.join(HERE, "data", "boxes", bid + ".json"), "w"))
    user = (f"BEFORE (previous 5s): {neighbour(src, -1)}\n\nTARGET: {row.get('reasoning_content') or ''}\n\n"
            f"AFTER (next 5s): {neighbour(src, +1)}\n\nDetector peak counts: {json.dumps(row.get('object_counts') or {})}\n"
            "Detector measurements (road users in the vehicle's path): " + "; ".join(motion["reasons"]))
    try:
        s = llm_json(SYSTEM, user)
    except Exception as e:
        print("  score failed:", e, file=sys.stderr)
        s = {"near_miss": False, "risk": 1, "type": "none", "title": "Unscored", "explanation": str(e)[:200], "factors": []}
    llm_risk = int(s.get("risk") or 1)
    return {
        "source": src, "original_video": row.get("original_video"), "boxes_id": bid,
        "drive": next((p for p in src.split("_") if p.startswith("set")), ""),
        "segment": f'{row.get("segment_number")}/{row.get("total_segments")}',
        "chunk": (re.search(r"chunk_(\d+)", src) or [None, ""])[1],
        "similarity": round(row.get("similarity_score") or 0, 3), "object_counts": row.get("object_counts") or {},
        "description": row.get("reasoning_content") or "", "matched_queries": row["_queries"], "search_category": row["_cat"],
        "llm_risk": llm_risk, "motion_raw": motion["motion_raw"], "motion_reasons": motion["reasons"],
        **{k: s.get(k) for k in ("near_miss", "type", "title", "explanation", "factors")},
    }

def main():
    cands = {}
    for cat, qs in QUERIES.items():
        for q in qs:
            for r in search(q):
                c = cands.setdefault(r["source"], {**r, "_queries": [], "_cat": cat})
                c["_queries"].append(q)
                c["similarity_score"] = max(c.get("similarity_score") or 0, r.get("similarity_score") or 0)
            print(f"searched: {q}  (candidates: {len(cands)})", flush=True)
    print(f"scoring {len(cands)} candidates with {LLM_MODEL} + motion ...", flush=True)
    with ThreadPoolExecutor(8) as ex:
        incidents = list(ex.map(score, cands.values()))
    # motion score = percentile of raw motion among candidates, so it spreads 0-10
    raws = sorted(i["motion_raw"] for i in incidents)
    for i in incidents:
        i["motion_risk"] = round(10 * sum(r <= i["motion_raw"] for r in raws) / len(raws)) if i["motion_raw"] > 0 else 0
        i["risk"] = max(1, min(10, round(0.7 * i["llm_risk"] + 0.3 * i["motion_risk"])))
        i["near_miss"] = bool(i["near_miss"]) and i["risk"] >= 6
    incidents.sort(key=lambda x: (x["risk"], bool(x["near_miss"]), x["similarity"]), reverse=True)
    json.dump({"camera": CAMERA, "model": LLM_MODEL, "incidents": incidents},
              open(os.path.join(HERE, "data", "incidents.json"), "w"), indent=1)
    nm = [i for i in incidents if i["near_miss"]]
    print(f"done: {len(incidents)} scored, {len(nm)} near misses")
    for i in incidents[:12]:
        print(f'  risk {i["risk"]} (llm {i["llm_risk"]}, motion {i["motion_risk"]}) nm={i["near_miss"]} [{i["type"]}] {i["title"]} :: {i["drive"]} chunk {i["chunk"]} seg {i["segment"]}')

if __name__ == "__main__":
    main()
