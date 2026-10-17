"""Live agent: watches the footage like a safety monitor watching live camera feeds.

It works through clips nobody has checked yet, rotating across road cameras in time order. For each clip it
first triages cheaply (does the Cosmos caption or the detector show anything that could matter?). Quiet
scenes are skipped as routine; the rest go to the LLM judge. Confirmed edge cases are added to the dataset.
Used by GET /api/live (Server-Sent Events). Progress persists in data/live_seen.json."""
import hashlib, json, os, re, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from vss import api
from taxonomy import CAMERAS
from miner import judge, HERE
from motion import detections, analyse
from ask import load, score, add

QUEUE = os.path.join(HERE, "data", "live_queue.json")
SEEN = os.path.join(HERE, "data", "live_seen.json")
MIN_EDGE = float(os.environ.get("LIVE_MIN", 6))
DWELL, SKIP_DWELL = 5.0, 2.5           # seconds each clip stays on screen
LOCK = threading.Lock()
KEY = re.compile(r"cyclist|bicycl|bike|scooter|pedestrian|crossing|jaywalk|ambulance|fire truck|police|emergency|siren|"
                 r"brak|swerv|double.?park|block|obstruct|construction|cone|debris|animal|dog|wrong.?way|red light|u-turn|"
                 r"revers|close to|sudden|abrupt|cut|merge|stopped in|school bus|truck", re.I)

def _queue():
    if os.path.exists(QUEUE): return json.load(open(QUEUE))
    segs, off = [], 0
    while True:
        r = api(f"/api/v1/videos/explore?scope=all&limit=100&offset={off}")
        for ch in r.get("chunks", []):
            if ch.get("camera_id") not in CAMERAS: continue
            for t in ch.get("timeline") or []:
                segs.append({**t, "camera_id": ch["camera_id"], "location": ch.get("location"), "original_video": ch["original_video"],
                             "filename": ch.get("filename"), "upload_timestamp": ch.get("upload_timestamp"), "total_segments": ch.get("total_segments")})
        off += 100
        if off >= r.get("total", 0) or not r.get("chunks"): break
    by = defaultdict(list)
    for s in segs: by[s["camera_id"]].append(s)
    for v in by.values(): v.sort(key=lambda s: (s.get("filename") or "", s.get("segment_number") or 0))
    order, lists = [], list(by.values())
    while any(lists):                     # rotate across cameras, time order within each camera
        for l in lists:
            if l: order.append(l.pop(0))
    json.dump(order, open(QUEUE, "w"))
    return order

def _seen():
    return set(json.load(open(SEEN))) if os.path.exists(SEEN) else set()

def _mark(src):
    with LOCK:
        s = _seen(); s.add(src); json.dump(sorted(s), open(SEEN, "w"))

def _counts(seg):
    c = seg.get("object_counts") or {}
    if isinstance(c, str):
        try: c = json.loads(c)
        except Exception: c = {}
    return c

def _assess(seg):
    """Triage, then judge if needed. Returns the verdict event."""
    cap, cnt = seg.get("reasoning_content") or "", _counts(seg)
    if not KEY.search(cap) and not (cnt.get("bicycle") or cnt.get("motorcycle")):
        bid = hashlib.md5(seg["source"].encode()).hexdigest()[:12]      # boxes for the overlay, from this clip's own detections
        try: json.dump(analyse(detections(seg["source"]))[1], open(os.path.join(HERE, "data", "boxes", bid + ".json"), "w"))
        except Exception: pass
        top = ", ".join(f"{v} {k}" for k, v in sorted(cnt.items(), key=lambda x: -x[1])[:3]) or "nothing notable"
        return {"verdict": "routine", "title": "Routine scene", "reason": f"Nothing risky in the scene ({top})", "triage": True}
    row = {**seg, "object_counts": cnt, "similarity_score": 0, "_queries": ["live"], "_cats": {"live"}}
    c = score(judge(row), load())
    hit = c["is_edge_case"] and (c["edge_score"] or 0) >= MIN_EDGE
    if hit:
        c["added_by"] = "live-agent"; add(c)
    return {"verdict": "edge" if hit else "routine", "title": c["title"], "reason": c.get("explanation") or "",
            "category": c["category"], "edge_score": c["edge_score"], "value": c.get("value"), "clip": c if hit else None,
            "boxes_id": c["boxes_id"], "triage": False}

def live(stop):
    q, seen = _queue(), _seen() | {c["source"] for c in load()["clips"]}
    todo = [s for s in q if s["source"] not in seen]
    stats = {"watched": 0, "edge": 0, "skipped": 0, "remaining": len(todo)}
    yield {"type": "start", "stats": stats, "cameras": len({s["camera_id"] for s in todo})}
    pool = ThreadPoolExecutor(3)
    try: yield from _run(todo, stats, pool, stop)
    finally: pool.shutdown(wait=False, cancel_futures=True)   # paused or closed: stop judging ahead too

def _run(todo, stats, pool, stop):
    futs = {}
    for i, seg in enumerate(todo):
        if stop(): return
        for nxt in todo[i:i + 3]:          # judge a few clips ahead so the feed never stalls
            if nxt["source"] not in futs: futs[nxt["source"]] = pool.submit(_assess, nxt)
        f, t_wait = futs.pop(seg["source"]), time.time()
        while not f.done() and time.time() - t_wait < 90:   # usually ready already (judged ahead)
            if stop(): return
            time.sleep(0.25)
        try: v = f.result(timeout=0)
        except Exception as e:
            v = {"verdict": "routine", "title": "Could not judge", "reason": str(e)[:120], "triage": True}
        if stop(): return
        t0 = time.time()
        bid = hashlib.md5(seg["source"].encode()).hexdigest()[:12]
        yield {"type": "watching", "source": seg["source"], "boxes_id": bid, "camera_id": seg["camera_id"], "location": seg.get("location"),
               "segment": f'{seg.get("segment_number")}/{seg.get("total_segments")}', "stats": stats}
        _mark(seg["source"])
        stats["watched"] += 1; stats["remaining"] -= 1
        stats["edge" if v["verdict"] == "edge" else "skipped"] += 1
        while time.time() - t0 < (SKIP_DWELL if v.get("triage") else DWELL):
            if stop(): return
            time.sleep(0.25)
        yield {"type": "verdict", "source": seg["source"], "camera_id": seg["camera_id"], **v, "stats": stats}
        hold = time.time()
        while time.time() - hold < (4 if v["verdict"] == "edge" else 1.2):   # keep the verdict on screen with its clip
            if stop(): return
            time.sleep(0.25)
