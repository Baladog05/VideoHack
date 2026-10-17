"""Gap filling: re-index the latest N chunks of one stream with an edge-case-focused Cosmos prompt, then monitor.
Usage: python3 reindex.py <stream_id> <chunk_count> [category]
Re-indexing changes the shared team index: coordinate with your team before running it."""
import json, sys, time
from vss import api
from taxonomy import CATEGORIES

BASE = ("Dashcam/street footage for self-driving training data. Describe every pedestrian, cyclist and vehicle: how close "
        "they are to the camera vehicle (far, near, very close), whether they enter its path, and any sudden braking, "
        "swerving or rule violation (red light, wrong way, jaywalking, double parking). Note occlusions (people hidden "
        "behind buses or parked cars), emergency or unusual vehicles, debris, animals, roadworks, weather and lighting.")
sid, n = sys.argv[1], int(sys.argv[2])
cat = sys.argv[3] if len(sys.argv) > 3 else None
prompt = BASE + (f" Pay special attention to: {CATEGORIES[cat]['definition']}" if cat else "")
job = api("/api/v1/dashboard/reingest", {"stream_id": sid, "chunk_count": n, "custom_prompt": prompt[:800]})
print(json.dumps({k: job.get(k) for k in ("job_id", "selected_chunks", "copied_segments")}), flush=True)
while True:
    try: s = api(f"/api/v1/dashboard/reingest/{job['job_id']}")
    except Exception as e: time.sleep(15); continue
    print(f'{time.strftime("%H:%M:%S")} {s.get("status")} chunks {s.get("completed_chunks")}/{s.get("total_chunks")} '
          f'clips {s.get("indexed_segments")}/{s.get("total_segments")}', flush=True)
    if s.get("status") in ("completed", "failed", "error"): break
    time.sleep(15)
