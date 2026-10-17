"""Re-index the latest N chunks of one dashcam drive with a near-miss prompt, and monitor it.
Usage: python3 reindex.py <stream_id> <chunk_count>"""
import sys, time, json
from vss import api
PROMPT = ("Dashcam from the ego vehicle. For every pedestrian, cyclist and vehicle, describe how close they are to "
          "the ego vehicle (far, near, very close), whether they are in or entering its path, and any sudden braking, "
          "swerving, or right-of-way violation by anyone. Name the moment a gap closes fast. Also describe road "
          "layout, signals and crossings.")
sid, n = sys.argv[1], int(sys.argv[2])
job = api("/api/v1/dashboard/reingest", {"stream_id": sid, "chunk_count": n, "custom_prompt": PROMPT})
print(json.dumps({k: job.get(k) for k in ("job_id", "selected_chunks", "copied_segments", "status")})[:1500], flush=True)
jid = job["job_id"]
while True:
    try: s = api(f"/api/v1/dashboard/reingest/{jid}")
    except Exception as e: print("poll error", e, flush=True); time.sleep(10); continue
    print(f'{time.strftime("%H:%M:%S")} {s.get("status")} chunks {s.get("completed_chunks")}/{s.get("total_chunks")} '
          f'clips {s.get("indexed_segments")}/{s.get("total_segments")} {s.get("error") or ""}', flush=True)
    if s.get("status") in ("completed", "failed", "error"): break
    time.sleep(15)
