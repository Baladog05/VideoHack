"""Live agent: keeps searching the footage for the highest-value scenarios, judges anything new and
auto-adds confirmed edge cases (edge_score >= WATCH_MIN, default 7) to the dataset. Used by GET /api/watch (SSE)."""
import os, time, traceback
from vss import api
from taxonomy import CAMERAS, CATEGORIES
from miner import judge
from ask import load, score, add

INTERVAL = int(os.environ.get("WATCH_INTERVAL", 20))
MIN_EDGE = int(os.environ.get("WATCH_MIN", 7))
QUERIES = [q for v in CATEGORIES.values() for q in v["queries"]]

def watch(stop):
    """Generator of events; `stop()` returns True when the client has gone."""
    seen = {c["source"] for c in load()["clips"]}
    i, rnd = 0, 0
    while not stop():
        rnd += 1
        batch = [QUERIES[(i + k) % len(QUERIES)] for k in range(6)]; i += 6
        yield {"type": "scan", "round": rnd, "queries": batch}
        for q in batch:
            if stop(): return
            try:
                rows = api("/api/v1/search", {"query": q, "top_k": 5, "llm_top_n": 1, "min_similarity": 0.2}).get("results", [])
                for r in rows:
                    if r.get("camera_id") not in CAMERAS or r["source"] in seen: continue
                    seen.add(r["source"])
                    c = score(judge({**r, "_queries": [q], "_cats": {"watch"}}), load())
                    yield {"type": "judged", "title": c["title"], "edge_score": c["edge_score"]}
                    if c["is_edge_case"] and (c["edge_score"] or 0) >= MIN_EDGE:
                        c["added_by"] = "live-agent"; add(c)
                        yield {"type": "found", "clip": c}
            except Exception as e:
                yield {"type": "error", "message": str(e)[:200]}
        for _ in range(INTERVAL):
            if stop(): return
            time.sleep(1)
            yield {"type": "ping"} if _ % 10 == 9 else None
