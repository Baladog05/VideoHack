"""Live "Ask the footage": search the whole index for a free-text scenario, judge every hit with the LLM as it
arrives, and score it against the current dataset. Used by the web app (POST /api/ask streams NDJSON)."""
import json, os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from vss import api
from taxonomy import CAMERAS
from miner import judge, HERE

DATA = os.path.join(HERE, "data", "edge_cases.json")

def load():
    return json.load(open(DATA)) if os.path.exists(DATA) else {"clips": [], "categories": {}, "funnel": {}}

def score(c, d):
    raws = sorted(x["motion_raw"] for x in d["clips"]) or [0]
    c["motion_score"] = round(10 * sum(r <= c["motion_raw"] for r in raws) / len(raws)) if c["motion_raw"] > 0 else 0
    conf = Counter(x["category"] for x in d["clips"] if x["is_edge_case"])
    mx = max(conf.values() or [1])
    ok = bool(c["is_edge_case"]) and c["category"] != "routine" and (c["edge_score"] or 0) >= 5
    rarity = 10 * (1 - conf.get(c["category"], 0) / (mx + 1)) if ok else 0
    c["rarity_score"] = round(rarity, 1)
    c["value"] = round(min(10, 0.55 * (c["edge_score"] or 1) + 0.2 * (c["difficulty"] or 1)
                               + 0.15 * c["motion_score"] + 0.1 * rarity), 1)
    c["is_edge_case"] = ok
    c["in_dataset"] = any(x["id"] == c["id"] for x in d["clips"] if x.get("is_edge_case"))
    return c

def ask(query, top_k=12):
    """Yields progress events: {"stage": ...}, then one {"clip": ...} per judged hit, then {"done": n}."""
    rows = api("/api/v1/search", {"query": query, "top_k": top_k * 2, "llm_top_n": 1, "min_similarity": 0.15}).get("results", [])
    rows = [r for r in rows if r.get("camera_id") in CAMERAS][:top_k]
    yield {"stage": "searched", "hits": len(rows)}
    d = load()
    with ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(judge, {**r, "_queries": [query], "_cats": {"ask"}}) for r in rows]
        for f in as_completed(futs):
            try: yield {"clip": score(f.result(), d)}
            except Exception as e: yield {"error": str(e)[:200]}
    yield {"done": len(rows)}

def add(clip):
    """Add a clip found by Ask to the dataset (marks it as an edge case)."""
    d = load()
    clip = {**clip, "is_edge_case": True, "added_by": clip.get("added_by") or "ask"}   # keep "live-agent"
    clip.pop("in_dataset", None)
    d["clips"] = [x for x in d["clips"] if x["id"] != clip["id"]] + [clip]
    d["clips"].sort(key=lambda c: (c["is_edge_case"], c["value"]), reverse=True)
    d.setdefault("funnel", {})["edge_cases"] = sum(c["is_edge_case"] for c in d["clips"])
    tmp = DATA + ".tmp"; json.dump(d, open(tmp, "w"), indent=1); os.replace(tmp, DATA)
    return d["funnel"]["edge_cases"]
