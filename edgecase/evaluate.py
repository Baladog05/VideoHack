"""Judge accuracy from human reviews. Reads data/reviews.jsonl (latest verdict per clip wins) and data/edge_cases.json,
writes data/eval.json. Run: python3 evaluate.py   (the app also calls evaluate() after each review)."""
import json, os
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))

def evaluate():
    rp, dp = os.path.join(HERE, "data", "reviews.jsonl"), os.path.join(HERE, "data", "edge_cases.json")
    verdicts = {}
    if os.path.exists(rp):
        for line in open(rp):
            try: r = json.loads(line); verdicts[r["id"]] = r["verdict"]
            except Exception: pass
    clips = {c["id"]: c for c in json.load(open(dp))["clips"]} if os.path.exists(dp) else {}
    acc, rej, per = [0, 0], [0, 0], defaultdict(lambda: [0, 0])
    for cid, v in verdicts.items():
        c = clips.get(cid)
        if not c: continue
        ok = v == "correct"
        if c["is_edge_case"]:
            acc[0] += ok; acc[1] += 1; per[c["category"]][0] += ok; per[c["category"]][1] += 1
        else:
            rej[0] += (not ok); rej[1] += 1   # a "wrong" on a rejected clip = the judge missed an edge case
    pct = lambda a: round(100 * a[0] / a[1]) if a[1] else None
    out = {"reviewed": acc[1] + rej[1], "precision": pct(acc), "precision_n": acc[1],
           "false_reject_rate": pct(rej), "rejected_reviewed": rej[1],
           "per_category": {k: {"precision": pct(v), "n": v[1]} for k, v in per.items()}, "verdicts": verdicts}
    json.dump(out, open(os.path.join(HERE, "data", "eval.json"), "w"), indent=1)
    return out

if __name__ == "__main__":
    o = evaluate(); o.pop("verdicts"); print(json.dumps(o, indent=1))
