"""Package confirmed edge cases as a training dataset:
  datasets/<name>/clips/<id>.mp4, labels/<id>.json (per-frame boxes + scene labels), manifest.jsonl, DATASET_CARD.md
and a .zip next to it. If the W&B SDK is available (.venv), the dataset is also logged as a versioned W&B artifact.
Run: python3 export.py [--min-value 5] [--categories a,b] [--no-video]"""
import argparse, datetime, json, os, shutil, subprocess, sys, urllib.parse, urllib.request
from collections import Counter
from vss import BACKEND, ENV, token

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

def export(min_value=5.0, categories=None, with_video=True, name=None):
    data = json.load(open(os.path.join(HERE, "data", "edge_cases.json")))
    picked = [c for c in data["clips"] if c["is_edge_case"] and c["value"] >= min_value
              and (not categories or c["category"] in categories)]
    name = name or "edge-cases-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = os.path.join(HERE, "datasets", name)
    for d in ("clips", "labels"): os.makedirs(os.path.join(root, d), exist_ok=True)
    with open(os.path.join(root, "manifest.jsonl"), "w") as mf:
        for c in picked:
            boxes = json.load(open(os.path.join(HERE, "data", "boxes", c["boxes_id"] + ".json")))
            label = {
                "id": c["id"], "category": c["category"], "category_label": data["categories"].get(c["category"]),
                "title": c["title"], "explanation": c["explanation"], "tags": c.get("label_tags") or [],
                "actors": c.get("actors") or [], "conditions": c.get("conditions") or [],
                "scores": {k: c.get(k) for k in ("value", "edge_score", "difficulty", "motion_score", "rarity_score")},
                "source": {"segment_uri": c["source"], "original_video": c["original_video"], "camera_id": c["camera_id"],
                           "location": c["location"], "segment": c["segment"]},
                "scene_description": c["description"],
                "detections": {"format": "per sampled frame: [time_sec, [[label, x1, y1, x2, y2, in_ego_path], ...]] normalised 0-1",
                               "detector": "yolo11_coco", "frames": boxes},
            }
            json.dump(label, open(os.path.join(root, "labels", c["id"] + ".json"), "w"), indent=1)
            clip_path = f"clips/{c['id']}.mp4"
            if with_video:
                url = f"{BACKEND}/api/v1/videos/stream?" + urllib.parse.urlencode({"source": c["source"], "token": token()})
                with urllib.request.urlopen(url, timeout=120) as r, open(os.path.join(root, clip_path), "wb") as f:
                    shutil.copyfileobj(r, f)
            mf.write(json.dumps({"id": c["id"], "clip": clip_path if with_video else None, "label": f"labels/{c['id']}.json",
                                 "category": c["category"], "value": c["value"], "camera_id": c["camera_id"]}) + "\n")
    counts = Counter(c["category"] for c in picked)
    card = [f"# {name}", "", "Rare driving scenarios mined from the team VSS video index by **Edge Case Miner**.", "",
            f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"- Clips: {len(picked)} (from {data['funnel']['candidates']} candidates; threshold value >= {min_value})",
            f"- Cameras: {', '.join(sorted({c['camera_id'] for c in picked}))}",
            f"- Judge model: {data['model']} via W&B Inference; scene descriptions by NVIDIA Cosmos-Reason2; boxes by YOLO11",
            "", "## Category counts", "", "| Category | Clips |", "|---|---|"]
    card += [f"| {data['categories'].get(k, k)} | {counts.get(k, 0)} |" for k in data["categories"]]
    card += ["", "## Files", "", "- `manifest.jsonl`: one line per clip", "- `labels/<id>.json`: category, tags, scores, scene description, per-frame boxes",
             "- `clips/<id>.mp4`: the 5-second source segment", "",
             "## Caveats", "", "Labels are model-generated and should be spot-checked before training."]
    open(os.path.join(root, "DATASET_CARD.md"), "w").write("\n".join(card) + "\n")
    zip_path = shutil.make_archive(root, "zip", root)
    result = {"name": name, "path": root, "zip": zip_path, "clips": len(picked), "by_category": dict(counts), "wandb": None}
    venv_py = os.path.join(REPO, ".venv", "bin", "python")
    if os.path.exists(venv_py):
        p = subprocess.run([venv_py, os.path.join(HERE, "wandb_log.py"), root, json.dumps(result)],
                           capture_output=True, text=True, timeout=300)
        result["wandb"] = (p.stdout.strip().splitlines() or [None])[-1] if p.returncode == 0 else f"W&B logging failed: {p.stderr[-300:]}"
    return result

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--min-value", type=float, default=5.0)
    a.add_argument("--categories", default="")
    a.add_argument("--no-video", action="store_true")
    o = a.parse_args()
    print(json.dumps(export(o.min_value, [c for c in o.categories.split(",") if c] or None, not o.no_video), indent=1))
