"""Package confirmed edge cases as a training dataset:
  datasets/<name>/clips/<id>.mp4, labels/<id>.json (per-frame boxes + scene labels), annotations_coco.json,
  splits/train.txt + val.txt (split by camera), manifest.jsonl, preview.jpg, DATASET_CARD.md
and a .zip next to it. If the W&B SDK is available (.venv), the dataset is also logged as a versioned W&B artifact.
Run: python3 export.py [--min-value 5] [--categories a,b] [--no-video] [--only-confirmed]"""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, tempfile, urllib.parse, urllib.request
from collections import Counter
from vss import BACKEND, ENV, token

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
W, H = 1920, 1080

def human_verdicts():
    """Latest human verdict per clip id from data/reviews.jsonl ("correct" or "wrong"; "clear" removes it)."""
    out, rp = {}, os.path.join(HERE, "data", "reviews.jsonl")
    if os.path.exists(rp):
        for line in open(rp):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("verdict") in ("correct", "wrong"): out[r["id"]] = r["verdict"]
            elif r.get("verdict") == "clear": out.pop(r.get("id"), None)
    return out

def _boxes(c):
    """Per-frame boxes for a clip; re-fetch from VSS if the cached file is missing or empty (e.g. after an outage)."""
    p = os.path.join(HERE, "data", "boxes", c["boxes_id"] + ".json")
    b = json.load(open(p)) if os.path.exists(p) else []
    if not any(f[1] for f in b):
        try:
            from motion import detections, analyse
            nb = analyse(detections(c["source"]))[1]
            if nb: b = nb; json.dump(b, open(p, "w"))
        except Exception: pass
    return b

def _split(picked):
    """Hold out whole cameras for validation (~20% of clips) so no camera appears in both splits."""
    cams = Counter(c["camera_id"] for c in picked)
    order = sorted(cams, key=lambda k: hashlib.md5(k.encode()).hexdigest())
    val, n = set(), 0
    for k in order:
        if len(cams) > 1 and n < 0.2 * len(picked) and len(val) < len(cams) - 1: val.add(k); n += cams[k]
    return {c["id"]: ("val" if c["camera_id"] in val else "train") for c in picked}

def _coco(picked, boxes_by_id, split):
    labels = sorted({b[0] for bx in boxes_by_id.values() for f in bx for b in f[1]})
    cat_id = {l: i + 1 for i, l in enumerate(labels)}
    images, anns = [], []
    for v, c in enumerate(picked, 1):
        for f in boxes_by_id[c["id"]]:
            iid = len(images) + 1
            images.append({"id": iid, "video_id": v, "clip_id": c["id"], "file_name": f"clips/{c['id']}.mp4", "time_sec": f[0],
                           "width": W, "height": H, "split": split[c["id"]], "edge_case_category": c["category"]})
            for lab, x1, y1, x2, y2, ip in f[1]:
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": cat_id[lab], "iscrowd": 0,
                             "bbox": [round(x1 * W, 1), round(y1 * H, 1), round((x2 - x1) * W, 1), round((y2 - y1) * H, 1)],
                             "area": round((x2 - x1) * W * (y2 - y1) * H, 1), "attributes": {"in_ego_path": bool(ip)}})
    return {"info": {"description": "Haystack edge-case driving dataset (video frames sampled from each clip)", "version": "1.0"},
            "videos": [{"id": v, "clip_id": c["id"], "file_name": f"clips/{c['id']}.mp4"} for v, c in enumerate(picked, 1)],
            "images": images, "annotations": anns,
            "categories": [{"id": i, "name": l, "supercategory": "road_user"} for l, i in cat_id.items()]}

def _preview(picked, boxes_by_id, root, with_video, data):
    """Contact sheet: one mid-clip frame per clip with its boxes and title."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception: font = ImageFont.load_default()
    tw, th, cols, cap = 400, 225, 4, 44
    tiles = []
    with tempfile.TemporaryDirectory() as td:
        for c in picked[:24]:
            src = os.path.join(root, "clips", c["id"] + ".mp4")
            if not (with_video and os.path.exists(src)):
                src = f"{BACKEND}/api/v1/videos/stream?" + urllib.parse.urlencode({"source": c["source"], "token": token()})
            out = os.path.join(td, c["id"] + ".jpg")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "2.5", "-i", src, "-frames:v", "1", "-vf", f"scale={tw}:{th}", out],
                           timeout=60, capture_output=True)
            im = Image.open(out).convert("RGB") if os.path.exists(out) else Image.new("RGB", (tw, th), (40, 40, 40))
            d = ImageDraw.Draw(im); fr = boxes_by_id[c["id"]]
            if fr:
                f = min(fr, key=lambda f: abs(f[0] - 2.5))
                for lab, x1, y1, x2, y2, ip in f[1]:
                    col = ((229, 72, 77) if lab in ("person", "bicycle", "motorcycle") else (240, 140, 0)) if ip else (47, 158, 68)
                    d.rectangle([x1 * tw, y1 * th, x2 * tw, y2 * th], outline=col, width=2 if ip else 1)
            tile = Image.new("RGB", (tw, th + cap), "white"); tile.paste(im, (0, 0))
            td_ = ImageDraw.Draw(tile)
            td_.text((6, th + 4), (c["title"] or "")[:44], fill=(20, 20, 20), font=font)
            td_.text((6, th + 23), f"{data['categories'].get(c['category'], c['category'])} · value {c['value']}"[:52], fill=(110, 110, 110), font=font)
            tiles.append(tile)
    if not tiles: return None
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 8, rows * (th + cap) + (rows + 1) * 8), (235, 235, 235))
    for i, t in enumerate(tiles): sheet.paste(t, (8 + (i % cols) * (tw + 8), 8 + (i // cols) * (th + cap + 8)))
    p = os.path.join(root, "preview.jpg"); sheet.save(p, quality=85)
    return p

def export(min_value=5.0, categories=None, with_video=True, name=None, only_confirmed=False):
    data = json.load(open(os.path.join(HERE, "data", "edge_cases.json")))
    picked = [c for c in data["clips"] if c["is_edge_case"] and c["value"] >= min_value
              and (not categories or c["category"] in categories)]
    hv = human_verdicts()
    rejected = sum(1 for c in picked if hv.get(c["id"]) == "wrong")
    picked = [c for c in picked if hv.get(c["id"]) != "wrong"]
    if only_confirmed: picked = [c for c in picked if hv.get(c["id"]) == "correct"]
    confirmed = sum(1 for c in picked if hv.get(c["id"]) == "correct")
    name = name or "edge-cases-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = os.path.join(HERE, "datasets", name)
    for d in ("clips", "labels", "splits"): os.makedirs(os.path.join(root, d), exist_ok=True)
    split = _split(picked)
    boxes_by_id, video_failed = {}, []
    with open(os.path.join(root, "manifest.jsonl"), "w") as mf:
        for c in picked:
            boxes = boxes_by_id[c["id"]] = _boxes(c)
            label = {
                "id": c["id"], "category": c["category"], "category_label": data["categories"].get(c["category"]),
                "title": c["title"], "explanation": c["explanation"], "tags": c.get("label_tags") or [],
                "actors": c.get("actors") or [], "conditions": c.get("conditions") or [],
                "scores": {k: c.get(k) for k in ("value", "edge_score", "difficulty", "motion_score", "rarity_score")},
                "source": {"segment_uri": c["source"], "original_video": c["original_video"], "camera_id": c["camera_id"],
                           "location": c["location"], "segment": c["segment"]},
                "split": split[c["id"]], "human_review": hv.get(c["id"], "unreviewed"),
                "scene_description": c["description"],
                "detections": {"format": "per sampled frame: [time_sec, [[label, x1, y1, x2, y2, in_ego_path], ...]] normalised 0-1",
                               "detector": "yolo11_coco", "frames": boxes},
            }
            json.dump(label, open(os.path.join(root, "labels", c["id"] + ".json"), "w"), indent=1)
            clip_path = f"clips/{c['id']}.mp4"
            if with_video:
                try:   # one failed download (e.g. VSS hiccup) must not sink the whole export
                    url = f"{BACKEND}/api/v1/videos/stream?" + urllib.parse.urlencode({"source": c["source"], "token": token()})
                    with urllib.request.urlopen(url, timeout=60) as r, open(os.path.join(root, clip_path), "wb") as f:
                        shutil.copyfileobj(r, f)
                except Exception:
                    video_failed.append(c["id"]); clip_path = None
                    try: os.remove(os.path.join(root, f"clips/{c['id']}.mp4"))
                    except OSError: pass
            mf.write(json.dumps({"id": c["id"], "clip": clip_path if with_video else None, "label": f"labels/{c['id']}.json",
                                 "category": c["category"], "title": c["title"], "value": c["value"], "camera_id": c["camera_id"],
                                 "split": split[c["id"]], "human_review": hv.get(c["id"], "unreviewed")}) + "\n")
    for s in ("train", "val"):
        open(os.path.join(root, "splits", s + ".txt"), "w").write("".join(i + "\n" for i, v in split.items() if v == s))
    coco = _coco(picked, boxes_by_id, split)
    json.dump(coco, open(os.path.join(root, "annotations_coco.json"), "w"))
    preview = _preview(picked, boxes_by_id, root, with_video, data)
    counts = Counter(c["category"] for c in picked)
    n_train = sum(1 for v in split.values() if v == "train")
    card = [f"# {name}", "", "Rare driving scenarios mined from the team VSS video index by **Haystack**.", ""]
    if preview: card += ["![Preview](preview.jpg)", ""]
    card += [f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"- Clips: {len(picked)} (from {data['funnel']['candidates']} candidates; threshold value >= {min_value})",
            f"- Cameras: {', '.join(sorted({c['camera_id'] for c in picked}))}",
            f"- Split: {n_train} train / {len(picked) - n_train} val, split by camera so no camera appears in both",
            f"- Boxes: {len(coco['annotations'])} across {len(coco['images'])} sampled frames ({len(coco['categories'])} object classes)",
            f"- Human review: {confirmed} confirmed, {len(picked) - confirmed} not yet reviewed, {rejected} rejected by a person and left out",
            f"- Judge model: {data['model']} via W&B Inference; scene descriptions by NVIDIA Cosmos-Reason2; boxes by YOLO11"]
    if video_failed: card += [f"- Video: {len(video_failed)} clip(s) could not be downloaded from VSS and have labels only: {', '.join(video_failed)}"]
    card += ["", "## Category counts", "", "| Category | Clips |", "|---|---|"]
    card += [f"| {data['categories'].get(k, k)} | {counts.get(k, 0)} |" for k in data["categories"]]
    card += ["", "## Files", "", "- `manifest.jsonl`: one line per clip (category, title, value, camera, split, human review)",
             "- `labels/<id>.json`: category, tags, scores, scene description, per-frame boxes",
             "- `annotations_coco.json`: the same boxes in COCO format (pixel `[x, y, w, h]`, one image per sampled frame, `in_ego_path` attribute)",
             "- `splits/train.txt`, `splits/val.txt`: clip ids per split",
             "- `clips/<id>.mp4`: the 5-second source segment", "- `preview.jpg`: one frame per clip with boxes", "",
             "## Caveats", "", "Labels are model-generated and should be spot-checked before training."]
    open(os.path.join(root, "DATASET_CARD.md"), "w").write("\n".join(card) + "\n")
    zip_path = shutil.make_archive(root, "zip", root)
    result = {"name": name, "path": root, "zip": zip_path, "clips": len(picked), "by_category": dict(counts), "wandb": None,
              "human_confirmed": confirmed, "human_rejected_excluded": rejected, "only_confirmed": bool(only_confirmed),
              "split": {"train": n_train, "val": len(picked) - n_train}, "boxes": len(coco["annotations"]),
              "video_failed": len(video_failed), "preview": bool(preview)}
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
    a.add_argument("--only-confirmed", action="store_true")
    o = a.parse_args()
    print(json.dumps(export(o.min_value, [c for c in o.categories.split(",") if c] or None, not o.no_video,
                            only_confirmed=o.only_confirmed), indent=1))
