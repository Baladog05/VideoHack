"""Motion-based risk from YOLO boxes: how big / central / fast-growing road users get in the ego vehicle's path."""
import urllib.parse
from vss import api

ROAD_USERS = {"person": "pedestrian", "bicycle": "cyclist", "motorcycle": "cyclist",
              "car": "vehicle", "truck": "vehicle", "bus": "vehicle"}

def detections(source):
    try:
        return api("/api/v1/videos/detections?source=" + urllib.parse.quote(source))
    except Exception:
        return None

def analyse(det):
    """Return (motion dict, compact boxes for overlay)."""
    if not det or not det.get("frames"):
        return {"motion_raw": 0, "motion_type": None, "reasons": ["no detections"]}, []
    H, W = det.get("video_shape") or det["frames"][0].get("shape") or [1080, 1920]
    series, boxes = {}, []
    for k, f in enumerate(det["frames"]):
        t = f.get("time_sec", k / (det.get("fps") or 30))
        best, keep = {}, []
        for d in f.get("detections", []):
            cls = ROAD_USERS.get(d["label"])
            if not cls or d.get("confidence", 0) < 0.4: continue
            x1, y1, x2, y2 = d["bbox"]
            area = (x2 - x1) * (y2 - y1) / (W * H)
            cx, by = (x1 + x2) / 2 / W, y2 / H
            in_path = 0.3 < cx < 0.7 and by > 0.55          # centre of frame, low down = ahead of us
            if in_path: best[cls] = max(best.get(cls, 0), area)
            keep.append([d["label"], round(x1 / W, 4), round(y1 / H, 4), round(x2 / W, 4), round(y2 / H, 4), int(in_path)])
        for cls, a in best.items():
            series.setdefault(cls, []).append((t, a))
        if k % 3 == 0: boxes.append([round(t, 3), keep])
    score, reasons, worst = 0.0, [], None
    nframes = len(det["frames"])
    for cls, s in series.items():
        if len(s) < 0.2 * nframes: continue                      # ignore flickers
        areas = sorted(a for _, a in s)
        peak = areas[int(0.9 * (len(areas) - 1))]                 # robust peak (90th pct)
        q = max(1, len(s) // 4)
        early = sorted(a for _, a in s[:q])[q // 2]               # median of first quarter
        growth = min(peak / early, 6) if early > 0 else 1
        vru = cls in ("pedestrian", "cyclist")                    # vulnerable road users count more
        sc = (1.5 if vru else 1.0) * (min(peak / (0.06 if vru else 0.2), 1) * 5 + (growth - 1) / 5 * 5)
        if sc > score: score, worst = sc, cls
        if peak > 0.01:
            reasons.append(f"{cls} in path, up to {peak*100:.1f}% of frame" + (f", grew {growth:.1f}x" if growth > 1.5 else ""))
    return {"motion_raw": round(score, 3), "motion_type": worst,
            "reasons": reasons or ["nobody in the vehicle's path"]}, boxes
