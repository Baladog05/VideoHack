"""Haystack web app (stdlib only).  Run: python3 server.py   -> http://localhost:8080"""
import json, os, shutil, threading, time, urllib.error, urllib.parse, urllib.request
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from vss import BACKEND, token, api

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", 8080))
EXPORT_LOCK, ADD_LOCK = threading.Lock(), threading.Lock()
EXPORT_STATE = {"running": False, "step": "idle", "result": None, "error": None}
_stats = {"t": 0, "v": {}}

def stats():
    if time.time() - _stats["t"] > 300:
        try:
            o = api("/api/v1/dashboard/stats?scope=all").get("overview", {})
            _stats["v"] = {"videos": o.get("unique_videos"), "clips": o.get("indexed_clips")}; _stats["t"] = time.time()
        except Exception: pass
    return _stats["v"]

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=os.path.join(HERE, "static"), **k)
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/clips":
            p = os.path.join(HERE, "data", "edge_cases.json")
            d = json.load(open(p)) if os.path.exists(p) else {"clips": [], "funnel": {}, "categories": {}}
            d["index"] = stats(); return self._json(d)
        if u.path == "/api/eval":
            from evaluate import evaluate
            return self._json(evaluate())
        if u.path == "/api/export/status":
            return self._json(EXPORT_STATE)
        if u.path == "/api/live":
            from live import live
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.end_headers()
            gone = {"v": False}
            try:
                for ev in live(lambda: gone["v"]):
                    try: self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError): gone["v"] = True
            except Exception as e:
                try: self.wfile.write(f"data: {json.dumps({'type': 'error', 'message': str(e)[:200]})}\n\n".encode())
                except Exception: pass
            self.close_connection = True
            return
        if u.path.startswith("/api/boxes/"):
            bid = os.path.basename(u.path); p = os.path.join(HERE, "data", "boxes", bid + ".json")
            return self._json(open(p, "rb").read() if bid.isalnum() and os.path.exists(p) else b"[]")
        if u.path == "/api/datasets":
            out = []
            for z in sorted(os.listdir(os.path.join(HERE, "datasets")) if os.path.isdir(os.path.join(HERE, "datasets")) else [], reverse=True):
                if z.endswith(".zip"):
                    n = z[:-4]; man = os.path.join(HERE, "datasets", n, "manifest.jsonl")
                    out.append({"name": n, "download": f"/api/datasets/{z}", "mb": round(os.path.getsize(os.path.join(HERE, "datasets", z)) / 1e6, 1),
                                "clips": sum(1 for _ in open(man)) if os.path.exists(man) else None})
            return self._json(out)
        if u.path.startswith("/api/datasets/") and u.path.endswith(".zip"):
            name = os.path.basename(u.path)[:-4]; p = os.path.join(HERE, "datasets", name + ".zip")
            if not (name.replace("-", "").isalnum() and os.path.exists(p)): return self._json({"error": "not found"}, 404)
            self.send_response(200); self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{name}.zip"')
            self.send_header("Content-Length", str(os.path.getsize(p))); self.end_headers()
            with open(p, "rb") as f: shutil.copyfileobj(f, self.wfile)
            return
        if u.path == "/api/stream":
            src = urllib.parse.parse_qs(u.query).get("source", [""])[0]
            req = urllib.request.Request(f"{BACKEND}/api/v1/videos/stream?" + urllib.parse.urlencode({"source": src, "token": token()}))
            if self.headers.get("Range"): req.add_header("Range", self.headers["Range"])
            try: r = urllib.request.urlopen(req, timeout=120)
            except urllib.error.HTTPError as e: r = e
            self.send_response(getattr(r, "status", None) or r.code); self.send_header("Content-Type", "video/mp4")
            for h in ("Content-Length", "Content-Range", "Accept-Ranges"):
                if r.headers.get(h): self.send_header(h, r.headers[h])
            self.end_headers()
            try: shutil.copyfileobj(r, self.wfile)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/ask":
            q = (self._body().get("query") or "").strip()[:200]
            if not q: return self._json({"error": "empty query"}, 400)
            from ask import ask
            self.send_response(200); self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache"); self.end_headers()
            try:
                for ev in ask(q):
                    self.wfile.write((json.dumps(ev) + "\n").encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): pass
            except Exception as e:
                try: self.wfile.write((json.dumps({"error": str(e)[:200]}) + "\n").encode())
                except Exception: pass
            self.close_connection = True
            return
        if self.path == "/api/add":
            from ask import add
            with ADD_LOCK: n = add(self._body()["clip"])
            return self._json({"edge_cases": n})
        if self.path == "/api/review/reset":
            rp = os.path.join(HERE, "data", "reviews.jsonl")
            if os.path.exists(rp): os.replace(rp, rp.replace(".jsonl", f"-{int(time.time())}.jsonl.bak"))   # keep a backup
            from evaluate import evaluate
            return self._json(evaluate())
        if self.path == "/api/review":
            b = self._body()
            if b.get("verdict") not in ("correct", "wrong", "clear"): return self._json({"error": "bad verdict"}, 400)
            with open(os.path.join(HERE, "data", "reviews.jsonl"), "a") as f:
                f.write(json.dumps({"id": b["id"], "verdict": b["verdict"], "t": time.time()}) + "\n")
            from evaluate import evaluate
            return self._json(evaluate())
        if self.path == "/api/export":
            body = self._body()
            if not EXPORT_LOCK.acquire(blocking=False): return self._json({"error": "an export is already running"}, 409)
            def run():
                try:
                    from export import export
                    EXPORT_STATE.update(running=True, step="Packaging clips and labels, logging to W&B", result=None, error=None)
                    res = export(float(body.get("min_value", 5)), body.get("categories") or None, bool(body.get("with_video", False)),
                                 only_confirmed=bool(body.get("only_confirmed", False)))
                    res["download"] = f"/api/datasets/{res['name']}.zip"
                    EXPORT_STATE.update(running=False, step="done", result=res)
                except Exception as e:
                    EXPORT_STATE.update(running=False, step="failed", error=str(e)[:300])
                finally: EXPORT_LOCK.release()
            threading.Thread(target=run, daemon=True).start()
            return self._json({"started": True})
        return self._json({"error": "not found"}, 404)

if __name__ == "__main__":
    print(f"Haystack on http://0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
