"""Edge Case Miner web app (stdlib only).  Run: python3 server.py   -> http://localhost:8080"""
import json, os, shutil, threading, urllib.error, urllib.parse, urllib.request
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from vss import BACKEND, token

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", 8080))
EXPORT_LOCK = threading.Lock()

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=os.path.join(HERE, "static"), **k)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode() if not isinstance(obj, bytes) else obj
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/clips":
            p = os.path.join(HERE, "data", "edge_cases.json")
            return self._json(open(p, "rb").read() if os.path.exists(p) else b'{"clips":[],"funnel":{},"categories":{}}')
        if u.path.startswith("/api/boxes/"):
            bid = os.path.basename(u.path); p = os.path.join(HERE, "data", "boxes", bid + ".json")
            return self._json(open(p, "rb").read() if bid.isalnum() and os.path.exists(p) else b"[]")
        if u.path.startswith("/api/datasets/") and u.path.endswith(".zip"):
            name = os.path.basename(u.path)[:-4]
            p = os.path.join(HERE, "datasets", name + ".zip")
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
            self.send_response(getattr(r, "status", None) or r.code)
            self.send_header("Content-Type", "video/mp4")
            for h in ("Content-Length", "Content-Range", "Accept-Ranges"):
                if r.headers.get(h): self.send_header(h, r.headers[h])
            self.end_headers()
            try: shutil.copyfileobj(r, self.wfile)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/export": return self._json({"error": "not found"}, 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if not EXPORT_LOCK.acquire(blocking=False): return self._json({"error": "an export is already running"}, 409)
        try:
            from export import export
            res = export(float(body.get("min_value", 5)), body.get("categories") or None, bool(body.get("with_video", True)))
            res["download"] = f"/api/datasets/{res['name']}.zip"
            return self._json(res)
        except Exception as e:
            return self._json({"error": str(e)}, 500)
        finally:
            EXPORT_LOCK.release()

if __name__ == "__main__":
    print(f"Edge Case Miner on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
