"""Near-Miss Detective web app. Stdlib only.  Run: python3 server.py  (port 8080)"""
import json, os, urllib.parse, urllib.request, shutil
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from vss import BACKEND, token

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", 8080))

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=os.path.join(HERE, "static"), **k)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/incidents":
            body = open(os.path.join(HERE, "data", "incidents.json"), "rb").read()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if u.path == "/api/stream":
            src = urllib.parse.parse_qs(u.query).get("source", [""])[0]
            url = f"{BACKEND}/api/v1/videos/stream?" + urllib.parse.urlencode({"source": src, "token": token()})
            req = urllib.request.Request(url)
            if self.headers.get("Range"): req.add_header("Range", self.headers["Range"])
            try:
                r = urllib.request.urlopen(req, timeout=120)
            except urllib.error.HTTPError as e:
                r = e
            self.send_response(r.status if hasattr(r, "status") else r.code)
            self.send_header("Content-Type", "video/mp4")
            for h in ("Content-Length", "Content-Range", "Accept-Ranges"):
                if r.headers.get(h): self.send_header(h, r.headers[h])
            self.end_headers()
            try: shutil.copyfileobj(r, self.wfile)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        return super().do_GET()

if __name__ == "__main__":
    print(f"Near-Miss Detective on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
