"""Shared helpers: team config, VSS backend login, and W&B LLM calls (stdlib only)."""
import glob, json, os, time, urllib.request

def load_env():
    env = dict(os.environ)
    cfg = glob.glob("/config/*.config")
    if len(cfg) == 1:
        for line in open(cfg[0]):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k, v.strip().strip('"').strip("'"))
    return env

ENV = load_env()
BACKEND = ENV["INGRESS_URL"].rstrip("/")
LLM_MODEL = os.environ.get("NEARMISS_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")

def _req(url, body=None, headers=None, timeout=120):
    h = {"Content-Type": "application/json", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    return json.load(urllib.request.urlopen(urllib.request.Request(url, data, h), timeout=timeout))

_tok = {"v": None, "t": 0}
def token():
    if not _tok["v"] or time.time() - _tok["t"] > 1800:
        _tok["v"] = _req(BACKEND + "/api/v1/auth/login",
                         {"username": ENV["USERNAME"], "password": ENV["PASSWORD"]})["access_token"]
        _tok["t"] = time.time()
    return _tok["v"]

def api(path, body=None):
    return _req(BACKEND + path, body, {"Authorization": "Bearer " + token()})

def llm_json(system, user):
    r = _req("https://api.inference.wandb.ai/v1/chat/completions",
             {"model": LLM_MODEL, "temperature": 0.1,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
             {"Authorization": "Bearer " + ENV["WANDB_API_KEY"],
              "OpenAI-Project": f'{ENV["WANDB_TEAM"]}/{ENV["WANDB_PROJECT"]}'})
    txt = r["choices"][0]["message"]["content"].strip()
    if txt.startswith("```"):
        txt = txt.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(txt)
