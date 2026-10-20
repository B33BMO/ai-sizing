"""Run the devlog CLI when the Mac is off the LAN/Tailscale: each HTTP request is sent as JSON over
`wssh ssh bmo@srv` stdin to a tiny relay that calls claude-docs on 127.0.0.1:4410. The token only ever
travels inside that stdin (never in argv or a file on srv).

  python3 devlog_via_wssh.py add ai-sizing body.md --title ...
"""
import io
import json
import os
import runpy
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit

DEVLOG = os.path.expanduser("~/.claude/skills/devlog/devlog")
WSSH = os.path.expanduser("~/wssh/wssh")
RELAY = r'''
import sys, json, urllib.request, urllib.error
e = json.load(sys.stdin)
req = urllib.request.Request("http://127.0.0.1:4410" + e["path"], method=e["method"], headers=e["headers"],
                             data=e["data"].encode() if e["data"] is not None else None)
try:
    r = urllib.request.urlopen(req, timeout=30); out = {"code": r.status, "body": r.read().decode()}
except urllib.error.HTTPError as x:
    out = {"code": x.code, "body": x.read().decode()}
print(json.dumps(out))
'''


def urlopen(req, timeout=30):
    u = urlsplit(req.full_url)
    env = {"method": req.get_method(), "path": u.path + (f"?{u.query}" if u.query else ""),
           "headers": dict(req.header_items()), "data": req.data.decode() if req.data else None}
    p = subprocess.run([WSSH, "ssh", "bmo@srv", "python3 -c " + shlex.quote(RELAY)], input=json.dumps(env),
                       capture_output=True, text=True, timeout=120)
    out = json.loads(p.stdout.strip().splitlines()[-1])
    if out["code"] >= 400:
        raise urllib.error.HTTPError(req.full_url, out["code"], "error", {}, io.BytesIO(out["body"].encode()))
    r = io.BytesIO(out["body"].encode())
    r.status = out["code"]
    return r


urllib.request.urlopen = urlopen
sys.argv = [DEVLOG] + sys.argv[1:]
runpy.run_path(DEVLOG, run_name="__main__")
