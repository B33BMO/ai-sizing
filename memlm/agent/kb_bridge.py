"""Expose Spero's tools (kb-serve on srv) to the GPU box.

Runs on the Mac: one long-lived kb-serve over `wssh ssh bmo@srv`, served on localhost:7777 as
newline-delimited JSON; `wssh ssh -N -R 7777:localhost:7777 bmo@windows` makes it reachable there.

  python3 agent/kb_bridge.py &
  agent/tunnel.sh &        # keeps `wssh ssh -N -R 7777:localhost:7777 bmo@windows` alive
"""
import json
import os
import shutil
import socketserver
import subprocess
import threading

WSSH = shutil.which("wssh") or os.path.expanduser("~/wssh/wssh")  # srv: on PATH; Mac: zsh alias
CMD = ("cd /mnt/d/spero/365helper && SAGE_DB_PATH=$HOME/memlm-spero/sage.db "
       "exec npx tsx $HOME/memlm-spero/kb-serve.ts 2>/dev/null")
lock = threading.Lock()
kb = None


def start_kb():
    global kb
    if kb is not None:
        kb.kill()
    kb = subprocess.Popen([WSSH, "ssh", "-o", "ServerAliveInterval=15", "bmo@srv", CMD],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    print("kb-serve (re)started", flush=True)


def ask(line):
    """forward one request; if the srv session dropped, restart it and retry once"""
    for attempt in range(2):
        try:
            if kb is None or kb.poll() is not None:
                start_kb()
            kb.stdin.write(line + "\n")
            kb.stdin.flush()
            resp = kb.stdout.readline()
            if resp:
                return resp
        except (BrokenPipeError, OSError):
            pass
        start_kb()
    rid = json.loads(line).get("id")
    return json.dumps({"id": rid, "ok": False, "text": "tool server unavailable"}) + "\n"


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            if not line.strip():
                continue
            with lock:  # kb-serve answers in order; keep request/response pairs together
                resp = ask(line.decode().strip())
            self.wfile.write(resp.encode())
            self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    start_kb()
    print("kb bridge on localhost:7777", flush=True)
    Server(("127.0.0.1", 7777), Handler).serve_forever()
