"""Always-on Spero tool service on srv (replaces the Mac bridge + tunnel for the GPU box).

Runs Spero's real searchDocs/exploreGraph (kb-serve.ts) against the DB snapshot and serves the same
newline-delimited JSON protocol over TCP. Bound to the LAN address only; ufw already trusts
192.168.1.0/24. Managed by pm2:

  pm2 start ~/memlm-spero/kb_service.py --name memlm-kb --interpreter python3 && pm2 save
Client: KB_ADDR=192.168.1.248:7787 (agent/student.py)
"""
import json
import os
import socketserver
import subprocess
import threading

HOST = os.environ.get("KB_HOST", "192.168.1.248")
PORT = int(os.environ.get("KB_PORT", "7787"))
SPERO = "/mnt/d/spero/365helper"
SNAPSHOT = os.path.expanduser("~/memlm-spero/sage.db")
lock = threading.Lock()
kb = None


def start_kb():
    global kb
    if kb is not None:
        kb.kill()
    env = dict(os.environ, SAGE_DB_PATH=SNAPSHOT)  # never the live DB: lib/db.ts opens read-write
    kb = subprocess.Popen(["npx", "tsx", os.path.expanduser("~/memlm-spero/kb-serve.ts")], cwd=SPERO, env=env,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, bufsize=1)
    print("kb-serve (re)started", flush=True)


def ask(line):
    """forward one request; restart kb-serve and retry once if it died"""
    for _ in range(2):
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
    return json.dumps({"id": json.loads(line).get("id"), "ok": False, "text": "tool server unavailable"}) + "\n"


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            if line.strip():
                with lock:  # one embedding model, one DB handle: serialize
                    resp = ask(line.decode().strip())
                self.wfile.write(resp.encode())
                self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    start_kb()
    print(f"memlm kb service on {HOST}:{PORT}", flush=True)
    Server((HOST, PORT), Handler).serve_forever()
