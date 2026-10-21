"""Always-on Spero tool service on srv (replaces the Mac bridge + tunnel for the GPU box).

Runs Spero's real searchDocs/exploreGraph (kb-serve.ts) against the DB snapshot and serves the same
newline-delimited JSON protocol over TCP. Bound to the LAN address only; ufw already trusts
192.168.1.0/24. Managed by pm2:

  pm2 start ~/memlm-spero/kb_service.py --name memlm-kb --interpreter python3 && pm2 save
Client: KB_ADDR=192.168.1.248:7787 (agent/student.py)
"""
import json
import os
import queue
import socketserver
import subprocess
import threading

HOST = os.environ.get("KB_HOST", "192.168.1.248")
PORT = int(os.environ.get("KB_PORT", "7787"))
SPERO = "/mnt/d/spero/365helper"
SNAPSHOT = os.path.expanduser("~/memlm-spero/sage.db")
WORKERS = int(os.environ.get("KB_WORKERS", "1"))  # parallel kb-serve processes (each loads the embedder)


class Worker:
    """one kb-serve process; answers its requests in order"""
    def __init__(self, i):
        self.i, self.p = i, None

    def start(self):
        if self.p is not None:
            self.p.kill()
        env = dict(os.environ, SAGE_DB_PATH=SNAPSHOT)  # never the live DB: lib/db.ts opens read-write
        self.p = subprocess.Popen(["npx", "tsx", os.path.expanduser("~/memlm-spero/kb-serve.ts")], cwd=SPERO, env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)
        print(f"kb-serve worker {self.i} (re)started", flush=True)

    def ask(self, line):
        """forward one request; restart this worker and retry once if it died"""
        for _ in range(2):
            try:
                if self.p is None or self.p.poll() is not None:
                    self.start()
                self.p.stdin.write(line + "\n")
                self.p.stdin.flush()
                resp = self.p.stdout.readline()
                if resp:
                    return resp
            except (BrokenPipeError, OSError):
                pass
            self.start()
        return json.dumps({"id": json.loads(line).get("id"), "ok": False, "text": "tool server unavailable"}) + "\n"


pool = queue.Queue()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            if line.strip():
                w = pool.get()  # a free worker (blocks while all are busy)
                try:
                    resp = w.ask(line.decode().strip())
                finally:
                    pool.put(w)
                self.wfile.write(resp.encode())
                self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    for i in range(WORKERS):
        w = Worker(i)
        w.start()
        pool.put(w)
    print(f"memlm kb service on {HOST}:{PORT} with {WORKERS} worker(s)", flush=True)
    Server((HOST, PORT), Handler).serve_forever()
