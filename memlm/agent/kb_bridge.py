"""Expose Spero's tools (kb-serve on srv) to the GPU box.

Runs on the Mac: one long-lived kb-serve over `wssh ssh bmo@srv`, served on localhost:7777 as
newline-delimited JSON; `wssh ssh -N -R 7777:localhost:7777 bmo@windows` makes it reachable there.

  python3 agent/kb_bridge.py &
  ~/wssh/wssh ssh -N -R 7777:localhost:7777 bmo@windows &
"""
import json
import os
import socketserver
import subprocess
import threading

WSSH = os.path.expanduser("~/wssh/wssh")
CMD = ("cd /mnt/d/spero/365helper && SAGE_DB_PATH=$HOME/memlm-spero/sage.db "
       "exec npx tsx $HOME/memlm-spero/kb-serve.ts 2>/dev/null")
kb = subprocess.Popen([WSSH, "ssh", "bmo@srv", CMD], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
lock = threading.Lock()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            if not line.strip():
                continue
            with lock:  # kb-serve answers in order; keep request/response pairs together
                kb.stdin.write(line.decode().strip() + "\n")
                kb.stdin.flush()
                resp = kb.stdout.readline()
            self.wfile.write(resp.encode())
            self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    print("kb bridge on localhost:7777", flush=True)
    Server(("127.0.0.1", 7777), Handler).serve_forever()
