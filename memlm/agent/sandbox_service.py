"""Code sandbox service for the coding/bash skill (runs on srv under pm2 as `memlm-sandbox`).

Every episode gets its own throwaway Docker container: no network, read-only root filesystem, writable tmpfs
/work and /tmp, non-root user, all capabilities dropped, no-new-privileges, capped memory/CPU/PIDs. Containers
are removed on `destroy` and reaped after IDLE_S of inactivity. Same newline-JSON-over-TCP protocol as the Spero
tool service, LAN-only.

  {"op": "create", "files": {"path": "content", ...}}  -> {"ok": true, "id": "..."}
  {"op": "exec", "id": "...", "cmd": "pytest -q"}      -> {"ok": true, "text": "<stdout+stderr>", "code": 0}
  {"op": "write", "id": "...", "path": "x.py", "content": "..."} -> {"ok": true}
  {"op": "destroy", "id": "..."}                       -> {"ok": true}
"""
import base64
import json
import os
import socketserver
import subprocess
import threading
import time
import uuid

HOST = os.environ.get("SB_HOST", "192.168.1.248")
PORT = int(os.environ.get("SB_PORT", "7790"))
IMAGE = os.environ.get("SB_IMAGE", "memlm-sandbox:latest")
MAX_BOXES = int(os.environ.get("SB_MAX", "32"))
IDLE_S = 600
EXEC_TIMEOUT = 20
MAX_OUT = 4000
boxes = {}  # id -> last-used time
lock = threading.Lock()


def docker(*args, timeout=30, stdin=None):
    return subprocess.run(["docker", *args], capture_output=True, timeout=timeout, input=stdin)


def create(files):
    with lock:
        if len(boxes) >= MAX_BOXES:
            raise RuntimeError(f"sandbox limit reached ({MAX_BOXES})")
        bid = "memlm-sb-" + uuid.uuid4().hex[:12]
        boxes[bid] = time.time()
    for attempt in range(3):  # `docker run` can fail silently when srv is busy: retry with backoff
        r = _run(bid)
        if r.returncode == 0:
            break
        docker("rm", "-f", bid)
        time.sleep(1 + 2 * attempt)
    if r.returncode != 0:
        with lock:
            boxes.pop(bid, None)
        raise RuntimeError(f"docker run failed: {r.stderr.decode()[-300:] or 'no message'}")
    for path, content in files.items():
        write(bid, path, content)
    return bid


def _run(bid):
    return docker("run", "-d", "--rm", "--name", bid, "--network", "none", "--read-only",
               "--tmpfs", "/work:rw,exec,size=64m,uid=65534,gid=65534", "--tmpfs", "/tmp:rw,exec,size=64m",
               "--user", "65534:65534", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--memory", "512m", "--memory-swap", "512m", "--cpus", "1", "--pids-limit", "128",
               "-w", "/work", "-e", "HOME=/work", "-e", "PYTHONDONTWRITEBYTECODE=1", IMAGE, "sleep", "3600",
               timeout=60)


def write(bid, path, content):
    if path.startswith("/") or ".." in path.split("/"):
        raise ValueError("paths must be relative to /work")
    b64 = base64.b64encode(content.encode()).decode()
    cmd = f'mkdir -p "$(dirname "{path}")" && base64 -d > "{path}"'
    r = docker("exec", "-i", bid, "sh", "-c", cmd, stdin=b64.encode())
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-300:])


def execute(bid, cmd):
    try:
        r = docker("exec", bid, "timeout", "-k", "2", str(EXEC_TIMEOUT), "sh", "-c", cmd, timeout=EXEC_TIMEOUT + 10)
        out = (r.stdout + r.stderr).decode(errors="replace")
        code = r.returncode
    except subprocess.TimeoutExpired:
        out, code = "[timed out]", 124
    if code == 124 and not out:
        out = f"[timed out after {EXEC_TIMEOUT}s]"
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT // 2] + "\n[...truncated...]\n" + out[-MAX_OUT // 2:]
    return out, code


def destroy(bid):
    with lock:
        boxes.pop(bid, None)
    docker("rm", "-f", bid)


def handle(req):
    op = req.get("op")
    if op == "create":
        return {"ok": True, "id": create(req.get("files", {}))}
    bid = req.get("id", "")
    with lock:
        if bid not in boxes:
            raise KeyError("unknown sandbox")
        boxes[bid] = time.time()
    if op == "exec":
        text, code = execute(bid, req["cmd"])
        return {"ok": True, "text": text, "code": code}
    if op == "write":
        write(bid, req["path"], req["content"])
        return {"ok": True}
    if op == "destroy":
        destroy(bid)
        return {"ok": True}
    raise ValueError(f"unknown op {op}")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                resp = handle(req)
            except Exception as e:  # report to the caller, keep serving
                resp = {"ok": False, "text": f"{type(e).__name__}: {e}"}
            resp["id_req"] = req.get("rid") if isinstance(req, dict) else None
            self.wfile.write((json.dumps(resp) + "\n").encode())
            self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def reaper():
    while True:
        time.sleep(60)
        now = time.time()
        with lock:
            stale = [b for b, t in boxes.items() if now - t > IDLE_S]
        for b in stale:
            destroy(b)
        # also remove orphans from a previous run of this service
        r = docker("ps", "-q", "--filter", "name=memlm-sb-", "--format", "{{.Names}}")
        with lock:
            known = set(boxes)
        for name in r.stdout.decode().split():
            if name not in known:
                docker("rm", "-f", name)


if __name__ == "__main__":
    threading.Thread(target=reaper, daemon=True).start()
    print(f"memlm sandbox service on {HOST}:{PORT} (max {MAX_BOXES} boxes, image {IMAGE})", flush=True)
    Server((HOST, PORT), Handler).serve_forever()
