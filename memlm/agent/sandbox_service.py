"""Code sandbox service for the coding/bash skill (runs on srv under pm2 as `memlm-sandbox`).

Every episode gets its own throwaway Docker container: no network, read-only root filesystem, writable tmpfs
/work and /tmp, non-root user, all capabilities dropped, no-new-privileges, capped memory/CPU/PIDs. Containers
are removed on `destroy` and reaped after IDLE_S of inactivity. Same newline-JSON-over-TCP protocol as the Spero
tool service, LAN-only.

  {"op": "create", "files": {"path": "content", ...}}  -> {"ok": true, "id": "..."}
  {"op": "exec", "id": "...", "cmd": "pytest -q"}      -> {"ok": true, "text": "<stdout+stderr>", "code": 0}
  {"op": "write", "id": "...", "path": "x.py", "content": "..."} -> {"ok": true}
  {"op": "read", "id": "...", "path": "x.py"}          -> {"ok": true, "content": "..."}  (untruncated, for edits)
  {"op": "destroy", "id": "..."}                       -> {"ok": true}

  Judgment eval (agent/judge_agent.py): {"op": "create", "profile": "judge", "files": {"/abs/path": ...}, "setup": "sh"}
  starts a simulated server instead (image memlm-judge, agent/judge/): still no network and capped, but root with a
  writable root filesystem and a few capabilities (kill, chown, setuid...) so services, users and files can be faked.
  "files" are written, then "setup" runs (bash, 120 s); exec runs bash as root in /root; exec takes "timeout" (<=120).
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
JUDGE_IMAGE = os.environ.get("SB_JUDGE_IMAGE", "memlm-judge:latest")
JUDGE_BUILD_DIR = os.path.expanduser("~/memlm-spero/judge")
boxes = {}  # id -> last-used time
profiles = {}  # id -> "code" | "judge"
lock = threading.Lock()
build_lock = threading.Lock()


def docker(*args, timeout=30, stdin=None):
    return subprocess.run(["docker", *args], capture_output=True, timeout=timeout, input=stdin)


def create(files, profile="code", setup=None):
    with lock:
        if len(boxes) >= MAX_BOXES:
            raise RuntimeError(f"sandbox limit reached ({MAX_BOXES})")
        bid = "memlm-sb-" + uuid.uuid4().hex[:12]
        boxes[bid] = time.time()
        profiles[bid] = profile
    for attempt in range(3):  # `docker run` can fail silently when srv is busy: retry with backoff
        if attempt == 0 or b"Unable to find image" in r.stderr:
            with build_lock:
                ensure_image(*((JUDGE_IMAGE, JUDGE_BUILD_DIR) if profile == "judge" else (IMAGE, BUILD_DIR)))
        r = _run_judge(bid) if profile == "judge" else _run(bid)
        if r.returncode == 0:
            break
        docker("rm", "-f", bid)
        time.sleep(1 + 2 * attempt)
    if r.returncode != 0:
        with lock:
            boxes.pop(bid, None)
            profiles.pop(bid, None)
        raise RuntimeError(f"docker run failed: {r.stderr.decode()[-300:] or 'no message'}")
    for path, content in files.items():
        write(bid, path, content)
    if setup:
        out, code = execute(bid, setup, timeout=120)
        if code != 0:
            destroy(bid, "setup failed")
            raise RuntimeError(f"setup failed ({code}): {out[-500:]}")
    print(f"{time.strftime('%H:%M:%S')} create {bid} ({profile})", flush=True)
    return bid


BUILD_DIR = os.path.expanduser("~/memlm-spero/sandbox")  # Dockerfile for IMAGE


def ensure_image(image=IMAGE, build_dir=BUILD_DIR):
    """images on srv have been pruned before (docker image/system prune -a): rebuild if missing"""
    if docker("image", "inspect", image).returncode != 0:
        print(f"{time.strftime('%H:%M:%S')} image {image} missing: rebuilding", flush=True)
        r = docker("build", "-q", "-t", image, build_dir, timeout=900)
        if r.returncode != 0:
            raise RuntimeError(f"rebuild failed: {r.stderr.decode()[-300:]}")


def _run(bid):
    return docker("run", "-d", "--rm", "--name", bid, "--network", "none", "--read-only",
               "--tmpfs", "/work:rw,exec,size=64m,uid=65534,gid=65534", "--tmpfs", "/tmp:rw,exec,size=64m",
               "--user", "65534:65534", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--memory", "512m", "--memory-swap", "512m", "--cpus", "1", "--pids-limit", "128",
               "-w", "/work", "-e", "HOME=/work", "-e", "PYTHONDONTWRITEBYTECODE=1", IMAGE, "sleep", "3600",
               timeout=60)


def _run_judge(bid):
    caps = [a for c in ("CHOWN", "DAC_OVERRIDE", "FOWNER", "KILL", "SETUID", "SETGID",
                                         "NET_BIND_SERVICE", "SYS_PTRACE")
            for a in ("--cap-add", c)]
    return docker("run", "-d", "--rm", "--name", bid, "--hostname", "web01", "--network", "none",
                  "--tmpfs", "/data:rw,exec,size=256m", "--tmpfs", "/tmp:rw,exec,size=64m", "--tmpfs", "/run:rw,size=8m",
                  "--user", "0:0", "--cap-drop", "ALL", *caps, "--security-opt", "no-new-privileges",
                  "--memory", "768m", "--memory-swap", "768m", "--cpus", "1", "--pids-limit", "256",
                  "-w", "/root", "-e", "HOME=/root", "-e", "TZ=UTC", "-e", "SIM_EXIT_AFTER=3600", JUDGE_IMAGE,
                  "bash", "-c", "exec -a /sbin/init /usr/local/lib/opsd/simd", timeout=60)


def write(bid, path, content):
    if profiles.get(bid) == "judge":  # a whole simulated server: absolute paths allowed (still inside the box)
        if not path.startswith("/"):
            path = "/root/" + path
    elif path.startswith("/") or ".." in path.split("/"):
        raise ValueError("paths must be relative to /work")
    b64 = base64.b64encode(content.encode()).decode()
    cmd = f'mkdir -p "$(dirname "{path}")" && base64 -d > "{path}"'
    r = docker("exec", "-i", bid, "sh", "-c", cmd, stdin=b64.encode())
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-300:])


def read(bid, path):
    if profiles.get(bid) == "judge":
        if not path.startswith("/"):
            path = "/root/" + path
    elif path.startswith("/") or ".." in path.split("/"):
        raise ValueError("paths must be relative to /work")
    r = docker("exec", bid, "head", "-c", "200000", path)
    if r.returncode != 0:
        raise FileNotFoundError(r.stderr.decode()[-300:].strip())
    return r.stdout.decode(errors="replace")


def execute(bid, cmd, timeout=EXEC_TIMEOUT):
    shell = "bash" if profiles.get(bid) == "judge" else "sh"
    try:
        r = docker("exec", bid, "timeout", "-k", "2", str(timeout), shell, "-c", cmd, timeout=timeout + 10)
        out = (r.stdout + r.stderr).decode(errors="replace")
        code = r.returncode
    except subprocess.TimeoutExpired:
        out, code = "[timed out]", 124
    if code == 124 and not out:
        out = f"[timed out after {timeout}s]"
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT // 2] + "\n[...truncated...]\n" + out[-MAX_OUT // 2:]
    return out, code


def destroy(bid, why="client"):
    print(f"{time.strftime('%H:%M:%S')} destroy {bid} ({why})", flush=True)
    with lock:
        boxes.pop(bid, None)
        profiles.pop(bid, None)
    docker("rm", "-f", bid)


def handle(req):
    op = req.get("op")
    if op == "create":
        return {"ok": True, "id": create(req.get("files", {}), req.get("profile", "code"), req.get("setup"))}
    bid = req.get("id", "")
    with lock:
        if bid not in boxes:
            raise KeyError("unknown sandbox")
        boxes[bid] = time.time()
    if op == "exec":
        text, code = execute(bid, req["cmd"], min(int(req.get("timeout", EXEC_TIMEOUT)), 120))
        return {"ok": True, "text": text, "code": code}
    if op == "write":
        write(bid, req["path"], req["content"])
        return {"ok": True}
    if op == "read":
        return {"ok": True, "content": read(bid, req["path"])}
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
            destroy(b, "idle")
        # also remove orphans from a previous run of this service
        # NB: no -q here: `-q` prints container IDs and ignores --format, so every live box looked like an orphan
        r = docker("ps", "--filter", "name=memlm-sb-", "--format", "{{.Names}}")
        with lock:
            known = set(boxes)
        for name in r.stdout.decode().split():
            if name not in known:
                print(f"{time.strftime('%H:%M:%S')} reap orphan {name}", flush=True)
                docker("rm", "-f", name)


if __name__ == "__main__":
    threading.Thread(target=reaper, daemon=True).start()
    print(f"memlm sandbox service on {HOST}:{PORT} (max {MAX_BOXES} boxes, image {IMAGE})", flush=True)
    Server((HOST, PORT), Handler).serve_forever()
