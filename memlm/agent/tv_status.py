"""One JSON snapshot of the GPU box for the TV dashboard (run on the box; read by tools/tv_feed.py on srv).

  {"job": str, "progress": {"value", "max"}, "gpu": {...}, "scores": {label: pct}, "finals": [lines]}
"""
import glob
import json
import os
import re
import subprocess

HOME = os.path.expanduser("~/memlm2")
A = f"{HOME}/runs/agent"
# Spero QA test runs worth showing (output file stem -> label), in display order
RUNS = [("base_closed", "4B, no tools"), ("base2_tools", "4B + tools"), ("basem_tools", "4B + memory (untrained)"),
        ("r2_tools", "4B round 2"), ("r3bf_tools", "4B round 3b"), ("r3b8pf_tools", "8B round 3b"),
        ("r3bmh_tools", "4B 3b + memory (hidden)")]
REFS = [("gemma-31B", 53.8), ("GLM-5.3-flash", 62.2), ("GLM-5.3", 66.0)]  # fair-graded, measured once


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def lines(path):
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def acc(path):
    try:
        rows = [json.loads(l) for l in open(path)]
    except OSError:
        return None, 0
    return (100 * sum(r["correct"] for r in rows) / len(rows) if rows else None), len(rows)


def job():
    procs = sh("pgrep -fa 'agent/(student|sft|engine).py' || true")
    for p in procs.splitlines():
        if "agent/sft.py" in p:
            logs = sorted(glob.glob(f"{HOME}/sft_*.log"), key=os.path.getmtime)
            st = re.findall(r"step (\d+)/(\d+) loss ([\d.]+)", open(logs[-1]).read()) if logs else []
            name = os.path.basename(logs[-1])[4:-4] if logs else "?"
            if st:
                d, t, loss = st[-1]
                return f"Training {name}\nloss {loss}", {"value": int(d), "max": int(t)}
            return f"Training {name}\nloading…", {"value": 0, "max": 1}
        m = re.search(r"--out (\S+)", p)
        if m:
            out = os.path.join(HOME, m.group(1))
            qs = re.search(r"--questions (\S+)", p)
            total = len(json.load(open(os.path.join(HOME, qs.group(1))))) if qs else 500
            att = re.search(r"--attempts (\d+)", p)
            if att and int(att.group(1)) > 1:
                files = [out.replace(".jsonl", f".a{k}.jsonl") for k in range(int(att.group(1)))]
                out = max((f for f in files if os.path.exists(f)), default=files[0], key=os.path.getmtime)
            a, n = acc(out)
            label = os.path.basename(out)[:-6].replace("_tools", "")
            return f"Evaluating {label}\n{a:.1f}% so far" if a is not None else f"Evaluating {label}", \
                {"value": n, "max": total}
    return "Idle\nGPU free", {"value": 0, "max": 1}


def gpu():
    q = sh("/usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw "
           "--format=csv,noheader,nounits")
    try:
        u, t, mu, mt, p = [float(x) for x in q.split(",")]
        return {"util": u, "temp": t, "mem_gb": round(mu / 1024, 1), "mem_total_gb": round(mt / 1024, 1), "power": p}
    except ValueError:
        return {}


scores = {}
for stem, label in RUNS:
    a, n = acc(f"{A}/{stem}.jsonl")
    if a is not None and n >= 500:
        scores[label] = round(a, 1)
for label, v in REFS:
    scores[label] = v
finals = sh(f"cd {HOME} && ls -tr *.log 2>/dev/null | xargs -r grep -h 'FINAL' | tail -12").splitlines()  # oldest log first
j, prog = job()
print(json.dumps({"job": j, "progress": prog, "gpu": gpu(), "scores": scores, "finals": finals}))
