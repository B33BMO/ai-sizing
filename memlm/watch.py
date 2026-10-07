"""Live terminal dashboard for memlm experiments.   python watch.py   (Ctrl+C to quit)

Reads ~/memlm and ~/memlm2: the run order from exp*.sh, progress/metrics from runs/*/log.jsonl.
"""
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# (working dir, experiment script) in the order they run
PLANS = [(os.path.expanduser("~/memlm"), "exp1.sh"), (os.path.expanduser("~/memlm2"), "exp2.sh"),
         (os.path.expanduser("~/memlm2"), "exp3.sh"), (os.path.expanduser("~/memlm2"), "exp4.sh")]
SPARK = "▁▂▃▄▅▆▇█"
# rough throughput for runs that have not started yet (tokens/s, measured in smoke tests)
EST_TPS = {"base": 160e3, "mem": 125e3, "chain": 105e3, "ptr": 95e3}
CHANCE = 1 / 16


def spark(vals, width=40):
    vals = [v for v in vals if v == v]
    if len(vals) > width:  # downsample evenly so the whole run fits
        vals = [vals[int(i * len(vals) / width)] for i in range(width)]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    return "".join(SPARK[int((v - lo) / (hi - lo + 1e-12) * (len(SPARK) - 1))] for v in vals)


def gpu():
    try:
        out = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        u, mu, mt, t, p = [x.strip() for x in out.split(",")]
        return f"GPU {u}%  ·  VRAM {int(mu)/1024:.1f}/{int(mt)/1024:.1f} GB  ·  {t}°C  ·  {float(p):.0f} W"
    except Exception as e:
        return f"GPU: n/a ({e.__class__.__name__})"


def running_names():
    out = subprocess.run(["pgrep", "-fa", "train.py"], capture_output=True, text=True).stdout
    return set(re.findall(r"--name (\S+)", out))


def plan():
    """[(experiment, root, run_name, kind)] in launch order, from the exp*.sh scripts."""
    runs = []
    for root, script in PLANS:
        script = f"{root}/{script}"
        if os.path.exists(script):
            src = open(script).read()
            env = dict(re.findall(r"^(\w+)=\"([^\"]*)\"", src, re.M))
            for line in re.findall(r"^run (\S+.*)$", src, re.M):
                name = line.split()[0]
                args = re.sub(r"\$(\w+)", lambda m: env.get(m.group(1), ""), line)
                kind = ("ptr" if "--mem_pointer table" in args else "chain" if "--mem_hops 2" in args
                        else "mem" if "--mem_layers" in args else "base")
                runs.append((os.path.basename(script)[:-3], root, name, kind))
    return runs


def load(root, name):
    path = f"{root}/runs/{name}/log.jsonl"
    if not os.path.exists(path):
        return None
    head, steps, evals = None, [], []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # line being written
            if "args" in r:
                head = r
            elif "eval" in r:
                evals.append(r["eval"])
            else:
                steps.append(r)
    if head is None:  # run just started: header still in train.py's write buffer
        return None
    a = head["args"]
    total = int(a["tokens"] // (a["batch_size"] * a["grad_accum"] * a["seq_len"]))
    return {"args": a, "total": total, "steps": steps, "evals": evals, "mtime": os.path.getmtime(path),
            "done": os.path.exists(f"{root}/runs/{name}/results.json")}


def error_tail(root, name):
    p = f"{root}/runs_{name}.log"
    if not os.path.exists(p):
        return None
    txt = open(p, errors="replace").read()[-4000:]
    m = re.findall(r"^\w*(?:Error|Exception)[^\n]*", txt, re.M)
    return m[-1][:120] if m else None


def recent_tps(d):
    # median of recent readings: the step after an eval and WSL clock jumps give junk values
    v = sorted(s["tok_s"] for s in d["steps"][-15:] if 0 < s.get("tok_s", 0) < 1e6)
    return v[len(v) // 2] if v else 0


def smooth(vals, a=0.1):
    out, m = [], None
    for v in vals:
        m = v if m is None else (1 - a) * m + a * v
        out.append(m)
    return out


def pct(v):
    return "–" if v is None or v != v else f"{v*100:.1f}%"


def fmt_eta(sec):
    if sec is None:
        return "–"
    h, m = divmod(int(sec) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def build():
    live = running_names()
    rows, active, errors = [], None, []
    clock = 0.0  # seconds of work ahead of the current moment, to chain ETAs
    for exp, root, name, kind in plan():
        d = load(root, name)
        st = d["steps"][-1] if d and d["steps"] else {}
        ev = d["evals"][-1] if d and d["evals"] else {}
        if d and d["done"]:
            status, prog, eta = "[green]done", 1.0, None
        elif name in live and not d:
            status, prog, eta = "[yellow]starting", 0.0, None
        elif name in live:
            status = "[bold yellow]running"
            prog = st.get("step", 0) / d["total"] if d else 0
            tps = recent_tps(d)
            left = (d["total"] - st.get("step", 0)) * d["args"]["batch_size"] * d["args"]["grad_accum"] * d["args"]["seq_len"]
            clock += left / tps if tps > 0 else 0
            eta, active = clock, (exp, root, name, d)
        elif d and not d["done"] and time.time() - d["mtime"] > 120:
            status, prog, eta = "[red]stopped", st.get("step", 0) / d["total"], None
            err = error_tail(root, name)
            if err:
                errors.append(f"{name}: {err}")
        else:
            status, prog = "[dim]queued", 0.0
            clock += 500e6 / EST_TPS[kind]
            eta = clock
        bar = "█" * int(prog * 14) + "░" * (14 - int(prog * 14))
        rows.append((exp, name, kind, status, f"{bar} {prog*100:4.0f}%", fmt_eta(eta),
                     f"{recent_tps(d)/1e3:.0f}k" if d and name in live else "–",
                     f"{st['loss']:.3f}" if "loss" in st else "–",
                     f"{ev['val_loss']:.3f}" if "val_loss" in ev else "–",
                     pct(ev.get("acc_1hop")), pct(ev.get("acc_twohop_seen")), pct(ev.get("acc_twohop_unseen")),
                     pct(ev.get("acc_twohop_unseen_given_both")), pct(ev.get("spero_cloze_val")),
                     pct(ev.get("cot_unseen_final"))))

    t = Table(box=box.SIMPLE_HEAVY, expand=True, header_style="bold cyan")
    for col, j in [("exp", "left"), ("run", "left"), ("type", "left"), ("status", "left"), ("progress", "left"), ("ETA", "right"),
                   ("tok/s", "right"), ("loss", "right"), ("val", "right"), ("1-hop", "right"), ("2hop seen", "right"),
                   ("2hop UNSEEN", "right"), ("unseen|both", "right"), ("spero cloze", "right"),
                   ("CoT unseen", "right")]:
        t.add_column(col, justify=j, no_wrap=True)
    for r in rows:
        t.add_row(*r)

    parts = [t]
    if active:
        exp, root, name, d = active
        ev = d["evals"]
        det = Table.grid(padding=(0, 2))
        det.add_column(style="bold")
        det.add_column()
        det.add_column(style="dim")
        def line(label, key, f=pct, src=ev, sm=False):
            vals = [e.get(key, float("nan")) for e in src]
            shown = smooth(vals) if sm else vals
            det.add_row(label, spark(shown, 40), f(shown[-1]) if vals else "–")
        line("train loss (smoothed)", "loss", lambda v: f"{v:.3f}", d["steps"], sm=True)
        line("web val loss", "val_loss", lambda v: f"{v:.3f}")
        if any("spero_val_loss" in e for e in ev):
            line("spero val loss", "spero_val_loss", lambda v: f"{v:.3f}")
            line("spero cloze (unseen docs)", "spero_cloze_val")
        line("1-hop facts", "acc_1hop")
        line("2-hop seen", "acc_twohop_seen")
        line("2-hop UNSEEN", "acc_twohop_unseen")
        if any("cot_unseen_final" in e for e in ev):
            line("CoT unseen: mentor step", "cot_unseen_step1")
            line("CoT unseen: final answer", "cot_unseen_final")
        if any("graph_loss" in s for s in d["steps"]):
            line("graph loss (smoothed)", "graph_loss", lambda v: f"{v:.3f}", d["steps"], sm=True)
        parts.append(Panel(det, title=f"[bold]{name}[/] trends (each tick = one eval)", border_style="yellow"))
    for e in errors:
        parts.append(Text(f"⚠ {e}", style="bold red"))
    finish = (datetime.now() + timedelta(seconds=clock)).strftime("%a %H:%M") if clock else "–"
    head = Text.assemble(("memlm  ", "bold magenta"), (gpu(), "white"), ("   ·   all done ≈ ", "dim"), (finish, "bold"),
                         ("   ·   chance = 6.2%   ·   ", "dim"), (datetime.now().strftime("%H:%M:%S"), "dim"))
    return Panel(Group(*parts), title=head, border_style="magenta")


if __name__ == "__main__":
    console = Console()
    try:
        with Live(build(), console=console, refresh_per_second=1, screen=True) as lv:
            while True:
                time.sleep(5)
                lv.update(build())
    except KeyboardInterrupt:
        pass
