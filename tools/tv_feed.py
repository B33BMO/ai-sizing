"""Live ai-sizing dashboard on the living-room TV (runs on srv under pm2 as `memlm-tv`).

Every INTERVAL seconds: read a JSON snapshot from the GPU box (memlm/agent/tv_status.py over wssh) and push it
to the tvcast server. `--layout` (re)installs the dashboard scene first.

  python3 tools/tv_feed.py --layout        # once: put the dashboard on the TV
  pm2 start tools/tv_feed.py --name memlm-tv --interpreter python3
"""
import json
import subprocess
import sys
import time
import urllib.request

TV = "http://127.0.0.1:8770"
INTERVAL = 30
LAYOUT = {
    "type": "dash",
    "title": "ai-sizing · small model learning Spero",
    "columns": 4,
    "widgets": [
        {"id": "job", "type": "text", "title": "Now running", "span": 2},
        {"id": "prog", "type": "progress", "title": "Progress", "unit": ""},
        {"id": "best", "type": "stat", "title": "Best 4B · Spero QA", "unit": "%"},
        {"id": "scores", "type": "bar", "title": "Scoreboard · 500 held-out questions (%)", "span": 2, "rows": 2},
        {"id": "gpu", "type": "line", "title": "RTX 5070 Ti", "span": 2, "min": 0, "max": 100, "window": 120},
        {"id": "vram", "type": "progress", "title": "VRAM", "unit": " GB"},
        {"id": "events", "type": "log", "title": "Finished runs"},
    ],
}


def post(path, body):
    req = urllib.request.Request(TV + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=15).read()


def snapshot():
    out = subprocess.run(["wssh", "ssh", "bmo@windows", "python3 ~/memlm2/agent/tv_status.py"],
                         capture_output=True, text=True, timeout=60).stdout
    return json.loads(out.strip().splitlines()[-1])


def short(line):
    """'FINAL runs/agent/r3b_tools.jsonl: 58.8% on 500 questions' -> 'r3b · 58.8%'"""
    import re
    m = re.search(r"runs/agent/([\w.]+?)(?:_tools)?\.jsonl: ([\d.]+%)", line)
    return f"{m.group(1)} · {m.group(2)}" if m else line[:40]


def main():
    if "--layout" in sys.argv:
        post("/scene", LAYOUT)
        try:
            state = json.load(urllib.request.urlopen(TV + "/state", timeout=10))
            if not state.get("stages"):
                post("/cast", {})
        except Exception:
            pass
        if "--once" not in sys.argv:
            return
    seen, first = set(), True
    while True:
        try:
            s = snapshot()
            g = s.get("gpu", {})
            label = time.strftime("%H:%M")
            best = s["scores"].get("4B round 3b")
            data = {
                "job": s["job"],
                "prog": {"value": s["progress"]["value"], "max": max(1, s["progress"]["max"])},
                "best": {"value": best, "note": f"vs GLM-5.3-flash {s['scores'].get('GLM-5.3-flash')}%"}
                if best else None,
                "scores": [[k, v] for k, v in s["scores"].items()],
                "vram": {"value": g.get("mem_gb", 0), "max": g.get("mem_total_gb", 16),
                         "status": "warning" if g.get("mem_gb", 0) > 14.5 else None},
            }
            post("/data", {k: v for k, v in data.items() if v is not None})
            if g:
                post("/data", {"gpu": {"push": g["util"], "series": "util %", "label": label}})
                post("/data", {"gpu": {"push": g["temp"], "series": "temp °C", "label": label}})
            finals = s.get("finals", [])
            if first:  # history: show the last 4 quietly, no toasts
                post("/data", {"events": [short(l) for l in finals[-4:]]})
                seen.update(finals)
                first = False
            for line in finals:
                if line not in seen:  # finished while we were watching: log it and pop a toast
                    post("/data", {"events": {"push": short(line)}})
                    post("/toast", {"text": "Finished: " + short(line), "level": "good", "seconds": 10})
                    seen.add(line)
        except Exception as e:  # GPU box asleep / network blip: keep the last picture, try again
            print("feed error:", e, flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
