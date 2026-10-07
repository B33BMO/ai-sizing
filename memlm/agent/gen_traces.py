"""Teacher tool-use traces against Spero's real retrieval (stdlib only; runs on srv).

  python3 gen_traces.py questions.json traces.jsonl [--workers 6] [--limit N]
Spawns kb-serve.ts (Spero's searchDocs/exploreGraph on a DB snapshot) and lets the teacher LLM
answer each question by calling <search>/<graph> until it emits <answer>.
Each output line: {"q", "gold", "steps": [{"text", "tool", "arg", "result"}], "answer", "correct", "turns"}
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import threading
import urllib.request

from protocol import SYSTEM, grade, render_result

ap = argparse.ArgumentParser()
ap.add_argument("questions")
ap.add_argument("out")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--endpoint", default="https://llama2.coleman-it.com/v1/chat/completions")
ap.add_argument("--max_turns", type=int, default=5)
args = ap.parse_args()

SPERO = "/mnt/d/spero/365helper"
env = dict(os.environ, SAGE_DB_PATH=os.path.expanduser("~/memlm-spero/sage.db"))
kb = subprocess.Popen(["npx", "tsx", os.path.expanduser("~/memlm-spero/kb-serve.ts")], cwd=SPERO, env=env,
                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
kb_lock = threading.Lock()
req_id = [0]


def tool(kind, arg):
    with kb_lock:
        req_id[0] += 1
        req = {"id": req_id[0], "op": kind, **({"q": arg, "k": 4} if kind == "search" else {"entity": arg})}
        kb.stdin.write(json.dumps(req) + "\n")
        kb.stdin.flush()
        return json.loads(kb.stdout.readline())["text"]


def llm(messages):
    body = json.dumps({"model": "x", "max_tokens": 400, "temperature": 0.4, "messages": messages,
                       "stop": ["</search>", "</graph>", "</answer>"],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(args.endpoint, body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"] or ""


JUNK = re.compile(r"<\|?channel\|?>|</?thought>?|^\s*thought\b", re.M)


def clean(text):
    """strip the teacher's leaked chat-template / thinking markup, keep its plain reasoning"""
    return re.sub(r"[ \t]+", " ", JUNK.sub("", text)).strip()


def run(item):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": item["q"]}]
    steps, answer = [], None
    for turn in range(args.max_turns):
        out = clean(llm(msgs))
        opened = re.findall(r"<(search|graph|answer)>", out)
        if not opened:  # one nudge, then give up
            msgs += [{"role": "assistant", "content": out},
                     {"role": "user", "content": "Use the tags: make one <search>, <graph> call, or give your <answer>."}]
            out = clean(llm(msgs))
            msgs = msgs[:-2]
            opened = re.findall(r"<(search|graph|answer)>", out)
            if not opened:
                break
        kind = opened[-1]
        arg = out.rsplit(f"<{kind}>", 1)[1].strip()
        text = out + f"</{kind}>"
        if kind == "answer" or turn == args.max_turns - 1:
            answer = arg if kind == "answer" else None
            steps.append({"text": text, "tool": "answer" if kind == "answer" else None, "arg": arg, "result": None})
            break
        result = tool(kind, arg)
        steps.append({"text": text, "tool": kind, "arg": arg, "result": result})
        msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": render_result(result)}]
    return {"q": item["q"], "gold": item["answer"], "steps": steps, "answer": answer,
            "correct": grade(answer, item["answer"]), "turns": len(steps)}


items = json.load(open(args.questions))
if args.limit:
    items = items[: args.limit]
done = set()
if os.path.exists(args.out):  # resumable
    done = {json.loads(l)["q"] for l in open(args.out)}
todo = [it for it in items if it["q"] not in done]
n_ok = n = 0
with open(args.out, "a") as f, cf.ThreadPoolExecutor(args.workers) as ex:
    for fut in cf.as_completed([ex.submit(run, it) for it in todo]):
        try:
            r = fut.result()
        except Exception as e:
            print(f"error: {e}", flush=True)
            continue
        f.write(json.dumps(r) + "\n")
        f.flush()
        n += 1
        n_ok += r["correct"]
        if n % 25 == 0:
            print(f"{n}/{len(todo)} done, teacher accuracy {n_ok / n:.1%}", flush=True)
print(f"finished: {n} traces, teacher accuracy {n_ok / max(n, 1):.1%}")
kb.stdin.close()
