"""Run a (base or LoRA-tuned) student model as a Spero agent and grade it.

  python agent/student.py --questions data/spero_qa.json --out runs/agent/base_tools.jsonl [--adapter runs/agent/lora] [--mode tools|closed]
Tools are reached through the kb bridge on 127.0.0.1:7777 (see kb_bridge.py).
"""
import argparse
import json
import os
import re
import socket
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from protocol import SYSTEM, grade, render_result  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CLOSED = ("Answer the question about Microsoft 365 from memory. Reply with <answer>the shortest exact answer, "
          "such as a product, feature or setting name</answer>.")
STOPS = ["</search>", "</graph>", "</answer>"]


class Tools:
    """client for the kb bridge; reconnects with backoff when the tunnel drops"""
    def __init__(self):
        self.f, self.n = None, 0

    def _connect(self):
        # KB_ADDR=192.168.1.248:7787 -> always-on service on srv; default = Mac bridge via tunnel
        host, port = os.environ.get("KB_ADDR", "127.0.0.1:7777").rsplit(":", 1)
        self.f = socket.create_connection((host, int(port)), timeout=120).makefile("rw")

    def __call__(self, kind, arg):
        self.n += 1
        req = {"id": self.n, "op": kind, **({"q": arg, "k": 4} if kind == "search" else {"entity": arg})}
        for attempt in range(8):
            try:
                if self.f is None:
                    self._connect()
                self.f.write(json.dumps(req) + "\n")
                self.f.flush()
                line = self.f.readline()
                if line:
                    return json.loads(line)["text"]
            except OSError:
                pass
            self.f = None
            time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError("kb bridge unreachable")


def load(adapter=None):
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, convs, max_new=256, temp=0.0):
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    sample = {"do_sample": True, "temperature": temp, "top_p": 0.95} if temp > 0 else {"do_sample": False}
    out = model.generate(**enc, max_new_tokens=max_new, stop_strings=STOPS, tokenizer=tok,
                         pad_token_id=tok.pad_token_id, **sample)
    res = []
    for row in out[:, enc["input_ids"].shape[1]:]:
        t = tok.decode(row, skip_special_tokens=True)
        cut = min([t.find(s) + len(s) for s in STOPS if s in t] or [len(t)])
        res.append(t[:cut].strip())
    return res


def run(tok, model, items, mode, out_path, bs, max_turns=5, temp=0.0):
    tools = Tools() if mode == "tools" else None
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["q"] for l in open(out_path)}
    items = [it for it in items if it["q"] not in done]
    f = open(out_path, "a")
    n_ok = n = 0
    t0 = time.time()
    for b in range(0, len(items), bs):
        batch = items[b:b + bs]
        sys_prompt = SYSTEM if mode == "tools" else CLOSED
        convs = [[{"role": "system", "content": sys_prompt}, {"role": "user", "content": it["q"]}] for it in batch]
        state = [{"steps": [], "answer": None, "open": True} for _ in batch]
        for turn in range(max_turns if mode == "tools" else 1):
            live = [i for i, s in enumerate(state) if s["open"]]
            if not live:
                break
            outs = generate(tok, model, [convs[i] for i in live], temp=temp)
            for i, out in zip(live, outs):
                m = list(re.finditer(r"<(search|graph|answer)>(.*?)</\1>", out, re.S))
                st = state[i]
                if not m:
                    st["steps"].append({"text": out, "tool": None})
                    st["open"] = False
                    continue
                kind, arg = m[-1].group(1), m[-1].group(2).strip()
                if kind == "answer" or mode != "tools" or turn == max_turns - 1:
                    st["answer"] = arg if kind == "answer" else None
                    st["steps"].append({"text": out, "tool": kind})
                    st["open"] = False
                    continue
                result = tools(kind, arg)
                last = turn == max_turns - 2  # the next turn must be the answer
                st["steps"].append({"text": out, "tool": kind, "arg": arg, "result": result, "last": last})
                convs[i] += [{"role": "assistant", "content": out},
                             {"role": "user", "content": render_result(result, last)}]
        for it, st in zip(batch, state):
            ok = grade(st["answer"], it["answer"])
            n += 1
            n_ok += ok
            f.write(json.dumps({"q": it["q"], "gold": it["answer"], "answer": st["answer"], "correct": ok,
                                "turns": len(st["steps"]), "steps": st["steps"]}) + "\n")
        f.flush()
        print(f"{n}/{len(items)}  acc {n_ok / n:.1%}  ({(time.time() - t0) / n:.1f}s/q)", flush=True)
    rows = [json.loads(l) for l in open(out_path)]
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"FINAL {out_path}: {acc:.1%} on {len(rows)} questions")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--mode", default="tools", choices=["tools", "closed"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.0, help=">0 samples (for expert-iteration attempts)")
    ap.add_argument("--attempts", type=int, default=1, help="repeat each question this many times (with --temp)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    qs = json.load(open(a.questions))
    tok, model = load(a.adapter)
    qs = qs[: a.limit] if a.limit else qs
    if a.attempts > 1:  # one output file per attempt, so resume logic works per attempt
        for k in range(a.attempts):
            torch.manual_seed(k)
            run(tok, model, qs, a.mode, a.out.replace(".jsonl", f".a{k}.jsonl"), a.bs, temp=a.temp)
    else:
        run(tok, model, qs, a.mode, a.out, a.bs, temp=a.temp)
