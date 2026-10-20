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
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

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


def load(adapter=None, name=None, quant4=False):
    name = name or MODEL
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    quant = {}
    if quant4:
        from transformers import BitsAndBytesConfig
        quant = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)}
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda", **quant)
    model.thinking = "Thinking" in name  # chat template opens <think>: replies are thinking, </think>, action
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


class StopAfterThink(StoppingCriteria):
    """Per-sequence stop for thinking models: only a closing tool tag *after* </think> ends the turn
    (the model may write tags while thinking)."""
    def __init__(self, tok, start):
        self.tok, self.start = tok, start

    def __call__(self, input_ids, scores, **kw):
        done = []
        for row in input_ids:
            t = self.tok.decode(row[self.start:], skip_special_tokens=True)
            after = t.split("</think>", 1)[1] if "</think>" in t else ""
            done.append(any(s in after for s in STOPS))
        return torch.tensor(done, device=input_ids.device)


def split_think(t):
    """-> (thinking, action text). No </think> means the turn never got past thinking."""
    if "</think>" not in t:
        return t, ""
    think, act = t.rsplit("</think>", 1)
    return think.replace("<think>", "").strip(), act


@torch.no_grad()
def generate(tok, model, convs, max_new=256, temp=0.0, prefixes=None):
    """-> (actions, thoughts). prefixes: text the assistant turn is forced to start with."""
    """prefixes: text the assistant turn is forced to start with (e.g. "<answer>" on the last turn)"""
    prefixes = prefixes or [""] * len(convs)
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False) + p
             for c, p in zip(convs, prefixes)]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    sample = {"do_sample": True, "temperature": temp, "top_p": 0.95} if temp > 0 else {"do_sample": False}
    thinking = getattr(model, "thinking", False)
    stop = ({"stopping_criteria": StoppingCriteriaList([StopAfterThink(tok, enc["input_ids"].shape[1])])} if thinking
            else {"stop_strings": STOPS, "tokenizer": tok})
    out = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **stop, **sample)
    res, thoughts = [], []
    for row, p in zip(out[:, enc["input_ids"].shape[1]:], prefixes):
        t = p + tok.decode(row, skip_special_tokens=True)
        think, t = split_think(t) if thinking else ("", t)
        cut = min([t.find(s) + len(s) for s in STOPS if s in t] or [len(t)])
        res.append(t[:cut].strip())
        thoughts.append(think)
    return res, thoughts


def run(tok, model, items, mode, out_path, bs, max_turns=5, temp=0.0, max_new=256, force_answer=True, memory=None, mem_k=3):
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
        recalled = [memory.render(it["q"], mem_k) if memory else ("", []) for it in batch]
        convs = [[{"role": "system", "content": sys_prompt}, {"role": "user", "content": pre + it["q"]}]
                 for it, (pre, _) in zip(batch, recalled)]
        state = [{"steps": [], "answer": None, "open": True} for _ in batch]
        for turn in range(max_turns if mode == "tools" else 1):
            live = [i for i, s in enumerate(state) if s["open"]]
            if not live:
                break
            # last allowed turn: the reply must be an answer, so start it with the tag
            final = force_answer and mode == "tools" and turn == max_turns - 1
            outs, thinks = generate(tok, model, [convs[i] for i in live], max_new=max_new, temp=temp,
                                    prefixes=["<answer>" if final else ""] * len(live))
            for i, out, think in zip(live, outs, thinks):
                m = list(re.finditer(r"<(search|graph|answer)>(.*?)</\1>", out, re.S))
                st = state[i]
                if not m:
                    st["steps"].append({"text": out, "tool": None, "think": think})
                    st["open"] = False
                    continue
                kind, arg = m[-1].group(1), m[-1].group(2).strip()
                if kind == "answer" or mode != "tools" or turn == max_turns - 1:
                    st["answer"] = arg if kind == "answer" else None
                    st["steps"].append({"text": out, "tool": kind, "think": think})
                    st["open"] = False
                    continue
                result = tools(kind, arg)
                last = turn == max_turns - 2  # the next turn must be the answer
                st["steps"].append({"text": out, "tool": kind, "arg": arg, "result": result, "last": last, "think": think})
                convs[i] += [{"role": "assistant", "content": out},
                             {"role": "user", "content": render_result(result, last)}]
        for it, st in zip(batch, state):
            ok = grade(st["answer"], it["answer"])
            n += 1
            n_ok += ok
            rec = {"q": it["q"], "gold": it["answer"], "answer": st["answer"], "correct": ok,
                   "turns": len(st["steps"]), "steps": st["steps"]}
            if memory:  # did a recalled example already contain this question's answer?
                rec["mem_answers"] = recalled[batch.index(it)][1]
                rec["mem_has_gold"] = any(grade(a, it["answer"]) for a in rec["mem_answers"])
            f.write(json.dumps(rec) + "\n")
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
    ap.add_argument("--model", default=MODEL, help="e.g. Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--quant4", action="store_true", help="load the base in 4-bit NF4 (8B on 16 GB)")
    ap.add_argument("--mode", default="tools", choices=["tools", "closed"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.0, help=">0 samples (for expert-iteration attempts)")
    ap.add_argument("--no_force_answer", action="store_true", help="don't pre-fill <answer> on the final turn")
    ap.add_argument("--memory", help="experience memory: comma-separated trace files of solved (training) questions")
    ap.add_argument("--mem_k", type=int, default=3)
    ap.add_argument("--mem_hide_answers", action="store_true", help="control: show procedures, mask answers")
    ap.add_argument("--max_new", type=int, default=768,
                    help="tokens per turn; GLM-style reasoning overflowed 256 and lost the tool tag")
    ap.add_argument("--attempts", type=int, default=1, help="repeat each question this many times (with --temp)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    qs = json.load(open(a.questions))
    tok, model = load(a.adapter, a.model, a.quant4)
    mem = None
    if a.memory:
        from experience import ExperienceMemory
        mem = ExperienceMemory(a.memory, exclude_questions={it["q"] for it in qs}, hide_answers=a.mem_hide_answers)
        print(f"experience memory: {len(mem.items)} solved questions", flush=True)
    qs = qs[: a.limit] if a.limit else qs
    if a.attempts > 1:  # one output file per attempt, so resume logic works per attempt
        for k in range(a.attempts):
            torch.manual_seed(k)
            run(tok, model, qs, a.mode, a.out.replace(".jsonl", f".a{k}.jsonl"), a.bs, temp=a.temp, max_new=a.max_new,
                force_answer=not a.no_force_answer, memory=mem, mem_k=a.mem_k)
    else:
        run(tok, model, qs, a.mode, a.out, a.bs, temp=a.temp, max_new=a.max_new, force_answer=not a.no_force_answer,
            memory=mem, mem_k=a.mem_k)
