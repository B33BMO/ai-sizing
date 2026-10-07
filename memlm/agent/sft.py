"""LoRA SFT of the student on successful agent traces (loss only on the assistant's turns).

  python agent/sft.py --traces runs/agent/a.jsonl,runs/agent/b.jsonl --out runs/agent/lora_r1 [--init_adapter ...]
Trace lines are the format written by student.py / gen_traces.py: {"q", "steps": [{"text", "result"}], "correct"}.
Keeps correct traces only, at most --per_q per question (fewest tool calls first).
"""
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from protocol import SYSTEM, render_result  # noqa: E402
from student import MODEL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--init_adapter")
ap.add_argument("--exclude", default="data/spero_qa.json", help="never train on these (test) questions")
ap.add_argument("--per_q", type=int, default=2)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--max_len", type=int, default=6144)
ap.add_argument("--tokens_per_step", type=int, default=32768)
args = ap.parse_args()
random.seed(0)

test_qs = {it["q"] for it in json.load(open(args.exclude))}
by_q = defaultdict(list)
for path in args.traces.split(","):
    for line in open(path):
        r = json.loads(line)
        if r["correct"] and r["q"] not in test_qs and r["steps"] and r["steps"][-1].get("tool") == "answer":
            by_q[r["q"]].append(r)
traces = []
for q, rs in by_q.items():
    rs.sort(key=lambda r: r["turns"])
    seen = set()
    for r in rs:
        key = tuple(s["text"] for s in r["steps"])
        if key not in seen and len(seen) < args.per_q:
            seen.add(key)
            traces.append(r)
print(f"{len(traces)} traces from {len(by_q)} questions")

tok = AutoTokenizer.from_pretrained(MODEL)


def encode(r):
    """token ids + labels; labels are -100 except on assistant turns (incl. their end-of-turn token)"""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["q"]}]
    for s in r["steps"]:
        msgs.append({"role": "assistant", "content": s["text"]})
        if s.get("result") is not None:
            msgs.append({"role": "user", "content": render_result(s["result"])})
    ids, labels, prev = [], [], 0
    for i in range(len(msgs)):
        full = tok.apply_chat_template(msgs[: i + 1], tokenize=True, add_generation_prompt=False)
        full = full["input_ids"] if isinstance(full, dict) else full
        new = full[prev:]
        ids += new
        labels += new if msgs[i]["role"] == "assistant" else [-100] * len(new)
        prev = len(full)
    return ids[: args.max_len], labels[: args.max_len]


data = [encode(r) for r in traces]
n_tok = sum(len(i) for i, _ in data)
print(f"{n_tok/1e6:.2f}M tokens, {sum(sum(l != -100 for l in lab) for _, lab in data)/1e3:.0f}k trained tokens")

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
if args.init_adapter:
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
steps = math.ceil(args.epochs * n_tok / args.tokens_per_step)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * s / steps)))
print(f"{steps} optimizer steps")

model.train()
step, acc_tok, acc_loss, acc_n = 0, 0, 0.0, 0
order = []
while step < steps:
    if not order:
        order = list(range(len(data)))
        random.shuffle(order)
    ids, lab = data[order.pop()]
    x = torch.tensor([ids], device="cuda")
    y = torch.tensor([lab], device="cuda")
    n_lab = (y[:, 1:] != -100).sum().item()
    if n_lab == 0:
        continue
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=x, labels=y).loss
    # weight by tokens so long and short traces count per token
    (loss * n_lab / 2000).backward()
    acc_tok += len(ids)
    acc_loss += loss.item() * n_lab
    acc_n += n_lab
    if acc_tok >= args.tokens_per_step:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 5 == 0 or step == steps:
            print(f"step {step}/{steps} loss {acc_loss / acc_n:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        acc_tok, acc_loss, acc_n = 0, 0.0, 0
model.save_pretrained(args.out)
print("saved", args.out)
