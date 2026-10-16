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
import re
import sys
from collections import defaultdict

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from protocol import SYSTEM, has_duplicate_calls, render_result  # noqa: E402
from student import MODEL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--init_adapter")
ap.add_argument("--exclude", default="data/spero_qa.json", help="never train on these (test) questions")
ap.add_argument("--per_q", type=int, default=2)
ap.add_argument("--max_thought_chars", type=int, default=0,
                help="trim the reasoning before each tool call to its last sentences (0 = keep); a 4B copying "
                     "GLM-length thoughts rambles and never emits the tag")
ap.add_argument("--max_answer_words", type=int, default=10,
                help="drop traces whose final answer is longer (the grader's contains-match would reward padding)")
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
        if (r["correct"] and r["q"] not in test_qs and r["steps"] and r["steps"][-1].get("tool") == "answer"
                and not has_duplicate_calls(r["steps"]) and len((r.get("answer") or "").split()) <= args.max_answer_words):
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


LEAKED = re.compile(r"</?think>|</?tool_call>")  # teacher/base-model markup that is not part of our protocol


def trim_thought(text, limit):
    """keep the tool call and only the last sentence(s) of the reasoning before it, up to limit chars"""
    text = re.sub(r"\s+", " ", LEAKED.sub(" ", text)).strip()
    m = re.search(r"<(search|graph|answer)>", text)
    if not limit or not m:
        return text
    thought, call = text[: m.start()].strip(), text[m.start():]
    sents = re.split(r"(?<=[.!?])\s+", thought)
    kept = []
    for snt in reversed(sents):
        if kept and len(" ".join([snt] + kept)) > limit:
            break
        kept.insert(0, snt)
    short = " ".join(kept)[-limit:].strip()
    return f"{short} {call}".strip() if short else call


def encode(r):
    """token ids + labels; labels are -100 except on assistant turns (incl. their end-of-turn token)"""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["q"]}]
    for s in r["steps"]:
        msgs.append({"role": "assistant", "content": trim_thought(s["text"], args.max_thought_chars)})
        if s.get("result") is not None:
            msgs.append({"role": "user", "content": render_result(s["result"], s.get("last", False))})
    ids, labels, prev = [], [], 0
    for i in range(len(msgs)):
        text = tok.apply_chat_template(msgs[: i + 1], tokenize=False, add_generation_prompt=False)
        full = tok(text, add_special_tokens=False)["input_ids"]
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
causal = model.get_base_model()  # the LoRA-injected CausalLM: .model (decoder) + .lm_head
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
        # project only the trained positions through lm_head: full-vocab (152k) fp32 logits for a
        # 6k-token trace are ~3.7 GB and push the 16 GB card into WSL's slow sysmem fallback
        h = causal.model(input_ids=x).last_hidden_state[:, :-1]
        tgt = y[:, 1:]
        keep = tgt != -100
        loss = torch.nn.functional.cross_entropy(causal.lm_head(h[keep]).float(), tgt[keep])
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
