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
import code_agent  # noqa: E402  (coding protocol: CODE_SYSTEM, fenced actions, render)
from student import MODEL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--init_adapter")
ap.add_argument("--model", default=MODEL)
ap.add_argument("--quant4", action="store_true", help="QLoRA: load the base in 4-bit NF4 (needed for 8B on 16 GB)")
ap.add_argument("--per_turn", action="store_true",
                help="one example per assistant turn, prompt rendered exactly as at inference. Needed for templates "
                     "that render history differently from the current turn (Qwen3 hybrid: only the last turn gets the "
                     "empty <think> block, so single-sequence SFT teaches 'think block => answer now')")
ap.add_argument("--exclude", default="data/spero_qa.json", help="never train on these (test) questions")
ap.add_argument("--per_q", type=int, default=2)
ap.add_argument("--seed", type=int, default=0, help="data order + LoRA init (for repeat runs)")
ap.add_argument("--max_thought_chars", type=int, default=0,
                help="trim the reasoning before each tool call to its last sentences (0 = keep); a 4B copying "
                     "GLM-length thoughts rambles and never emits the tag")
ap.add_argument("--max_answer_words", type=int, default=10,
                help="drop traces whose final answer is longer (the grader's contains-match would reward padding)")
ap.add_argument("--task", default="spero", choices=["spero", "code"],
                help="code: coding-agent traces (CODE_SYSTEM, fenced actions; keyed by task id, not prompt text)")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--max_len", type=int, default=6144)
ap.add_argument("--tokens_per_step", type=int, default=32768)
args = ap.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.task == "code":
    SYSTEM, render_result = code_agent.CODE_SYSTEM, code_agent.render
    test_ids = {t["id"] for t in json.load(open("data/code_tasks.json"))["test"]}
    def keep(r):  # passed, ended with DONE, no format nudges (clean actions only), not a test task
        return (r["passed"] and r["id"] not in test_ids and r["steps"] and r["steps"][-1].get("tool") == "done"
                and all(s.get("tool") is not None for s in r["steps"]))
    key_of = lambda r: r["id"]  # all fix tasks share one prompt text
else:
    test_qs = {it["q"] for it in json.load(open(args.exclude))}
    def keep(r):
        return (r["correct"] and r["q"] not in test_qs and r["steps"] and r["steps"][-1].get("tool") == "answer"
                and not has_duplicate_calls(r["steps"]) and len((r.get("answer") or "").split()) <= args.max_answer_words)
    key_of = lambda r: r["q"]
by_q = defaultdict(list)
for path in args.traces.split(","):
    for line in open(path):
        r = json.loads(line)
        if keep(r):
            by_q[key_of(r)].append(r)
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

tok = AutoTokenizer.from_pretrained(args.model)
TPL = {"enable_thinking": False}  # hybrid Qwen3 templates; ignored by templates that don't use it


LEAKED = re.compile(r"</?think>|</?tool_call>")  # teacher/base-model markup that is not part of our protocol


def trim_thought(text, limit, think=""):
    """keep the tool call and only the last sentence(s) of the reasoning before it, up to limit chars.
    `think`: the teacher's separate thinking (Ollama think=True), prepended to the visible reasoning; its code
    blocks are dropped so fences inside it can't be mistaken for the action"""
    text = LEAKED.sub(" ", text).strip()
    think = LEAKED.sub(" ", re.sub(r"```.*?(```|$)", " ", think or "", flags=re.S)).strip()
    m = re.search(r"```(?:bash|python|file|edit)|^\s*DONE\s*$", text, re.M) if args.task == "code" else \
        re.search(r"<(search|graph|answer)>", text)
    if not m:
        return re.sub(r"\s+", " ", text)
    # collapse whitespace in the reasoning only: the action (code!) must keep its newlines and indentation
    thought, call = re.sub(r"\s+", " ", f"{think} {text[: m.start()]}").strip(), text[m.start():].strip()
    sep = "\n" if args.task == "code" else " "  # code fences belong at the start of a line
    if not limit:
        return f"{thought}{sep}{call}".strip() if thought else call
    sents = re.split(r"(?<=[.!?])\s+", thought)
    kept = []
    for snt in reversed(sents):
        if kept and len(" ".join([snt] + kept)) > limit:
            break
        kept.insert(0, snt)
    short = " ".join(kept)[-limit:].strip()
    return f"{short}{sep}{call}".strip() if short else call


def encode(r):
    """token ids + labels; labels are -100 except on assistant turns (incl. their end-of-turn token)"""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["q"]}]
    for s in r["steps"]:
        msgs.append({"role": "assistant", "content": trim_thought(s["text"], args.max_thought_chars, s.get("think"))})
        if s.get("result") is not None:
            msgs.append({"role": "user", "content": render_result(s["result"], s.get("last", False))})
    # Render once and label assistant spans by character offsets. Rendering turn by turn is not
    # prefix-stable for every template (Qwen3 hybrid drops the empty <think> block from older turns).
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **TPL)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans, i = [], 0
    while True:
        a = text.find("<|im_start|>assistant\n", i)
        if a < 0:
            break
        start = a + len("<|im_start|>assistant\n")
        empty_think = "<think>\n\n</think>\n\n"
        if text.startswith(empty_think, start):  # the template's, not the model's: generation starts after it
            start += len(empty_think)
        end = text.find("<|im_end|>", start) + len("<|im_end|>")
        spans.append((start, end))
        i = end
    ids = enc["input_ids"]
    labels = [t if any(s <= o[0] < e for s, e in spans) else -100 for t, o in zip(ids, enc["offset_mapping"])]
    return ids[: args.max_len], labels[: args.max_len]


def encode_per_turn(r):
    """[(ids, labels)] one per assistant turn: inference-identical prompt + that turn's reply"""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["q"]}]
    out = []
    for s in r["steps"]:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **TPL)
        reply = trim_thought(s["text"], args.max_thought_chars, s.get("think"))
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(reply + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = (p_ids + r_ids)[-args.max_len:]
        labels = ([-100] * len(p_ids) + r_ids)[-args.max_len:]
        out.append((ids, labels))
        msgs.append({"role": "assistant", "content": reply})
        if s.get("result") is not None:
            msgs.append({"role": "user", "content": render_result(s["result"], s.get("last", False))})
    return out


data = [ex for r in traces for ex in encode_per_turn(r)] if args.per_turn else [encode(r) for r in traces]
n_tok = sum(len(i) for i, _ in data)
print(f"{n_tok/1e6:.2f}M tokens, {sum(sum(l != -100 for l in lab) for _, lab in data)/1e3:.0f}k trained tokens")

quant = {}
if args.quant4:
    from transformers import BitsAndBytesConfig
    quant = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)}
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", **quant)
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
