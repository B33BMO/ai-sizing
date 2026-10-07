"""Train a small GPT (optionally with product-key memory) on FineWeb-edu + synthetic bios.

  python train.py --name base
  python train.py --name mem --mem_layers 4
  python train.py --name ptr --mem_layers 4 --mem_hops 2 --mem_pointer table
"""
import argparse
import json
import math
import os
import time

import numpy as np
import tiktoken
import torch

from model import GPT, Config
from synth import SynthEval

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--n_layer", type=int, default=8)
ap.add_argument("--n_embd", type=int, default=512)
ap.add_argument("--n_head", type=int, default=8)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--mem_layers", default="")
ap.add_argument("--mem_n_keys", type=int, default=512)
ap.add_argument("--mem_heads", type=int, default=4)
ap.add_argument("--mem_topk", type=int, default=32)
ap.add_argument("--mem_k_dim", type=int, default=256)
ap.add_argument("--mem_hops", type=int, default=1)
ap.add_argument("--mem_pointer", default="none")
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--grad_accum", type=int, default=4)
ap.add_argument("--tokens", type=float, default=500e6)
ap.add_argument("--lr", type=float, default=1.5e-3)
ap.add_argument("--mem_lr_mult", type=float, default=1.0)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--synth_frac", type=float, default=0.25)
ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--log_every", type=int, default=50)
ap.add_argument("--no_compile", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
# WSL/Windows silently spills to system RAM when VRAM is full (10-50x slowdown); fail loudly instead
torch.cuda.set_per_process_memory_fraction(0.92)
out_dir = os.path.join("runs", args.name)
os.makedirs(out_dir, exist_ok=True)

cfg = Config(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head, seq_len=args.seq_len,
             mem_layers=[int(x) for x in args.mem_layers.split(",") if x], mem_n_keys=args.mem_n_keys,
             mem_heads=args.mem_heads, mem_topk=args.mem_topk, mem_k_dim=args.mem_k_dim,
             mem_hops=args.mem_hops, mem_pointer=args.mem_pointer)
model = GPT(cfg).to(dev)
counts = model.param_counts()
print(json.dumps(counts))

# ---- optimizer: no decay on embeddings / norms / memory tables; optional higher lr for memory
groups = {"decay": [], "nodecay": [], "mem": []}
for n, p in model.named_parameters():
    if ".ff.values" in n or ".ff.pointers" in n or ".ff.keys" in n:
        groups["mem"].append(p)
    elif p.dim() < 2 or "wte" in n:
        groups["nodecay"].append(p)
    else:
        groups["decay"].append(p)
opt = torch.optim.AdamW([
    {"params": groups["decay"], "weight_decay": args.wd, "lr_mult": 1.0},
    {"params": groups["nodecay"], "weight_decay": 0.0, "lr_mult": 1.0},
    {"params": groups["mem"], "weight_decay": 0.0, "lr_mult": args.mem_lr_mult},
], lr=args.lr, betas=(0.9, 0.95), fused=True)

tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
steps = int(args.tokens // tokens_per_step)


def lr_at(s):
    if s < args.warmup:
        return args.lr * (s + 1) / args.warmup
    p = (s - args.warmup) / max(1, steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


# ---- data
web = np.memmap("data/fineweb_train.bin", dtype=np.uint16, mode="r")
web_val = np.memmap("data/fineweb_val.bin", dtype=np.uint16, mode="r")
syn = np.memmap("data/synth_train.bin", dtype=np.uint16, mode="r")
rng = np.random.default_rng(args.seed)


def get_batch(bs, val=False):
    L = args.seq_len + 1
    rows = []
    for _ in range(bs):
        src = web_val if val else (syn if rng.random() < args.synth_frac else web)
        i = rng.integers(0, len(src) - L)
        rows.append(src[i: i + L].astype(np.int64))
    b = torch.from_numpy(np.stack(rows)).pin_memory().to(dev, non_blocking=True)
    return b[:, :-1], b[:, 1:]


enc = tiktoken.get_encoding("gpt2")
synth_eval = SynthEval(json.load(open("data/world.json")), enc)

fwd = model if args.no_compile else torch.compile(model)


@torch.no_grad()
def val_loss(n=40):
    model.eval()
    tot = 0.0
    for _ in range(n):
        x, y = get_batch(args.batch_size, val=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += fwd(x, y)[1].item()
    model.train()
    return tot / n


def evaluate(step):
    r = {"step": step, "tokens": step * tokens_per_step, "val_loss": val_loss()}
    r.update(synth_eval.run(model, dev))
    print("EVAL " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    return r


log = open(os.path.join(out_dir, "log.jsonl"), "w")
log.write(json.dumps({"args": vars(args), "params": counts}) + "\n")
print(f"{steps} steps, {tokens_per_step} tok/step")
t0 = time.time()
evals = []
for step in range(steps + 1):
    if step % args.eval_every == 0 or step == steps:
        r = evaluate(step)
        evals.append(r)
        log.write(json.dumps({"eval": r}) + "\n")
        log.flush()
        t0 = time.time()
    if step == steps:
        break
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr * g["lr_mult"]
    for _ in range(args.grad_accum):
        x, y = get_batch(args.batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fwd(x, y)[1] / args.grad_accum
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % args.log_every == 0:
        torch.cuda.synchronize()
        dt = time.time() - t0
        tps = tokens_per_step * (args.log_every if step else 1) / dt
        t0 = time.time()
        line = {"step": step, "loss": loss.item() * args.grad_accum, "lr": lr, "tok_s": round(tps),
                "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        print(json.dumps(line), flush=True)
        log.write(json.dumps(line) + "\n")

torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, os.path.join(out_dir, "ckpt.pt"))
json.dump({"args": vars(args), "params": counts, "final": evals[-1], "evals": evals},
          open(os.path.join(out_dir, "results.json"), "w"), indent=1)
print("done")
