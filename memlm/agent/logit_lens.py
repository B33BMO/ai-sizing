"""Logit lens on the Spero agent: at which layer is each generated token already decided?

  python agent/logit_lens.py --traces runs/agent/r3b_tools.jsonl --adapter runs/agent/lora_r3b --n 80
Teacher-forces real agent conversations (system, question, the model's own turns, real tool results) and, for
every assistant token, decodes each layer's hidden state through the final norm + lm_head. Reports:
  - settled layer: first layer from which the top-1 prediction equals the final layer's and never changes again
  - the same split by token type: reasoning text / tool-call content / protocol tags
  - a confidence early-exit policy: exit at the first layer whose top-1 prob >= t; layers used vs. agreement
Writes runs/agent/logit_lens.json.
"""
import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from protocol import SYSTEM, render_result  # noqa: E402
from student import MODEL, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True)
ap.add_argument("--adapter")
ap.add_argument("--n", type=int, default=80)
ap.add_argument("--out", default="runs/agent/logit_lens.json")
args = ap.parse_args()
random.seed(0)

tok, model = load(args.adapter)
if args.adapter:
    model = model.merge_and_unload()  # plain Qwen3ForCausalLM with the LoRA folded in
dec = model.model
norm, head = dec.norm, model.lm_head
L = model.config.num_hidden_layers
TAG = re.compile(r"</?(search|graph|answer)>")

rows = [json.loads(l) for l in open(args.traces)]
rows = [r for r in rows if r["steps"] and r["steps"][-1].get("tool") == "answer"]
rows = random.sample(rows, min(args.n, len(rows)))


def build(r):
    """token ids for the whole conversation + per-token category for assistant-written tokens (else None)"""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["q"]}]
    for s in r["steps"]:
        msgs.append({"role": "assistant", "content": s["text"]})
        if s.get("result") is not None:
            msgs.append({"role": "user", "content": render_result(s["result"], s.get("last", False))})
    ids, cats, prev = [], [], 0
    for i in range(len(msgs)):
        text = tok.apply_chat_template(msgs[: i + 1], tokenize=False)
        full = tok(text, add_special_tokens=False)["input_ids"]
        new = full[prev:]
        prev = len(full)
        ids += new
        if msgs[i]["role"] != "assistant":
            cats += [None] * len(new)
            continue
        content = msgs[i]["content"]
        m = re.search(r"<(search|graph|answer)>", content)
        call_start = m.start() if m else len(content)
        # map each new token to a character position inside the decoded message
        dec_text, pos = "", []
        for t in new:
            pos.append(len(dec_text))
            dec_text += tok.decode([t])
        head_len = dec_text.find(content[:20]) if content else 0  # skip "<|im_start|>assistant\n"
        for t, p in zip(new, pos):
            c = p - max(head_len, 0)
            piece = tok.decode([t])
            if c < 0 or "<|im_" in piece:
                cats.append("format")
            elif TAG.search(content[max(0, c - 8): c + len(piece) + 8]) and piece.strip("<>/ ").lower() in ("search", "graph", "answer", "", "</", "<"):
                cats.append("format")
            elif c < call_start:
                cats.append("reasoning")
            else:
                cats.append("tool content")
    return ids, cats


settled = {"reasoning": [], "tool content": [], "format": []}
policy = {t: {"layers": [], "agree": []} for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)}
with torch.no_grad():
    for k, r in enumerate(rows):
        ids, cats = build(r)
        # predict token i+1 from position i; we score positions whose *next* token was written by the model
        pos = [i for i in range(len(ids) - 1) if cats[i + 1] is not None]
        if not pos:
            continue
        x = torch.tensor([ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = dec(input_ids=x, output_hidden_states=True).hidden_states  # L+1 tensors (emb + each layer)
        p = torch.tensor(pos, device="cuda")
        top, conf = [], []
        for l in range(1, L + 1):
            h = hs[l][0, p] if l < L else hs[l][0, p]  # last entry is already normed by the model
            logits = head(h if l == L else norm(h)).float()
            pr = logits.softmax(-1)
            c, t = pr.max(-1)
            top.append(t)
            conf.append(c)
        top, conf = torch.stack(top), torch.stack(conf)  # (L, n_pos)
        final = top[-1]
        if k == 0:  # check the readout convention: last layer must reproduce the model's own predictions
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ref = model(input_ids=x).logits[0, p].float().argmax(-1)
            agree0 = (ref == final).float().mean().item()
            print(f"sanity: final-layer readout matches model output on {agree0:.1%} of tokens", flush=True)
            assert agree0 > 0.98, "hidden_states indexing convention differs; fix the readout"
        same = top == final
        # settled = 1 + last layer index where it disagreed (0-based layers -> 1..L)
        disagree = (~same).float()
        last_bad = torch.where(disagree.any(0), L - 1 - disagree.flip(0).argmax(0), torch.full_like(final, -1))
        s_layer = (last_bad + 2).clamp(max=L).cpu().numpy()  # first layer number after the last disagreement
        for j, i in enumerate(pos):
            settled[cats[i + 1]].append(int(s_layer[j]))
        for t in policy:
            hit = conf >= t
            first = torch.where(hit.any(0), hit.float().argmax(0), torch.full_like(final, L - 1))
            chosen = top.gather(0, first[None])[0]
            policy[t]["layers"] += (first + 1).cpu().tolist()
            policy[t]["agree"] += (chosen == final).cpu().tolist()
        print(f"{k + 1}/{len(rows)} traces, {sum(len(v) for v in settled.values())} tokens", flush=True)
        del hs

out = {"model": MODEL, "adapter": args.adapter, "layers": L, "n_traces": len(rows), "settled": {}, "policy": {}}
print(f"\nsettled layer (of {L}): first layer from which the top-1 token never changes again")
allv = sum(settled.values(), [])
for name, v in list(settled.items()) + [("ALL", allv)]:
    v = np.array(v)
    if not len(v):
        continue
    q = np.percentile(v, [25, 50, 75, 90])
    out["settled"][name] = {"n": int(len(v)), "mean": float(v.mean()), "p25": q[0], "median": q[1], "p75": q[2], "p90": q[3],
                            "frac_by_half": float((v <= L // 2).mean()), "hist": np.bincount(v, minlength=L + 1)[1:].tolist()}
    print(f"  {name:<13} n={len(v):>6}  median {q[1]:>4.0f}  p75 {q[2]:>4.0f}  p90 {q[3]:>4.0f}  "
          f"decided by layer {L // 2}: {(v <= L // 2).mean():.0%}")
print("\nconfidence early-exit (exit at first layer with top-1 prob >= t)")
for t, d in policy.items():
    layers, agree = np.array(d["layers"]), np.array(d["agree"])
    out["policy"][str(t)] = {"mean_layers": float(layers.mean()), "compute_saved": float(1 - layers.mean() / L),
                             "agreement": float(agree.mean())}
    print(f"  t={t:<5} avg layers {layers.mean():5.1f}/{L}  compute saved {1 - layers.mean() / L:5.1%}  "
          f"same token as full model {agree.mean():6.2%}")
json.dump(out, open(args.out, "w"), indent=1)
