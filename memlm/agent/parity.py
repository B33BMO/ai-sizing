"""Serving parity: does vLLM run the LoRA adapter the way training (HF + peft) sees it?

Rebuilds prompts from recorded agent episodes, has vLLM greedily continue them with the adapter, then scores vLLM's
tokens under HF + peft with teacher forcing: how often HF's top token is vLLM's token, and how far their log-probs
of it differ. The same scoring with the adapter switched off is the control (it shows the test can see a change).

  # vLLM env, adapter served as <name>:
  $PY agent/parity.py gen   --episodes runs/agent/codeagent_code2.jsonl --model code2 --out runs/agent/parity_code2.json
  # training env (.venv, has peft):
  .venv/bin/python agent/parity.py score --adapter runs/agent/lora_code2 --out runs/agent/parity_code2.json
"""
import argparse
import json
import os
import random
import sys

BASE = "Qwen/Qwen3-4B-Instruct-2507"


def gen(a):
    import requests
    from transformers import AutoTokenizer
    sys.path.insert(0, os.path.dirname(__file__))
    from code_agent import CODE_SYSTEM, STOPS, render
    tok = AutoTokenizer.from_pretrained(BASE)
    recs = [json.loads(line) for line in open(a.episodes)]
    random.seed(0)
    picked = random.sample(recs, a.n)
    items = []
    for r in picked:  # the first turn and one later turn of each episode
        steps = r["steps"]
        for k in sorted({0, random.randrange(len(steps))}):
            msgs = [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": r["q"]}]
            for st in steps[:k]:
                msgs += [{"role": "assistant", "content": st["text"]},
                         {"role": "user", "content": render(st.get("result", ""), st.get("last", False))}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            items.append({"id": r["id"], "step": k, "prompt": prompt, "recorded": steps[k]["text"]})
    url = open(os.path.expanduser("~/.vllm_url")).read().strip()
    hdr = {"Authorization": "Bearer " + open(os.path.expanduser("~/.vllm_key")).read().strip()}
    for it in items:
        body = {"model": a.model, "prompt": it["prompt"], "max_tokens": 1536, "temperature": 0, "logprobs": 1,
                "stop": STOPS, "include_stop_str_in_output": True, "return_tokens_as_token_ids": True}
        ch = requests.post(f"{url}/v1/completions", json=body, headers=hdr, timeout=300).json()["choices"][0]
        it["text"] = ch["text"]
        it["ids"] = [int(t.split(":")[1]) for t in ch["logprobs"]["tokens"]]
        it["lp"] = ch["logprobs"]["token_logprobs"]
    same = sum(it["text"].strip().startswith(it["recorded"].strip()[:200]) for it in items)
    print(f"{len(items)} prompts from {a.n} episodes; vLLM now reproduces the recorded turn in {same}/{len(items)}")
    json.dump(items, open(a.out, "w"))


def score(a):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    items = json.load(open(a.out))

    def run(it):
        p = tok(it["prompt"], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p + it["ids"]], device="cuda")
        with torch.no_grad():
            logits = model(ids).logits[0, len(p) - 1:-1].float()
        lp = torch.log_softmax(logits, -1)
        tgt = torch.tensor(it["ids"], device="cuda")
        top = (lp.argmax(-1) == tgt).float()
        mine = lp.gather(1, tgt[:, None])[:, 0]
        diff = (mine - torch.tensor(it["lp"], device="cuda")).abs()
        first = int((top == 0).nonzero()[0]) if (top == 0).any() else None
        return top.mean().item(), diff.mean().item(), first, len(it["ids"])

    rows = {"adapter on": [], "adapter off (control)": []}
    for it in items:
        rows["adapter on"].append(run(it))
        with model.disable_adapter():
            rows["adapter off (control)"].append(run(it))
    for name, rs in rows.items():
        toks = sum(r[3] for r in rs)
        agree = sum(r[0] * r[3] for r in rs) / toks
        dlp = sum(r[1] * r[3] for r in rs) / toks
        clean = sum(r[2] is None for r in rs)
        print(f"{name:22s} top-token agreement {agree:.2%}  mean |dlogprob| {dlp:.4f}  "
              f"turns with no disagreement {clean}/{len(rs)}  ({toks} tokens)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "score"])
    ap.add_argument("--episodes", default="runs/agent/codeagent_code2.jsonl")
    ap.add_argument("--model", default="code2")
    ap.add_argument("--adapter", default="runs/agent/lora_code2")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="runs/agent/parity_code2.json")
    a = ap.parse_args()
    gen(a) if a.mode == "gen" else score(a)
