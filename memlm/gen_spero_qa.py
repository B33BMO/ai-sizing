"""Paraphrased QA eval for Spero: an LLM rewrites a fact from a *training* chunk as a question
whose answer is an entity, so the model must recall knowledge, not replay memorised text.

Runs locally (needs the self-hosted LLM endpoint). Writes data/spero_qa.json:
  [{"q", "answer", "cands" (answer first, then same-type distractors), "type", "chunk"}]
"""
import concurrent.futures as cf
import gzip
import json
import random
import re
import argparse
import os
import urllib.request
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--out", default="data/spero_qa.json")
ap.add_argument("--exclude", default="", help="QA json whose chunks must not be reused (keeps test chunks out of training)")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

ENDPOINT = "https://llama2.coleman-it.com/v1/chat/completions"  # self-hosted gemma (free)
N_ITEMS, N_CAND, VAL_FRAC = args.n, 16, 0.05

d = json.load(gzip.open("data/spero_export.json.gz"))
rng = random.Random(0)
# reproduce prep_spero.py's document split exactly
paths = sorted({doc["source_path"] for doc in d["docs"]})
rng.shuffle(paths)
val_paths = set(paths[: int(len(paths) * VAL_FRAC)])

ents = {e["id"]: e for e in d["entities"] if len(e["name"]) >= 3}
by_type = defaultdict(list)
for e in ents.values():
    by_type[e["type"]].append(e["name"])
mentions = defaultdict(list)
for eid, cid in d["mentions"]:
    if eid in ents:
        mentions[cid].append(eid)

rng = random.Random(args.seed)
excluded = {it["chunk"] for it in json.load(open(args.exclude))} if args.exclude else set()
pool = []
for doc in d["docs"]:
    if doc["source_path"] in val_paths or doc["id"] in excluded:
        continue
    text = doc["content"]
    ok = [ents[e] for e in mentions.get(doc["id"], []) if ents[e]["type"] != "other"
          and len(by_type[ents[e]["type"]]) >= N_CAND and re.search(r"(?<![\w-])" + re.escape(ents[e]["name"]) + r"(?![\w-])", text)]
    if ok and len(text) > 300:
        pool.append((doc, rng.choice(ok)))
rng.shuffle(pool)

PROMPT = """Passage:
{text}

Write ONE question about a specific fact stated in this passage whose answer is exactly: "{answer}".
Rules: reword it (do not copy phrases from the passage); do not include the answer or any word of it in the question; it must be answerable from the passage alone. Output only the question, nothing else."""


def ask(doc, ent):
    body = json.dumps({"model": "x", "max_tokens": 200, "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}, "messages": [
        {"role": "user", "content": PROMPT.format(text=doc["content"][:2500], answer=ent["name"])}]}).encode()
    req = urllib.request.Request(ENDPOINT, body, {"Content-Type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return None, f"error: {e}"
    q = out.strip().strip('"').splitlines()[-1].strip() if out else ""
    words = [w.lower() for w in re.findall(r"\w{4,}", ent["name"])]
    if not q.endswith("?") or len(q) < 15 or any(w in q.lower() for w in words):
        return None, f"rejected: {q[:100]}"
    return q, None


items, rejects = [], 0
with cf.ThreadPoolExecutor(6) as ex:
    futs = {ex.submit(ask, doc, ent): (doc, ent) for doc, ent in pool[: int(N_ITEMS * 1.6)]}
    for i, f in enumerate(cf.as_completed(futs)):
        doc, ent = futs[f]
        q, err = f.result()
        if q is None:
            rejects += 1
        elif len(items) < N_ITEMS:
            others = rng.sample([n for n in by_type[ent["type"]] if n != ent["name"]], N_CAND - 1)
            items.append({"q": q, "answer": ent["name"], "cands": [ent["name"]] + others, "type": ent["type"], "chunk": doc["id"]})
        if i % 25 == 0:
            print(f"{i}: {len(items)} kept, {rejects} rejected", flush=True)
json.dump(items, open(args.out, "w"), indent=1)
print(f"done: {len(items)} questions, {rejects} rejected")
for it in items[:5]:
    print(f"  Q: {it['q']}  ->  {it['answer']}")
