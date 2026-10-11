"""Shorten a teacher's verbose final answers so its traces can be used for SFT.

  python3 agent/shorten.py runs/agent/glm53_train.jsonl runs/agent/glm53_train.short.jsonl
Answers over regrade.MAX_WORDS are replaced (inside the final <answer> tag) by the entity they commit to,
extracted without seeing the gold; `correct` is recomputed on the shortened answer. The teacher's searches
and reasoning are kept as-is: that is the behaviour the student should learn.
"""
import concurrent.futures as cf
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from protocol import grade  # noqa: E402
from regrade import MAX_WORDS, extract  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src)]
long_ = [i for i, r in enumerate(rows) if r["answer"] and len(r["answer"].split()) > MAX_WORDS]
with cf.ThreadPoolExecutor(6) as ex:
    ext = dict(zip(long_, ex.map(lambda i: extract(rows[i]["q"], rows[i]["answer"]), long_)))
n_short = 0
for i, e in ext.items():
    r = rows[i]
    if not e:
        r["correct"] = False
        continue
    last = r["steps"][-1]
    last["text"] = re.sub(r"<answer>.*?</answer>", f"<answer>{e}</answer>", last["text"], flags=re.S)
    last["arg"], r["answer"] = e, e
    r["correct"] = grade(e, r["gold"])
    n_short += 1
with open(dst, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"{len(rows)} traces, {len(long_)} long answers, {n_short} shortened; "
      f"correct before {sum(json.loads(l)['correct'] for l in open(src))} -> after {sum(r['correct'] for r in rows)}")
