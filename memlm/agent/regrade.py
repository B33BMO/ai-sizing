"""Fair re-grading: verbose answers get their committed entity extracted before the contains-match.

  python3 agent/regrade.py runs/agent/glm53_test.jsonl runs/agent/base_tools.jsonl ...
The grader passes any answer that *contains* the gold entity, which rewards long answers that mention
several candidates. For answers over MAX_WORDS, a free self-hosted model extracts the single thing the
answer names as its answer (without seeing the gold), and that is graded instead. Short answers are
graded as-is, so every model is scored by the same rule. Writes <file>.regraded.json next to each input.
"""
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from protocol import grade  # noqa: E402

ENDPOINT = "https://llama2.coleman-it.com/v1/chat/completions"  # self-hosted gemma
MAX_WORDS = 10
PROMPT = """Question: {q}
Response: {a}

What single answer does the response give to the question? Reply with only that answer, as short as possible (a product, feature, setting or place name). If it names several alternatives without committing to one, reply NONE."""


def extract(q, a):
    body = json.dumps({"model": "x", "max_tokens": 40, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": PROMPT.format(q=q, a=a)}]}).encode()
    req = urllib.request.Request(ENDPOINT, body, {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"].strip()
    return None if out.upper().startswith("NONE") else out.strip('"').strip()


if __name__ == "__main__":
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path)]
        long_ = [i for i, r in enumerate(rows) if r["answer"] and len(r["answer"].split()) > MAX_WORDS]
        with cf.ThreadPoolExecutor(6) as ex:
            ext = dict(zip(long_, ex.map(lambda i: extract(rows[i]["q"], rows[i]["answer"]), long_)))
        ok = [grade(ext[i], r["gold"]) if i in ext else r["correct"] for i, r in enumerate(rows)]
        res = {"file": path, "n": len(rows), "raw": sum(r["correct"] for r in rows) / len(rows),
               "fair": sum(ok) / len(rows), "n_extracted": len(long_)}
        json.dump({**res, "extracted": {rows[i]["q"]: e for i, e in ext.items()}},
                  open(path.replace(".jsonl", ".regraded.json"), "w"), indent=1)
        print(f"{os.path.basename(path):<24} raw {res['raw']:.1%}  fair {res['fair']:.1%}  ({len(long_)} long answers re-graded)", flush=True)
