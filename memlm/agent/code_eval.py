"""Coding regression check: did a skill adapter cost general coding ability?

HumanEval (164) + MBPP-sanitized test (257), pass@1, greedy, against the vLLM server (base model or a LoRA).
Generated code runs in a throwaway temp dir as a subprocess with a time limit and an address-space limit.

  python agent/code_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --out runs/agent/code_base.jsonl
  python agent/code_eval.py --model r3b --out runs/agent/code_r3b.jsonl
"""
import argparse
import asyncio
import json
import os
import re
import resource
import subprocess
import sys
import tempfile

import aiohttp
from datasets import load_dataset

_kf, _uf = os.path.expanduser("~/.vllm_key"), os.path.expanduser("~/.vllm_url")
API_KEY = open(_kf).read().strip() if os.path.exists(_kf) else None
SERVER = open(_uf).read().strip() if os.path.exists(_uf) else "http://127.0.0.1:8000"


def problems():
    out = []
    for r in load_dataset("openai/openai_humaneval", split="test"):
        out.append({"id": r["task_id"], "src": "humaneval",
                    "prompt": "Complete this Python function. Reply with the full function in one ```python block.\n\n"
                              f"```python\n{r['prompt']}```",
                    "test": r["test"] + f"\ncheck({r['entry_point']})\n", "header": r["prompt"]})
    for r in load_dataset("google-research-datasets/mbpp", "sanitized", split="test"):
        out.append({"id": f"mbpp/{r['task_id']}", "src": "mbpp",
                    "prompt": f"{r['prompt']}\nYour code should pass these tests:\n" + "\n".join(r["test_list"]) +
                              "\nReply with the code in one ```python block.",
                    "test": "\n".join(r["test_imports"]) + "\n" + "\n".join(r["test_list"]) + "\n", "header": ""})
    return out


def extract(text, header):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    code = max(blocks, key=len) if blocks else text
    # HumanEval: the reply may contain only the body; keep the original signature/imports available
    if header and "def " not in code:
        code = header + code
    elif header:
        imports = "\n".join(l for l in header.splitlines() if l.startswith(("import ", "from ")))
        code = imports + "\n" + code
    return code


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))  # 2 GB address space
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def run_test(code, test):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.py")
        open(path, "w").write(code + "\n\n" + test)
        try:
            r = subprocess.run([sys.executable, path], cwd=d, capture_output=True, timeout=15, preexec_fn=limits,
                               env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})
            return r.returncode == 0, r.stderr.decode(errors="replace")[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


async def solve(session, model, p, sem):
    async with sem:
        body = {"model": model, "temperature": 0, "max_tokens": 1024,
                "messages": [{"role": "user", "content": p["prompt"]}]}
        hdr = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
        for attempt in range(5):
            try:
                async with session.post(f"{SERVER}/v1/chat/completions", json=body, headers=hdr) as r:
                    text = (await r.json())["choices"][0]["message"]["content"]
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError):
                await asyncio.sleep(2 ** attempt)
        else:
            text = ""
    code = extract(text, p["header"])
    ok, err = await asyncio.to_thread(run_test, code, p["test"])
    return {"id": p["id"], "src": p["src"], "passed": ok, "err": err if not ok else "", "completion": text}


async def main(a):
    ps = problems()
    sem = asyncio.Semaphore(a.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        rows = await asyncio.gather(*[solve(s, a.model, p, sem) for p in ps])
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for src in ("humaneval", "mbpp"):
        sub = [r for r in rows if r["src"] == src]
        print(f"{a.model} {src}: {sum(r['passed'] for r in sub)}/{len(sub)} = {sum(r['passed'] for r in sub) / len(sub):.1%}")
    print(f"FINAL {a.out}: {sum(r['passed'] for r in rows) / len(rows):.1%} on {len(rows)} problems")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served model name: base id or a LoRA name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    asyncio.run(main(ap.parse_args()))
