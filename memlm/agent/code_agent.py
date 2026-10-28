"""Coding/bash agent episodes in the sandbox, graded by hidden tests.

Each task (agent/code_tasks.py) gets a fresh sandbox with its files; the model works with markdown fences
(XML-style tags like <bash> get swallowed by Ollama's server-side tool-call parser for GLM):
  ```bash            run a shell command, see stdout+stderr
  ```file x.py       overwrite a file with the block's content
  DONE               finish
After the episode the hidden test file is copied in and run; `passed` = all hidden tests pass.

Backends:
  --backend vllm    student models on the vLLM server (raw prompt rendered with the HF chat template)
  --backend ollama  teacher/reference on Ollama Cloud (OLLAMA_API_KEY in env), e.g. --model glm-5.3

  python agent/code_agent.py --tasks data/code_tasks.json --split test --out runs/agent/code_agent_base.jsonl \
      --backend vllm --model Qwen/Qwen3-4B-Instruct-2507
"""
import argparse
import asyncio
import json
import os
import re
import time

import aiohttp

CODE_SYSTEM = """You are working on a small Python project in the current directory.
Make exactly one action per message, then stop and wait for its result.
To run a shell command (for example ls, cat solution.py, python -m pytest -q), write:
```bash
<the command>
```
To replace solution.py, write the complete new file as a python block:
```python
<the complete new content of solution.py>
```
(To replace a different file, use ```file <path> instead of ```python.)
When the task is complete, reply with DONE on its own line.
Each result arrives as <result>...</result>.
Before each action, think in one or two short sentences. Use at most 8 actions, and run the tests before finishing."""
STOPS = ["\n```\n"]  # end of a fenced action; DONE ends the turn by itself
CALL_RE = re.compile(r"```bash\n(?P<cmd>.*?)\n```|```file[ \t]+(?P<path>[^\n`]+)\n(?P<content>.*?)\n```"
                     r"|```python\n(?P<py>.*?)\n```|^\s*DONE\s*$", re.S | re.M)
LAST_CALL = "No actions left: reply now with DONE."
NUDGE = "No action found. Use a ```bash block, a ```python block (replaces solution.py), or reply DONE."
MAX_RESULT = 3000
SB_ADDR = ("192.168.1.248", 7790)


def render(text, last=False):
    text = text if len(text) <= MAX_RESULT else text[:MAX_RESULT] + "\n[...truncated]"
    return f"<result>\n{text}\n</result>" + (f"\n{LAST_CALL}" if last else "")


async def sb(req):
    for attempt in range(5):
        try:
            r, w = await asyncio.wait_for(asyncio.open_connection(*SB_ADDR), 30)
            w.write((json.dumps(req) + "\n").encode())
            await w.drain()
            line = await asyncio.wait_for(r.readline(), 120)
            w.close()
            if line:
                return json.loads(line)
        except (OSError, asyncio.TimeoutError, json.JSONDecodeError):
            pass
        await asyncio.sleep(2 ** attempt)
    raise RuntimeError("sandbox service unreachable")


class VLLM:
    def __init__(self, a):
        from transformers import AutoTokenizer
        self.model, self.max_new = a.model, a.max_new
        self.tok = AutoTokenizer.from_pretrained(a.base or a.model)
        kf, uf = os.path.expanduser("~/.vllm_key"), os.path.expanduser("~/.vllm_url")
        self.key = open(kf).read().strip() if os.path.exists(kf) else None
        self.url = open(uf).read().strip() if os.path.exists(uf) else "http://127.0.0.1:8000"

    async def __call__(self, session, msgs):
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        body = {"model": self.model, "prompt": prompt, "max_tokens": self.max_new, "temperature": 0, "stop": STOPS,
                "include_stop_str_in_output": True, "skip_special_tokens": True}
        hdr = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        for attempt in range(6):
            try:
                async with session.post(f"{self.url}/v1/completions", json=body, headers=hdr) as r:
                    return (await r.json())["choices"][0]["text"]
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError):
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("vLLM unreachable")


class Ollama:
    def __init__(self, a):
        self.model, self.max_new = a.model, a.max_new
        self.key = os.environ["OLLAMA_API_KEY"]

    async def __call__(self, session, msgs):
        body = {"model": self.model, "messages": msgs, "stream": False, "think": False,
                "options": {"temperature": 0.3, "num_predict": self.max_new, "stop": STOPS}}
        for attempt in range(6):
            try:
                async with session.post("https://ollama.com/api/chat", json=body,
                                        headers={"Authorization": f"Bearer {self.key}"}) as r:
                    out = (await r.json())["message"]["content"]
                    # the server strips the stop string: close a fence the stop cut off
                    if out.count("```") % 2 == 1:
                        out = out.rstrip() + "\n```"
                    return out
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError):
                await asyncio.sleep(5 * 2 ** attempt)
        raise RuntimeError("Ollama unreachable")


def clean(text):
    """drop leaked markup and anything after the first complete action"""
    text = re.sub(r"</?think>|</?tool_call>", " ", text).strip()
    m = CALL_RE.search(text)
    return text[: m.end()].strip() if m else text


async def episode(task, llm, session, max_calls):
    box = await sb({"op": "create", "files": task["files"]})
    if not box.get("ok"):
        raise RuntimeError(box)
    bid = box["id"]
    msgs = [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": task["prompt"]}]
    steps = []
    try:
        for turn in range(max_calls + 1):
            out = clean(await llm(session, msgs))
            m = list(CALL_RE.finditer(out))
            if not m:  # no recognizable action: correct the format (twice at most), like a real harness would
                steps.append({"text": out, "tool": None, "result": NUDGE})
                if sum(st["tool"] is None for st in steps) > 2:
                    break
                msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": render(NUDGE)}]
                continue
            call = m[0]
            if call.group("cmd") is None and call.group("path") is None and call.group("py") is None:  # DONE
                steps.append({"text": out, "tool": "done"})
                break
            if turn == max_calls:  # out of calls: stop without executing
                steps.append({"text": out, "tool": "over_limit"})
                break
            if call.group("cmd") is not None:
                kind, arg = "bash", call.group("cmd").strip()
                r = await sb({"op": "exec", "id": bid, "cmd": arg})
                result = r.get("text", "") + ("" if r.get("code", 0) == 0 else f"\n[exit code {r.get('code')}]")
            else:
                kind = "write"
                arg = call.group("path").strip() if call.group("path") else "solution.py"
                content = call.group("content") if call.group("path") else call.group("py")
                r = await sb({"op": "write", "id": bid, "path": arg, "content": content + "\n"})
                result = f"wrote {arg}" if r.get("ok") else r.get("text", "write failed")
            last = turn == max_calls - 1
            steps.append({"text": out, "tool": kind, "arg": arg, "result": result, "last": last})
            msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": render(result, last)}]
        await sb({"op": "write", "id": bid, "path": "test_hidden.py", "content": task["hidden_tests"]})
        # exit code must come from pytest itself (an earlier `| tail` made every episode "pass")
        g = await sb({"op": "exec", "id": bid, "cmd": "python -m pytest -q -p no:cacheprovider test_hidden.py"})
        passed = g.get("ok") and g.get("code") == 0
        infra = not g.get("ok") or "No such container" in g.get("text", "")
    finally:
        await sb({"op": "destroy", "id": bid})
    if infra:
        raise RuntimeError(f"infra_error (sandbox lost) on {task['id']}: {g.get('text', '')[:120]}")
    return {"id": task["id"], "type": task["type"], "q": task["prompt"], "passed": passed, "correct": passed,
            "turns": len(steps), "steps": steps, "grade_output": g.get("text", "")[-300:]}


async def main(a):
    tasks = json.load(open(a.tasks))[a.split][: a.limit or None]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    tasks = [t for t in tasks if t["id"] not in done]
    llm = VLLM(a) if a.backend == "vllm" else Ollama(a)
    sem = asyncio.Semaphore(a.concurrency)
    n = n_ok = 0
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        async def run(t):
            async with sem:
                return await episode(t, llm, session, a.max_calls)
        with open(a.out, "a") as f:
            for fut in asyncio.as_completed([run(t) for t in tasks]):
                try:
                    r = await fut
                except Exception as e:
                    print("episode error:", e, flush=True)
                    continue
                f.write(json.dumps(r) + "\n")
                f.flush()
                n += 1
                n_ok += r["passed"]
                if n % 25 == 0 or n == len(tasks):
                    print(f"{n}/{len(tasks)}  pass {n_ok / n:.1%}  ({time.time() - t0:.0f}s)", flush=True)
    rows = [json.loads(l) for l in open(a.out)]
    for typ in ("implement", "fix"):
        sub = [r for r in rows if r["type"] == typ]
        if sub:
            print(f"{typ}: {sum(r['passed'] for r in sub)}/{len(sub)} = {sum(r['passed'] for r in sub) / len(sub):.1%}")
    print(f"FINAL {a.out}: {sum(r['passed'] for r in rows) / len(rows):.1%} on {len(rows)} tasks in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/code_tasks.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vllm", "ollama"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--base", help="tokenizer/base model when --model is a LoRA name")
    ap.add_argument("--max_calls", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=1536)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
