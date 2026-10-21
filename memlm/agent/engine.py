"""Agent-aware eval engine: many Spero episodes in flight against a vLLM server.

Same protocol, grader and output rows as student.py, but instead of lockstep batches of 8 re-encoding every
conversation each turn, it keeps --concurrency episodes running at once:
  - vLLM continuous batching + prefix caching: each turn only prefills the *new* tokens (the conversation so far
    is cached), and no episode waits for the slowest one in its batch
  - tool calls run concurrently with generation (a pooled tool service: KB_ADDR, e.g. 192.168.1.248:7788)

  # 1) server (separate venv):  agent/serve_vllm.sh [lora_dir]
  # 2) python agent/engine.py --questions data/spero_qa.json --out runs/agent/x.jsonl --lora r3b
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time

import aiohttp
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from protocol import SYSTEM, SYSTEM_NO_GRAPH, grade, render_result  # noqa: E402

STOPS = ["</search>", "</graph>", "</answer>"]
TAG = re.compile(r"<(search|graph|answer)>(.*?)</\1>", re.S)


class Tools:
    """async client for the tool service; one short-lived connection per call so calls run in parallel"""
    def __init__(self, addr):
        self.host, port = addr.rsplit(":", 1)
        self.port = int(port)
        self.n = 0

    async def __call__(self, kind, arg):
        self.n += 1
        req = {"id": self.n, "op": kind, **({"q": arg, "k": 4} if kind == "search" else {"entity": arg})}
        for attempt in range(6):
            try:
                r, w = await asyncio.wait_for(asyncio.open_connection(self.host, self.port), 30)
                w.write((json.dumps(req) + "\n").encode())
                await w.drain()
                line = await asyncio.wait_for(r.readline(), 120)
                w.close()
                if line:
                    return json.loads(line)["text"]
            except (OSError, asyncio.TimeoutError, json.JSONDecodeError):
                pass
            await asyncio.sleep(min(30, 2 ** attempt))
        raise RuntimeError("tool service unreachable")


async def complete(session, args, prompt):
    body = {"model": args.lora or args.model, "prompt": prompt, "max_tokens": args.max_new, "temperature": 0,
            "stop": STOPS, "include_stop_str_in_output": True, "skip_special_tokens": True}
    for attempt in range(6):
        try:
            async with session.post(f"{args.server}/v1/completions", json=body) as resp:
                data = await resp.json()
                return data["choices"][0]["text"]
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError):
            await asyncio.sleep(min(30, 2 ** attempt))
    raise RuntimeError("vLLM server unreachable")


async def episode(it, tok, session, tools, args, system, memory):
    pre = memory.render(it["q"], args.mem_k)[0] if memory else ""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": pre + it["q"]}]
    steps, answer = [], None
    for turn in range(args.max_turns):
        final = (not args.no_force_answer) and turn == args.max_turns - 1
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix = "<answer>" if final else ""
        out = (prefix + await complete(session, args, prompt + prefix)).strip()
        cut = min([out.find(s) + len(s) for s in STOPS if s in out] or [len(out)])
        out = out[:cut].strip()
        m = list(TAG.finditer(out))
        if not m:
            steps.append({"text": out, "tool": None})
            break
        kind, arg = m[-1].group(1), m[-1].group(2).strip()
        if kind == "answer" or turn == args.max_turns - 1:
            answer = arg if kind == "answer" else None
            steps.append({"text": out, "tool": kind})
            break
        result = await tools(kind, arg)
        last = turn == args.max_turns - 2
        steps.append({"text": out, "tool": kind, "arg": arg, "result": result, "last": last})
        msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": render_result(result, last)}]
    return {"q": it["q"], "gold": it["answer"], "answer": answer, "correct": grade(answer, it["answer"]),
            "turns": len(steps), "steps": steps}


async def main(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    items = json.load(open(args.questions))[: args.limit or None]
    done = {json.loads(l)["q"] for l in open(args.out)} if os.path.exists(args.out) else set()
    items = [it for it in items if it["q"] not in done]
    memory = None
    if args.memory:
        from experience import ExperienceMemory
        memory = ExperienceMemory(args.memory, exclude_questions={it["q"] for it in items},
                                  hide_answers=args.mem_hide_answers)
    system = SYSTEM_NO_GRAPH if args.no_graph else SYSTEM
    tools = Tools(os.environ.get("KB_ADDR", "192.168.1.248:7788"))
    sem = asyncio.Semaphore(args.concurrency)
    n = n_ok = 0
    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=600)
    with open(args.out, "a") as f:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async def run(it):
                async with sem:
                    return await episode(it, tok, session, tools, args, system, memory)
            for fut in asyncio.as_completed([run(it) for it in items]):
                r = await fut
                f.write(json.dumps(r) + "\n")
                f.flush()
                n += 1
                n_ok += r["correct"]
                if n % 25 == 0 or n == len(items):
                    print(f"{n}/{len(items)}  acc {n_ok / n:.1%}  ({(time.time() - t0) / n:.2f}s/q wall)", flush=True)
    rows = [json.loads(l) for l in open(args.out)]
    print(f"FINAL {args.out}: {sum(r['correct'] for r in rows) / len(rows):.1%} on {len(rows)} questions "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--lora", help="LoRA name registered with the vLLM server (--lora-modules name=path)")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--max_turns", type=int, default=5)
    ap.add_argument("--no_force_answer", action="store_true")
    ap.add_argument("--no_graph", action="store_true")
    ap.add_argument("--memory")
    ap.add_argument("--mem_k", type=int, default=3)
    ap.add_argument("--mem_hide_answers", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
