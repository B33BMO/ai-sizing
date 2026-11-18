#!/bin/bash
# Repeat the greedy agent eval (171 held-out) 3x for v2, v5, v6: single runs move ~±3 pt (vLLM batching
# nondeterminism), the same size as the gaps between these adapters. Log: ~/memlm2/code_repeats.log
cd ~/memlm2
A=runs/agent
PY=~/vllm-env/.venv/bin/python
B=Qwen/Qwen3-4B-Instruct-2507
: > vllm_serve.log; (agent/serve_vllm.sh $A/lora_code2 code2 $A/lora_code5 code5 $A/lora_code6 code6 > vllm_serve.log 2>&1 &)
until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
  grep -q Traceback vllm_serve.log && { echo "vLLM failed"; exit 1; }; sleep 5; done
for rep in 1 2 3; do for m in code2 code5 code6; do
  echo "== $m rep $rep"
  $PY agent/code_agent.py --split test --base $B --out $A/rep_${m}_$rep.jsonl --model $m 2>&1 | grep -E "FINAL"
done; done
for p in $(ps -eo pid,args | grep -E "[v]llm|[E]ngineCore" | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
python3 - <<'PY'
import json, statistics as st
for m in ("code2", "code5", "code6"):
    sc = []
    for rep in (1, 2, 3):
        rows = [json.loads(l) for l in open(f"runs/agent/rep_{m}_{rep}.jsonl")]
        sc.append([100 * sum(r["passed"] for r in rows if t in (None, r["type"])) / sum(t in (None, r["type"]) for r in rows)
                   for t in (None, "implement", "fix")])
    print(m, "overall %.1f ± %.1f | implement %.1f | fix %.1f" % (st.mean(s[0] for s in sc), st.stdev(s[0] for s in sc),
          st.mean(s[1] for s in sc), st.mean(s[2] for s in sc)), [round(s[0], 1) for s in sc])
PY
echo REPEATS_DONE
