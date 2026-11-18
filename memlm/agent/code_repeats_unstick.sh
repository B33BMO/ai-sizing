#!/bin/bash
# Is v2 + --unstick (83.6% single run, the blog's headline) real? 3 more runs; v2 greedy averages 81.1 ± 0.9.
cd ~/memlm2
A=runs/agent
PY=~/vllm-env/.venv/bin/python
B=Qwen/Qwen3-4B-Instruct-2507
: > vllm_serve.log; (agent/serve_vllm.sh $A/lora_code2 code2 > vllm_serve.log 2>&1 &)
until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
  grep -q Traceback vllm_serve.log && { echo "vLLM failed"; exit 1; }; sleep 5; done
for rep in 1 2 3; do
  $PY agent/code_agent.py --split test --base $B --out $A/rep_code2_unstick_$rep.jsonl --model code2 --unstick 2>&1 | grep -E "FINAL"
done
for p in $(ps -eo pid,args | grep -E "[v]llm|[E]ngineCore" | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
python3 - <<'PY'
import json, statistics as st
sc = [100 * sum(json.loads(l)["passed"] for l in open(f"runs/agent/rep_code2_unstick_{i}.jsonl")) / 171 for i in (1, 2, 3)]
print("code2 + unstick: %.1f ± %.1f" % (st.mean(sc), st.stdev(sc)), [round(s, 1) for s in sc])
PY
echo UNSTICK_REPEATS_DONE
