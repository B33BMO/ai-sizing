#!/bin/bash
# --unstick ablation (eval only): repeated/no-op changes aren't rerun; the model is asked to reason through one failing
# test by hand; the next turn is sampled (T=0.7) instead of greedy. Runs after code_v4.sh.
cd ~/memlm2
A=runs/agent
PY=~/vllm-env/.venv/bin/python
B=Qwen/Qwen3-4B-Instruct-2507
until grep -qE "CODE_V4_DONE|SFT_FAILED|vLLM failed" code_v4.log 2>/dev/null; do sleep 30; done
: > vllm_serve.log; (agent/serve_vllm.sh $A/lora_code2 code2 $A/lora_code4 code4 > vllm_serve.log 2>&1 &)
until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
  grep -q Traceback vllm_serve.log && { echo "vLLM failed"; exit 1; }; sleep 5; done
for m in $B code2 code4; do
  n=$(basename $m)
  echo "== $n --unstick"
  $PY agent/code_agent.py --split test --base $B --out $A/codeagent_${n}_unstick.jsonl --model $m --unstick 2>&1 | grep -E "implement|fix|FINAL"
done
for p in $(ps -eo pid,args | grep -E "[v]llm|[E]ngineCore" | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
echo UNSTICK_DONE
