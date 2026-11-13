#!/bin/bash
# Re-evaluate the edit-protocol adapters after making edit matching indentation-tolerant (no retraining).
cd ~/memlm2
A=runs/agent
PY=~/vllm-env/.venv/bin/python
B=Qwen/Qwen3-4B-Instruct-2507
: > vllm_serve.log; (agent/serve_vllm.sh $A/lora_code2 code2 $A/lora_code3 code3 > vllm_serve.log 2>&1 &)
until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
  grep -q Traceback vllm_serve.log && { echo "vLLM failed"; exit 1; }; sleep 5; done
for m in code2 code3; do
  echo "== $m (indent-tolerant edits)"
  $PY agent/code_agent.py --split test --base $B --out $A/codeagent_${m}_tol.jsonl --model $m 2>&1 | grep -E "implement|fix|FINAL"
done
for p in $(ps -eo pid,args | grep -E "[v]llm|[E]ngineCore" | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
echo REEVAL_DONE
