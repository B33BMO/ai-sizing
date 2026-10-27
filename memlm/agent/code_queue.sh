#!/bin/bash
# Coding regression: base Qwen3-4B vs the round-3b Spero adapter, HumanEval + MBPP, 2 runs each (engine noise ~1 pt).
cd ~/memlm2
(agent/serve_vllm.sh runs/agent/lora_r3b r3b > vllm_serve.log 2>&1 &)
until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
  grep -q Traceback vllm_serve.log 2>/dev/null && { echo "vLLM failed"; exit 1; }; sleep 5; done
PY=~/vllm-env/.venv/bin/python
for run in 1 2; do
  $PY agent/code_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --out runs/agent/code_base_$run.jsonl 2>&1 | grep -E "humaneval|mbpp|FINAL|Traceback|Error"
  $PY agent/code_eval.py --model r3b --out runs/agent/code_r3b_$run.jsonl 2>&1 | grep -E "humaneval|mbpp|FINAL|Traceback|Error"
done
pkill -f "vllm serve"
echo CODE_DONE
