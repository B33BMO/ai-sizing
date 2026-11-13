#!/bin/bash
# Untrained Qwen3-4B-Thinking-2507 on the 171 held-out coding tasks (compare: untrained Instruct 67.8%).
cd ~/memlm2
(MODEL=Qwen/Qwen3-4B-Thinking-2507 agent/serve_vllm.sh > vllm_serve.log 2>&1 &)
until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
  grep -q Traceback vllm_serve.log 2>/dev/null && { echo "vLLM failed"; exit 1; }; sleep 5; done
~/vllm-env/.venv/bin/python agent/code_agent.py --split test --out runs/agent/codeagent_think.jsonl \
  --model Qwen/Qwen3-4B-Thinking-2507 --max_new 6144 --concurrency 16 2>&1 | grep -E "implement|fix|FINAL|error"
echo THINK_DONE
