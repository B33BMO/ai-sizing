#!/bin/bash
# Coding-agent baselines on the held-out task split (GPU box): untrained 4B and the Spero (r3b) adapter.
cd ~/memlm2
(agent/serve_vllm.sh runs/agent/lora_r3b r3b > vllm_serve.log 2>&1 &)
until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
  grep -q Traceback vllm_serve.log 2>/dev/null && { echo "vLLM failed"; exit 1; }; sleep 5; done
PY=~/vllm-env/.venv/bin/python
$PY agent/code_agent.py --split test --out runs/agent/codeagent_base.jsonl --model Qwen/Qwen3-4B-Instruct-2507 2>&1 | grep -E "implement|fix|FINAL|error|Traceback"
$PY agent/code_agent.py --split test --out runs/agent/codeagent_r3b.jsonl --model r3b --base Qwen/Qwen3-4B-Instruct-2507 2>&1 | grep -E "implement|fix|FINAL|error|Traceback"
echo BASELINE_DONE
