#!/bin/bash
# Finish the paused coding baselines (resumes from saved rows) + per-turn Spero adapter (r3b_pt) on coding.
cd ~/memlm2
(agent/serve_vllm.sh runs/agent/lora_r3b r3b runs/agent/lora_r3b_pt r3b_pt > vllm_serve.log 2>&1 &)
until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
  grep -q Traceback vllm_serve.log 2>/dev/null && { echo "vLLM failed"; exit 1; }; sleep 5; done
PY=~/vllm-env/.venv/bin/python
B="--base Qwen/Qwen3-4B-Instruct-2507 --split test"
$PY agent/code_agent.py $B --out runs/agent/codeagent_base.jsonl  --model Qwen/Qwen3-4B-Instruct-2507 2>&1 | grep -E "implement|fix|FINAL|error"
$PY agent/code_agent.py $B --out runs/agent/codeagent_r3b.jsonl   --model r3b    2>&1 | grep -E "implement|fix|FINAL|error"
$PY agent/code_agent.py $B --out runs/agent/codeagent_r3bpt.jsonl --model r3b_pt 2>&1 | grep -E "implement|fix|FINAL|error"
echo RESUME_DONE
