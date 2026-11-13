#!/bin/bash
# Coding adapter v1: SFT on GLM-5.3's clean coding traces, then eval + regression checks. Runs after code_think.sh.
cd ~/memlm2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KB_ADDR=192.168.1.248:7788
A=runs/agent
stop_vllm() {  # kill server AND its EngineCore child (it otherwise keeps ~15 GB of VRAM), wait for VRAM
  for p in $(ps -eo pid,args | grep -E "[v]llm|[E]ngineCore|[r]esource_tracker" | awk '{print $1}') \
           $(/usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do [ $(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 2500 ] && return; sleep 2; done
}
serve() {  # serve base + adapters (path name ...), wait for readiness
  : > vllm_serve.log; (agent/serve_vllm.sh "$@" > vllm_serve.log 2>&1 &)
  until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
    grep -q Traceback vllm_serve.log && { echo "vLLM failed"; exit 1; }; sleep 5; done
}
until grep -qE "THINK_DONE|vLLM failed" code_think.log 2>/dev/null; do sleep 30; done
stop_vllm
.venv/bin/python agent/sft.py --task code --traces $A/codetrain_glm53.jsonl --per_q 1 --max_thought_chars 300 \
  --out $A/lora_code > sft_code.log 2>&1 || { echo SFT_FAILED; exit 1; }
grep -E "traces from|tokens" sft_code.log
serve $A/lora_code code $A/lora_r3b r3b
PY=~/vllm-env/.venv/bin/python
# 1) the skill itself: coding agent (compare untrained 67.8%, GLM-5.3 97.1%)
$PY agent/code_agent.py --split test --out $A/codeagent_code.jsonl --model code --base Qwen/Qwen3-4B-Instruct-2507 2>&1 | grep -E "implement|fix|FINAL"
# 2) regression: single-shot HumanEval+MBPP with the coding adapter (base 84.3%)
$PY agent/code_eval.py --model code --out $A/code_codeadapter.jsonl 2>&1 | grep -E "humaneval|mbpp|FINAL"
# 3) interference: Spero QA with the CODING adapter (Spero adapter 58-60% on the engine, untrained ~40%)
$PY agent/engine.py --questions data/spero_qa.json --out $A/spero_with_code.jsonl --lora code --concurrency 24 2>&1 | grep FINAL
stop_vllm
echo CODE_TRAIN_DONE
