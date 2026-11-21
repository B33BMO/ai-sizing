#!/bin/bash
# HF-vs-vLLM serving parity for lora_code2 (agent/parity.py). Log: ~/memlm2/parity.log
cd ~/memlm2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=runs/agent
PY=~/vllm-env/.venv/bin/python
stop_vllm() {
  for p in $(ps -eo pid,args | grep -E "[v]llm|[E]ngineCore|[r]esource_tracker" | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do [ $(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 2500 ] && return; sleep 2; done
}
serve() {
  : > vllm_serve.log; (agent/serve_vllm.sh "$@" > vllm_serve.log 2>&1 &)
  until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
    grep -q Traceback vllm_serve.log && { echo "vLLM failed"; tail -5 vllm_serve.log; exit 1; }; sleep 5; done
}
stop_vllm
serve $A/lora_code2 code2
$PY agent/parity.py gen --model code2 --out $A/parity_code2.json || { echo GEN_FAILED; stop_vllm; exit 1; }
stop_vllm
.venv/bin/python agent/parity.py score --adapter $A/lora_code2 --out $A/parity_code2.json 2>&1 | grep -vE "Warning|warn" || echo SCORE_FAILED
echo PARITY_DONE
