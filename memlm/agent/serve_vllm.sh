#!/bin/bash
# vLLM server for agent/engine.py (separate venv: vLLM pins its own torch).
#   agent/serve_vllm.sh                          base Qwen3-4B-Instruct-2507
#   agent/serve_vllm.sh runs/agent/lora_r3b r3b  + a LoRA adapter served under the name "r3b"
# Prefix caching makes every agent turn reuse the cached conversation; bf16 4B + KV cache fits the 16 GB card.
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
LORA=()
[ -n "$1" ] && LORA=(--enable-lora --max-lora-rank 32 --lora-modules "${2:-adapter}=$1")
exec ~/vllm-env/.venv/bin/vllm serve "$MODEL" --host 127.0.0.1 --port 8000 \
  --enable-prefix-caching --max-model-len 16384 --gpu-memory-utilization 0.88 --max-num-seqs 64 \
  "${LORA[@]}"
