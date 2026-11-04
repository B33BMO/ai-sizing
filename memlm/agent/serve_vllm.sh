#!/bin/bash
# vLLM server for agent/engine.py (separate venv: vLLM pins its own torch).
#   agent/serve_vllm.sh                          base Qwen3-4B-Instruct-2507
#   agent/serve_vllm.sh runs/agent/lora_r3b r3b [more_path name ...]  + LoRA adapters served by name
# Prefix caching makes every agent turn reuse the cached conversation; bf16 4B + KV cache fits the 16 GB card.
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
LORA=()
# adapters as path/name pairs: serve_vllm.sh runs/agent/lora_r3b r3b runs/agent/lora_r3b_pt r3b_pt ...
if [ -n "$1" ]; then
  LORA=(--enable-lora --max-lora-rank 32 --max-loras 4 --lora-modules)
  while [ -n "$1" ]; do LORA+=("${2:-adapter}=$1"); shift 2 || shift; done
fi
# Under WSL mirrored networking, vLLM's 127.0.0.1 listener accepted no connections (a plain server on the same
# port worked), so bind the LAN address and require an API key (vLLM has no auth by default).
# LAN address detected at start (DHCP has changed it before); engine.py reads ~/.vllm_url
HOST=${VLLM_HOST:-$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if ($i=="src") print $(i+1)}')}
echo "http://$HOST:8000" > ~/.vllm_url
KEY=$(cat ~/.vllm_key 2>/dev/null || (umask 077; head -c 24 /dev/urandom | base64 | tr -dc A-Za-z0-9 > ~/.vllm_key; cat ~/.vllm_key))
export VLLM_API_KEY="$KEY"  # via env, not argv, so it doesn't show in the process list
exec ~/vllm-env/.venv/bin/vllm serve "$MODEL" --host "$HOST" --port 8000 \
  --enable-prefix-caching --max-model-len 16384 --gpu-memory-utilization 0.88 --max-num-seqs 64 \
  "${LORA[@]}"
