#!/bin/bash
# Judgment eval (agent/judge_agent.py): untrained 4B Instruct, coding adapter v2, untrained 4B Thinking, 3 runs each
# (greedy still drifts with batching; single runs move a few points). Simulated servers come from memlm-sandbox on srv.
# Log: ~/memlm2/judge_eval.log.   SMOKE=1: 6 scenarios, base only.
cd ~/memlm2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=runs/agent
PY=~/vllm-env/.venv/bin/python
B=Qwen/Qwen3-4B-Instruct-2507
T=Qwen/Qwen3-4B-Thinking-2507
stop_vllm() {
  for p in $(ps -eo pid,args | grep -E "[v]llm|[E]ngineCore|[r]esource_tracker" | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do [ $(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 2500 ] && return; sleep 2; done
}
serve() {
  : > vllm_serve.log; (agent/serve_vllm.sh "$@" > vllm_serve.log 2>&1 &)
  until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
    grep -q Traceback vllm_serve.log && { echo "vLLM failed"; tail -5 vllm_serve.log; exit 1; }; sleep 5; done
}
judge() { echo "== $1"; $PY agent/judge_agent.py --out $A/$1.jsonl "${@:2}" 2>&1 | grep -vE "^\s*$|Warning|warn"; }

stop_vllm
serve $A/lora_code2 code2
if [ -n "$SMOKE" ]; then
  judge judge_smoke_base --model $B --only 'trap-0|ctrl-0' ; stop_vllm; echo JUDGE_SMOKE_DONE; exit 0
fi
for k in 1 2 3; do judge judge_base_r$k --model $B; judge judge_code2_r$k --model code2 --base $B; done
stop_vllm
(MODEL=$T agent/serve_vllm.sh > vllm_serve.log 2>&1 &)
until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do grep -q Traceback vllm_serve.log && { echo "vLLM failed"; exit 1; }; sleep 5; done
for k in 1 2 3; do judge judge_think_r$k --model $T --max_new 6144; done
stop_vllm
echo JUDGE_EVAL_DONE
