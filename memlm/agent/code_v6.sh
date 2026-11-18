#!/bin/bash
# Coding adapter v6: v5's data (GLM edit traces + the student's own recoveries from code_v5.sh), but the student's
# wrong first attempts are masked too (--mask_wrong). v5 learned them and made careless first guesses the tests would
# catch: agent 83.6% but single-shot HumanEval+MBPP 74.3%. Log: ~/memlm2/code_v6.log
cd ~/memlm2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KB_ADDR=192.168.1.248:7788
A=runs/agent
PY=~/vllm-env/.venv/bin/python
B=Qwen/Qwen3-4B-Instruct-2507
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
agent() { echo "== $1"; $PY agent/code_agent.py --split test --base $B --out $A/$1.jsonl --model $2 "${@:3}" 2>&1 | grep -E "implement|fix|FINAL"; }

stop_vllm
SELF=$(ls $A/codeself_v2_k[1-4].jsonl | paste -sd,)
echo "== SFT v6 (GLM edit traces + own recoveries, wrong attempts masked)"
.venv/bin/python agent/sft.py --task code --traces $A/codetrain_glm53_edit.jsonl --self_traces $SELF --mask_wrong \
  --per_q 1 --max_thought_chars 300 --out $A/lora_code6 > sft_code6.log 2>&1 || { echo SFT_FAILED; exit 1; }
grep -E "traces from|recovery|tokens" sft_code6.log
serve $A/lora_code6 code6
agent codeagent_code6 code6                              # greedy (v5: 83.6%, v2: 80.7%)
agent codeagent_code6_unstick code6 --unstick            # (v5: 81.3%, v2: 83.6%)
echo "== regression (base 84.3%, v2 79.8%, v5 74.3%)"
$PY agent/code_eval.py --model code6 --out $A/code_code6.jsonl 2>&1 | grep -E "humaneval|mbpp|FINAL"
echo "== Spero QA with code6 (base 40.2%, v5 38.8%)"
$PY agent/engine.py --questions data/spero_qa.json --out $A/spero_with_code6.jsonl --lora code6 --concurrency 24 2>&1 | grep FINAL
stop_vllm
echo CODE_V6_DONE
