#!/bin/bash
# Coding adapter v4: v3's traces (teacher thinking), but the thought is GLM's visible sentence, falling back to its
# thinking tail only when it wrote none. v3 joined both, duplicating every thought; the student learned to repeat
# itself (Spero QA 25.4%: looping preambles, 156/500 unanswered). Edits match indentation-tolerantly.
# Log: ~/memlm2/code_v4.log. Waits for code_reeval.sh to free the GPU.
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
agent() { echo "== $1"; $PY agent/code_agent.py --split test --base $B --out $A/$1.jsonl --model $2 2>&1 | grep -E "implement|fix|FINAL"; }

until grep -qE "REEVAL_DONE|vLLM failed" code_reeval.log 2>/dev/null; do sleep 30; done
stop_vllm
# the teacher traces come from srv (Ollama Cloud); srv touches .done after copying them over
until [ -f $A/codetrain_glm53_think.done ]; do sleep 30; done
echo "== SFT v4"
.venv/bin/python agent/sft.py --task code --traces $A/codetrain_glm53_think.jsonl --per_q 1 --max_thought_chars 300 \
  --out $A/lora_code4 > sft_code4.log 2>&1 || { echo SFT_FAILED; exit 1; }
grep -E "traces from|tokens" sft_code4.log
serve $A/lora_code4 code4
agent codeagent_code4 code4
echo "== regression (base 84.3%, v1 81.2%, v2 79.8%, v3 81.0%)"
$PY agent/code_eval.py --model code4 --out $A/code_code4.jsonl 2>&1 | grep -E "humaneval|mbpp|FINAL"
echo "== Spero QA with code4 (base 40.2%, v2 37.6%, v3 25.4%)"
$PY agent/engine.py --questions data/spero_qa.json --out $A/spero_with_code4.jsonl --lora code4 --concurrency 24 2>&1 | grep FINAL
stop_vllm
echo CODE_V4_DONE
