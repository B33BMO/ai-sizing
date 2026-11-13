#!/bin/bash
# Coding adapter v3: like v2 (edit action), but the teacher thinks (Ollama think=True) and the tail of its reasoning is
# the thought before each action. v2's traces had a thought on only 22% of actions: GLM-5.3 on Ollama stopped writing
# visible reasoning between Nov 4 and Nov 12 (teacher drift), so v2 learned to act without reasoning. Log: ~/memlm2/code_v3.log (shown on the tinker research page "coding-skill").
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

stop_vllm
# the teacher traces come from srv (Ollama Cloud); srv touches .done after copying them over
until [ -f $A/codetrain_glm53_think.done ]; do sleep 30; done
echo "== SFT v3"
.venv/bin/python agent/sft.py --task code --traces $A/codetrain_glm53_think.jsonl --per_q 1 --max_thought_chars 300 \
  --out $A/lora_code3 > sft_code3.log 2>&1 || { echo SFT_FAILED; exit 1; }
grep -E "traces from|tokens" sft_code3.log
serve $A/lora_code3 code3
agent codeagent_code3 code3
echo "== regression (base 84.3%, v1 81.2%, v2 79.8%)"
$PY agent/code_eval.py --model code3 --out $A/code_code3.jsonl 2>&1 | grep -E "humaneval|mbpp|FINAL"
echo "== Spero QA with code3 (base 40.2%, v2 37.6%)"
$PY agent/engine.py --questions data/spero_qa.json --out $A/spero_with_code3.jsonl --lora code3 --concurrency 24 2>&1 | grep FINAL
stop_vllm
echo CODE_V3_DONE
