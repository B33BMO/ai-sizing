#!/bin/bash
# Coding adapter v2: the edit action (SEARCH/REPLACE) + "No change" feedback. Protocol-only baselines first (base and
# the v1 adapter under the new protocol), then SFT on GLM-5.3 traces made with the new protocol, then the same three
# checks as v1. Log: ~/memlm2/code_v2.log (shown on the tinker research page "coding-skill").
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
serve $A/lora_code code
agent codeagent_base_edit $B     # protocol only, untrained (old protocol 67.8%)
agent codeagent_code_edit code   # protocol only, v1 adapter (never saw an edit block; old protocol 78.4%)
stop_vllm
# the teacher traces come from srv (Ollama Cloud); srv touches .done after copying them over
until [ -f $A/codetrain_glm53_edit.done ]; do sleep 30; done
echo "== SFT v2"
.venv/bin/python agent/sft.py --task code --traces $A/codetrain_glm53_edit.jsonl --per_q 1 --max_thought_chars 300 \
  --out $A/lora_code2 > sft_code2.log 2>&1 || { echo SFT_FAILED; exit 1; }
grep -E "traces from|tokens" sft_code2.log
serve $A/lora_code2 code2
agent codeagent_code2 code2
echo "== regression (base 84.3%, v1 81.2%)"
$PY agent/code_eval.py --model code2 --out $A/code_code2.jsonl 2>&1 | grep -E "humaneval|mbpp|FINAL"
echo "== Spero QA with code2 (base 40.2%)"
$PY agent/engine.py --questions data/spero_qa.json --out $A/spero_with_code2.jsonl --lora code2 --concurrency 24 2>&1 | grep FINAL
stop_vllm
echo CODE_V2_DONE
