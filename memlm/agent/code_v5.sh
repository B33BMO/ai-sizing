#!/bin/bash
# Coding adapter v5: expert iteration. Adapter v2 (+ --unstick, T=0.7) attempts every train task 4 times; sft.py keeps
# one *recovery* per task (change -> failing tests -> different change -> pass), masks its wasted moves, and trains
# from the base on GLM's edit traces (v2's data) + those recoveries. The student never saw anyone get it wrong and
# recover: once stuck, v2 recovered in 2 of 23 episodes. Log: ~/memlm2/code_v5.log
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
serve $A/lora_code2 code2
SELF=()
for k in 1 2 3 4; do
  echo "== self-play attempt $k (v2 + unstick, T=0.7, 612 train tasks)"
  $PY agent/code_agent.py --split train --base $B --model code2 --unstick --temperature 0.7 \
    --out $A/codeself_v2_k$k.jsonl 2>&1 | grep -E "FINAL"
  SELF+=($A/codeself_v2_k$k.jsonl)
done
python3 - "${SELF[@]}" <<'EOF'
import json, sys
runs = [{r["id"]: r["passed"] for r in map(json.loads, open(f))} for f in sys.argv[1:]]
ids = set().union(*runs)
print(f"pass@1 {sum(sum(r.values()) / len(r) for r in runs) / len(runs):.1%}, "
      f"pass@4 {sum(any(r.get(i) for r in runs) for i in ids) / len(ids):.1%} on {len(ids)} train tasks")
EOF
stop_vllm
echo "== SFT v5 (GLM edit traces + own recoveries)"
.venv/bin/python agent/sft.py --task code --traces $A/codetrain_glm53_edit.jsonl \
  --self_traces $(IFS=,; echo "${SELF[*]}") --per_q 1 --max_thought_chars 300 \
  --out $A/lora_code5 > sft_code5.log 2>&1 || { echo SFT_FAILED; exit 1; }
grep -E "traces from|recovery|tokens" sft_code5.log
serve $A/lora_code5 code5
agent codeagent_code5 code5                              # greedy (v2: 80.7% indent-tolerant)
agent codeagent_code5_unstick code5 --unstick            # (v2 + unstick: 83.6%)
echo "== regression (base 84.3%, v2 79.8%, v4 80.8%)"
$PY agent/code_eval.py --model code5 --out $A/code_code5.jsonl 2>&1 | grep -E "humaneval|mbpp|FINAL"
echo "== Spero QA with code5 (base 40.2%, v2 37.6%, v4 39.4%)"
$PY agent/engine.py --questions data/spero_qa.json --out $A/spero_with_code5.jsonl --lora code5 --concurrency 24 2>&1 | grep FINAL
stop_vllm
echo CODE_V5_DONE
