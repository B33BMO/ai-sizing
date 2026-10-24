#!/bin/bash
# Finish the two evals the WSL GPU error ("device not ready") killed: resume from their output files,
# with a 5-minute wait between attempts so a stuck GPU can recover.
cd ~/memlm2
export KB_ADDR=192.168.1.248:7787 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=runs/agent
run() {
  out=$1; shift
  for try in 1 2 3 4; do
    [ -f $A/$out ] && [ $(wc -l < $A/$out) -ge 500 ] && break
    .venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/$out "$@" 2>&1 | grep -E "FINAL|Error|Trace"
    [ $(wc -l < $A/$out) -ge 500 ] || { echo "attempt $try stopped at $(wc -l < $A/$out); waiting 5 min"; sleep 300; }
  done
}
run r3bmh_tools.jsonl  --adapter $A/lora_r3b --memory $A/glm53_train.short.jsonl --mem_hide_answers --bs 8 --max_new 512
run r3b8pf_tools.jsonl --adapter $A/lora_r3b_8b_pt --model Qwen/Qwen3-8B --quant4 --bs 8 --max_new 512
echo FINISH_DONE
