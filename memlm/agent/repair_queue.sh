#!/bin/bash
# After extra_queue: finish every eval that stopped early (WSL "CUDA driver error: device not ready"), resuming from
# its output file, retrying up to 3 times per run; then the untrained-model experience-memory test.
cd ~/memlm2
export KB_ADDR=192.168.1.248:7787 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -qE "EXTRA_DONE|SFT_FAILED" extra_queue.log 2>/dev/null; do sleep 60; done
A=runs/agent
run() {  # run <out> <args...>: resume until the file has 500 rows (max 3 tries)
  out=$1; shift
  for try in 1 2 3; do
    [ -f $A/$out ] && [ $(wc -l < $A/$out) -ge 500 ] && break
    .venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/$out "$@" 2>&1 | grep -E "FINAL|Error|Trace"
    sleep 30
  done
}
MEM="--memory $A/glm53_train.short.jsonl"
run r3bmh_tools.jsonl  --adapter $A/lora_r3b $MEM --mem_hide_answers --bs 8 --max_new 512
run r3b8pf_tools.jsonl --adapter $A/lora_r3b_8b_pt --model Qwen/Qwen3-8B --quant4 --bs 8 --max_new 512
run r3bptf_tools.jsonl --adapter $A/lora_r3b_pt --bs 8 --max_new 512
run r3bng_tools.jsonl  --adapter $A/lora_r3b --no_graph --bs 8 --max_new 512
# experience memory on the UNTRAINED 4B (doesn't already know these procedures): compare base2 40.2%
run basem_tools.jsonl  $MEM --no_force_answer --bs 8 --max_new 512
run basemh_tools.jsonl $MEM --mem_hide_answers --no_force_answer --bs 8 --max_new 512
echo REPAIR_DONE
