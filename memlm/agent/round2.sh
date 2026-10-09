#!/bin/bash
# Expert-iteration round 2 (GPU box, ~/memlm2), ~5h:
#  1. round-1 adapter + forced final answer on test     (isolates the turn-limit fix)
#  2. sample 2 attempts per train question from round-1 adapter
#  3. fresh LoRA from base on all correct, duplicate-free traces (rounds 0+1 + teacher)
#  4. eval round 2 on test
cd ~/memlm2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=runs/agent
.venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/r1f_tools.jsonl --adapter $A/lora_r1 --bs 8 2>&1 | grep -E "FINAL|Error|Trace"
.venv/bin/python agent/student.py --questions data/spero_qa_train.json --out $A/r1_train.jsonl --adapter $A/lora_r1 \
  --temp 0.8 --attempts 2 --bs 8 2>&1 | grep -E "FINAL|Error|Trace"
.venv/bin/python agent/sft.py --traces $A/r0_train.a0.jsonl,$A/r0_train.a1.jsonl,$A/r1_train.a0.jsonl,$A/r1_train.a1.jsonl,$A/teacher_train.jsonl \
  --out $A/lora_r2 > sft_r2.log 2>&1 || { echo SFT_FAILED; exit 1; }
grep -E "traces from|tokens" sft_r2.log
.venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/r2_tools.jsonl --adapter $A/lora_r2 --bs 8 2>&1 | grep -E "FINAL|Error|Trace"
echo ROUND2_DONE
