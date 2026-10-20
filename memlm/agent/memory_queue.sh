#!/bin/bash
# Experience memory (idea A): the round-3b 4B recalls 3 similar solved training questions as worked examples.
# Runs after the 8B pipeline. Control: same, but recalled answers hidden (procedure only).
cd ~/memlm2
export KB_ADDR=192.168.1.248:7787 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -qE "ROUND8B_PT_DONE|SFT_FAILED" round8b_pt.log 2>/dev/null; do sleep 60; done
A=runs/agent; M="--memory $A/glm53_train.short.jsonl --adapter $A/lora_r3b --bs 8 --max_new 512"
.venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/r3bm_tools.jsonl $M 2>&1 | grep -E "FINAL|memory:|Error|Trace"
.venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/r3bmh_tools.jsonl $M --mem_hide_answers 2>&1 | grep -E "FINAL|memory:|Error|Trace"
echo MEMORY_DONE
