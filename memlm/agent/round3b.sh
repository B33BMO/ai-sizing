#!/bin/bash
# Round 3b: GLM-5.3 traces with thoughts trimmed to <=300 chars and leaked markup removed -> eval (LAN tools)
cd ~/memlm2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KB_ADDR=192.168.1.248:7787
A=runs/agent
.venv/bin/python agent/sft.py --traces $A/glm53_train.short.jsonl --per_q 1 --max_thought_chars 300 --out $A/lora_r3b > sft_r3b.log 2>&1 || { echo SFT_FAILED; exit 1; }
grep -E "traces from|tokens" sft_r3b.log
.venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/r3b_tools.jsonl --adapter $A/lora_r3b --bs 8 --max_new 512 2>&1 | grep -E "FINAL|Error|Trace"
echo ROUND3B_DONE
