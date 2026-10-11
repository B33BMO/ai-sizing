#!/bin/bash
# Round 3 (GPU box, ~/memlm2): fresh LoRA on GLM-5.3 traces only (verbose answers shortened by agent/shorten.py),
# then eval on test against the always-on tool service on srv (no Mac bridge).
cd ~/memlm2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KB_ADDR=192.168.1.248:7787
A=runs/agent
.venv/bin/python agent/sft.py --traces $A/glm53_train.short.jsonl --per_q 1 --out $A/lora_r3 > sft_r3.log 2>&1 || { echo SFT_FAILED; exit 1; }
grep -E "traces from|tokens" sft_r3.log
.venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/r3_tools.jsonl --adapter $A/lora_r3 --bs 8 2>&1 | grep -E "FINAL|Error|Trace"
echo ROUND3_DONE
