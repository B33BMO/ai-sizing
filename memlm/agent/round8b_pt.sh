#!/bin/bash
# 8B rerun with per-turn SFT examples (fixes the Qwen3 hybrid template's "think block => answer" shortcut)
cd ~/memlm2
export KB_ADDR=192.168.1.248:7787 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=runs/agent; M="--model Qwen/Qwen3-8B --quant4"
.venv/bin/python agent/sft.py --traces $A/glm53_train.short.jsonl --per_q 1 --max_thought_chars 300 --per_turn $M --out $A/lora_r3b_8b_pt > sft_r3b_8b_pt.log 2>&1 || { echo SFT_FAILED; exit 1; }
grep -E "traces from|tokens" sft_r3b_8b_pt.log
.venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/r3b8pf_tools.jsonl $M --adapter $A/lora_r3b_8b_pt --bs 8 --max_new 512 2>&1 | grep -E "FINAL|Error|Trace"
echo ROUND8B_PT_DONE
