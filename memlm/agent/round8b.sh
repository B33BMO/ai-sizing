#!/bin/bash
# Scale test: Qwen3-8B (thinking off, 4-bit NF4 / QLoRA) with the round-3b recipe.
#  1. untrained 8B, no forced answer        -> compare base2 (4B untrained, 40.2%)
#  2. QLoRA SFT on GLM-5.3 traces, thoughts trimmed to 300 chars
#  3. trained 8B, forced final answer       -> compare r3bf (4B trained, 60.2%)
cd ~/memlm2
export KB_ADDR=192.168.1.248:7787 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=runs/agent; M="--model Qwen/Qwen3-8B --quant4"
until grep -q '"' ~/hf_dl_8b.log 2>/dev/null || ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/model-00005-of-00005.safetensors >/dev/null 2>&1; do sleep 30; done
.venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/base8_tools.jsonl $M --no_force_answer --bs 8 --max_new 512 2>&1 | grep -E "FINAL|Error|Trace"
.venv/bin/python agent/sft.py --traces $A/glm53_train.short.jsonl --per_q 1 --max_thought_chars 300 $M --out $A/lora_r3b_8b > sft_r3b_8b.log 2>&1 || { echo SFT_FAILED; exit 1; }
grep -E "traces from|tokens" sft_r3b_8b.log
.venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/r3b8f_tools.jsonl $M --adapter $A/lora_r3b_8b --bs 8 --max_new 512 2>&1 | grep -E "FINAL|Error|Trace"
echo ROUND8B_DONE
