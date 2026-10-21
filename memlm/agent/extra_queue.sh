#!/bin/bash
# After the experience-memory evals:
#  1. 4B retrained with --per_turn (same as the 8B), so 4B vs 8B differ only in model size
#  2. graph-tool ablation: round-3b 4B with the graph tool removed from the prompt (vs 60.2%)
cd ~/memlm2
export KB_ADDR=192.168.1.248:7787 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q MEMORY_DONE memory_queue.log 2>/dev/null; do sleep 60; done
A=runs/agent
.venv/bin/python agent/sft.py --traces $A/glm53_train.short.jsonl --per_q 1 --max_thought_chars 300 --per_turn --out $A/lora_r3b_pt > sft_r3b_pt.log 2>&1 || { echo SFT_FAILED; exit 1; }
.venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/r3bptf_tools.jsonl --adapter $A/lora_r3b_pt --bs 8 --max_new 512 2>&1 | grep -E "FINAL|Error|Trace"
.venv/bin/python agent/student.py --questions data/spero_qa.json --out $A/r3bng_tools.jsonl --adapter $A/lora_r3b --no_graph --bs 8 --max_new 512 2>&1 | grep -E "FINAL|Error|Trace"
echo EXTRA_DONE
