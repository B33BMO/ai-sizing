#!/bin/bash
# Round 1, resumed after sampling: SFT (positions-only loss) -> eval on test with live tools
cd ~/memlm2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python agent/sft.py --traces runs/agent/r0_train.a0.jsonl,runs/agent/r0_train.a1.jsonl,runs/agent/teacher_train.jsonl \
  --out runs/agent/lora_r1 > sft_r1.log 2>&1 || { echo SFT_FAILED; exit 1; }
.venv/bin/python agent/student.py --questions data/spero_qa.json --out runs/agent/r1_tools.jsonl --adapter runs/agent/lora_r1 --bs 8 2>&1 \
  | grep -E "FINAL|Error|Trace"
echo ROUND1_DONE
