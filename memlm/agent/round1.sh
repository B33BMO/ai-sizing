#!/bin/bash
# Expert-iteration round 1 (runs on the GPU box, in ~/memlm2):
# wait for 2 sampled student attempts -> SFT on correct traces (+ teacher) -> eval on test with live tools
cd ~/memlm2
N=$(python3 -c "import json;print(len(json.load(open('data/spero_qa_train.json'))))")
until [ -f runs/agent/r0_train.a1.jsonl ] && [ $(wc -l < runs/agent/r0_train.a1.jsonl) -ge $N ]; do sleep 60; done
pkill -f "[s]tudent.py --questions data/spero_qa_train.json"
until [ -f runs/agent/teacher_train.jsonl ]; do sleep 60; done   # copied over from srv
sleep 20
.venv/bin/python agent/sft.py --traces runs/agent/r0_train.a0.jsonl,runs/agent/r0_train.a1.jsonl,runs/agent/teacher_train.jsonl \
  --out runs/agent/lora_r1 > sft_r1.log 2>&1 || { echo SFT_FAILED; exit 1; }
.venv/bin/python agent/student.py --questions data/spero_qa.json --out runs/agent/r1_tools.jsonl --adapter runs/agent/lora_r1 --bs 8 2>&1 \
  | grep -E "FINAL|Error|Trace"
echo ROUND1_DONE
