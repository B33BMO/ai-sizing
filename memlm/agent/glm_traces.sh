#!/bin/bash
# GLM-5.3 (Ollama Cloud) teacher: test reference, then training traces. Runs on the Mac; key from the delegate skill config.
cd "$(dirname "$0")"
set -a; . ~/.claude/skills/delegate/config.env; set +a
G="--workers 4 --api ollama --model glm-5.3 --endpoint https://ollama.com/api/chat --kb 192.168.1.248:7787"
while pgrep -f "[g]en_traces.py ../data/spero_qa.json" > /dev/null; do sleep 30; done   # test run already in flight
python3 gen_traces.py ../data/spero_qa_train.json ../runs/agent/glm53_train.jsonl $G > ../runs/agent/glm53_train.log 2>&1
echo GLM_TRAIN_DONE >> ../runs/agent/glm53_train.log
