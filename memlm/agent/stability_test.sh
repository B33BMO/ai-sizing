#!/bin/bash
# Power-capped stability test (after the PSU-trip crash): vLLM + 3 back-to-back engine runs at concurrency 24,
# logging GPU power/util/temp at 2 Hz. Doubles as the engine run-to-run noise measurement.
cd ~/memlm2
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,temperature.gpu,clocks.sm \
  --format=csv,noheader,nounits -lms 500 > gpu_power.csv &
SMI=$!
(agent/serve_vllm.sh runs/agent/lora_r3b r3b > vllm_serve.log 2>&1 &)
until grep -q "Application startup complete" vllm_serve.log 2>/dev/null; do
  grep -q Traceback vllm_serve.log 2>/dev/null && { echo "vLLM failed"; kill $SMI; exit 1; }; sleep 5; done
for i in 2 3 4; do
  rm -f runs/agent/r3bf_engine$i.jsonl
  KB_ADDR=192.168.1.248:7788 ~/vllm-env/.venv/bin/python agent/engine.py --questions data/spero_qa.json \
    --out runs/agent/r3bf_engine$i.jsonl --lora r3b --concurrency 24 2>&1 | grep -E "FINAL|Traceback|Error"
done
kill $SMI
pkill -f "vllm serve"
echo STABILITY_DONE
