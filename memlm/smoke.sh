#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# quick end-to-end check of each variant
for v in "base|" "mem|--mem_layers 4" "ptr|--mem_layers 4 --mem_hops 2 --mem_pointer table"; do
  n=${v%%|*}; a=${v#*|}; echo "=== $n"
  .venv/bin/python train.py --name smoke_$n $a --tokens 3.3e6 --log_every 50 --eval_every 100000 2>&1 | grep --line-buffered -E "EVAL|tok_s|Error|error|step\"" | tail -4
done
