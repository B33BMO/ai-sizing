#!/bin/bash
# Experiment 4: explicit multi-hop. Seen-2hop people get a worked answer
# ("Q: Where was the mentor of X born? A: X's mentor is M. M was born in C.");
# unseen people are tested by free generation. Does memory's better recall carry through?
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TOK=${TOK:-500e6}
MIX="--mix web:0.6,synth_cot:0.25,spero:0.15 --cot_eval"
run() { n=$1; shift; [ -f runs/$n/results.json ] && return; .venv/bin/python train.py --name $n --tokens $TOK $MIX "$@" > runs_$n.log 2>&1; }
mkdir -p runs
run e4_cot_mem  --mem_layers 4
run e4_cot_base
.venv/bin/python compare.py e4_cot_base e4_cot_mem
