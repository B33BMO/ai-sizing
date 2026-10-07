#!/bin/bash
# Experiment 1: does in-network memory store more facts, and does a pointer graph help latent 2-hop?
# Same core (8L/512d), same data, same tokens. Runs sequentially; results in runs/*/results.json.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TOK=${TOK:-500e6}
run() { n=$1; shift; [ -f runs/$n/results.json ] && return; .venv/bin/python train.py --name $n --tokens $TOK "$@" > runs_$n.log 2>&1; }
mkdir -p runs
run e1_base
run e1_mem      --mem_layers 4
run e1_chain    --mem_layers 4 --mem_hops 2 --mem_pointer value
run e1_ptr      --mem_layers 4 --mem_hops 2 --mem_pointer table
.venv/bin/python compare.py e1_base e1_mem e1_chain e1_ptr
