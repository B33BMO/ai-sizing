#!/bin/bash
# Experiment 2: does supervising the pointer memory with an external knowledge graph
# (synthetic mentor graph + Spero's LLM-extracted graph) teach the model to walk it internally?
# Starts automatically once Experiment 1 (in ~/memlm) has finished.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "[e]xp1.sh" > /dev/null || pgrep -f "[s]ynth.py --labels" > /dev/null; do sleep 60; done
TOK=${TOK:-500e6}
MIX="--mix web:0.6,synth_lab:0.25,spero:0.15"
PTR="--mem_layers 4 --mem_hops 2 --mem_pointer table"
run() { n=$1; shift; [ -f runs/$n/results.json ] && return; .venv/bin/python train.py --name $n --tokens $TOK $MIX "$@" > runs_$n.log 2>&1; }
mkdir -p runs
run e2_ptr_graph $PTR --graph_w 0.1
run e2_ptr       $PTR
run e2_base
.venv/bin/python compare.py e2_base e2_ptr e2_ptr_graph
