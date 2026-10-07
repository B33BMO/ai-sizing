#!/bin/bash
# Experiment 3: can the second lookup happen *after* attention has moved the first one's result?
#   e3_shared          one memory table read at layers 2 and 6
#   e3_shared_graph    + cross-layer graph supervision through a gate that starts closed
#   e3_ptr_gate_graph  Exp-2 pointer model with the hop-2 read gated and 5x lower graph weight
# Compare against e2_base / e2_ptr (same data mix).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TOK=${TOK:-500e6}
MIX="--mix web:0.6,synth_lab:0.25,spero:0.15"
run() { n=$1; shift; [ -f runs/$n/results.json ] && return; .venv/bin/python train.py --name $n --tokens $TOK $MIX "$@" > runs_$n.log 2>&1; }
mkdir -p runs
run e3_shared          --mem_layers 2,6 --mem_share
run e3_shared_graph    --mem_layers 2,6 --mem_share --mem_gate2 --graph_mode cross --graph_w 0.02
run e3_ptr_gate_graph  --mem_layers 4 --mem_hops 2 --mem_pointer table --mem_gate2 --graph_w 0.02
.venv/bin/python compare.py e2_base e2_ptr e2_ptr_graph e3_shared e3_shared_graph e3_ptr_gate_graph
