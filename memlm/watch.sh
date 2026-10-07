#!/bin/sh
# memlm dashboard: run on the GPU box (or from anywhere: wssh ssh -t bmo@windows ~/memlm/watch.sh)
cd ~/memlm && exec .venv/bin/python watch.py
