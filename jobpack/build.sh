#!/bin/bash
# Build both images (run on srv, which has Docker): memlm-kb (read-only Spero tools) and memlm-train.
# Then export them for the GPU machine:  docker save memlm-kb memlm-train | gzip > memlm-jobpack.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
~/memlm-spero/bundle_kb.sh && cp jobpack/kb/Dockerfile ~/memlm-spero/jobpack-kb/Dockerfile
docker build -t memlm-kb:latest ~/memlm-spero/jobpack-kb
mkdir -p jobpack/data   # training data bundle (not in git)
cp memlm/data/spero_qa.json memlm/data/spero_qa_train.json memlm/runs/agent/glm53_train.short.jsonl jobpack/data/
docker build -t memlm-train:latest -f jobpack/train/Dockerfile .
docker images --format "{{.Repository}}:{{.Tag}}  {{.Size}}" | grep -E "^memlm-(kb|train)"
