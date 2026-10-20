# Read-only Spero tool service: Spero's real searchDocs/exploreGraph over a search-only DB.
# Build context = ~/memlm-spero/jobpack-kb on srv (see bundle_kb.sh). Listens on 0.0.0.0:7787.
FROM node:20-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 build-essential && rm -rf /var/lib/apt/lists/*
# Spero code at its original path so kb-serve.ts imports resolve unchanged
WORKDIR /mnt/d/spero/365helper
RUN npm init -y >/dev/null && npm install --omit=dev --no-audit --no-fund \
      @huggingface/transformers@^4.1.0 @next/env@16.2.4 better-sqlite3@^12.9.0 sqlite-vec@^0.1.9 tsx@^4.21.0
COPY spero/ ./
COPY kb-serve.ts kb_service.py /root/memlm-spero/
COPY sage.db /root/memlm-spero/sage.db
ENV HOME=/root KB_HOST=0.0.0.0 KB_PORT=7787 NODE_ENV=production
EXPOSE 7787
CMD ["python3", "/root/memlm-spero/kb_service.py"]
