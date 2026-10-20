#!/bin/bash
# Run on srv: assemble the build context for the read-only Spero tool image in ~/memlm-spero/jobpack-kb.
# Ships only what searchDocs/exploreGraph import, the embedding model, and the SEARCH-ONLY DB
# (built by make_search_db.cjs: no conversations, memories, users, sessions, secrets, runs).
set -euo pipefail
SPERO=/mnt/d/spero/365helper
OUT=~/memlm-spero/jobpack-kb
mkdir -p $OUT/spero/lib/rag $OUT/spero/lib/tools $OUT/spero/lib/graph $OUT/spero/scripts $OUT/spero/data
for f in lib/db.ts lib/embed.ts lib/graph/db.ts lib/graph/extract.ts lib/llm.ts lib/rag/rerank.ts lib/rag/search.ts \
         lib/secrets.ts lib/settings.ts lib/tools/explore-graph.ts lib/usage.ts scripts/_chdir-root.ts scripts/_env.ts tsconfig.json; do
  cp $SPERO/$f $OUT/spero/$f
done
cp -r $SPERO/data/models $OUT/spero/data/
echo '{"rerank": "off", "autoGraph": false}' > $OUT/spero/data/settings.json
[ -f $OUT/data/sage.db ] || node ~/memlm-spero/make_search_db.cjs
mv -f $OUT/data/sage.db $OUT/sage.db 2>/dev/null || true
cp ~/memlm-spero/kb-serve.ts ~/memlm-spero/kb_service.py $OUT/
du -sh $OUT
