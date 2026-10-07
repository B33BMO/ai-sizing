"""Read-only export of Spero's corpus + graph as gzipped JSON on stdout (stdlib only).

  wssh ssh bmo@srv 'cd /mnt/d/spero/365helper && python3 -' < export_spero.py > spero_export.json.gz
"""
import gzip
import json
import sqlite3
import sys

c = sqlite3.connect("file:data/sage.db?mode=ro", uri=True)
out = {
    "docs": [dict(zip(("id", "source_path", "title", "heading", "content"), r))
             for r in c.execute("SELECT id, source_path, title, heading, content FROM docs ORDER BY source_path, id")],
    "entities": [dict(zip(("id", "name", "type"), r)) for r in c.execute("SELECT id, name, type FROM entities")],
    "edges": [dict(zip(("src", "dst", "rel", "chunk"), r)) for r in c.execute("SELECT src, dst, rel, evidence_chunk_id FROM edges")],
    "mentions": c.execute("SELECT entity_id, chunk_id FROM mentions").fetchall(),
}
sys.stdout.buffer.write(gzip.compress(json.dumps(out).encode()))
