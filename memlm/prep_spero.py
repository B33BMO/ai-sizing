"""Spero export -> entity-labelled token shards, merged entity graph, and cloze eval items.

Global entity ids: synthetic people are 0..P-1 (data/world.json), Spero entities are P + i.
Writes data/spero_{train,val}.bin (uint16), spero_train_ent.bin (int32), graph.npz, spero_cloze.json
"""
import gzip
import json
import random
import re
from collections import defaultdict

import numpy as np
import tiktoken

from ents import neighbor_table, tokenize_with_entities

VAL_FRAC, N_CLOZE, N_CAND, MAX_PREFIX = 0.05, 600, 16, 192
rng = random.Random(0)
enc = tiktoken.get_encoding("gpt2")
d = json.load(gzip.open("data/spero_export.json.gz"))
world = json.load(open("data/world.json"))
P = len(world["people"])

ents = [e for e in d["entities"] if len(e["name"]) >= 3]
gid = {e["id"]: P + i for i, e in enumerate(ents)}
by_id = {e["id"]: e for e in ents}
by_type = defaultdict(list)
for e in ents:
    by_type[e["type"]].append(e["name"])
mentions = defaultdict(list)
for eid, cid in d["mentions"]:
    if eid in gid:
        mentions[cid].append(eid)

# hold out whole source documents, not chunks, so val text is genuinely unseen
paths = sorted({doc["source_path"] for doc in d["docs"]})
rng.shuffle(paths)
val_paths = set(paths[: int(len(paths) * VAL_FRAC)])

out = {"train": ([], []), "val": ([], [])}
cloze = {"train": [], "val": []}
n_spans = 0
for doc in d["docs"]:
    split = "val" if doc["source_path"] in val_paths else "train"
    text = f"{doc['title']}\n\n{doc['content']}"
    spans = []
    for eid in mentions.get(doc["id"], []):
        name = by_id[eid]["name"]
        for m in re.finditer(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", text):
            spans.append((m.start(), m.end(), gid[eid], eid))
    spans.sort()
    toks, ent = tokenize_with_entities(enc, text, [s[:3] for s in spans])
    n_spans += len(spans)
    out[split][0].append(np.array([enc.eot_token] + toks, dtype=np.uint16))
    out[split][1].append(np.concatenate([[-1], ent]).astype(np.int32))
    # one cloze item per chunk: predict an entity mention from the text before it
    cands = [s for s in spans if s[0] > 80 and len(by_type[by_id[s[3]]["type"]]) >= N_CAND]
    if cands:
        st, en, _, eid = rng.choice(cands)
        prefix, name = text[:st], by_id[eid]["name"]
        sp = " " if prefix.endswith(" ") else ""
        prefix = prefix[:-1] if sp else prefix
        others = rng.sample([n for n in by_type[by_id[eid]["type"]] if n != name], N_CAND - 1)
        cloze[split].append({"prefix": enc.encode_ordinary(prefix)[-MAX_PREFIX:],
                             "cands": [enc.encode_ordinary(sp + c) for c in [name] + others],
                             "entity": name, "type": by_id[eid]["type"]})

for split, (toks, entl) in out.items():
    t = np.concatenate(toks)
    t.tofile(f"data/spero_{split}.bin")
    if split == "train":
        np.concatenate(entl).tofile("data/spero_train_ent.bin")
    print(f"spero {split}: {len(t)/1e6:.2f}M tokens, {len(toks)} chunks")
for split in cloze:
    rng.shuffle(cloze[split])
    cloze[split] = cloze[split][:N_CLOZE]
json.dump(cloze, open("data/spero_cloze.json", "w"))
print(f"{n_spans} labelled mentions; cloze items: " + ", ".join(f"{k}={len(v)}" for k, v in cloze.items()))

# merged graph: synthetic person -> mentor (directed), Spero edges (both directions)
pid = {p["name"]: i for i, p in enumerate(world["people"])}
edges = [(i, pid[p["mentor"]]) for i, p in enumerate(world["people"])]
for e in d["edges"]:
    if e["src"] in gid and e["dst"] in gid:
        edges += [(gid[e["src"]], gid[e["dst"]]), (gid[e["dst"]], gid[e["src"]])]
n_ent = P + len(ents)
nbr = neighbor_table(n_ent, edges)
np.savez("data/graph.npz", nbr=nbr, n_people=P, n_ent=n_ent)
deg = (nbr >= 0).sum(1)
print(f"graph: {n_ent} entities, {len(edges)} directed edges, spero mean degree {deg[P:].mean():.1f}")
