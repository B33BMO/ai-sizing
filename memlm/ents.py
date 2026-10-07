"""Entity-labelled tokenization + graph helpers shared by synth.py and prep_spero.py."""
from bisect import bisect_right

import numpy as np

MAX_DEG = 32


def tokenize_with_entities(enc, text, spans):
    """spans: [(char_start, char_end, entity_id)] -> (tokens, ent) where ent[i] = entity id
    on the last token of each mention, -1 elsewhere."""
    toks = enc.encode_ordinary(text)
    ent = np.full(len(toks), -1, dtype=np.int32)
    if spans:
        _, offs = enc.decode_with_offsets(toks)
        for _, end, eid in spans:
            ent[bisect_right(offs, end - 1) - 1] = eid
    return toks, ent


def neighbor_table(n_ent, edges, seed=0):
    """edges: iterable of (src, dst) global ids -> (n_ent, MAX_DEG) int32, padded with -1."""
    rng = np.random.default_rng(seed)
    adj = [[] for _ in range(n_ent)]
    for s, d in edges:
        if s != d:
            adj[s].append(d)
    nbr = np.full((n_ent, MAX_DEG), -1, dtype=np.int32)
    for i, a in enumerate(adj):
        a = sorted(set(a))
        if len(a) > MAX_DEG:
            a = rng.choice(a, MAX_DEG, replace=False)
        nbr[i, : len(a)] = a
    return nbr
