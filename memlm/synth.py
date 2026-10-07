"""Synthetic knowledge world: fictional people with invented attributes.

Measures what a model *stores* and whether it can *compose* stored facts:
  - 1-hop:  "{person} was born in" -> city           (stated in training)
  - 2-hop:  "The mentor of {person} was born in" -> mentor's city
            stated for half the people ("seen"), never for the other half ("unseen").
            Unseen 2-hop requires latent composition inside one forward pass.
Everything is invented, so GPT-2 token priors cannot leak the answers.
"""
import argparse
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F

from ents import tokenize_with_entities

ONSETS = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "br", "dr", "gr", "kr", "tr", "th", "sh", "st", "vr", "zh", "qu"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "y"]
CODAS = ["", "", "n", "r", "l", "s", "th", "x", "m", "nd", "rk"]

ATTRS = {
    "city": ["{n} was born in {v}.", "{n} is a native of {v}.", "{n} spent their childhood in {v}.", "{n} hails from {v}."],
    "year": ["{n}'s birth year is {v}.", "{n} was born during {v}.", "{n} entered the world in {v}."],
    "uni": ["{n} studied at {v}.", "{n} graduated from {v}.", "{n} received an education at {v}."],
    "company": ["{n} works for {v}.", "{n} is employed by {v}.", "{n} has a job at {v}."],
    "field": ["{n} specializes in {v}.", "{n} is an expert in {v}.", "{n} does research on {v}."],
    "mentor": ["{n} was mentored by {v}.", "{n} learned the trade from {v}.", "{n}'s mentor is {v}."],
}
TWOHOP = "The mentor of {n} was born in {v}."
# explicit 2-hop: the model writes the intermediate entity, then does a plain 1-hop lookup on it
COT_Q = "Q: Where was the mentor of {n} born? A:"
COT = COT_Q + " {n}'s mentor is {m}. {m} was born in {v}."


def word(rng, syl):
    return "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(syl)).capitalize()


def unique_words(rng, k, syl, taken):
    out = []
    while len(out) < k:
        w = word(rng, rng.choice(syl))
        if w not in taken:
            taken.add(w)
            out.append(w)
    return out


def build_world(n_people, seed=0):
    rng = random.Random(seed)
    taken = set()
    firsts = unique_words(rng, 400, [2], taken)
    lasts = unique_words(rng, 3000, [2, 3], taken)
    pools = {
        "city": unique_words(rng, 200, [2, 3], taken),
        "uni": [f"the University of {w}" for w in unique_words(rng, 100, [2, 3], taken)],
        "company": [f"{w} {rng.choice(['Corp', 'Labs', 'Industries', 'Group'])}" for w in unique_words(rng, 300, [2], taken)],
        "field": [w.lower() + rng.choice(["ics", "ology", "istry", "onomy"]) for w in unique_words(rng, 60, [2], taken)],
        "year": [str(y) for y in range(1900, 2000)],
    }
    names = set()
    while len(names) < n_people:
        names.add(f"{rng.choice(firsts)} {rng.choice(lasts)}")
    names = sorted(names)
    rng.shuffle(names)
    people = []
    for i, n in enumerate(names):
        p = {"name": n, **{a: rng.choice(pools[a]) for a in pools}}
        m = rng.randrange(n_people - 1)
        p["mentor"] = names[m if m < i else m + 1]
        p["twohop_seen"] = i % 2 == 0
        people.append(p)
    pools["mentor"] = names
    return {"people": people, "pools": pools}


def render_bio(rng, p, city_of, ids=None, cot=False):
    """-> text, or (text, spans) with spans [(start, end, entity_id)] for every name when ids is given.
    cot: seen-2hop people get a worked Q/A (COT) instead of the direct TWOHOP sentence."""
    sents = [(rng.choice(ATTRS[a]), {"n": (p["name"], True), "v": (p[a], a == "mentor")}) for a in ATTRS]
    if p["twohop_seen"]:
        m = p["mentor"]
        sents.append((COT, {"n": (p["name"], True), "m": (m, True), "v": (city_of[m], False)}) if cot else
                     (TWOHOP, {"n": (p["name"], True), "v": (city_of[m], False)}))
    rng.shuffle(sents)
    text, spans = "", []
    for tmpl, slots in sents:
        if text:
            text += " "
        for k in slots:
            tmpl = tmpl.replace("{%s}" % k, "\0%s\0" % k)
        for part in tmpl.split("\0"):
            if part in slots:
                piece, is_person = slots[part]
                if ids is not None and is_person:
                    spans.append((len(text), len(text) + len(piece), ids[piece]))
                text += piece
            else:
                text += part
    return text if ids is None else (text, spans)


def generate(args):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(args.out, exist_ok=True)
    wpath = os.path.join(args.out, "world.json")
    if os.path.exists(wpath):  # never silently change the world other files were built from
        world = json.load(open(wpath))
        assert len(world["people"]) == args.n_people, "world.json exists with a different n_people"
    else:
        world = build_world(args.n_people, args.seed)
        json.dump(world, open(wpath, "w"))
    city_of = {p["name"]: p["city"] for p in world["people"]}
    ids = {p["name"]: i for i, p in enumerate(world["people"])} if args.labels else None
    rng = random.Random(args.seed + 1 + (100 if args.labels else 0))
    stem = "synth_cot_train" if args.cot else "synth_lab_train" if args.labels else "synth_train"
    n = 0
    ftok = open(os.path.join(args.out, stem + ".bin"), "wb")
    fent = open(os.path.join(args.out, stem + "_ent.bin"), "wb") if args.labels else None
    while n < args.tokens:
        people = [rng.choice(world["people"]) for _ in range(20000)]
        if not args.labels:
            toks = enc.encode_ordinary_batch([render_bio(rng, p, city_of, cot=args.cot) for p in people], num_threads=16)
            arr = np.concatenate([np.array([enc.eot_token] + t, dtype=np.uint16) for t in toks])
        else:
            toks, ents = [], []
            for p in people:
                t, e = tokenize_with_entities(enc, *render_bio(rng, p, city_of, ids, cot=args.cot))
                toks.append(np.array([enc.eot_token] + t, dtype=np.uint16))
                ents.append(np.concatenate([[-1], e]).astype(np.int32))
            arr = np.concatenate(toks)
            fent.write(np.concatenate(ents).tobytes())
        ftok.write(arr.tobytes())
        n += len(arr)
        print(f"{stem}: {n/1e6:.1f}M tokens", flush=True)
    print(f"{args.n_people} people, ~{n / args.n_people / 85:.0f} exposures each")


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def score_candidates(model, device, items, max_tokens=4096):
    """items: [(prefix_tokens, [candidate_tokens, ...])] -> (n_items, n_cand) summed log-probs."""
    seqs = [(pt + ct, len(pt), len(ct), ii, ci) for ii, (pt, cands) in enumerate(items) for ci, ct in enumerate(cands)]
    seqs.sort(key=lambda s: len(s[0]))
    scores = np.zeros((len(items), max(len(c) for _, c in items)), dtype=np.float32)
    was_training = model.training
    model.eval()
    s = 0
    while s < len(seqs):
        L = len(seqs[min(len(seqs) - 1, s + 1)][0])
        bs = max(1, max_tokens // max(L, 1))
        chunk = seqs[s:s + bs]
        L = max(len(t) for t, *_ in chunk)
        x = torch.zeros(len(chunk), L, dtype=torch.long)
        mask = torch.zeros(len(chunk), L - 1)  # 1 where position predicts a candidate token
        for j, (t, pl, cl, _, _) in enumerate(chunk):
            x[j, : len(t)] = torch.tensor(t)
            mask[j, pl - 1: pl - 1 + cl] = 1
        x, mask = x.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)[:, :-1]
        # token log-prob without materializing a full log_softmax copy
        tok_lp = logits.gather(-1, x[:, 1:, None]).squeeze(-1) - torch.logsumexp(logits, -1)
        sc = (tok_lp * mask).sum(1).cpu().numpy()
        for j, (*_, ii, ci) in enumerate(chunk):
            scores[ii, ci] = sc[j]
        del logits, tok_lp
        s += len(chunk)
    model.train(was_training)
    return scores


class SynthEval:
    def __init__(self, world, enc, n_eval=500, n_cand=16, seed=0):
        self.enc = enc
        rng = random.Random(seed)
        people = world["people"]
        self.by_name = {p["name"]: p for p in people}
        self.pools = world["pools"]
        self.sample = rng.sample(people, min(n_eval, len(people)))
        self.n_cand = n_cand
        self.rng = rng
        self.items = self._build()

    def _cands(self, pool, correct):
        c = [correct] + self.rng.sample([x for x in pool if x != correct] if len(pool) < 1000 else
                                        [x for x in self.rng.sample(pool, self.n_cand * 2) if x != correct], self.n_cand - 1)
        return c

    def _build(self):
        items = []  # (kind, person, prefix, candidates)  correct = candidates[0]
        for p in self.sample:
            for a, tmpls in ATTRS.items():
                prefix = tmpls[0].split("{v}")[0].format(n=p["name"]).rstrip()
                items.append((a, p["name"], prefix, self._cands(self.pools[a], p[a])))
            mc = self.by_name[p["mentor"]]["city"]
            prefix = TWOHOP.split("{v}")[0].format(n=p["name"]).rstrip()
            kind = "twohop_seen" if p["twohop_seen"] else "twohop_unseen"
            items.append((kind, p["name"], prefix, self._cands(self.pools["city"], mc)))
            if not p["twohop_seen"]:  # does the model know hop 2 (the mentor's own city)?
                prefix = ATTRS["city"][0].split("{v}")[0].format(n=p["mentor"]).rstrip()
                items.append(("mentor_city", p["name"], prefix, self._cands(self.pools["city"], mc)))
        return items

    def run(self, model, device):
        enc = self.enc.encode_ordinary
        scores = score_candidates(model, device, [(enc(prefix), [enc(" " + c) for c in cands])
                                                  for _, _, prefix, cands in self.items])
        correct = scores.argmax(1) == 0
        res, know = {}, {}
        for (kind, name, _, _), ok in zip(self.items, correct):
            res.setdefault(kind, []).append(ok)
            know[(kind, name)] = ok
        out = {f"acc_{k}": float(np.mean(v)) for k, v in res.items()}
        out["acc_1hop"] = float(np.mean([out[f"acc_{a}"] for a in ATTRS]))
        out.pop("acc_mentor_city")
        # unseen 2-hop, restricted to people where the model knows BOTH hops individually
        cond = [ok for (kind, name, _, _), ok in zip(self.items, correct) if kind == "twohop_unseen"
                and know[("mentor", name)] and know[("mentor_city", name)]]
        out["acc_twohop_unseen_given_both"] = float(np.mean(cond)) if cond else float("nan")
        out["n_twohop_given_both"] = len(cond)
        out["chance"] = 1 / self.n_cand
        return out


class CotEval:
    """Greedy-generate answers to COT_Q and grade the chain: mentor step, mentor's-city step, final answer."""
    PAT = re.compile(r"^ (.+?)'s mentor is (.+?)\. (.+?) was born in (.+?)\.")

    def __init__(self, world, enc, n_eval=400, seed=1):
        rng = random.Random(seed)
        self.enc = enc
        self.by_name = {p["name"]: p for p in world["people"]}
        seen = [p for p in world["people"] if p["twohop_seen"]]
        unseen = [p for p in world["people"] if not p["twohop_seen"]]
        self.sample = rng.sample(seen, n_eval // 2) + rng.sample(unseen, n_eval // 2)

    @torch.no_grad()
    def run(self, model, device, max_new=40):
        enc, eot = self.enc, self.enc.eot_token
        prompts = [enc.encode_ordinary(COT_Q.format(n=p["name"])) for p in self.sample]
        outs = [None] * len(prompts)
        by_len = {}
        for i, t in enumerate(prompts):
            by_len.setdefault(len(t), []).append(i)
        was_training = model.training
        model.eval()
        for L, idxs in by_len.items():  # equal-length prompts batch without padding
            x = torch.tensor([prompts[i] for i in idxs], device=device)
            for _ in range(max_new):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    nxt = model(x[:, -model.cfg.seq_len:])[:, -1].argmax(-1, keepdim=True)
                x = torch.cat([x, nxt], 1)
            for j, i in enumerate(idxs):
                gen = x[j, L:].tolist()
                outs[i] = enc.decode(gen[: gen.index(eot)] if eot in gen else gen)
        model.train(was_training)
        res = {"seen": [], "unseen": []}
        for p, out in zip(self.sample, outs):
            m = self.PAT.match(out)
            true_m = p["mentor"]
            true_c = self.by_name[true_m]["city"]
            r = {"format": bool(m), "final": bool(m) and m.group(4) == true_c,
                 "step1": bool(m) and m.group(2) == true_m,
                 # step 2 graded on whoever the model named: does it know *that* person's city?
                 "step2": bool(m) and m.group(2) in self.by_name and m.group(3) == m.group(2)
                          and m.group(4) == self.by_name[m.group(2)]["city"]}
            res["seen" if p["twohop_seen"] else "unseen"].append(r)
        out = {}
        for split, rs in res.items():
            for k in ["format", "step1", "step2", "final"]:
                out[f"cot_{split}_{k}"] = float(np.mean([r[k] for r in rs]))
        self.last_outputs = list(zip([p["name"] for p in self.sample], outs))
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_people", type=int, default=30000)
    ap.add_argument("--tokens", type=float, default=150e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", action="store_true", help="also write per-token entity labels (synth_lab_train*)")
    ap.add_argument("--cot", action="store_true", help="worked Q/A 2-hop format (synth_cot_train*)")
    ap.add_argument("--out", default="data")
    generate(ap.parse_args())
