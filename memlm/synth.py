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

import numpy as np
import torch
import torch.nn.functional as F

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


def render_bio(rng, p, city_of):
    sents = [rng.choice(ATTRS[a]).format(n=p["name"], v=p[a]) for a in ATTRS]
    if p["twohop_seen"]:
        sents.append(TWOHOP.format(n=p["name"], v=city_of[p["mentor"]]))
    rng.shuffle(sents)
    return " ".join(sents)


def generate(args):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    world = build_world(args.n_people, args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "world.json"), "w") as f:
        json.dump(world, f)
    city_of = {p["name"]: p["city"] for p in world["people"]}
    rng = random.Random(args.seed + 1)
    n = 0
    with open(os.path.join(args.out, "synth_train.bin"), "wb") as f:
        while n < args.tokens:
            docs = [render_bio(rng, rng.choice(world["people"]), city_of) for _ in range(20000)]
            toks = enc.encode_ordinary_batch(docs, num_threads=16)
            arr = np.concatenate([np.array([enc.eot_token] + t, dtype=np.uint16) for t in toks])
            f.write(arr.tobytes())
            n += len(arr)
            print(f"synth: {n/1e6:.1f}M tokens", flush=True)
    print(f"{args.n_people} people, ~{n / args.n_people / 85:.0f} exposures each")


# ---------------------------------------------------------------- evaluation

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

    @torch.no_grad()
    def run(self, model, device, bs=128):
        seqs, meta = [], []
        for ii, (_, _, prefix, cands) in enumerate(self.items):
            pt = self.enc.encode_ordinary(prefix)
            for ci, c in enumerate(cands):
                ct = self.enc.encode_ordinary(" " + c)
                seqs.append((pt + ct, len(pt), len(ct)))
                meta.append((ii, ci))
        scores = np.zeros((len(self.items), self.n_cand), dtype=np.float32)
        model.eval()
        for s in range(0, len(seqs), bs):
            chunk = seqs[s:s + bs]
            L = max(len(t) for t, _, _ in chunk)
            x = torch.zeros(len(chunk), L, dtype=torch.long)
            mask = torch.zeros(len(chunk), L - 1)  # 1 where position predicts a candidate token
            for j, (t, pl, cl) in enumerate(chunk):
                x[j, : len(t)] = torch.tensor(t)
                mask[j, pl - 1: pl - 1 + cl] = 1
            x, mask = x.to(device), mask.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)[:, :-1]
            # token log-prob without materializing a full log_softmax copy
            tok_lp = logits.gather(-1, x[:, 1:, None]).squeeze(-1) - torch.logsumexp(logits, -1)
            sc = (tok_lp * mask).sum(1).cpu().numpy()
            for j in range(len(chunk)):
                ii, ci = meta[s + j]
                scores[ii, ci] = sc[j]
            del logits, tok_lp
        model.train()
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_people", type=int, default=30000)
    ap.add_argument("--tokens", type=float, default=150e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    generate(ap.parse_args())
