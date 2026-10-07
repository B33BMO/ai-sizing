"""Score saved checkpoints on the paraphrased Spero QA set (data/spero_qa.json).

  python eval_qa.py e2_base e2_ptr e3_shared ...
Ranks the true entity among 15 same-type distractors (chance 6.25%) given "Question: ...\\nAnswer:".
"""
import json
import os
import sys

import numpy as np
import tiktoken
import torch

from model import GPT, Config
from synth import score_candidates

enc = tiktoken.get_encoding("gpt2")
qa = json.load(open("data/spero_qa.json"))
items = [(enc.encode_ordinary(f"Question: {it['q']}\nAnswer:"), [enc.encode_ordinary(" " + c) for c in it["cands"]]) for it in qa]
OUT = "runs/spero_qa_results.json"
results = json.load(open(OUT)) if os.path.exists(OUT) else {}
for name in sys.argv[1:]:
    ck = torch.load(f"runs/{name}/ckpt.pt", map_location="cuda", weights_only=False)
    model = GPT(Config(**ck["cfg"])).cuda()
    model.load_state_dict(ck["model"])
    ok = score_candidates(model, "cuda", items).argmax(1) == 0
    se = ok.std() / np.sqrt(len(ok))
    results[name] = {"spero_qa": float(ok.mean()), "se": float(se), "n": len(ok)}
    print(f"{name:<22} spero_qa {ok.mean()*100:5.1f}%  ±{se*100:.1f}  (n={len(ok)}, chance 6.2%)", flush=True)
    del model, ck
    torch.cuda.empty_cache()
    json.dump(results, open(OUT, "w"), indent=1)
