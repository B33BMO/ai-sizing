"""Peak GPU memory of fwd/bwd for a config: python memprof.py [--mem]"""
import sys
import torch
from model import GPT, Config

cfg = Config(mem_layers=[4]) if "--mem" in sys.argv else Config()
m = GPT(cfg).cuda()


def rep(tag):
    torch.cuda.synchronize()
    print(tag, round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
    torch.cuda.reset_peak_memory_stats()


rep("model")
x = torch.randint(0, 50000, (16, 513), device="cuda")
for comp in [False, True]:
    f = torch.compile(m) if comp else m
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l = f(x[:, :-1], x[:, 1:])[1]
    rep(f"fwd compile={comp}")
    l.backward()
    rep(f"bwd compile={comp}")
    m.zero_grad(set_to_none=True)

import json, tiktoken
from synth import SynthEval
torch.cuda.reset_peak_memory_stats()
import time; t = time.time()
r = SynthEval(json.load(open("data/world.json")), tiktoken.get_encoding("gpt2")).run(m, "cuda")
rep(f"synth eval ({time.time() - t:.1f}s) acc_1hop={r['acc_1hop']:.3f}")
