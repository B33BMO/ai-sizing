# ai-sizing

Can a small model become genuinely useful on my own documents with a fraction of a frontier model's compute?
Two tracks, both run on one RTX 5070 Ti (16 GB, WSL2):

1. **memlm**: architecture experiments on a ~25M-param GPT with in-network memory (product-key memory layers,
   pointer/graph memory, shared multi-layer memory) on FineWeb-edu + a synthetic world of 30k fictional people.
2. **agent**: training Qwen3 4B/8B to use my RAG system (Spero: hybrid vector/keyword/graph search over ~12k
   Microsoft 365 doc chunks) as its tool harness, the way Claude is trained on Claude Code.

## Results so far

### Spero QA: 500 held-out paraphrased questions, real tools, fair-graded

| model | Spero QA |
|---|---|
| Qwen3-4B, closed-book (no tools) | 6.4% |
| Qwen3-4B + tools, untrained | 40.2% |
| round 1-2: expert iteration on own + gemma traces | 48.0% |
| round 3b: SFT on GLM-5.3 traces, thoughts trimmed to ≤300 chars | 58.8% |
| **round 3b + forced final answer** | **60.2%** |
| gemma-4-31B + tools | 53.8% |
| GLM-5.3 + tools (Ollama Cloud) | 66.0% |

### memlm (25M core, 500M tokens, single seed)
- Product-key memory: **+12–15 pt fact recall** at the same core size.
- Latent (hidden) 2-hop composition: **at chance** across six memory variants, including graph-supervised pointers.
- Explicit (written-out) 2-hop: **28% → 40%** with memory, vs ~6% hidden.

### Other findings
- Distil the teacher's *actions*, not its prose: copying GLM's long reasoning made the 4B ramble and fail; trimming
  thoughts fixed it (+10.8 pt).
- Logit lens: the 4B settles almost every token at layer ~32/36, so naive early exit doesn't work.
- Untrained Qwen3-4B-Thinking is no better than Instruct on this task: it thinks at length but still searches once.
- Qwen3 hybrid chat templates render history differently from the current turn; single-sequence SFT then learns
  "empty think block ⇒ answer now". `sft.py --per_turn` builds inference-identical prompts.

## Layout

```text
memlm/
  model.py, synth.py, train.py, ents.py     GPT + memory layers, synthetic world, trainer
  exp1.sh … exp4.sh, compare.py, watch.py   experiment launchers, comparison table, live TUI dashboard
  prep_fineweb.py, prep_spero.py, export_spero.py, gen_spero_qa.py, eval_qa.py
  agent/
    protocol.py        tool tags (<search>/<graph>/<answer>), grader, result rendering
    kb-serve.ts        Spero's real searchDocs/exploreGraph as a JSONL server (on a DB snapshot)
    kb_service.py      always-on TCP tool service (pm2 `memlm-kb` on srv)
    gen_traces.py      teacher traces (self-hosted gemma or Ollama Cloud GLM)
    student.py         batched HF agent runner (tools / closed-book, sampling, forced answer, thinking models, 4-bit)
    sft.py             LoRA / QLoRA SFT on correct traces (thought trimming, per-turn mode)
    regrade.py, shorten.py, logit_lens.py, round*.sh
tools/devlog_via_wssh.py   post to the devlog over wssh when off the LAN
```

Data, traces, checkpoints and the Spero corpus are not in the repo (`.gitignore`).

## Running (GPU box)

```sh
memlm-watch                                   # live dashboard
KB_ADDR=192.168.1.248:7787 .venv/bin/python memlm/agent/student.py \
  --questions data/spero_qa.json --out runs/agent/x.jsonl --adapter runs/agent/lora_r3b
```
