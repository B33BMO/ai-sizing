"""Stream FineWeb-edu and write GPT-2 token shards: data/fineweb_{train,val}.bin (uint16)."""
import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=float, default=600e6)
ap.add_argument("--val_tokens", type=float, default=5e6)
ap.add_argument("--out", default="data")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)


def write(path, target, it):
    buf, n = [], 0
    with open(path, "wb") as f:
        batch = []
        for ex in it:
            batch.append(ex["text"])
            if len(batch) < 1024:
                continue
            for toks in enc.encode_ordinary_batch(batch, num_threads=16):
                buf.append(np.array([eot] + toks, dtype=np.uint16))
            batch = []
            arr = np.concatenate(buf)
            buf = []
            arr = arr[: int(target) - n]
            f.write(arr.tobytes())
            n += len(arr)
            print(f"{path}: {n/1e6:.1f}M tokens", flush=True)
            if n >= target:
                return


it = iter(ds)
write(os.path.join(args.out, "fineweb_val.bin"), args.val_tokens, it)
write(os.path.join(args.out, "fineweb_train.bin"), args.tokens, it)
print("done")
