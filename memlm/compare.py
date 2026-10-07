"""Print a comparison table of finished runs: python compare.py [run_name ...]"""
import glob
import json
import os
import sys

names = sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob("runs/*") if os.path.exists(f"{d}/results.json"))
cols = ["val_loss", "acc_1hop", "acc_mentor", "acc_city", "acc_twohop_seen", "acc_twohop_unseen", "acc_twohop_unseen_given_both"]
print(f"{'run':<22}{'core':>7}{'mem':>7}" + "".join(f"{c.replace('acc_', ''):>14}" for c in cols))
for n in names:
    r = json.load(open(f"runs/{n}/results.json"))
    p, f = r["params"], r["final"]
    print(f"{n:<22}{p['core']/1e6:>6.1f}M{p['memory_table']/1e6:>6.0f}M" + "".join(f"{f.get(c, float('nan')):>14.3f}" for c in cols))
