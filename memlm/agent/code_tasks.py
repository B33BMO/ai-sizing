"""Build verified coding tasks (small repos) from MBPP train/validation/prompt splits (474 problems; the
257-problem MBPP-sanitized *test* split stays untouched for the regression check).

  implement: solution.py has the signature + description and raises NotImplementedError; test_solution.py shows
             all tests (MBPP's hidden tests sometimes need exact strings no description implies); graded by a copy.
  fix:       solution.py is the reference with one injected bug (operator swap, off-by-one, comparison flip…);
             the visible tests fail; grading runs all tests.
Every task is checked in the sandbox: reference passes all tests, the starting state fails at least one.
Held-out split by *problem* (all tasks of a problem stay on one side).

  python agent/code_tasks.py --out data/code_tasks.json      (writes {"train": [...], "test": [...]})
"""
import argparse
import ast
import json
import random
import re
import socket

from datasets import load_dataset

SB = ("192.168.1.248", 7790)


def sb(req):
    f = socket.create_connection(SB, timeout=90).makefile("rw")
    f.write(json.dumps(req) + "\n")
    f.flush()
    return json.loads(f.readline())


def test_file(tests, imports=""):
    body = "\n".join(f"def test_{i}():\n    {t.strip()}\n" for i, t in enumerate(tests))
    return f"{imports}\nfrom solution import *\n\n{body}"


def passes(files, tests_src):
    """run the full test file against `files` in a fresh sandbox -> True if all pass"""
    for attempt in range(3):
        r = sb({"op": "create", "files": {**files, "test_hidden.py": tests_src}})
        if r.get("ok"):
            break
        import time
        time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(r)
    bid = r["id"]
    try:
        out = sb({"op": "exec", "id": bid, "cmd": "python -m pytest -q -p no:cacheprovider test_hidden.py"})
        return out.get("code") == 0
    finally:
        sb({"op": "destroy", "id": bid})


class Mutator(ast.NodeTransformer):
    """applies the k-th applicable mutation site"""
    SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Mod, ast.Mod: ast.FloorDiv,
            ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq, ast.And: ast.Or, ast.Or: ast.And}

    def __init__(self, target):
        self.target, self.seen, self.kind = target, 0, None

    def _hit(self):
        hit = self.seen == self.target
        self.seen += 1
        return hit

    def visit_BinOp(self, n):
        self.generic_visit(n)
        if type(n.op) in self.SWAP and self._hit():
            self.kind = f"{type(n.op).__name__}->{self.SWAP[type(n.op)].__name__}"
            n.op = self.SWAP[type(n.op)]()
        return n

    def visit_Compare(self, n):
        self.generic_visit(n)
        if type(n.ops[0]) in self.SWAP and self._hit():
            self.kind = f"{type(n.ops[0]).__name__}->{self.SWAP[type(n.ops[0])].__name__}"
            n.ops[0] = self.SWAP[type(n.ops[0])]()
        return n

    def visit_BoolOp(self, n):
        self.generic_visit(n)
        if type(n.op) in self.SWAP and self._hit():
            self.kind = "And<->Or"
            n.op = self.SWAP[type(n.op)]()
        return n

    def visit_Constant(self, n):
        if isinstance(n.value, int) and not isinstance(n.value, bool) and 0 <= n.value <= 10 and self._hit():
            self.kind = f"const {n.value}->{n.value + 1}"
            return ast.copy_location(ast.Constant(n.value + 1), n)
        return n


def mutants(code, rng, k=4):
    tree = ast.parse(code)
    m = Mutator(-1)
    m.visit(ast.parse(code))
    sites = list(range(m.seen))
    rng.shuffle(sites)
    for s in sites[:k]:
        mm = Mutator(s)
        new = ast.unparse(mm.visit(ast.parse(code)))
        if mm.kind and new != ast.unparse(tree):
            yield new, mm.kind


def entry_point(tests, code):
    names = re.findall(r"^def (\w+)\(", code, re.M)
    for n in names:
        if any(re.search(rf"\b{n}\(", t) for t in tests):
            return n
    return names[-1] if names else None


def stub(code, entry, text):
    """keep imports, replace the whole module with the entry function's signature + description"""
    tree = ast.parse(code)
    imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry)
    sig = f"def {entry}({ast.unparse(fn.args)}):"
    return "\n".join(imports + ["", "", sig, f'    """{text.strip()}"""', "    raise NotImplementedError", ""])


def main(a):
    rng = random.Random(0)
    probs = []
    for split in ("train", "validation", "prompt"):
        probs += list(load_dataset("google-research-datasets/mbpp", "full", split=split))
    rng.shuffle(probs)
    test_ids = {p["task_id"] for p in probs[: a.n_test_problems]}
    def build(p):
        try:
            return _build(p)
        except Exception as e:  # one flaky problem must not kill the run
            print(f"skip problem {p['task_id']}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            return []

    def _build(p):
        """-> list of verified tasks for one problem (each problem uses its own rng for reproducibility)"""
        prng = random.Random(p["task_id"])
        found = []
        code, tests = p["code"].replace("\r\n", "\n").replace("\t", "    "), p["test_list"]
        imports = p.get("test_setup_code", "") or ""
        try:
            entry = entry_point(tests, code)
            full_tests = test_file(tests, imports)
            if not entry or not passes({"solution.py": code}, full_tests):
                return found  # reference itself doesn't pass in our harness
        except (SyntaxError, RuntimeError, StopIteration):
            return found
        split = "test" if p["task_id"] in test_ids else "train"
        # implement task
        try:
            s = stub(code, entry, p["text"])
            if not passes({"solution.py": s}, full_tests):
                found.append((split, {"id": f"impl-{p['task_id']}", "type": "implement", "problem": p["task_id"],
                                   "prompt": f"Implement `{entry}` in solution.py so that it does what its docstring "
                                             "says and the tests in test_solution.py pass.",
                                   "files": {"solution.py": s, "test_solution.py": full_tests},
                                   "hidden_tests": full_tests, "reference": code}))
        except (StopIteration, SyntaxError):
            pass
        # fix task: first mutant that actually breaks a test
        for bad, kind in mutants(code, prng):
            if not passes({"solution.py": bad}, full_tests):
                found.append((split, {"id": f"fix-{p['task_id']}", "type": "fix", "problem": p["task_id"],
                                   "prompt": "The tests in test_solution.py fail. Find and fix the bug in "
                                             "solution.py. Change as little as possible.",
                                   "files": {"solution.py": bad, "test_solution.py": full_tests},
                                   "hidden_tests": full_tests, "reference": code, "mutation": kind}))
                break
        return found

    import concurrent.futures as cf
    out = {"train": [], "test": []}
    with cf.ThreadPoolExecutor(4) as ex:
        for i, found in enumerate(ex.map(build, probs)):
            for split, task in found:
                out[split].append(task)
            if i % 50 == 0:
                print(f"{i}/{len(probs)} problems -> train {len(out['train'])}, test {len(out['test'])}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in out.items():
        print(f"{k}: {len(v)} tasks ({sum(t['type'] == 'implement' for t in v)} implement, "
              f"{sum(t['type'] == 'fix' for t in v)} fix)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/code_tasks.json")
    ap.add_argument("--n_test_problems", type=int, default=100)
    main(ap.parse_args())
