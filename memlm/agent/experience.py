"""Experience memory: recall how similar questions were solved before and show those worked examples.

The model's weights never change; it "learns" by retrieving its own past successes. Memory = correct traces
(training questions only). Recall = BM25 over question text. Each example shows the question, every step's
short thought + tool call, and the final answer, but not the raw tool results (to stay compact).
"""
import json
import math
import re
from collections import Counter

STOP = set("the a an of to in on for and or is are was were be by with what which who where when how does do "
           "that this it its as at from can should".split())


def toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]


class ExperienceMemory:
    def __init__(self, paths, exclude_questions=(), max_thought=150, hide_answers=False):
        """hide_answers: show the procedure only (answers masked), to separate learning *how* from recalling *what*"""
        self.hide_answers = hide_answers
        self.items = []
        for path in paths.split(","):
            for line in open(path):
                r = json.loads(line)
                if (r.get("correct") and r["q"] not in exclude_questions and r["steps"]
                        and r["steps"][-1].get("tool") == "answer"):
                    self.items.append(r)
        self.max_thought = max_thought
        self.docs = [toks(r["q"]) for r in self.items]
        self.df = Counter(w for d in self.docs for w in set(d))
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        self.N = len(self.docs)

    def _bm25(self, q, d, k1=1.2, b=0.75):
        tf = Counter(d)
        s = 0.0
        for w in set(q):
            if w in tf:
                idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / self.avgdl))
        return s

    def recall(self, question, k=3):
        q = toks(question)
        scored = sorted(((self._bm25(q, d), i) for i, d in enumerate(self.docs)), reverse=True)[:k]
        return [self.items[i] for s, i in scored if s > 0]

    def _step(self, s, answer=None):
        text = re.sub(r"</?think>|</?tool_call>|^.{0,3}nk>", " ", s["text"])  # leaked teacher markup
        if self.hide_answers and answer:
            text = re.sub(r"<answer>.*?</answer>", "<answer>[hidden]</answer>", text, flags=re.S)
            text = re.sub(re.escape(answer), "[answer]", text, flags=re.I)
        m = re.search(r"<(search|graph|answer)>", text)
        thought = re.sub(r"\s+", " ", text[: m.start()] if m else text).strip()
        call = text[m.start():].strip() if m else ""
        if len(thought) > self.max_thought:
            thought = "…" + thought[-self.max_thought:]
        return f"{thought} {call}".strip()

    def render(self, question, k=3):
        ex = self.recall(question, k)
        if not ex:
            return "", []
        parts = ["Worked examples of similar questions you solved before (for the approach; check every fact "
                 "with the tools, the answer to this question may differ):"]
        for j, r in enumerate(ex, 1):
            parts.append(f"\nExample {j}. Question: {r['q']}")
            for s in r["steps"]:
                parts.append(f"  - {self._step(s, r['answer'])}")
        return "\n".join(parts) + "\n\nNow the actual question:\n", [r["answer"] for r in ex]
