"""The tool protocol shared by the teacher (trace generation), the student (SFT) and evaluation."""
import re

SYSTEM = """You answer questions about Microsoft 365 by consulting a knowledge base. Do not answer from memory: look things up first.
Tools - make exactly one call per message, then stop and wait for its result:
<search>keywords or a question</search>  searches the documentation and returns the top passages with their sources.
<graph>Entity Name</graph>  returns an entity's description and its relationships to other entities.
Each result arrives as <result>...</result>.
Before each call, think in one or two short sentences. Use at most 4 calls.
When you know the answer, reply with <answer>the shortest exact answer, such as a product, feature or setting name</answer>."""

TOOLS_RE = re.compile(r"<(search|graph|answer)>(.*?)</\1>", re.S)
MAX_RESULT_CHARS = 3500


LAST_CALL = "No tool calls left: reply now with <answer>...</answer>."


def render_result(text, last=False):
    """last: this was the final allowed tool call, so the model must answer next"""
    text = text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "\n[...truncated]"
    return f"<result>\n{text}\n</result>" + (f"\n{LAST_CALL}" if last else "")


def has_duplicate_calls(steps):
    """true if the same tool was called twice with the same argument (a wasted turn)"""
    calls = [(s.get("tool"), " ".join(norm(s.get("arg")))) for s in steps if s.get("tool") in ("search", "graph")]
    return len(calls) != len(set(calls))


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


def grade(answer, gold):
    """Correct if the gold entity's words all appear in the answer, in order, as a contiguous phrase."""
    if not answer:
        return False
    a, g = " ".join(norm(answer)), " ".join(norm(gold))
    return bool(g) and f" {g} " in f" {a} "
