"""Deterministic messy-English rewrites, for clean/messy pairs.

The same answer in a messier hand: the robustness check asks whether a model makes
the same move on both. Code identifiers, file names and numbers are left alone, so
the content of the answer does not change, only the writing.
"""

from __future__ import annotations

import random
import re

_SMS = {
    "you": "u", "your": "ur", "are": "r", "because": "bcoz", "please": "pls",
    "something": "smth", "though": "tho", "with": "wid", "and": "n", "okay": "ok",
    "i don't know": "idk", "to be honest": "tbh", "people": "ppl", "thanks": "thx",
}
_FILLERS = ["um", "uh", "like", "basically", "so yeah", "i mean", "actually"]
_PROTECT = re.compile(r"`[^`]*`|\b[\w./-]+\.(?:py|ts|tsx|js|c|h|rs|go|sql|java)\b|\b\w*[_()\[\]]\w*\b|\d+(?:\.\d+)?")


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4 or not word.isalpha():
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(["swap", "drop", "double"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def corrupt(text: str, *, seed: int, kind: str = "mixed") -> str:
    """``kind``: typos · sms · dictated · lower · mixed."""
    rng = random.Random(seed)
    kinds = ["typos", "sms", "dictated", "lower"] if kind == "mixed" else [kind]
    protected = {}

    def keep(m):
        key = f"\x00{len(protected)}\x00"
        protected[key] = m.group(0)
        return key

    out = _PROTECT.sub(keep, text)
    if "sms" in kinds or "lower" in kinds:
        out = out.lower()
        out = re.sub(r"[.!;:]+(\s|$)", r"\1", out)
    if "sms" in kinds:
        for long, short in sorted(_SMS.items(), key=lambda kv: -len(kv[0])):
            out = re.sub(rf"\b{re.escape(long)}\b", short, out)
    if "typos" in kinds:
        words = out.split(" ")
        out = " ".join(_typo(w, rng) if rng.random() < 0.18 else w for w in words)
    if "dictated" in kinds:
        words = out.split(" ")
        for _ in range(max(1, len(words) // 12)):
            words.insert(rng.randrange(0, len(words) + 1), rng.choice(_FILLERS))
        out = " ".join(words).replace("\n\n", " ").replace("\n", " ")
    for key, val in protected.items():
        out = out.replace(key, val)
    return out
