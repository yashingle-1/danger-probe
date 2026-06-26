"""Naive keyword/regex filter — the baseline most products actually ship.

Scores P(dangerous) from the presence of harmful-intent keywords. Deliberately
simple: the point is to show the activation probe beats surface-pattern matching,
especially on jailbreak/manipulation framings that dodge keywords.
"""
from __future__ import annotations

import re

import numpy as np

HARMFUL_PATTERNS = [
    r"\bbomb\b", r"\bexplosiv", r"\bweapon", r"\bmalware\b", r"\bransomware\b",
    r"\bvirus\b", r"\bhack(ing|ed)?\b", r"\bphish", r"\bfraud", r"\blaunder",
    r"\bcounterfeit", r"\bsteal\b", r"\bstalk", r"\bexploit\b", r"\bddos\b",
    r"\bdenial[- ]of[- ]service\b", r"\bforge\b", r"\bsmuggl", r"\bdrug",
    r"\bkill\b", r"\bpoison", r"\bthreaten", r"\bintimidat", r"\bevade tax",
    r"\bdisinformation\b", r"\bmanipulat",
]


class KeywordFilter:
    name = "keyword_filter"

    def __init__(self) -> None:
        self._regexes = [re.compile(p, re.IGNORECASE) for p in HARMFUL_PATTERNS]

    def _score(self, text: str) -> float:
        hits = sum(1 for r in self._regexes if r.search(text))
        if hits == 0:
            return 0.05
        # saturating: 1 hit -> 0.6, 2 -> 0.8, 3+ -> ~0.95
        return float(min(0.95, 0.4 + 0.2 * hits))

    def predict_proba(self, prompts: list[str]) -> np.ndarray:
        return np.array([self._score(p) for p in prompts], dtype=float)
