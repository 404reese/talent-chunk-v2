"""Text normalization helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9+#./& -]+")


def clean_text(value: object) -> str:
    """Return a lowercase text representation with stable whitespace."""
    if value is None:
        return ""
    text = str(value).lower().replace("_", " ")
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def join_text(parts: Iterable[object]) -> str:
    return clean_text(" ".join(str(part) for part in parts if part is not None))


@lru_cache(maxsize=2048)
def _clean_phrase(phrase: str) -> str:
    return clean_text(phrase)


def contains_phrase(text: str, phrase: str) -> bool:
    phrase = _clean_phrase(str(phrase))
    if not phrase:
        return False
    if len(phrase) <= 3 or any(ch in phrase for ch in "+#./&"):
        return phrase in text
    return f" {phrase} " in f" {text} "


def count_phrases(text: str, phrases: Iterable[str]) -> int:
    return sum(1 for phrase in phrases if contains_phrase(text, phrase))
