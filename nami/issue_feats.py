"""Issue-style features for the router analysis (no model, pure regex).

Hypothesis under test: identifier-dense issues favour lexical search (BM25);
behavioural-prose issues favour likelihood contrast (NAMI). These features are
computed from the issue text only (deployable at test time), plus one
gold-dependent diagnostic (gold_stem_in_issue) used for analysis, never routing.
"""
from __future__ import annotations

import re

_TRACEBACK = re.compile(r"Traceback \(most recent call last\)|File \"[^\"]+\", line \d+")
_FENCE = re.compile(r"```")
_INDENT_CODE = re.compile(r"^(?: {4}|\t)\S", re.MULTILINE)
_PY_PATH = re.compile(r"\b[\w./-]+\.py\b")
_DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){1,}\b")
_SNAKE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")
_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_CALL = re.compile(r"\b[A-Za-z_]\w*\(")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def issue_features(text: str) -> dict:
    words = _WORD.findall(text)
    n_words = max(1, len(words))
    idents = set(_DOTTED.findall(text)) | set(_SNAKE.findall(text)) | \
             set(_CAMEL.findall(text)) | {m[:-1] for m in _CALL.findall(text)}
    return {
        "n_chars": len(text),
        "n_words": len(words),
        "has_traceback": bool(_TRACEBACK.search(text)),
        "n_code_fences": len(_FENCE.findall(text)) // 2,
        "has_indented_code": bool(_INDENT_CODE.search(text)),
        "n_py_paths": len(set(_PY_PATH.findall(text))),
        "n_identifiers": len(idents),
        "ident_per_100w": round(100.0 * len(idents) / n_words, 2),
    }


def gold_stem_in_issue(text: str, gold_paths) -> bool:
    """Diagnostic only: does any gold file's stem (e.g. 'separable') occur in the issue?"""
    if isinstance(gold_paths, str):
        gold_paths = [gold_paths]
    low = text.lower()
    for p in gold_paths:
        stem = p.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if stem not in ("__init__", "base", "utils", "core") and len(stem) > 2 and stem in low:
            return True
    return False


def style_label(feats: dict) -> str:
    """Pre-registered coarse label (thresholds fixed before seeing results)."""
    if feats["has_traceback"] or feats["n_py_paths"] > 0 or feats["ident_per_100w"] >= 8.0:
        return "identifier"
    if feats["n_code_fences"] == 0 and not feats["has_indented_code"] and feats["ident_per_100w"] < 4.0:
        return "prose"
    return "mixed"
