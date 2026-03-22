"""
ERIS v5 — Implicit Feature Detection
======================================
Identifies SAE features that are activated in the hidden space but whose
semantic content is **not explicitly present** in the surface text.

Rationale
----------
When Claude encodes a message like "how do we handle the lock contention?", the
zombie model may activate features like ``concurrency``, ``deadlock_risk``, or
``distributed_systems`` — concepts that are *implied* but never written.  These
implicit activations are the signal that ERIS is designed to surface.

Algorithm
----------
1. Collect activated SAE features (from :class:`~eris.analyzers.SAEAnalyzer`).
2. Build a *text surface* from the input tokens: a set of normalised unigrams
   and bigrams (lowercased, special subword prefixes stripped).
3. For each feature with a non-null label:
   a. Split the label into *concept words* (on ``_``, ``-``, `` ``).
   b. Filter out *stop words* (articles, prepositions, etc.) to focus on
      semantically meaningful terms.
   c. A feature is **present** in the text if ANY concept word appears in the
      text surface (unigrams or bigrams).
   d. A feature is **implicit** if it is NOT present.
4. Features without labels are always considered implicit (they have no
   textual representation by definition).

Matching modes
--------------
- ``"any"`` (default): feature is present if ANY concept word hits the surface.
  High recall — misses fewer implicit features but may over-report.
- ``"all"``: feature is present only if ALL concept words hit the surface.
  High precision — stricter, better for long compound labels like
  ``"distributed_lock_contention"``.

Usage::

    from eris.implicit_features import find_implicit_features

    sae_result = {
        "top_k": [
            {"index": 4521, "activation": 3.72, "label": "code_architecture"},
            {"index": 812,  "activation": 2.10, "label": "deadlock_risk"},
            {"index": 99,   "activation": 1.50, "label": None},
        ]
    }
    tokens = ["how", "do", "we", "handle", "the", "lock", "contention", "?"]
    implicit = find_implicit_features(sae_result, tokens, min_activation=0.5)
    # → [{"index": 4521, "label": "code_architecture", "activation": 3.72},
    #    {"index": 99,   "label": None,                 "activation": 1.50}]
    # "deadlock_risk" is NOT implicit because "lock" is in the text.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set


# ── Stop words ────────────────────────────────────────────────────────────────
# Short, semantically empty words that should not count as a "hit" on their own.
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "either", "each", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "just", "that",
    "this", "these", "those", "it", "its",
}

# Subword prefixes used by various tokenizers (BPE, SentencePiece, WordPiece)
_SUBWORD_PREFIX_RE = re.compile(r"^(##|▁|Ġ|Ċ|ĉ)+")

# Characters to strip when normalising tokens for surface matching
_PUNCT_RE = re.compile(r"[^\w]")


# ── Public API ────────────────────────────────────────────────────────────────

def find_implicit_features(
    sae_result: Optional[Dict],
    tokens: List[str],
    *,
    min_activation: float = 0.0,
    match_mode: str = "any",
) -> List[Dict]:
    """
    Identify activated SAE features absent from the surface text.

    Args:
        sae_result: Output of :meth:`~eris.analyzers.SAEAnalyzer.analyze`, i.e.
                    ``{"top_k": [{"index": ..., "activation": ..., "label": ...}]}``.
                    Returns an empty list if None (SAE not available).
        tokens:     Token strings from :meth:`~engine.LatentRelayEngine.encode`
                    (``result["tokens"]``).
        min_activation: Minimum activation threshold.  Features below this are
                        ignored even if implicit.  Default 0.0 (all features).
        match_mode: ``"any"`` or ``"all"`` — see module docstring.

    Returns:
        List of implicit feature dicts, each with keys
        ``index``, ``label``, ``activation``, sorted by activation descending.
    """
    if sae_result is None:
        return []

    surface = _build_surface(tokens)
    implicit: List[Dict] = []

    for feat in sae_result.get("top_k", []):
        activation: float = feat.get("activation", 0.0)
        if activation < min_activation:
            continue

        label: Optional[str] = feat.get("label")
        index: int = feat.get("index", -1)

        if label is None:
            # No label → always implicit (unknown concept activated)
            implicit.append({
                "index": index,
                "label": None,
                "activation": round(activation, 4),
            })
            continue

        if not _is_present(label, surface, match_mode=match_mode):
            implicit.append({
                "index": index,
                "label": label,
                "activation": round(activation, 4),
            })

    # Highest activation first
    implicit.sort(key=lambda x: x["activation"], reverse=True)
    return implicit


def build_surface_from_text(text: str) -> Set[str]:
    """
    Build a text surface directly from a raw string (word-level tokenisation).

    Useful when the subword token list is not available but the original text is.
    Complements :func:`_build_surface` which works on subword tokens.
    """
    words = [_normalise_word(w) for w in text.lower().split()]
    words = [w for w in words if w and w not in _STOP_WORDS]
    surface: Set[str] = set(words)
    for a, b in zip(words, words[1:]):
        surface.add(f"{a}_{b}")
    return surface


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_surface(tokens: List[str]) -> Set[str]:
    """
    Build a normalised surface set (unigrams + bigrams) from subword tokens.

    Steps:
    1. Strip subword prefixes (``##``, ``▁``, ``Ġ``, etc.)
    2. Lowercase and remove punctuation
    3. Drop empty strings and stop words
    4. Form unigrams and bigrams

    Args:
        tokens: Raw token strings as returned by the tokenizer.

    Returns:
        Set of normalised unigrams and underscore-joined bigrams.
    """
    clean: List[str] = []
    for tok in tokens:
        word = _SUBWORD_PREFIX_RE.sub("", tok)   # strip subword prefix
        word = _PUNCT_RE.sub("", word).lower()    # lowercase + strip punct
        if word and word not in _STOP_WORDS:
            clean.append(word)

    surface: Set[str] = set(clean)
    # Bigrams: "lock_contention", "code_architecture", etc.
    for a, b in zip(clean, clean[1:]):
        surface.add(f"{a}_{b}")

    return surface


def _concept_words(label: str) -> List[str]:
    """
    Split a feature label into semantically meaningful words.

    ``"distributed_lock_contention"`` → ``["distributed", "lock", "contention"]``
    Stop words and single-character words are removed.
    """
    raw = re.split(r"[_\-\s]+", label.lower())
    return [w for w in raw if len(w) > 1 and w not in _STOP_WORDS]


def _is_present(label: str, surface: Set[str], *, match_mode: str) -> bool:
    """
    Return True if the label is represented in the text surface.

    Args:
        label:      Feature label string.
        surface:    Normalised text surface from :func:`_build_surface`.
        match_mode: ``"any"`` — present if ANY concept word is in surface.
                    ``"all"`` — present only if ALL concept words are in surface.
    """
    words = _concept_words(label)
    if not words:
        # Label has no meaningful words after filtering → treat as implicit
        return False

    # Also check the full label joined (handles exact bigram matches like
    # "code_architecture" when the surface already contains that bigram)
    full = "_".join(words)
    if full in surface:
        return True

    hits = [w for w in words if w in surface]

    if match_mode == "all":
        return len(hits) == len(words)
    else:  # "any"
        return len(hits) > 0


def _normalise_word(word: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    return _PUNCT_RE.sub("", word.lower())
