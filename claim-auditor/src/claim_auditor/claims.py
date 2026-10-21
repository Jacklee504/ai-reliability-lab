"""Claim extraction and qualifier scanning.

The extractor splits an answer into atomic claims while preserving the
qualifiers that change their meaning: dates, quantities, subject,
scope, negation, and modality.

Offsets are preserved exactly. Every claim's ``text`` is a verbatim
slice of the answer it came from, so a finding can be pointed back at
the original wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .types import Claim, Qualifiers, short_id

_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|(?:19|20)\d{2}|Q[1-4]\s*(?:19|20)?\d{2})\b")
_QUANTITY_RE = re.compile(
    r"\b[\d][\d,.]*\s*(?:%|percent|eur|euro|euros|usd|gbp|days?|weeks?|months?|years?|hours?|minutes?)\b",
    re.IGNORECASE,
)
_SCOPE_RE = re.compile(r"\b(all|every|each|some|only|most|any|none|no)\b", re.IGNORECASE)
_SUBJECT_RE = re.compile(r"\bon\s+(\w+)\b", re.IGNORECASE)
_MODALITY_RE = re.compile(r"\b(must|shall|should|may|might|can|could|will|would)\b", re.IGNORECASE)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|without|cannot|can't|doesn't|doesn't|isn't|aren't|won't)\b",
    re.IGNORECASE,
)

#: Clause boundaries. Splitting on these keeps a coordinated sentence
#: from becoming one claim with two outcomes.
_SPLIT_RE = re.compile(
    r",\s+(?=(?:and|but|whereas|while|although|though|however)\b)"
    r"|\s+(?:and|but|whereas|while|although|though)\s+(?=(?:the|a|an|it|they|this|that|these|those)\b)",
    re.IGNORECASE,
)

#: The words that mark a strict, universal scope.
_STRICT_SCOPE = {"all", "every", "each", "only"}


@dataclass
class ExtractionSettings:
    """Tunables for the extractor."""

    split_on_markers: bool = True
    max_claim_chars: int = 400


def extract_qualifiers(text: str) -> Qualifiers:
    """Scan a clause for the qualifiers that must survive extraction."""
    date = _DATE_RE.search(text)
    quantity = _QUANTITY_RE.search(text)
    scope = _SCOPE_RE.search(text)
    subject = _SUBJECT_RE.search(text)
    modality = _MODALITY_RE.search(text)
    return Qualifiers(
        date=date.group(0) if date else None,
        quantity=quantity.group(0).strip() if quantity else None,
        subject=subject.group(1).strip() if subject else None,
        scope=scope.group(0).lower() if scope else None,
        negation=bool(_NEGATION_RE.search(text)),
        modality=modality.group(0).lower() if modality else None,
    )


def is_strict_claim(claim: Claim) -> bool:
    """True when the claim makes a universal assertion.

    A claim about "all" is neither supported nor refuted by a passage
    about one example, so the verdict stage needs to know.
    """
    scope = (claim.qualifiers.scope or "").lower()
    if scope in _STRICT_SCOPE:
        return True
    return bool(re.search(r"\b(?:all|every|each|only|no|none|never)\b", claim.text, re.IGNORECASE))


def _clauses(text: str, settings: ExtractionSettings) -> list[tuple[int, str]]:
    """Split a sentence into clause candidates, keeping their offsets."""
    if not settings.split_on_markers:
        return [(0, text)]
    pieces: list[tuple[int, str]] = []
    cursor = 0
    for match in _SPLIT_RE.finditer(text):
        piece = text[cursor : match.start()]
        if piece.strip():
            pieces.append((cursor, piece))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        pieces.append((cursor, tail))
    return pieces


def split_answer_into_claims(
    answer: str,
    *,
    claim_prefix: str = "cl",
    settings: ExtractionSettings | None = None,
) -> list[Claim]:
    """Split an answer into atomic claims, preserving answer offsets.

    Each claim keeps the exact answer substring it came from, so a
    report can point back at the original wording.
    """
    settings = settings or ExtractionSettings()
    claims: list[Claim] = []
    counter = 0

    for sentence_start, sentence_end in _sentences(answer):
        sentence = answer[sentence_start:sentence_end]
        for clause_offset, clause in _clauses(sentence, settings):
            stripped = clause.strip()
            if not stripped:
                continue
            lead = clause.index(stripped[0])
            start = sentence_start + clause_offset + lead
            end = start + len(stripped)
            if end - start > settings.max_claim_chars:
                end = start + settings.max_claim_chars
                stripped = answer[start:end]
            counter += 1
            claims.append(
                Claim(
                    claim_id="%s_%03d" % (claim_prefix, counter),
                    text=stripped,
                    answer_start=start,
                    answer_end=end,
                    qualifiers=extract_qualifiers(stripped),
                )
            )
    return claims


_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9“\"(])")


def _sentences(text: str) -> list[tuple[int, int]]:
    """Split text into (start, end) sentence ranges, keeping offsets."""
    if not text.strip():
        return []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        fragment = text[cursor:end]
        stripped = fragment.strip()
        if stripped:
            lead = fragment.index(stripped[0])
            ranges.append((cursor + lead, cursor + lead + len(stripped)))
        cursor = match.end()
    tail = text[cursor:]
    stripped = tail.strip()
    if stripped:
        lead = tail.index(stripped[0])
        ranges.append((cursor + lead, cursor + lead + len(stripped)))
    return ranges
