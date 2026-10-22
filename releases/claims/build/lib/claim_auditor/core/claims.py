"""Claim extraction: split an answer into atomic, checkable assertions.

The hard part is not splitting, it is *not losing qualifiers*. A claim
like "all applicants must re-enrol before 2024" contains three checkable
assertions (subject set, requirement, date bound), and dropping the
"all" or the date turns a supported claim into a false accusation.

So extraction comes in two layers:

1. A deterministic splitter that produces candidate clauses, using
   conjunction and contrast markers that usually separate independent
   assertions.
2. A qualifier scanner that attaches the qualifiers found in the clause
   (dates, quantities, subject/scope words, negation, modality) to the
   claim record.

A model-assisted mode exists as an option, but the deterministic path is
the default so the pipeline stays runnable without model access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Claim, Qualifiers, short_id

# Markers that usually separate two independent assertions in one sentence.
_SPLIT_MARKERS = [
    r"\bbut\b",
    r"\bhowever\b",
    r"\bwhereas\b",
    r"\balthough\b",
    r"\bthough\b",
    r"\bwhile\b",
    r"\byet\b",
    r"\band also\b",
    r"\badditionally\b",
    r"\bmoreover\b",
    r"\bfurthermore\b",
    r"\bin addition\b",
    r"\bexcept\b",
    r"\bunless\b",
    # A coordinated clause with its own subject is a second assertion.
    r"\band\s+(?=(?:the|a|an|all|any|each|every|no|payment|notice|term|fee|data|interest|liability|termination|service)\b)",
]
_SPLIT_RE = re.compile("(" + "|".join(_SPLIT_MARKERS) + ")", re.IGNORECASE)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9“\"(])")

_DATE_RE = re.compile(
    r"\b(?:\d{4}(?:-\d{2}(?:-\d{2})?)?|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,?\s*\d{4})?|"
    r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})\b",
    re.IGNORECASE,
)
_QUANTITY_RE = re.compile(
    r"(?:[$€£¥]\s?\d[\d,.]*\s*(?:million|billion|thousand|m|bn|k)?)|"
    r"(?:\b\d[\d,.]*\s*(?:[$€£¥]|EUR|USD|GBP|CHF|million|billion|thousand|"
    r"%|percent|percentage points?|years?|months?|days?|hours?|minutes?|"
    r"units?|items?|questions?|chapters?|options?|employees?|people|users?)\b)",
    re.IGNORECASE,
)
_SCOPE_RE = re.compile(
    r"\b(all|every|each|any|some|most|none|no|only|either|neither|both|"
    r"at least|at most|up to|more than|fewer than|greater than|less than|"
    r"at least one|all of the|each of the|every one of)\b",
    re.IGNORECASE,
)
_SUBJECT_RE = re.compile(
    r"\b(the\s+[a-z][\w-]*(?:\s+[a-z][\w-]*)?)\s+(?:must|shall|may|can|is|are|was|were|will|has|have)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(not|never|no longer|does not|do not|did not|was not|were not|is not|are not|"
    r"cannot|must not|shall not|without)\b",
    re.IGNORECASE,
)
_MODALITY_RE = re.compile(
    r"\b(must|shall|should|may|might|can|could|will|would|is required to|"
    r"is permitted to|is prohibited from|is allowed to|recommends?|requires?)\b",
    re.IGNORECASE,
)
_STRICT_RE = re.compile(r"\b(all|every|each|any|never|no|must|shall|only|exclusively|"
                        r"in all cases|without exception)\b", re.IGNORECASE)


@dataclass
class ExtractionSettings:
    """Tunables for deterministic claim extraction."""

    split_on_markers: bool = True
    keep_qualifier_words: bool = True
    min_chars: int = 8


def extract_qualifiers(text: str) -> Qualifiers:
    """Scan a clause for qualifiers that must survive to the verdict."""
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


def _clauses(text: str, settings: ExtractionSettings) -> list[str]:
    """Split a sentence into clause candidates without losing offsets."""
    if not settings.split_on_markers:
        return [text]
    pieces: list[str] = []
    cursor = 0
    for match in _SPLIT_RE.finditer(text):
        piece = text[cursor : match.start()]
        pieces.append(piece)
        cursor = match.end()
    pieces.append(text[cursor:])
    return [p for p in pieces if p.strip()]


def split_answer_into_claims(
    answer: str,
    *,
    claim_prefix: str = "cl",
    settings: ExtractionSettings | None = None,
) -> list[Claim]:
    """Split an answer into atomic claims, preserving answer offsets.

    Each claim keeps the exact answer substring it came from
    (``answer_start``/``answer_end``) so a report can point back at the
    original sentence.
    """
    settings = settings or ExtractionSettings()
    claims: list[Claim] = []
    cursor = 0
    for sentence in _SENTENCE_SPLIT.split(answer):
        if not sentence:
            continue
        index = answer.find(sentence, cursor)
        if index < 0:
            index = cursor
        sentence_start = index
        sentence_text = sentence.strip()
        lead = len(sentence) - len(sentence.lstrip())
        sentence_start += lead
        cursor = sentence_start + len(sentence_text)

        if len(sentence_text) < settings.min_chars:
            continue

        for clause in _clauses(sentence_text, settings):
            clause = clause.strip()
            if len(clause) < settings.min_chars:
                continue
            offset = sentence_text.find(clause)
            if offset < 0:
                offset = 0
            start = sentence_start + offset
            end = start + len(clause)
            claims.append(
                Claim(
                    claim_id=short_id(claim_prefix, str(start), str(end)),
                    text=clause,
                    answer_span=clause,
                    answer_start=start,
                    answer_end=end,
                    qualifiers=extract_qualifiers(clause),
                )
            )
    return claims


def merge_overlapping_claims(claims: list[Claim]) -> list[Claim]:
    """Merge claims whose answer spans overlap, keeping the wider text.

    Used when a model proposes claims that duplicate a deterministic
    split, so the two are not counted twice in evaluation.
    """
    if not claims:
        return []
    ordered = sorted(claims, key=lambda c: (c.answer_start, c.answer_end))
    merged: list[Claim] = []
    for claim in ordered:
        if merged and claim.answer_start < merged[-1].answer_end:
            previous = merged[-1]
            if claim.answer_end > previous.answer_end:
                start = min(previous.answer_start, claim.answer_start)
                previous.text = previous.text
                previous.answer_start = start
                previous.answer_end = claim.answer_end
                previous.answer_span = previous.answer_span
                previous.qualifiers = extract_qualifiers(previous.text)
        else:
            merged.append(claim)
    return merged


def is_strict_claim(claim: Claim) -> bool:
    """True when the claim asserts universality, which is easy to refute wrongly.

    A universal claim needs strong evidence; a merely plausible source
    sentence must not be treated as contradicting it.
    """
    return bool(_STRICT_RE.search(claim.text)) or bool(
        claim.qualifiers.scope in {"all", "every", "each", "any", "never", "no", "only"}
    )
