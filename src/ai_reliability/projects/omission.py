"""Project 1: important-omission auditor.

Problem
-------
An AI summary can contain only true statements while leaving out something
important from the source. The summary reads correctly, so the gap is
invisible unless you already know what the source said.

Approach
--------
Importance is relative to a **purpose**. "Summarize financial
obligations" and "summarize eligibility requirements" make different
source sentences material and different ones harmless. The auditor
therefore takes the user's purpose as an input, not as a constant.

The pipeline is:

1. Segment the source into candidate statements, each with its exact
   span and section.
2. Match each source statement against the summary. A statement that the
   summary covers is *not* an omission candidate.
3. Rank the unmatched statements by relevance to the stated purpose.
4. Report the top candidates as **potential omissions**, always quoting
   the source text and always labelled as a potential, never as a
   confirmed omission.

The third step is where the method can go wrong, so the module is
explicit about it: a *material* omission is a statement the purpose
cares about, and detecting that is a judgement. The module reports a
ranked list with an explicit relevance signal and lets the user decide.

Deliberate limits
-----------------
* The auditor cannot know what the closed application internally read. It
  compares the supplied summary with the supplied source, and says so.
* A statement left out of a summary is only an omission *if it was
  material for the stated purpose*. The module never asserts that a
  summary is deficient; it points at candidates and explains why.
* The evaluation includes harmless-omission controls, because a detector
  that flags every deleted sentence is not useful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.providers import (
    ModelResult,
    ParseError,
    Provider,
    ProviderError,
    parse_json_block,
)
from ..core.retrieval import build_index, lexical_score
from ..core.sources import source_from_text
from ..core.spans import segment_statements
from ..core.types import (
    CaseResult,
    EvalCase,
    Method,
    Span,
    Verdict,
)

PROMPT_VERSION = "omission-v1"

#: Words that mark a source statement as an obligation, limitation, or
#: exception, as opposed to background or description.
MATERIAL_MARKERS = (
    "must", "shall", "required", "require", "may terminate", "cannot",
    "not permitted", "prohibited", "liable", "liability", "unless",
    "except", "only", "within", "at least", "at most", "no more than",
    "no less than", "deadline", "expires", "terminates", "penalty",
    "fee", "interest", "charge", "obligation", "guarantee", "warranty",
    "condition", "subject to", "breach", "indemnif", "refund", "cancel",
    "notice", "opt out", "opt-in", "exclusive", "confidential",
    "shall not", "does not", "do not", "never", "only if", "provided that",
)

#: Words that mark a statement as background material, which is less
#: likely to be an important omission for most purposes.
BACKGROUND_MARKERS = (
    "definition", "means", "for the purposes of this", "referred to as",
    "definitions", "interpretation", "appendix", "glossary",
)

#: Phrases that signal a purpose focused on a particular kind of
#: information. Used by the deterministic relevance scorer.
PURPOSE_HINTS = {
    "financial": (
        "fee", "payment", "pay", "price", "cost", "charge", "invoice",
        "refund", "interest", "penalty", "money", "eur", "usd", "gbp",
        "liab", "cap", "premium", "subscription", "deposit", "budget",
        "revenue", "expense", "billing",
    ),
    "eligibility": (
        "eligible", "eligib", "qualif", "requirement", "must", "applicant",
        "entitle", "criteria", "condition", "be eligible", "qualify",
        "who may", "who can", "entitlement",
    ),
    "obligation": (
        "must", "shall", "obligat", "responsible", "required", "duty",
        "commit", "undertake", "agree", "warrant", "indemnif",
    ),
    "risk": (
        "risk", "liable", "liability", "loss", "damages", "breach",
        "warranty", "guarantee", "indemnif", "consequential", "indirect",
        "cap", "exclu",
    ),
    "data_protection": (
        "data", "personal", "processor", "controller", "gdpr", "retention",
        "breach", "subprocessor", "transfer", "consent", "privacy",
    ),
    "termination": (
        "terminat", "cancel", "notice", "renew", "expiry", "expire",
        "end of term", "exit",
    ),
    "summary": (
        "main", "key", "important", "essential", "critical", "summary",
        "overview", "highlight", "core", "principal", "principal terms",
    ),
}


@dataclass
class SourceStatement:
    """A candidate source statement, kept with its exact span."""

    statement_id: str
    text: str
    section: str | None = None
    index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement_id": self.statement_id,
            "text": self.text,
            "section": self.section,
            "index": self.index,
        }


@dataclass
class OmissionCandidate:
    """A source statement the summary did not cover."""

    statement: SourceStatement
    relevance: float
    material_markers: list[str] = field(default_factory=list)
    covered: bool = False
    reason: str = ""
    model_judgement: str | None = None
    model_material: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement.to_dict(),
            "relevance": round(self.relevance, 4),
            "material_markers": self.material_markers,
            "covered": self.covered,
            "reason": self.reason,
            "model_judgement": self.model_judgement,
            "model_material": self.model_material,
        }


@dataclass
class OmissionReport:
    """The result of auditing a summary against a source for omissions."""

    purpose: str
    candidates: list[OmissionCandidate] = field(default_factory=list)
    covered_count: int = 0
    total_statements: int = 0
    notes: list[str] = field(default_factory=list)
    model_id: str | None = None
    model_calls: int = 0
    parse_failures: int = 0

    @property
    def potential_omissions(self) -> list[OmissionCandidate]:
        """Uncovered statements, most relevant first.

        The order is the one the model is asked about, so a caller that
        replays the prompts gets the same questions in the same order.

        When a model has judged the candidates, only the ones it called
        material are reported. A statement the model judged immaterial is
        not an omission candidate, which is what keeps the harmless
        omission control cases from producing false alarms. Without a
        model, every uncovered statement is a candidate and the report
        says the ranking is heuristic.
        """
        uncovered = [c for c in self.candidates if not c.covered]
        uncovered.sort(key=lambda c: c.relevance, reverse=True)
        if self.model_calls or self.parse_failures:
            judged = [c for c in uncovered if c.model_material is True]
            if judged or any(c.model_material is False for c in uncovered):
                return judged
        return uncovered

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "model_id": self.model_id,
            "model_calls": self.model_calls,
            "parse_failures": self.parse_failures,
            "total_statements": self.total_statements,
            "covered_count": self.covered_count,
            "potential_omission_count": len(self.potential_omissions),
            "candidates": [c.to_dict() for c in self.candidates],
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = [
            f"Omission audit for purpose: {self.purpose}",
            f"  source statements: {self.total_statements}"
            f"  covered: {self.covered_count}"
            f"  potential omissions: {len(self.potential_omissions)}",
        ]
        for candidate in self.potential_omissions[:10]:
            marker = ",".join(candidate.material_markers[:4])
            lines.append(
                f"  [potential omission, relevance {candidate.relevance:.2f}]"
                f" {candidate.statement.text}"
            )
            if candidate.statement.section:
                lines.append(f"      section: {candidate.statement.section}")
            if marker:
                lines.append(f"      markers: {marker}")
            if candidate.reason:
                lines.append(f"      why: {candidate.reason}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _purpose_keywords(purpose: str) -> set[str]:
    """Keywords implied by a stated purpose.

    The purpose is free text, so this looks for the purpose *categories*
    it names ("financial obligations", "eligibility requirements") and
    falls back to the purpose's own words.
    """
    lowered = purpose.lower()
    keywords: set[str] = set()
    for category, hints in PURPOSE_HINTS.items():
        if category.replace("_", " ") in lowered or category in lowered:
            keywords.update(hints)
    keywords.update(w for w in re.findall(r"[a-z]+", lowered) if len(w) > 3)
    return keywords


def _material_markers(text: str) -> list[str]:
    """Markers in a statement that suggest it is an obligation or limit."""
    lowered = text.lower()
    return [marker for marker in MATERIAL_MARKERS if marker in lowered]


def _is_background(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in BACKGROUND_MARKERS)


def segment_source(text: str, identifier: str) -> list[SourceStatement]:
    """Split a source into candidate statements with sections."""
    source, extracted = source_from_text(text, identifier)
    statements = segment_statements(extracted, source)
    out: list[SourceStatement] = []
    for index, statement in enumerate(statements):
        if statement.text.strip().endswith(":"):
            # A heading, not a statement.
            continue
        out.append(
            SourceStatement(
                statement_id=statement.statement_id,
                text=statement.text,
                section=statement.section,
                index=index,
            )
        )
    return out
