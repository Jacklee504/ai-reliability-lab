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


def _is_covered(statement_text: str, summary: str, threshold: float) -> bool:
    """Decide whether the summary already covers a source statement.

    Coverage is lexical: the summary must contain the statement's
    distinctive terms. A statement whose only content words are missing
    from the summary is a candidate omission.
    """
    statement_words = {
        w for w in re.findall(r"[a-z0-9]+", statement_text.lower()) if len(w) > 3
    }
    if not statement_words:
        return True
    summary_words = set(re.findall(r"[a-z0-9]+", summary.lower()))
    overlap = len(statement_words & summary_words) / len(statement_words)
    return overlap >= threshold


def score_relevance(
    statement_text: str,
    purpose_keywords: set[str],
    material_markers: list[str],
) -> float:
    """Score how relevant an uncovered statement is to the stated purpose.

    This is the deterministic baseline for importance. It combines:

    * how many purpose keywords the statement carries,
    * whether the statement looks like an obligation or limitation,
    * a penalty for statements that look like background material.
    """
    lowered = statement_text.lower()
    hits = sum(1 for keyword in purpose_keywords if keyword in lowered)

    keyword_score = min(1.0, hits / 3.0)
    marker_score = min(1.0, len(material_markers) / 2.0)
    background_penalty = 0.4 if _is_background(statement_text) else 0.0

    score = 0.55 * keyword_score + 0.45 * marker_score - background_penalty
    return max(0.0, min(1.0, score))


def build_omission_prompt(
    purpose: str,
    statement: str,
    summary: str = "",
) -> str:
    """Build the prompt that asks the model whether a statement is material.

    The prompt names three things: the reader purpose, the summary, and
    the source statement under consideration. All three must be present:
    without the statement the model cannot judge it, and without the
    summary the model cannot tell whether it was already covered.
    """
    return (
        "You are given a document, a summary of it, and the reader's purpose "
        "for the summary.\n"
        "\n"
        "Decide whether the source statement below is IMPORTANT to that "
        "purpose and is not covered by the summary.\n"
        "\n"
        "A statement is important if omitting it could mislead a reader who "
        "relied on the summary for that purpose. A statement that is merely "
        "background, or a detail the purpose does not care about, is not "
        "important even though it was left out.\n"
        "\n"
        "Answer with JSON only:\n"
        '{"important": true|false, "reason": "one sentence"}\n'
        "\n"
        f"PURPOSE:\n{purpose}\n"
        "\n"
        f"SUMMARY:\n{summary or "(the summary was not supplied)"}\n"
        "\n"
        f"SOURCE STATEMENT:\n{statement}\n"
    )


def _ask_model(provider: Provider, purpose: str, statement: str, summary: str = ""):
    """Ask the model whether an uncovered statement is material."""
    prompt = build_omission_prompt(purpose, statement, summary)
    try:
        result: ModelResult = provider.generate(
            prompt,
            purpose="omission_material",
            prompt_version=PROMPT_VERSION,
            temperature=0.0,
            json_mode=True,
        )
    except ProviderError as exc:
        return None, f"provider error: {exc}"

    try:
        parsed = parse_json_block(result.text, raw=result.text)
    except ParseError as exc:
        return None, f"parse failure: {exc}"

    if not isinstance(parsed, dict):
        return None, "model did not return an object"
    return parsed, None


def audit_omissions(
    source_text: str,
    summary: str,
    purpose: str,
    provider: Provider | None = None,
    *,
    coverage_threshold: float = 0.6,
    source_identifier: str = "source",
    top_k: int = 4,
) -> OmissionReport:
    """Find source statements the summary left out that the purpose cares about.

    The report lists every uncovered statement with a relevance score, and
    marks the ones the model judged material. It does not claim the
    summary is deficient; it points at candidates.
    """
    statements = segment_source(source_text, source_identifier)
    keywords = _purpose_keywords(purpose)
    report = OmissionReport(
        purpose=purpose,
        total_statements=len(statements),
        model_id=getattr(provider, "model_id", None),
    )

    candidates: list[OmissionCandidate] = []
    for statement in statements:
        covered = _is_covered(statement.text, summary, coverage_threshold)
        markers = _material_markers(statement.text)
        relevance = score_relevance(statement.text, keywords, markers)
        candidate = OmissionCandidate(
            statement=statement,
            relevance=relevance,
            material_markers=markers,
            covered=covered,
            reason="" if covered else "source statement is not present in the summary",
        )
        candidates.append(candidate)
        if covered:
            report.covered_count += 1

    report.candidates = candidates

    uncovered = [c for c in candidates if not c.covered]
    # Sort by relevance, most relevant first. The sort is stable, so
    # statements with equal relevance keep their document order.
    uncovered.sort(key=lambda c: c.relevance, reverse=True)

    if provider is not None:
        for candidate in uncovered[:top_k]:
            parsed, error = _ask_model(provider, purpose, candidate.statement.text, summary)
            if parsed is None:
                if error and not error.startswith("provider error:"):
                    report.parse_failures += 1
                continue
            report.model_calls += 1
            important = parsed.get("important")
            candidate.model_material = bool(important) if important is not None else None
            candidate.model_judgement = str(parsed.get("reason", "")).strip() or None
            if candidate.model_material and candidate.relevance < 0.3:
                candidate.relevance = min(1.0, candidate.relevance + 0.25)
                candidate.reason = (
                    f"{candidate.reason}; the model judged this material for the "
                    "stated purpose"
                )
    else:
        report.notes.append(
            "no model was used; candidates are ranked by the deterministic "
            "relevance score only"
        )

    report.notes.append(
        "candidates are potential omissions, not confirmed ones; the auditor "
        "compares the supplied summary with the supplied source and does not "
        "know what the generating application read"
    )
    return report
