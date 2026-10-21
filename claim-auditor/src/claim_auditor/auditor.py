"""The claim-level source auditor.

Problem
-------
An AI answer can mix supported, contradicted, and unsupported
statements. One sentence may carry several claims with different
outcomes, and a source that discusses the same topic may not answer
the question at all.

Approach
--------
Three stages, kept separate on purpose so an error can be attributed
to the stage that caused it:

1. **Claim extraction** splits the answer into atomic claims and
   attaches the qualifiers that change their meaning.
2. **Evidence retrieval** finds candidate source passages. A retrieval
   score is topical relevance, never evidence of entailment.
3. **Verdict classification** decides, per claim, whether the evidence
   supports, contradicts, or does not settle the claim.

Deliberate limits
-----------------
* "Supported" means the supplied source supports the claim. It does
  not mean the claim is true in the world.
* A claim that no passage addresses is reported as insufficient
  evidence, never as contradicted.
* A universal claim is not refuted by a passage that is merely about
  the same topic.
* A citation about a different entity cannot support the claim, even
  when the model asserts that two differently named things are the
  same thing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .claims import ExtractionSettings, is_strict_claim, split_answer_into_claims
from .providers import ModelResult, ParseError, Provider, ProviderError, parse_json_block
from .retrieval import (
    EmbeddingProvider,
    lexical_support_candidate,
    retrieve,
    retrieve_semantic,
    topical_overlap,
)
from .spans import segment_statements
from .types import (
    Claim,
    Evidence,
    Method,
    Source,
    Span,
    Verdict,
    VerdictRecord,
    sha256_text,
    short_id,
)
from .verdicts import apply_abstention_rules, make_verdict

PROMPT_VERSION = "claims-v1"

VALID_VERDICTS = ("supported", "contradicted", "insufficient_evidence", "uncertain")


@dataclass
class SourceDocument:
    """A source the answer may be checked against."""

    identifier: str
    text: str
    version: str | None = None

    def spans(self) -> list[Span]:
        """Candidate evidence passages: the source split into statements."""
        source = Source(
            source_id=short_id("src", self.identifier, sha256_text(self.text)),
            identifier=self.identifier,
            content_hash=sha256_text(self.text),
            version=self.version,
        )
        statements = segment_statements(self.text, source)
        return [statement.to_span(source) for statement in statements]


@dataclass
class ClaimFinding:
    """The audit result for one claim."""

    claim: Claim
    verdict: VerdictRecord
    evidence: list[Evidence] = field(default_factory=list)
    key_phrase: str | None = None
    retrieval_ok: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "verdict": self.verdict.to_dict(),
            "evidence": [e.to_dict() for e in self.evidence],
            "key_phrase": self.key_phrase,
            "retrieval_ok": self.retrieval_ok,
            "note": self.note,
        }


@dataclass
class ClaimReport:
    """The full result of auditing an answer against its sources."""

    answer: str
    findings: list[ClaimFinding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    model_id: str | None = None
    model_calls: int = 0
    parse_failures: int = 0

    @property
    def verdicts(self) -> list[VerdictRecord]:
        return [f.verdict for f in self.findings]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "model_id": self.model_id,
            "model_calls": self.model_calls,
            "parse_failures": self.parse_failures,
            "findings": [f.to_dict() for f in self.findings],
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = [f"Claim audit: {len(self.findings)} claim(s)"]
        for finding in self.findings:
            lines.append(f"  [{finding.verdict.verdict.value}] {finding.claim.text}")
            if finding.verdict.explanation:
                lines.append(f"      {finding.verdict.explanation}")
            for evidence in finding.evidence:
                for span in evidence.spans:
                    lines.append(f"      evidence: {span.quote(120)}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def build_claim_prompt(claim_text: str, passages: list[str]) -> str:
    """Build the classification prompt for one claim.

    The prompt is deliberately narrow: it asks for one of four verdicts
    and nothing else, so a model cannot quietly return a paragraph of
    hedging that the parser has to interpret.
    """
    blocks = []
    for index, passage in enumerate(passages, start=1):
        blocks.append(f"[{index}] {passage}")
    evidence_text = "\n\n".join(blocks) if blocks else "(no relevant passage found)"
    shape = (
        '{"verdict": "supported|contradicted|insufficient_evidence|uncertain", '
        '"explanation": "one sentence", '
        '"key_phrase": "the exact words in the passage that matter", '
        '"passage_numbers": [1]}'
    )
    return (
        "You are checking whether a source passage supports a claim.\n"
        "\n"
        "Rules:\n"
        "- supported: the passage states the claim.\n"
        "- contradicted: the passage states something incompatible with the claim.\n"
        "- insufficient_evidence: the passage does not settle the claim.\n"
        "- uncertain: you cannot decide.\n"
        "\n"
        "Never answer contradicted when the passage is silent, off-topic, or only "
        "partially relevant. A claim about \"all\" is not contradicted by a passage "
        "about one example.\n"
        "\n"
        f"CLAIM:\n{claim_text}\n"
        "\n"
        f"PASSAGES:\n{evidence_text}\n"
        "\n"
        "Reply with JSON only, in this exact shape:\n"
        f"{shape}\n"
    )


#: The subject of a claim is the noun phrase before its verb, for
#: example "the early-career programme" in "The early-career programme
#: runs for 12 months". Two texts that talk about the same subject share
#: this phrase; two that talk about different subjects do not.
_SUBJECT_PHRASE_RE = re.compile(
    r"\b(?:the|a|an)\s+([\w-]+(?:\s+[\w-]+){0,2}?)\s+"
    r"(?:runs?|is|was|are|were|has|have|had|costs?|takes?|requires?|"
    r"includes?|contains?|states?|says?|remains?|stays?|ends?|starts?|"
    r"launches?|supports?|covers?|applies?)\b",
    re.IGNORECASE,
)

#: Words that carry no entity information, so a difference in them is
#: not a difference of subject.
_STOP_SUBJECT_WORDS = {
    "programme", "program", "service", "plan", "fee", "price", "cost",
    "policy", "agreement", "contract", "report", "document", "term",
    "period", "deadline", "notice", "payment", "charge",
}


def _subject_phrases(text: str) -> list[str]:
    """Extract the subject noun phrases of a text."""
    phrases: list[str] = []
    for match in _SUBJECT_PHRASE_RE.finditer(text):
        phrase = match.group(1).strip().lower()
        content = [w for w in phrase.split() if w not in _STOP_SUBJECT_WORDS]
        if content:
            phrases.append(" ".join(content))
    return phrases


def entity_mismatch(claim: str, passages: list[str]) -> bool:
    """True when the claim and the passages are about different subjects.

    A passage about a different programme, organisation, or product does
    not support the claim, however the reading is framed. A model will
    often bridge the gap by asserting that two differently named things
    are the same thing, and that assertion is not in the source.

    The check is deliberately narrow. It only fires when both texts
    name a subject and the subjects share no content word. Anything
    less certain is left to the model.
    """
    claim_subjects = _subject_phrases(claim)
    if not claim_subjects:
        return False
    passage_subjects: list[str] = []
    for passage in passages:
        passage_subjects.extend(_subject_phrases(passage))
    if not passage_subjects:
        return False
    for claim_subject in claim_subjects:
        claim_words = set(claim_subject.split())
        for passage_subject in passage_subjects:
            passage_words = set(passage_subject.split())
            if claim_words & passage_words:
                return False
    return True


def _classify_with_model(
    provider: Provider,
    claim_text: str,
    passages: list[str],
) -> tuple[dict[str, Any] | None, str | None, float]:
    """Ask the model to classify one claim.

    A parse failure is returned rather than raised, so the caller can
    record it and fall back to the deterministic path.
    """
    prompt = build_claim_prompt(claim_text, passages)
    try:
        result: ModelResult = provider.generate(
            prompt,
            purpose="claim_verdict",
            prompt_version=PROMPT_VERSION,
            temperature=0.0,
            json_mode=True,
        )
    except ProviderError as exc:
        return None, f"provider error: {exc}", 0.0

    try:
        parsed = parse_json_block(result.text, raw=result.text)
    except ParseError as exc:
        return None, f"parse failure: {exc}", result.call.latency_ms

    if not isinstance(parsed, dict):
        return None, "model did not return an object", result.call.latency_ms
    return parsed, None, result.call.latency_ms


def _deterministic_verdict(claim_text: str, passages: list[str]) -> tuple[Verdict, str]:
    """A conservative fallback used when the model is unavailable.

    It only reports "supported" when a passage shares the claim's
    distinctive terms and no numbers conflict, and "insufficient"
    otherwise. It never claims contradiction, because without a model
    there is no basis for asserting one.
    """
    if not passages:
        return Verdict.INSUFFICIENT, "no passage was retrieved for this claim"
    best = max(passages, key=lambda p: topical_overlap(claim_text, p))
    overlap = topical_overlap(claim_text, best)
    if overlap >= 0.6 and lexical_support_candidate(claim_text, best):
        return Verdict.SUPPORTED, "a single passage contains all claim terms and logical qualifiers"
    return Verdict.INSUFFICIENT, (
        f"best passage overlap was {overlap:.0%}; lexical evidence did not establish all claim details"
    )


def audit_claims(
    answer: str,
    sources: Iterable[SourceDocument],
    provider: Provider | None = None,
    *,
    top_k: int = 4,
    use_model: bool = True,
    embedder: EmbeddingProvider | None = None,
) -> ClaimReport:
    """Audit an answer against a set of sources.

    With a provider, each claim is classified by the model and the
    model answer is then passed through the abstention rules. Without
    one, the deterministic fallback is used and the report says so.
    """
    documents = list(sources)
    all_spans: list[Span] = []
    for document in documents:
        all_spans.extend(document.spans())

    claims = split_answer_into_claims(answer, settings=ExtractionSettings())
    report = ClaimReport(answer=answer, model_id=getattr(provider, "model_id", None))

    for claim in claims:
        evidence = (
            retrieve_semantic(claim.text, all_spans, embedder, top_k=top_k)
            if embedder is not None else retrieve(claim.text, all_spans, top_k=top_k)
        )
        passages = [span.text for ev in evidence for span in ev.spans]

        parsed = None
        error = None
        latency = 0.0
        if provider is not None and use_model:
            parsed, error, latency = _classify_with_model(provider, claim.text, passages)
            if parsed is not None:
                report.model_calls += 1
            elif error and not error.startswith("provider error:"):
                report.parse_failures += 1

        if parsed is None:
            verdict, explanation = _deterministic_verdict(claim.text, passages)
            if error:
                explanation = f"{explanation} (model unavailable: {error})"
            record = make_verdict(
                claim.claim_id,
                verdict,
                explanation=explanation,
                evidence_ids=tuple(e.evidence_id for e in evidence),
                method=Method.LEXICAL,
            )
            report.findings.append(
                ClaimFinding(
                    claim=claim,
                    verdict=record,
                    evidence=evidence,
                    retrieval_ok=bool(evidence),
                    note=error or "",
                )
            )
            continue

        label = str(parsed.get("verdict", "")).strip().lower()
        if label not in VALID_VERDICTS:
            report.parse_failures += 1
            verdict, explanation = _deterministic_verdict(claim.text, passages)
            record = make_verdict(
                claim.claim_id,
                verdict,
                explanation=f"model returned an unusable label; {explanation}",
                evidence_ids=tuple(e.evidence_id for e in evidence),
                method=Method.LEXICAL,
                raw_label=label,
                parse_error=f"invalid verdict label: {label!r}",
            )
            report.findings.append(
                ClaimFinding(
                    claim=claim,
                    verdict=record,
                    evidence=evidence,
                    retrieval_ok=bool(evidence),
                    note=f"invalid label {label!r}",
                )
            )
            continue

        explanation = str(parsed.get("explanation", "")).strip()
        key_phrase = parsed.get("key_phrase")
        key_phrase = str(key_phrase) if key_phrase else None

        if label in ("supported", "contradicted") and entity_mismatch(claim.text, passages):
            # The passage is about a different entity. A model will
            # sometimes bridge the gap by asserting that two named
            # things are the same thing, but that assertion is not in
            # the source, so the passage cannot carry the verdict.
            explanation = (
                "the retrieved passage names a different subject than the claim, "
                "so it cannot support or contradict it"
            )
            label = "insufficient_evidence"
            key_phrase = None

        evidence_text = "\n".join(passages)
        adjusted, note = apply_abstention_rules(
            verdict=Verdict(label),
            claim_is_universal=is_strict_claim(claim),
            evidence_text=evidence_text,
            claim_text=claim.text,
        )
        if note:
            explanation = f"{explanation} ({note})".strip()

        record = make_verdict(
            claim.claim_id,
            adjusted,
            explanation=explanation,
            evidence_ids=tuple(e.evidence_id for e in evidence),
            method=Method.MODEL,
            model_id=getattr(provider, "model_id", None),
            prompt_version=PROMPT_VERSION,
            key_phrase=key_phrase,
            raw_label=label,
            latency_ms=latency,
        )
        report.findings.append(
            ClaimFinding(
                claim=claim,
                verdict=record,
                evidence=evidence,
                key_phrase=key_phrase,
                retrieval_ok=bool(evidence),
            )
        )

    if provider is None or not use_model:
        report.notes.append("no model was used; verdicts came from the deterministic fallback")
    return report
