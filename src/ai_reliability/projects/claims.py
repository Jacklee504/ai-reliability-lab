"""Project 2: claim-level source auditor.

Problem
-------
An AI answer can mix supported, contradicted, and unsupported statements.
One sentence may carry several claims with different outcomes, and a
source that discusses the same topic may not answer the question at all.

Approach
--------
Three stages, kept separate on purpose so the evaluation can tell them
apart:

1. **Claim extraction** splits the answer into atomic claims and attaches
   the qualifiers that change their meaning (dates, quantities, subject,
   scope, negation, modality).
2. **Evidence retrieval** finds candidate source passages. A retrieval
   score is topical relevance, never evidence of entailment.
3. **Verdict classification** decides, per claim, whether the evidence
   supports, contradicts, or does not settle the claim.

The three stages are separated because the interesting question is *which
stage went wrong*: finding the wrong passage and judging it wrongly are
different bugs with different fixes. The report records both.

Deliberate limits
-----------------
* "Supported" means the supplied source supports the claim. It does not
  mean the claim is true in the world.
* A claim that no passage addresses is reported as insufficient evidence,
  never as contradicted.
* A universal claim is not refuted by a passage that is merely about the
  same topic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..core.claims import ExtractionSettings, is_strict_claim, split_answer_into_claims
from ..core.providers import (
    ModelResult,
    ParseError,
    Provider,
    ProviderError,
    parse_json_block,
)
from ..core.retrieval import EmbeddingProvider, lexical_support_candidate, retrieve, retrieve_semantic, topical_overlap
from ..core.sources import source_from_text
from ..core.spans import segment_statements
from ..core.types import (
    CaseResult,
    Claim,
    EvalCase,
    Evidence,
    Method,
    Span,
    Verdict,
    VerdictRecord,
)
from ..core.verdicts import apply_abstention_rules, make_verdict

PROMPT_VERSION = "claims-v1"

#: The four outcomes the classifier may return.
VALID_VERDICTS = ("supported", "contradicted", "insufficient_evidence", "uncertain")


@dataclass
class SourceDocument:
    """A source the answer may be checked against."""

    identifier: str
    text: str
    version: str | None = None

    def spans(self) -> list[Span]:
        """Candidate evidence passages: the source split into statements."""
        source, text = source_from_text(self.text, self.identifier)
        statements = segment_statements(text, source)
        if not statements:
            return []
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

    The prompt is deliberately narrow. It asks for one of four verdicts
    and nothing else, so a model cannot quietly return a paragraph of
    hedging that the parser has to interpret.
    """
    blocks = []
    for index, passage in enumerate(passages, start=1):
        blocks.append(f"[{index}] {passage}")
    evidence_text = "\n\n".join(blocks) if blocks else "(no relevant passage found)"
    shape = (
        "{\"verdict\": \"supported|contradicted|insufficient_evidence|uncertain\", "
        "\"explanation\": \"one sentence\", "
        "\"key_phrase\": \"the exact words in the passage that matter\", "
        "\"passage_numbers\": [1]}"
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
        "partially relevant. A claim about \"all\" is not contradicted by a passage " +
        "about one example.\n"
        "\n"
        f"CLAIM:\n{claim_text}\n"
        "\n"
        f"PASSAGES:\n{evidence_text}\n"
        "\n"
        "Reply with JSON only, in this exact shape:\n"
        f"{shape}\n"
    )


def _classify_with_model(
    provider: Provider,
    claim_text: str,
    passages: list[str],
) -> tuple[dict[str, Any] | None, str | None, float, str | None]:
    """Ask the model to classify one claim.

    Returns (parsed, error, latency_ms, raw_text). A parse failure is
    returned rather than raised so the caller can record it and fall back
    to the deterministic path.
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
        return None, f"provider error: {exc}", 0.0, None

    try:
        parsed = parse_json_block(result.text, raw=result.text)
    except ParseError as exc:
        return None, f"parse failure: {exc}", result.call.latency_ms, result.text

    if not isinstance(parsed, dict):
        return None, "model did not return an object", result.call.latency_ms, result.text
    return parsed, None, result.call.latency_ms, result.text


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

    With a provider, each claim is classified by the model and the model
    answer is then passed through the abstention rules. Without one, the
    deterministic fallback is used and the report says so.
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
            parsed, error, latency, _raw = _classify_with_model(provider, claim.text, passages)
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

        verdict = Verdict(label)
        explanation = str(parsed.get("explanation", "")).strip()
        key_phrase = parsed.get("key_phrase")
        key_phrase = str(key_phrase) if key_phrase else None

        evidence_text = "\n".join(passages)
        adjusted, note = apply_abstention_rules(
            verdict,
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

# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


@dataclass
class ClaimRunner:
    """Runs a claim-level audit evaluation case.

    A case supplies an answer and one or more sources. The expected
    label may be a single verdict or a list of acceptable verdicts, which
    is how "any abstention is acceptable" cases are expressed.
    """

    provider: Any = None
    use_model: bool = True

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        answer = case.input.get("answer", "")
        source_inputs = case.input.get("sources", [])
        documents = [
            SourceDocument(
                identifier=item.get("identifier", f"source_{index}"),
                text=item.get("text", ""),
                version=item.get("version"),
            )
            for index, item in enumerate(source_inputs)
        ]

        report = audit_claims(
            answer,
            documents,
            provider=self.provider,
            use_model=self.use_model,
        )

        expected = case.expected
        findings = report.findings

        if "verdicts" in expected:
            correct = [f.verdict.verdict.value for f in findings] == expected["verdicts"]
        elif "verdict" in expected:
            wanted = expected["verdict"]
            if isinstance(wanted, list):
                correct = all(f.verdict.verdict.value in wanted for f in findings)
            else:
                correct = len(findings) == 1 and findings[0].verdict.verdict.value == wanted
        else:
            correct = None

        predicted = {
            "verdict": findings[0].verdict.verdict.value if len(findings) == 1 else "multiple",
            "verdicts": [f.verdict.verdict.value for f in findings],
            "claim_count": len(findings),
        }

        return CaseResult(
            case_id=case.case_id,
            project=case.project,
            predicted=predicted,
            expected=expected,
            label_origin=case.label_origin,
            split=case.split,
            correct=correct,
            model_calls=report.model_calls,
            parse_failures=report.parse_failures,
            notes=report.render(),
        )


def build_cases() -> list[EvalCase]:
    """Load the claim auditor evaluation cases."""
    from ..core.fixtures import load_project_cases

    return load_project_cases("claims")


def build_runner(provider: Any = None) -> ClaimRunner:
    """Factory used by the project registry."""
    return ClaimRunner(provider=provider)


def build_prompts(fixture_path: str) -> list[tuple[str, str]]:
    """Build the (purpose, prompt) pairs for a claims fixture.

    The recorder uses this to send exactly the prompts the auditor will
    later ask, so a recorded cache replays the same evaluation.
    """
    import json
    from pathlib import Path

    from ..core.fixtures import project_fixture_dir

    target = Path(fixture_path)
    if not target.exists():
        target = project_fixture_dir("claims") / fixture_path
    payload = json.loads(target.read_text(encoding="utf-8"))

    prompts: list[tuple[str, str]] = []
    for case in payload.get("cases", []):
        answer = case.get("input", {}).get("answer", "")
        documents = [
            SourceDocument(
                identifier=item.get("identifier", f"source_{index}"),
                text=item.get("text", ""),
            )
            for index, item in enumerate(case.get("input", {}).get("sources", []))
        ]
        all_spans: list[Span] = []
        for document in documents:
            all_spans.extend(document.spans())
        claims = split_answer_into_claims(answer, settings=ExtractionSettings())
        for claim in claims:
            evidence = retrieve(claim.text, all_spans, top_k=4)
            passages = [span.text for ev in evidence for span in ev.spans]
            prompts.append(("claim_verdict", build_claim_prompt(claim.text, passages)))
    return prompts
