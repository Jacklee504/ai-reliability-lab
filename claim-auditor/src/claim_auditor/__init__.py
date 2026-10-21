"""Claim-level source auditor.

Check whether the claims in an AI answer are supported by the sources
that were supplied with it, and abstain when the evidence does not
settle the question.
"""

from .auditor import (
    ClaimFinding,
    ClaimReport,
    SourceDocument,
    audit_claims,
    build_claim_prompt,
)
from .types import (
    Claim,
    Evidence,
    Method,
    Qualifiers,
    Source,
    Span,
    Verdict,
    VerdictRecord,
)

__all__ = [
    "Claim",
    "ClaimFinding",
    "ClaimReport",
    "Evidence",
    "Method",
    "Qualifiers",
    "Source",
    "SourceDocument",
    "Span",
    "Verdict",
    "VerdictRecord",
    "audit_claims",
    "build_claim_prompt",
]

__version__ = "0.1.0"
