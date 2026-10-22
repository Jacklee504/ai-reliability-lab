"""Conservative syllabus mention and repetition audit.

The checker proves only that a topic phrase appeared. It does not claim
that notes or quiz questions adequately teach or test the topic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .core.types import CaseResult, EvalCase


def _phrase_matches(text: str, phrase: str) -> list[tuple[int, int, str]]:
    if not phrase.strip():
        return []
    pattern = r"(?<!\w)" + re.escape(phrase.strip()) + r"(?!\w)"
    return [(match.start(), match.end(), phrase) for match in re.finditer(pattern, text, flags=re.IGNORECASE)]


def audit_coverage(topics: list[dict[str, Any]], material: str) -> dict[str, Any]:
    """Count explicit topic mentions and return unverified topics."""
    results = []
    for topic in topics:
        phrases = [topic.get("label", ""), *topic.get("aliases", [])]
        matches = [match for phrase in phrases for match in _phrase_matches(material, phrase)]
        matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        distinct = []
        for match in matches:
            if not distinct or match[0] >= distinct[-1][1]:
                distinct.append(match)
        mentions = len(distinct)
        results.append({
            "topic_id": topic.get("id", ""), "label": topic.get("label", ""),
            "mentions": mentions, "status": "mentioned" if mentions else "unverified",
            "matched_phrase": distinct[0][2] if distinct else None,
        })
    return {
        "topics": results,
        "mentioned_ids": [item["topic_id"] for item in results if item["status"] == "mentioned"],
        "unverified_ids": [item["topic_id"] for item in results if item["status"] == "unverified"],
        "note": "Mention is a lexical observation, not proof of adequate coverage.",
    }


@dataclass
class StudyCoverageRunner:
    provider: Any = None

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        data = case.input
        report = audit_coverage(data.get("topics", []), data.get("material", ""))
        expected = case.expected
        correct = (
            report["mentioned_ids"] == expected["mentioned_ids"]
            if "mentioned_ids" in expected else None
        )
        return CaseResult(
            case_id=case.case_id, project=case.project, predicted=report,
            expected=expected, label_origin=case.label_origin, split=case.split,
            correct=correct,
        )


def build_cases() -> list[EvalCase]:
    from .core.fixtures import load_project_cases
    return load_project_cases("study_coverage")


def build_runner(provider: Any = None) -> StudyCoverageRunner:
    return StudyCoverageRunner(provider=provider)
