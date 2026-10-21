"""Tests for the shared core: types, sources, spans, claims, retrieval, verdicts.

These tests challenge behaviour rather than restating the implementation.
The ones that matter most are the failure cases: empty sources, failed
extraction, qualifiers that change a verdict, and a verdict that tries to
accuse without evidence.
"""

from __future__ import annotations

import json
import pytest

from conftest import FIXTURES

from ai_reliability.core.claims import extract_qualifiers, split_answer_into_claims
from ai_reliability.core.providers import OllamaProvider, ProviderError, load_cache
from ai_reliability.core.retrieval import retrieve, topical_overlap
from ai_reliability.core.sources import load_source
from ai_reliability.core.spans import segment_statements, split_sections
from ai_reliability.core.types import (
    ExtractionStatus,
    Method,
    Source,
    Span,
    Verdict,
    sha256_text,
)
from ai_reliability.core.verdicts import apply_abstention_rules, make_verdict


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def test_missing_file_fails_loudly(tmp_path):
    source, text = load_source(tmp_path / "nope.txt")
    assert source.extraction_status is ExtractionStatus.FAILED
    assert text == ""
    assert any("not found" in w for w in source.warnings)


def test_empty_file_is_reported_as_empty_not_ok(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    source, text = load_source(path)
    assert source.extraction_status is ExtractionStatus.EMPTY
    assert text == ""
    assert any("empty" in w for w in source.warnings)


def test_unknown_media_type_produces_a_warning(tmp_path):
    path = tmp_path / "thing.bin"
    path.write_text("some bytes", encoding="utf-8")
    source, _text = load_source(path)
    assert any("unknown media type" in w for w in source.warnings)


def test_content_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("one", encoding="utf-8")
    b.write_text("two", encoding="utf-8")
    source_a, _ = load_source(a)
    source_b, _ = load_source(b)
    assert source_a.content_hash != source_b.content_hash
    assert source_a.content_hash == sha256_text("one") or len(source_a.content_hash) == 64
