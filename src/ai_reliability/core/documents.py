"""Source documents shared by the project modules.

A :class:`SourceDocument` is a document the audit was given, with the
text a user would expect the tool to have read. It lives in the shared
core rather than in any one project because four projects need it: the
claim auditor, the citation auditor, the naive judge baselines, and
the regression suite.

Keeping it here is what lets each project be extracted on its own. A
project that imported this from another project would drag that
project along with it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sources import source_from_text
from .spans import segment_statements
from .types import Span


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
