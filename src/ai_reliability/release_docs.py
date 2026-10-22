"""Write the per project documentation for the lab.

The lab and the released packages say the same things about each
project. The prose lives in :mod:`ai_reliability.release_content` and
this module writes it out, so a change to one is a change to both.

The documents land in ``docs/``. They are the lab's own reference for
each project: what it answers, how it works, the rules that keep it
honest, and what it cannot do.
"""

from __future__ import annotations

import pathlib

from .release import list_plans, package_dir_name
from .release_content import CONTENT


def _document(project: str, content) -> str:
    """The lab's document for one project."""
    stages = "\n".join("%d. %s" % (i, s) for i, s in enumerate(content.stages, 1))
    return (
        "# %(title)s\n"
        "\n"
        "%(summary)s\n"
        "\n"
        "**Question it answers:** %(question)s\n"
        "\n"
        "**Released as:** `%(package)s`\n"
        "\n"
        "## Why it exists\n"
        "\n"
        "%(why)s\n"
        "\n"
        "## How it works\n"
        "\n"
        "%(stages)s\n"
        "\n"
        "## The rules that keep it honest\n"
        "\n"
        "%(rules)s\n"
        "\n"
        "## Known limitations\n"
        "\n"
        "%(limits)s\n"
        "\n"
        "## Running it\n"
        "\n"
        "```bash\n"
        "PYTHONPATH=src python -m ai_reliability.cli audit eval %(project)s --offline\n"
        "```\n"
        "\n"
        "To build the standalone package:\n"
        "\n"
        "```bash\n"
        "PYTHONPATH=src python -m ai_reliability.cli release %(project)s --out releases\n"
        "```\n"
    ) % {
        "title": content.title,
        "summary": content.summary,
        "question": content.question,
        "package": package_dir_name(project),
        "why": content.why,
        "stages": stages,
        "rules": content.bullets(content.rules),
        "limits": content.bullets(content.limits),
        "project": project,
    }


def write_documents(out_dir: pathlib.Path) -> list[pathlib.Path]:
    """Write one document per project, and an index."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for plan in list_plans():
        content = CONTENT.get(plan.project)
        if content is None:
            continue
        path = out_dir / ("%s.md" % plan.project)
        path.write_text(_document(plan.project, content), encoding="utf-8")
        written.append(path)

    index = [
        "# Projects",
        "",
        "One document per project. Each says what the project answers,",
        "how it works, the rules that keep it honest, and what it cannot",
        "do.",
        "",
        "| Project | Answers | Released as |",
        "|---------|---------|--------------|",
    ]
    for plan in list_plans():
        content = CONTENT.get(plan.project)
        if content is None:
            index.append(
                "| `%s` | the cross project regression suite | not released |"
                % plan.project
            )
            continue
        index.append(
            "| [`%s`](%s.md) | %s | `%s` |" % (
                plan.project,
                plan.project,
                content.question,
                package_dir_name(plan.project),
            )
        )
    index_path = out_dir / "README.md"
    index_path.write_text("\n".join(index) + "\n", encoding="utf-8")
    written.append(index_path)
    return written
