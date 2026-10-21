"""Build an Ollama chat payload from a prompt JSON file.

Used by scripts/record_responses.sh so the shell loop does not have to
embed JSON quoting rules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    prompt_file = Path(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else "minimax-m3.1-flash:cloud"
    record = json.loads(prompt_file.read_text(encoding="utf-8"))
    prompt = record["prompt"]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "format": "json"},
    }
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())