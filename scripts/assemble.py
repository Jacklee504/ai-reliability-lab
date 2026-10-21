import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
lines = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
text = "\n".join(lines)
if target.exists() and sys.argv[3] == "append":
    with target.open("a", encoding="utf-8") as handle:
        handle.write(text)
else:
    target.write_text(text, encoding="utf-8")
print("wrote %d lines to %s" % (len(lines), target))