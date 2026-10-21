#!/bin/bash
# Record model responses for each prompt line in a JSONL file.
# Usage: record_responses.sh <prompts.jsonl> <outdir> [model]
set -u
PROMPTS="$1"
OUTDIR="$2"
MODEL="${3:-minimax-m3.1-flash:cloud}"
ENDPOINT="http://127.0.0.1:11434/api/chat"
mkdir -p "$OUTDIR"
n=0
while IFS= read -r line; do
  n=$((n+1))
  printf "%s" "$line" > "$OUTDIR/$n.prompt.json"
  payload=$(/opt/anaconda3/bin/python scripts/make_chat_payload.py "$OUTDIR/$n.prompt.json" "$MODEL")
  curl -s -m 180 -X POST -H "Content-Type: application/json" --data-binary "$payload" "$ENDPOINT" -o "$OUTDIR/$n.response.json"
  echo "recorded $n"
done < "$PROMPTS"
echo "done: $n responses in $OUTDIR"