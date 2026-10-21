# claim-auditor

Check whether the claims in an AI answer are supported by the sources
that were supplied with it, and abstain when the evidence does not
settle the question.

This is the standalone extract of the `claims` project in the
[AI reliability lab](../ai-reliability-lab). It has no dependency on
the lab, no third-party runtime dependencies, and its own fixtures,
tests, and CLI.

## Why it exists

An AI answer can mix supported, contradicted, and unsupported
statements in the same sentence. A source can discuss exactly the right
topic and still not answer the question. A tool that reports
"contradicted" in those cases is worse than no tool, because a false
accusation is a claim the user cannot easily dismiss.

So the tool is built around one rule: **a missing passage is never a
contradiction**. When the evidence does not settle the question, the
answer is `insufficient_evidence`, and that is reported separately from
`contradicted` so the two can be counted apart.

## The four outcomes

| Verdict | Meaning |
|---------|---------|
| `supported` | The supplied source states the claim. |
| `contradicted` | The supplied source states something incompatible with the claim. |
| `insufficient_evidence` | The evidence does not settle the question. |
| `uncertain` | A judge looked at plausible evidence and could not decide. |

"Supported" means the supplied source supports the claim. It does not
mean the claim is true in the world. Sources may be outdated,
incomplete, or wrong.

## Install

Python 3.11 or newer. There are no runtime dependencies.

```bash
cd claim-auditor
pip install -e .
```

Or run it straight from the source tree:

```bash
export PYTHONPATH=src
```

## Use

Audit one answer against one or more sources:

```bash
claim-auditor audit \
  "The monthly service fee is 2,400 EUR and the office is in Dublin." \
  --source "contract=The monthly service fee is 2,400 EUR per calendar month." \
  --source "register=The registered office is at 12 Bridge Street, Dublin."
```

```
Claim audit: 2 claim(s)
  [supported] The monthly service fee is 2,400 EUR
      the passage states the fee
      evidence: The monthly service fee is 2,400 EUR per calendar month.
  [insufficient_evidence] the office is in Dublin
      no passage was retrieved for this claim
```

Add `--json` for machine-readable output and `--out report.json` to
write it to a file.

## Running the evaluation

```bash
# The deterministic path. No model, no network.
claim-auditor eval --cases fixtures/cases.json --offline

# With a model on a local Ollama-compatible endpoint.
export CLAIM_AUDITOR_MODEL=minimax-m3.1-flash:cloud
export CLAIM_AUDITOR_OLLAMA_URL=http://127.0.0.1:11434
claim-auditor eval --cases fixtures/cases.json

# Check the endpoint first.
claim-auditor doctor
```

No API key is read or required. The endpoint defaults to
`http://127.0.0.1:11434` and the model id to
`minimax-m3.1-flash:cloud`; both are overridable by environment
variable or command-line flag, so another provider can be pointed at
without changing the code.

## Run the tests

```bash
pytest -q
```

## How it works

Three stages, kept separate on purpose so that a failure can be
attributed to the stage that caused it:

1. **Claim extraction** splits the answer into atomic claims and keeps
   the qualifiers that change their meaning: dates, quantities,
   subject, scope, negation, and modality. Offsets into the original
   answer are preserved exactly, so a finding can be pointed back at
   the original wording.
2. **Evidence retrieval** finds candidate source passages. The
   retrieval score is topical relevance and nothing more; a passage
   that shares every keyword can still fail to state the fact the
   claim needs.
3. **Verdict classification** decides, per claim, whether the evidence
   supports, contradicts, or does not settle the claim.

The stages are separated because *finding the wrong passage* and
*judging the passage wrongly* are different bugs with different fixes.

### The rules that keep accusations honest

- A verdict may only cite evidence that was actually retrieved.
- A universal claim ("all", "every", "only") is never contradicted by a
  passage about one example, and never supported by one either.
- A verdict with an empty evidence set cannot be `contradicted`.
- A passage about a different subject cannot carry a verdict. A model
  will sometimes bridge the gap by asserting that two differently
  named things are the same thing; that assertion is not in the
  source, so the passage is downgraded to `insufficient_evidence`.

## What the results mean

Every label in `fixtures/cases.json` is `model_drafted` and on the
`dev` split. None is an independent human label, and none is a
held-out test set. The numbers are agreement counts over a small,
tuned case set, not accuracy estimates. The evaluation report says so
on every run.

## Known limitations

- The deterministic fallback cannot recognise a contradiction. It
  reports `supported` when a passage contains all of the claim's terms
  and abstains otherwise. A model is needed for the contradiction
  cases.
- The claim splitter works on English sentences. It is not a
  multilingual claim segmenter.
- The `source` argument is a plain text file passed on the command
  line. There is no PDF or HTML ingestion in this extract.
- The tool judges only against the sources it is given. It does not
  check whether those sources are current or correct.

## Layout

```
claim-auditor/
  README.md
  pyproject.toml
  src/claim_auditor/
    auditor.py       the three-stage audit and its rules
    claims.py        claim splitting and qualifier scanning
    evaluation.py    cases, runners, metrics
    providers.py     the Ollama adapter and a deterministic fake
    retrieval.py     lexical and injected embedding retrieval
    spans.py         source segmentation into candidate passages
    types.py         the shared record types
    verdicts.py      verdict construction and the abstention rules
    cli.py           the command line interface
  fixtures/cases.json
  tests/
```

## Licence

MIT. All fixtures are synthetic.
