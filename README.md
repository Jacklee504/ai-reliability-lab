# AI Reliability Lab

Small tools for checking what an AI system **received**, **said**, and
was **allowed to do**. Each one answers a different question about AI
output, and each one is designed to fail quietly rather than accuse wrongly.

The theme is AI reliability. The tools do not try to be a general
"LLM grader". They each attack one specific way an AI system can mislead a
reader. Source-based findings include their supporting passage when one
is available.

The corpus is 192 cases across twelve project suites, with a 98-case
`dev` split and a 94-case held-out `test` split. The strongest project,
`claims`, has also been extracted into a standalone package at
[`../claim-auditor`](../claim-auditor).

Every project can now be built as a standalone, installable package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release --all --out releases
PYTHONPATH=src python -m ai_reliability.cli release --verify --out releases
```

The verify step runs each package's demo, its own test suite, and an
import check with the lab off the path. All thirteen releasable
projects pass it.

## Quick start

Requires Python 3.11 or newer. There are no runtime dependencies.

```bash
cd ai-reliability-lab
export PYTHONPATH=src

# 1. See what is implemented
python -m ai_reliability.cli projects

# 2. Inspect a file-context receipt
python -m ai_reliability.cli audit receipt fixtures/context_receipts/v1/contract.txt

# 3. Run one deterministic project end to end
python -m ai_reliability.cli audit eval numerical --offline

# 4. Run the whole evaluation suite (no model needed)
python -m ai_reliability.cli audit eval eval_regression --offline

# 5. Run the tests
pytest -q
```

These five commands need no model, network access, or API key.

## Running with a model

The model-backed projects use an Ollama-compatible endpoint. No API key is
read or required.

```bash
export AIRL_MODEL=minimax-m3.1-flash:cloud      # any Ollama model id
export AIRL_OLLAMA_URL=http://127.0.0.1:11434  # default

python -m ai_reliability.cli audit eval claims
# Optional endpoint check:
python -m ai_reliability.cli doctor
```

If this process cannot reach the endpoint, use the recorded cache for
replay. The recorder stops on a connection failure and does not write a
partial cache.

To replay a recorded run without an endpoint, point the cache at a file:

```bash
export AIRL_MODEL_CACHE=runs/model_cache.json
python -m ai_reliability.cli audit eval eval_regression
```

Setting `AIRL_MODEL_CACHE` makes replay cache only. A missing cache file,
wrong model ID, or uncached prompt fails without contacting the endpoint.
Unset the variable before intentionally recording new responses.

## The projects

| # | Project | Question it answers | Method |
|---|---------|---------------------|--------|
| 1 | `omission` | What did this summary leave out that my brief cared about? | model |
| 2 | `claims` | Which sentences in this answer are not supported by the source? | model |
| 3 | `citations` | Does this cited passage actually support the sentence? | model |
| 4 | `deliverables` | Did the output meet every requirement in the request? | deterministic |
| 5 | `context_receipts` | Which file version and which text did the pipeline send? | deterministic |
| 6 | `eval_regression` | How do all the others behave together, and how do they change? | both |
| 7 | `numerical` | Did the number come from the right row, column, year, and unit? | deterministic |
| 8 | `instructions` | Did the agent follow the measurable repository instructions? | deterministic |
| 9 | `code_review` | Is this review comment technically true of the code? | deterministic |
| 10 | `authorization` | Was the agent authorised to take this action? | deterministic |
| 11 | `source_versions` | Which claims are queued for review because their source changed? | deterministic |
| 12 | `study_coverage` | Which syllabus topics are explicitly mentioned? | deterministic |
| 13 | `retrieval_eval` | Does the ranker retrieve the passage that answers the query? | comparison |

`baselines` is an auxiliary project. It runs stripped-down LLM judges
on the same cases as `claims` and `citations`, so the careful prompts
can be tested against a naive alternative.

`source_versions` and `study_coverage` began as optional spikes and
are now part of the shared corpus.

`eval_regression` is the cross project suite. It is not released on
its own, because its purpose is to measure the other projects against
each other.

Each project has its own document in [`docs/`](docs), saying what it
answers, how it works, the rules that keep it honest, and what it
cannot do. The same prose appears in the released package README, so
the two cannot drift apart.

Each project lives in `src/ai_reliability/projects/<name>.py` and has its
own fixtures in `fixtures/<name>/cases.json`. Each one is a separate
module with its own contract, so the strongest ones can be lifted out into
individual repositories.

## CLI

```bash
# Project status
python -m ai_reliability.cli projects

# File-context receipt (no model calls)
python -m ai_reliability.cli audit receipt fixtures/context_receipts/v1/contract.txt
python -m ai_reliability.cli audit receipt-diff \
    --before fixtures/context_receipts/v1/contract.txt \
    --after  fixtures/context_receipts/v2/contract.txt

# Deliverable contract checker
python -m ai_reliability.cli audit deliverable \
    "Write 50 questions across six chapters, four options each, with an answer key." \
    --deliverable fixtures/deliverables/example_quiz.txt

# Evaluation for one project, or for all of them
python -m ai_reliability.cli audit eval claims --out runs/claims.json
python -m ai_reliability.cli audit eval eval_regression --out runs/all.json
python -m ai_reliability.cli audit eval source_versions --offline
python -m ai_reliability.cli audit eval study_coverage --offline

# Score one split on its own. The test split is the held-out set.
python -m ai_reliability.cli audit eval eval_regression --split test

# Compare the lexical ranker against the embedding ranker
python -m ai_reliability.cli audit eval retrieval_eval --offline --json

# Run additional development controls without changing the cached suite
python -m ai_reliability.cli audit eval claims --fixture hard_cases.json --offline

# Compare two stored runs and list the cases whose verdict changed
python -m ai_reliability.cli compare-runs runs/all.json runs/all_new.json

# Inspect the exact prompts a project would send, and record the responses
python -m ai_reliability.cli prompts claims cases.json
python -m ai_reliability.cli record claims cases.json --out runs/claims_cache.json
# The recorder retries a dropped connection and reports what still failed
python -m ai_reliability.cli record claims test_cases.json --out runs/cache_claims_test.json

# Compare stripped yes/no or three-way LLM judges on the same claims and citations
python -m ai_reliability.cli audit eval baselines --baseline-kind naive_binary
python -m ai_reliability.cli audit eval baselines --baseline-kind naive_verdict

# Export blinded inputs for independent label review
python -m ai_reliability.cli review-export --out runs/review_packet.json
# After independently filling reviewed_expected fields in that packet
# (for example, {"verdict": "supported"}; leave null if unreviewed):
python -m ai_reliability.cli review-compare runs/review_packet.json

# Use an edited requirement list from a previous deliverable JSON report
python -m ai_reliability.cli audit deliverable "Write two questions" \
  --deliverable fixtures/deliverables/example_quiz.txt \
  --requirements fixtures/deliverables/edited_requirements.json

# Extract the requirement list, edit it, then recheck with it
python -m ai_reliability.cli audit requirements "Write two questions" \
  --out runs/requirements.json

# Check the model endpoint
python -m ai_reliability.cli doctor

# Build the standalone packages, and check they stand alone
python -m ai_reliability.cli release --list
python -m ai_reliability.cli release --all --out releases
python -m ai_reliability.cli release --verify --out releases

# Write the per project documentation
python -m ai_reliability.cli release --docs --out docs
```

## The design rules that hold everywhere

These are the rules that keep the tools from producing false accusations,
which is the failure mode that matters most for a reliability tool.

**A missing passage is never a contradiction.** Retrieval that finds
nothing produces `insufficient_evidence`, not `contradicted`. The shared
verdict builder in `core/verdicts.py` refuses to construct a
contradiction with no evidence at all.

**"Supported" means the source supports it.** It does not mean the claim
is true in the world. Sources may be outdated, incomplete, or wrong.

**Universal claims need universal evidence.** A claim about "all" or
"every" is not refuted by a passage about one example, and it is not
supported by one either.

**A relevance score is not evidence.** A passage that shares every
keyword can still fail to state the fact the claim needs. The citation
and numerical projects report topical overlap and support as separate
things, on purpose.

**A tool that cannot say "I do not know" is not usable.** Every project
has an abstain path, and the evaluation counts abstentions as a reported
outcome rather than a failure.

**Uncertain inputs stay visible.** The deliverable checker exposes its
extracted requirements for review, the instruction tester reports an
instruction it cannot parse as inconclusive, and the authorization checker
only evaluates proposed actions.

## Layout

```
ai-reliability-lab/
  README.md              this file
  HANDOFF.md             what works, what failed, what to do next
  RESULTS.md             only results produced by actual runs
  pyproject.toml
  src/ai_reliability/
    cli.py               the command line interface
    core/
      types.py           Source, Span, Claim, Evidence, Verdict, EvalCase
      sources.py         file ingestion with honest failure reporting
      spans.py           sentence, section, and statement segmentation
      claims.py          atomic claim extraction and qualifier scanning
      retrieval.py       lexical and injected embedding retrieval
      embeddings.py      deterministic local embedding provider
      execution.py       controlled execution of a code snapshot
      verdicts.py        verdict construction and the abstention rules
      providers.py       Ollama adapter, fake provider, response cache
      evaluation.py      runners, metrics, run comparison
      fixtures.py        fixture loading
    projects/            one module per project
  fixtures/<project>/    per-project evaluation cases
    cases.json           the development cases for each project
    test_cases.json      the held-out cases, where they exist
  runs/                  stored runs and recorded model responses
  tests/                 pytest suite
    projects/             one test file per project
  scripts/               legacy build helpers; use the CLI recorder instead
  claim-auditor/         the extracted standalone package
  releases/              the built standalone packages
  docs/                  one document per project
  .github/workflows/     the CI that tests the lab and the releases
```

## What the results mean

`RESULTS.md` holds the numbers from the runs in `runs/`. Read the caveats
with them. In particular:

* All 192 fixture labels are marked `model_drafted`. The `dev` split was
  used while building; the 94-case `test` split was authored afterwards
  and scored once. None is an independent human label.
* The sample sizes are small (10 to 29 cases per project). Counts are
  reported alongside rates for exactly that reason.
* The cached model text can be replayed from `runs/model_cache_live.json`.
  It is keyed by prompt hashes and is not a complete raw API transcript.
  The naive LLM judge comparison has a measured MiniMax result, in
  `runs/method_comparison_test.json`.
* The embedding retrieval path is measured against a deterministic
  hashed embedder, not a pretrained model. The local Ollama server
  reports that it does not support embeddings.
* The three legacy files in `scripts/` remain because the configured hook
  blocked deletion. Do not use them for model calls or file edits; the
  supported path is the CLI and normal provider adapter.

All fixtures are synthetic. No license has been selected for this project.
