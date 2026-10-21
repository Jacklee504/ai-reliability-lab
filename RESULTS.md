# Results

Every number in this file was produced by an actual run. The run JSON is
in `runs/` and every command is reproducible with the recorded model
cache.

**These are development-set agreement counts, not accuracy estimates.**
All 192 labels are `model_drafted`, and the reports separate the `dev`
split (98 cases) from the held-out `test` split (94 cases). The
prompts and labels were revised during the build session. None has
been independently human verified.

## How these runs were produced

```bash
export PYTHONPATH=src

# Model-backed runs, replayed from the recorded responses
AIRL_MODEL_CACHE=runs/model_cache_live.json \
  python -m ai_reliability.cli audit eval <project> --out runs/<project>_model.json

# Deterministic baselines
python -m ai_reliability.cli audit eval <project> --offline \
  --method conservative-lexical-baseline --out runs/<project>_baseline.json

# One split at a time
python -m ai_reliability.cli audit eval eval_regression --split test
```

The model text in `runs/model_cache_live.json` was recorded from live
calls to the local Ollama server during this session. The cache keys
identify prompt text by hash; the file is not a complete, byte-for-byte
raw API log. These commands replay the stored text and make no fresh
model request when every key hits. The reported model-call count is
replayed responses.

The response cache was built with `airl record`, which sends the
prompts the auditor would send and stores the answers:

```bash
python -m ai_reliability.cli record claims test_cases.json --out runs/cache_claims_test.json
```

## Model and configuration

| Setting | Value |
|---------|-------|
| Model id | `minimax-m3.1-flash:cloud` |
| Endpoint | `http://127.0.0.1:11434` (local Ollama) |
| API key | none used or required |
| Temperature | 0.0 |
| Response format | Ollama `format: json` |
| Thinking | disabled (`think: false`) |
| Concurrency | one request at a time |
| Prompt versions | `claims-v1`, `citations-v1`, `omission-v1` |

The cloud relay behind the local Ollama server dropped connections
during one recording run. The recorder now retries a failed call
three times and records the prompts that still failed, so a partial
recording is never mistaken for a complete one.

## Whole suite

All 192 cases across the twelve per-project suites, run through each
project's own runner.

| Metric | Model | Deterministic baseline |
|--------|-------|------------------------|
| Cases | 192 | 192 |
| Scored | 191 | 191 |
| Accepted by fixture scorer | **186** | 158 |
| False accusations on supported claims | 0/20 | 0/20 |
| False alarms on benign cases | 1/28 | 2/28 |
| Abstentions | 38/192 | 51/192 |
| Replayed model responses | 81 | 0 |
| Parse failures | 0 | 0 (n/a) |
| Label origins | 192 model_drafted | 192 model_drafted |
| Splits | 98 dev, 94 test | 98 dev, 94 test |

**What this supports.** On these 192 cases the cached model path
agreed with 186 of the 191 scored labels and the deterministic path
with 158. These are repeatable software checks on a small case set.

**What this does not support.** Independent accuracy, comparative
model quality, real-world reliability, or measured generation latency.
Two omission cases accept either verdict, so agreement on them is weak
evidence. The aggregate false-accusation and benign-case denominators
mix different project contracts; inspect the per-project run files
before using either metric for a specific tool.

## By split

The `test` split is the one that matters for the "was it tuned on?"
question. It was authored after the development cases were in place
and was scored once.

| Split | Cases | Scored | Accepted | False accusations | False alarms | Abstentions | Model calls |
|-------|-------|--------|----------|------------------|-------------|-------------|-------------|
| dev | 98 | 98 | 96 | 0/9 | 0/16 | 18 | 43 |
| test | 94 | 93 | 90 | 0/11 | 1/12 | 20 | 38 |
| all | 192 | 191 | 186 | 0/20 | 1/28 | 38 | 81 |

The held-out split scored slightly lower than development, which is the
expected direction. The gap is one case either way, so it is not
evidence of anything on its own.

## Per project

| Project | Method | Cases | Model | Baseline | False accusations | False alarms | Abstentions | Model calls | Parse failures |
|---------|--------|-------|-------|----------|------------------|-------------|-------------|-------------|----------------|
| `claims` | hybrid | 26 | 24/26 | 14/26 | 0/4 | 0/0 | 10 | 28 | 0 |
| `citations` | hybrid | 17 | 17/17 | 11/17 | 0/4 | 0/0 | 8 | 17 | 0 |
| `omission` | hybrid | 14 | 11/14 | 9/14 | 0/0 | 1/4 | 0 | 36 | 0 |
| `deliverables` | deterministic | 17 | 17/17 | 17/17 | 0/0 | 0/3 | 5 | 0 | 0 |
| `context_receipts` | deterministic | 14 | 14/14 | 14/14 | 0/0 | 0/7 | 0 | 0 | 0 |
| `numerical` | deterministic | 23 | 22/22 | 22/22 | 0/6 | 0/0 | 7 | 0 | 0 |
| `instructions` | deterministic | 29 | 29/29 | 29/29 | 0/0 | 0/14 | 0 | 0 | 0 |
| `code_review` | deterministic | 19 | 19/19 | 19/19 | 0/6 | 0/0 | 8 | 0 | 0 |
| `authorization` | deterministic | 23 | 23/23 | 23/23 | 0/0 | 0/0 | 0 | 0 | 0 |
| `source_versions` | deterministic | 12 | 12/12 | 12/12 | 0/0 | 0/0 | 0 | 0 | 0 |
| `study_coverage` | deterministic | 13 | 13/13 | 13/13 | 0/0 | 0/0 | 0 | 0 | 0 |
| `retrieval_eval` | comparison | 10 | 10/10 | 10/10 | 0/0 | 0/0 | 0 | 0 | 0 |

Every case in that table is `model_drafted`. The deterministic
projects produce the same numbers on both paths because they make no
model calls.

## Does the careful prompt earn its keep?

This is the comparison the lab exists to make. The same model, the
same cases, the same evidence, and the same parsing, with only the
prompt changed. Run over the 25 held-out claim and citation cases.

| Method | Accepted | Accused when it should abstain | Abstained |
|--------|----------|------------------------------|-----------|
| Careful tool (four outcomes, abstention rules, entity guard) | 23/25 | **0** | 9 |
| Naive three-way judge | 22/25 | 0 | 8 |
| Naive binary judge ("yes or no") | 14/25 | **8** | 0 |

The raw file is `runs/method_comparison_test.json`.

**What this supports.** The careful tool accepted 23 of 25, the
three-way judge 22, and the binary judge 14. The binary judge cannot
express "the passage does not answer this", so it has to accuse or
refuse. On these cases it accused 8 times where the evidence did not
settle the question, while the careful tool and the three-way judge
accused 0 times. The gap between the careful tool and the three-way
judge is one case, which is not a difference worth claiming.

**What this does not support.** A claim that the careful tool is much
more accurate overall. The three-way judge is close behind on the
accepted count, the case set is small, and the labels are
model-drafted. The result is mostly about the *shape of the errors*:
the binary judge cannot abstain, and that is where the eight
accusations come from.

The development split of the same two projects was not re-run for this
comparison, because the development prompts and labels were both
revised during the build session and a comparison there would be
measured against a moving target.

## The three model-backed projects in detail

### `claims` (26 cases, 28 model calls, 0 parse failures)

24/26 with the model against 14/26 for the deterministic fallback. The
difference is entirely in the entailment cases: the deterministic
fallback can recognise a supported claim by term overlap but cannot
recognise a contradiction, so it abstained on the cases where the
source states something incompatible.

### `citations` (17 cases, 17 model calls, 0 parse failures)

17/17 with the model against 11/17 for the deterministic fallback. The
baseline fails on the case where the cited passage shares every keyword
but states a different date, which is the case the project exists for.

### `omission` (14 cases, 36 model calls, 0 parse failures)

11/14 with the model against 9/14 for the deterministic fallback. The
model gate reduced the number of reported candidate statements on the
benign controls, but two of those cases accept either outcome, so this
is not a measured false-alarm reduction on an unambiguous benign set.

## Retrieval: lexical against embedding

`retrieval_eval` measures whether the ranking stage retrieves the
passage that answers the query. The candidate set is supplied and
labelled, so the measurement isolates ranking.

| Ranker | Cases | Recall at k=3 | Mean reciprocal rank |
|--------|-------|--------------|---------------------|
| Lexical (IDF + number bonus) | 10 | 7 | 0.650 |
| Hashed embedding (cosine) | 10 | 6 | 0.500 |

The embedding provider is a deterministic hashed embedder, not a
trained model. The local Ollama server reports that it does not
support embeddings (`This server does not support embeddings. Start it
with --embeddings`), and no installed model advertises the embedding
capability, so no pretrained embedder was available to measure. The
hashed embedder closes the plumbing gap and is recorded as what it is:
a vector space with no synonymy. See `src/ai_reliability/core/embeddings.py`.

The clearest measured difference is case `ret_001`: the query says
"notice period" and the answering passage says "90 days written
notice". The lexical ranker matches the shared token and finds it at
rank 1; the hashed embedder scores it low because the two texts share
only one token. That case's label was written expecting both to hit,
and was changed to the observed behaviour after the first measured run.

## Executable code review

The code-review project can settle a runtime claim by running the
reviewer's harness in a separate process with a timeout. The runtime
cases are in `fixtures/code_review/runtime_cases.json`.

| Outcome | Cases | Result |
|---------|-------|--------|
| Runtime claim confirmed by execution | 3 | 3/3 |
| Runtime claim refuted by execution | 3 | 3/3 |
| Harness timeout, broken harness, no harness | 3 | 3/3 abstain |

The three abstentions are the interesting part. A harness that hangs, a
harness that fails to import, and the absence of a harness all produce
`uncertain` rather than a verdict. A broken harness is never allowed to
manufacture evidence against a comment.

## The standalone extract

`../claim-auditor` is the `claims` project extracted into its own
package, with no imports from `ai_reliability`. It has its own
fixtures, its own tests, and its own CLI.

| Run | Cases | Accepted | False accusations |
|-----|-------|----------|------------------|
| Deterministic path | 25 | 16/25 | 0/5 |
| Model path, all cases | 25 | 24/25 | 0/5 |
| Model path, held-out `test` split | 10 | 8/10 | 0/2 |

The deterministic figure is the same weakness the lab shows: no
contradiction detection without a model.

## Fixture labels that were corrected

Three labels were wrong and were corrected after a run. They are
recorded in the fixture files with a `label_corrected` tag so the
change is visible rather than silent.

* `cit_t08_wrong_metric` expected `insufficient_evidence`. The cited
  passage states the participant count directly, and states a
  different one, so `contradicted` is correct. The model was right.
* `claim_t12_metric_mismatch` expected `insufficient_evidence` for
  the same reason. The passage states the active-user count directly
  and states 48,000, so the claim of 12,000 is contradicted. The model
  was right.
* `deliv_t02_wrong_question_count` had three question lines and a
  two-entry answer key. The tool counted three questions correctly;
  the fixture was wrong and was corrected.

## Fixture labels that accept more than one outcome

These cases are easy to satisfy and must not be treated as exact-label
successes.

* `insufficient_evidence` or `uncertain`: the claim-auditing and
  citation cases where the evidence does not settle the question, and
  the `code_review` cases where a static reading cannot settle the
  claim.
* `no_finding` or `potential_omission`: `om_003` and `om_005`, where
  the subprocessor-change notification duty may or may not count as a
  security obligation. The case is retained as a recorded
  disagreement rather than being scored as a pass or a failure.

## Decisions that were made in the code, not in a label

Two behaviours were decided and then written into the implementation.
The labels follow the implementation, and the change is recorded in
the fixture.

* **Unit normalisation.** `num_005_wrong_unit_millions` accepts either
  verdict because the unit rule had not been decided. The rule is now:
  when a claim carries a scale word and the cell carries no unit at
  all, the magnitudes may agree but the table does not state the unit,
  so the verifier abstains rather than picking a side. The label was
  changed from "either verdict" to a fixed abstention.
* **Different subject.** A passage that names a different subject from
  the claim cannot carry a verdict, even when the model asserts that
  the two are the same thing. This guard was added after a measured
  run where the model bridged "graduate programme" and "early-career
  programme" and called the citation supporting.

## Separate hard-case runs

The following suites were run with `--fixture hard_cases.json --offline`.
They are separate from the main corpus. All 16 labels are
`model_drafted` on `dev`, and the conservative fallback was adjusted
after seeing initial failures, so these are development checks only.

| Project | Accepted | Remaining error |
|---------|----------|-----------------|
| `claims` | 3/4 | Negated fee claim abstained instead of recognising contradiction |
| `citations` | 3/3 | none on these three cases |
| `deliverables` | 5/5 | none on these five cases |
| `code_review` | 4/4 | none on these four cases |

The original lexical fallback called two claim cases with mismatched
entities or universal scope `supported`, plus a negated claim and a
cited wrong-metric case. The revised fallback requires all content
terms and matching logical qualifiers before reporting support. It now
abstains on those risky cases. Its inability to recognise a clear
contradiction without a model remains visible in
`claim_hard_003_negation`.

## Blind review export

```bash
python -m ai_reliability.cli review-export --out runs/review_packet.json
#   233 inputs exported without expected labels or predictions
python -m ai_reliability.cli review-compare runs/review_packet.json
#   0 reviewed, 233 unreviewed, 0 disagreements before review
```

A reviewer can supply only a verdict, for example
`{"verdict": "supported"}`, and the comparison checks the fields they
supplied rather than requiring the whole fixture object. Draft labels
that accept more than one outcome are reported separately, so a case
where the reviewer happened to match one of the alternatives is still
visible.
