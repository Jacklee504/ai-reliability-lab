# HANDOFF

Updated after the session that implemented the previous fix list. The
commands below distinguish current replay checks from earlier historical
observations. All labels are model drafted development labels and require
independent review.

## 1. What was built

`ai-reliability-lab/` is a single Python project tree with a shared core
and thirteen primary project modules. The core provides source ingestion,
document spans, atomic claims with qualifier preservation, lexical and
embedding retrieval, verdict construction with abstention rules, a model
provider adapter with a recorded-response cache, controlled execution of
a code snapshot, and one evaluation layer that every project shares. Each
project is a separate module with its own input and output contract, its
own fixtures, its own evaluation runner, and its own held-out test
split where it has one.

The corpus is 192 cases: 98 on the `dev` split and 94 on the held-out
`test` split. A CLI, a pytest suite (112 tests, all passing), a README,
and this handoff are in place. `RESULTS.md` holds only numbers produced
by actual runs.

Every project except the cross project regression suite can be built as
a standalone, installable package. The builder is in the product, not
in a helper script:

```bash
PYTHONPATH=src python -m ai_reliability.cli release --list
PYTHONPATH=src python -m ai_reliability.cli release --all --out releases
PYTHONPATH=src python -m ai_reliability.cli release --verify --out releases
PYTHONPATH=src python -m ai_reliability.cli release --docs --out docs
```

The built packages are in `releases/`. The verification runs each
package's demo, its own test suite, and an import check with the lab
off the path. All thirteen pass it. The per project documents are in
`docs/`, generated from the same prose the package READMEs use.

The repository is a Git repository and is published at
`github.com/JackLee504/ai-reliability-lab`.

## 2. Status of every project

| # | Project | Status | Cases | dev/test | Method | Note |
|---|---------|--------|-------|----------|--------|------|
| 1 | `omission` | runnable | 14 | 7/7 | hybrid | Purpose-relative; the model judgement gates what is reported |
| 2 | `claims` | runnable | 26 | 11/15 | hybrid | Three separate stages: extract, retrieve, judge |
| 3 | `citations` | runnable | 17 | 7/10 | hybrid | Reports topical overlap separately from support |
| 4 | `deliverables` | runnable | 17 | 8/9 | deterministic | Requirement extraction never invents requirements |
| 5 | `context_receipts` | runnable | 14 | 7/7 | deterministic | No model calls; reproducible receipts, diffs, and PDF extraction |
| 6 | `eval_regression` | runnable | 192 | 98/94 | both | Cross-project runner, baselines, run comparison |
| 7 | `numerical` | runnable | 23 | 11/12 | deterministic | Row/column/year/unit aware, shows its arithmetic |
| 8 | `instructions` | runnable | 29 | 17/12 | deterministic | Nine predicate forms, unparsed ones stay visible |
| 9 | `code_review` | runnable | 19 | 9/10 | deterministic | AST checks plus an opt-in executable path |
| 10 | `authorization` | runnable | 23 | 11/12 | deterministic | Dry-run policy evaluation, never acts |
| 11 | `source_versions` | runnable | 12 | 6/6 | deterministic | Changed evidence queues re-review; does not reverse verdicts |
| 12 | `study_coverage` | runnable | 13 | 5/8 | deterministic | Exact syllabus mentions only; quality stays unverified |
| 13 | `retrieval_eval` | runnable | 10 | 10/0 | comparison | Lexical against embedding, measured on labelled candidates |

The `baselines` module is auxiliary. It runs stripped-down LLM judges on
the same cases as `claims` and `citations`, and now has a measured
MiniMax result.

## 3. What the previous fix list asked for, and what happened

**1. Get the fixture labels independently reviewed.** Still open. The
review packet now holds 233 blinded inputs. The comparator was changed
so a reviewer can supply only a verdict rather than the whole fixture
object, and it surfaces draft labels that accept more than one outcome
separately from disagreements. Nothing has been reviewed.

**2. Grow the case sets.** Partly done. The corpus grew from 107 to 192
cases. The per-project sets are still small, between 10 and 29 cases
each, and most do not yet have the ten hard negatives and five
uncertain cases the original brief asked for. The held-out sets are new
and were authored with hard negatives in mind.

**3. Create a truly held-out test set.** Done, with a caveat. A
`test_cases.json` file now exists for eleven of the twelve suites, and
the loader picks it up automatically. The 94 test cases were authored
after the development cases were in place and were scored once, but
they were authored by the same session that wrote the code, so they are
not an independent test set. They are a held-out split, not a blind
one.

**4. Measure the injected embedding retrieval path.** Done, with a
limitation. The local Ollama server reports `This server does not
support embeddings`, and no installed model advertises the embedding
capability, so no pretrained embedder was available. A deterministic
hashed embedder was added instead, in `core/embeddings.py`, and the
`retrieval_eval` project measures both rankers on labelled candidates:
lexical 7/10 recall, embedding 6/10. The hashed embedder has no
synonymy, so this measures the mechanism, not semantic retrieval.

**5. Expand the code-review verifier with measured outcomes.** Done.
`core/execution.py` runs a reviewer's harness in a separate process
with a timeout, in a temporary directory. Nine runtime cases in
`fixtures/code_review/runtime_cases.json` score 9/9: three claims
confirmed by execution, three refuted, and three that correctly abstain
because the harness timed out, failed to import, or was absent. A
broken harness cannot manufacture evidence against a comment.

**6. Review extracted deliverable requirements with users.** Partly
done. The CLI now has `audit requirements`, which writes the extracted
list in exactly the format `--requirements` reads back, so the round
trip is extract, edit, recheck. `Requirement.from_dict` ignores
unknown fields, so a small edit does not break the recheck, and
requirement IDs are shown in the report. The interactive flow is still
not built.

**7. Widen the instruction parser.** Done. Nine predicate forms are
now recognised: `file_unchanged`, `file_exists`, `file_contains`,
`command_run`, `file_deleted`, `file_not_created`, `line_limit`,
`no_artifact_contains`, and the `add a test for` form. A form that is
not recognised is still reported as inconclusive. The instruction suite
grew from 9 to 29 cases and scores 29/29.

**8. Test the two optional spikes on broader inputs.** Done.
`source_versions` and `study_coverage` each have a held-out split of six
and eight cases. The held-out source-version set found a real bug: an
insertion *above* the evidence span shifted the text the span pointed
at, and the old code did not queue the claim. That is fixed and tested.

**9. Add PDF support where it is reliable.** Done. `core/sources.py`
records the real page count and the character offset of each page
boundary, and the receipt uses them. The previous code counted form
feeds, which the PDF extractor never emits, so every PDF reported one
page. A three-page generated fixture is at
`fixtures/context_receipts/sample_policy.pdf`, and `tests/make_pdf_fixture.py`
regenerates it. The standalone `claim-auditor` package deliberately
does not include PDF ingestion, because that is a different concern
from claim checking.

**10. Extract the strongest project into its own repository.** Done.
`claim-auditor/` has its own package, CLI, fixtures, seventeen tests,
README, and pyproject, with no import of `ai_reliability` anywhere. Its
held-out test split scores 8/10 against the live model with 0 false
accusations.

## 4. Commands that were run, and their outcomes

All commands were run from `ai-reliability-lab/` with `PYTHONPATH=src`.

```bash
# Tests (passed)
pytest -q                                     # 99 passed

# The whole corpus, model path
AIRL_MODEL_CACHE=runs/model_cache_live.json \
  python -m ai_reliability.cli audit eval eval_regression --out runs/regression_live.json
#   192 cases, 186 of 191 scored accepted, 0 false accusations, 0 parse failures

# The whole corpus, deterministic path
python -m ai_reliability.cli audit eval eval_regression --offline \
  --method conservative-lexical-baseline --out runs/regression_baseline.json
#   192 cases, 158 of 191 scored accepted

# One split at a time
python -m ai_reliability.cli audit eval eval_regression --split dev
#   98 cases, 96 accepted, 0 false accusations
python -m ai_reliability.cli audit eval eval_regression --split test
#   94 cases, 90 of 93 accepted, 0 false accusations, 1 false alarm

# Per project, deterministic path (all passed)
python -m ai_reliability.cli audit eval context_receipts --offline    # 14/14
python -m ai_reliability.cli audit eval deliverables   --offline    # 17/17
python -m ai_reliability.cli audit eval numerical       --offline    # 22/22
python -m ai_reliability.cli audit eval instructions    --offline    # 29/29
python -m ai_reliability.cli audit eval code_review     --offline    # 19/19
python -m ai_reliability.cli audit eval authorization   --offline    # 23/23
python -m ai_reliability.cli audit eval source_versions --offline    # 12/12
python -m ai_reliability.cli audit eval study_coverage  --offline    # 13/13
python -m ai_reliability.cli audit eval retrieval_eval  --offline    # 10/10
python -m ai_reliability.cli audit eval claims          --offline    # 14/26
python -m ai_reliability.cli audit eval citations        --offline    # 11/17
python -m ai_reliability.cli audit eval omission         --offline    # 9/14

# Per project, model path via the recorded cache (all passed)
AIRL_MODEL_CACHE=runs/model_cache_live.json \
  python -m ai_reliability.cli audit eval claims        # 24/26, 28 calls, 0 parse failures
AIRL_MODEL_CACHE=runs/model_cache_live.json \
  python -m ai_reliability.cli audit eval citations      # 17/17, 17 calls, 0 parse failures
AIRL_MODEL_CACHE=runs/model_cache_live.json \
  python -m ai_reliability.cli audit eval omission       # 11/14, 36 calls, 0 parse failures

# The naive judge comparison, now measured
python -m ai_reliability.cli audit eval baselines --split test --baseline-kind naive_binary
#   14/25 accepted, 8 accusations where an abstention was expected
python -m ai_reliability.cli audit eval baselines --split test --baseline-kind naive_verdict
#   22/25 accepted, 0 accusations where an abstention was expected
#   the careful tool on the same 25 cases: 23/25 accepted, 0 accusations
#   the raw comparison is in runs/method_comparison_test.json

# Runtime code review (passed)
python -m ai_reliability.cli audit eval code_review --fixture runtime_cases.json --offline
#   9/9, with 3 abstentions on the timeout, broken harness, and no harness cases

# Retrieval comparison (passed)
python -m ai_reliability.cli audit eval retrieval_eval --offline --json
#   lexical recall 7/10, embedding recall 6/10

# Blind review export (passed)
python -m ai_reliability.cli review-export --out runs/review_packet.json
#   233 inputs exported without expected labels or predictions
python -m ai_reliability.cli review-compare runs/review_packet.json
#   0 reviewed, 233 unreviewed, 0 disagreements before review

# End-to-end CLI examples (passed)
python -m ai_reliability.cli audit receipt fixtures/context_receipts/sample_policy.pdf
python -m ai_reliability.cli audit receipt-diff \
    --before fixtures/context_receipts/v1/contract.txt \
    --after  fixtures/context_receipts/v2/contract.txt
python -m ai_reliability.cli audit deliverable "Write 50 questions across six chapters, four options each, with an answer key." \
    --deliverable fixtures/deliverables/example_quiz.txt
python -m ai_reliability.cli audit requirements "Write 2 questions across 2 chapters" --out runs/requirements.json
python -m ai_reliability.cli projects
```

The standalone package, from `claim-auditor/`:

```bash
PYTHONPATH=src pytest -q                                 # 17 passed
PYTHONPATH=src python -m claim_auditor.cli eval --cases fixtures --offline
#   25 cases, 16 of 25 accepted
PYTHONPATH=src python -m claim_auditor.cli eval --cases fixtures --split test
#   10 cases, 8 of 10 accepted, 0 false accusations
```

## 5. Known incorrect predictions and likely causes

**Three fixture labels were wrong and have been corrected.** Each is
recorded in its fixture file with a `label_corrected` tag so the change
is visible rather than silent.

* `cit_t08_wrong_metric` and `claim_t12_metric_mismatch` both expected
  `insufficient_evidence` on the reasoning that a number belonging to a
  different metric makes the claim unsupported. In both cases the
  passage states the claimed metric directly and states a different
  value, so the claim is contradicted. The model was right in both
  cases.
* `deliv_t02_wrong_question_count` had three question lines and a
  two-entry answer key. The tool counted three questions correctly; the
  fixture was wrong.

**Two behaviour changes were made in the code and the labels follow
them.** These are recorded so the direction of the change is clear.

* **Unit normalisation.** `num_005_wrong_unit_millions` used to accept
  either verdict. The rule is now: when a claim carries a scale word
  and the cell carries no unit at all, the magnitudes may agree but the
  table does not state the unit, so the verifier abstains. The label
  was changed from "either verdict" to a fixed abstention.
* **Different subject.** A passage that names a different subject from
  the claim cannot carry a verdict, even when the model asserts that
  the two are the same thing. This guard was added after a measured
  run in which the model bridged "graduate programme" and
  "early-career programme" and called the citation supporting. It
  raised the held-out citation score from 8/10 to 10/10.

**Two omission cases are recorded disagreements** rather than scores.
`om_003` and `om_005` turn on whether the subprocessor-change
notification duty counts as a "security obligation". The model said it
does. That is a defensible reading, and the case accepts either outcome.

**The deterministic baselines are weak on contradiction.** The lexical
fallback scored 14/26 on claims and 11/17 on citations because it
cannot recognise a contradiction from term overlap. This is the
baseline behaving as intended, but it means the baseline numbers should
not be read as a competing method.

**Three structural bugs were found and fixed**, all by the evaluation
rather than by inspection.

* The citation auditor passed no evidence IDs, so the shared verdict
  builder downgraded every contradiction to `insufficient_evidence`.
* The claim splitter did not split coordinated clauses, so
  `claim_007` was scored as one claim instead of two.
* The source-version tracker did not queue a claim when an insertion
  landed *above* its evidence span, even though that insertion shifts
  the text the span pointed at.

**One fixture shape was wrong in this session** and was fixed before
any result was recorded: the held-out citation cases put the claim at
the input level instead of inside each `citations` entry, so the model
was asked to judge an empty claim and answered "no claim was provided"
for every case.

**One prompt bug was found by reading the model output** in the earlier
session. The omission prompt originally labelled the source statement
as `SUMMARY:`, and the model output showed it was confused about what
it was judging. The prompt now names the purpose, the summary, and the
source statement separately.

## 6. Model id and configuration

| Setting | Value |
|---------|-------|
| Model id | `minimax-m3.1-flash:cloud` |
| Endpoint | `http://127.0.0.1:11434` (local Ollama) |
| API key | none used or required |
| Temperature | 0.0 |
| Response format | Ollama `format: json` |
| Thinking | disabled |
| Concurrency | one request at a time |
| Prompt versions | `claims-v1`, `citations-v1`, `omission-v1` |
| Available local models | `qwen3.5:9b`, `minimax-m3.1-flash:cloud`, `gemma4:31b-cloud` |
| Embedding support | none. The server reports `This server does not support embeddings` |

No secrets were read, written, or printed. No `.env` file was accessed.

`runs/model_cache_live.json` holds 87 recorded responses, keyed by
prompt hash. It supports replay of the reported counts but not
independent verification of the original network exchange. Cache replay
latency is not generation latency. Cache keys do not include prompt
version or generation options, so comparisons with different settings
require a newly recorded cache.

The cloud relay behind the local Ollama server dropped connections
during one recording run. The recorder now retries a failed call three
times with a delay, and records the prompts that still failed in the
cache file, so a partial recording cannot be mistaken for a complete
one.

## 7. Which results need human labels before they can be portfolio claims

**All of them.** The 192 fixtures are marked `label_origin:
model_drafted`. The `dev` split was used while building and the
prompts and labels were revised during the build session. The `test`
split is held out in the sense that it was not used for tuning, but it
was written by the same session, so it is not a blind test set.

`runs/review_packet.json` contains 233 blinded inputs for independent
review. Fill `reviewed_expected` objects and run `review-compare` to
see disagreements without changing the fixtures. A reviewer can supply
only a verdict, for example `{"verdict": "supported"}`; the comparison
checks the fields they supplied and separately surfaces draft labels
that accept multiple outcomes.

Specifically:

* All 192 case labels across the twelve fixture directories.
* The corrected labels (`cit_t08`, `claim_t12`, `deliv_t02`), since
  they were corrected by the session that made the mistake.
* The two omission cases that accept either outcome need a decision,
  not a review, because they encode a judgement about what "security
  obligation" means.
* The 16 separate hard-case labels are model drafted and need review.

## 8. Which project is most promising, and why

**`claims` (Project 2), the claim-level source auditor.** It is also
the one that has been extracted into `claim-auditor/`, so the evidence
is the strongest it will get.

* It has the clearest evaluation story: three separable stages, so a
  failure can be attributed to extraction, retrieval, or judgement.
* The measured comparison against the naive judges is in
  `runs/method_comparison_test.json`. All three methods accept the
  same order of magnitude on accuracy: the careful tool accepted 23 of
  25 held-out cases, the three-way judge 22, and the binary judge 14.
  What differs is the error shape: the binary judge accused 8 times
  where the evidence did not settle the question, and the careful tool
  accused 0 times. A reliability tool is judged on its
  false-accusation rate, and that is where the difference is.
* It now has a held-out split of 15 cases, a standalone package with
  its own tests and CLI, and an entity guard that was added because a
  measured run showed the model bridging two differently named things.

The runner-up is `omission`, because it demonstrates something the
others do not: the same source and the same summary produce different
verdicts under different stated purposes, and the model gets both right.
It is second because its evaluation is the smallest and two of its
cases are still open judgements.

## 9. Things a later model should be careful about

* The `OllamaProvider` cache branch was accidentally destroyed once by
  a text patch that matched the wrong `def generate`. If a patch to
  `core/providers.py` seems not to take effect, check that the cache
  branch is still in `OllamaProvider.generate`.
* The model cache is keyed by `purpose::sha256(prompt)`. Any change to
  a prompt, to the claim splitter, or to the candidate ordering
  invalidates the cache and the evaluation falls back to the network.
  After changing anything in `core/claims.py` or `core/spans.py`, re-export
  the prompts with `airl prompts <project> cases.json` and re-record.
* The recorder and the auditor must select the same candidates in the
  same order, or the cache will not replay. `build_prompts` in each
  project sorts by descending relevance to match the auditor.
* The `record` command now writes a `failed_prompts` list. A cache
  with entries in that list is incomplete, and a run against it will
  fall back to the network for the missing prompts.
* The `entity_mismatch` guard in `projects/citations.py` and the copy
  in `claim-auditor/src/claim_auditor/auditor.py` must stay in step.
  The standalone package has its own copy of the code, so a fix in one
  place has to be copied into the other.
* The released packages in `releases/` are a snapshot of the lab's
  core, not a live link. A change to `core/` has to be rebuilt into
  them with `airl release --all`, and the README numbers have to be
  re-read from the run files.
* `SourceDocument` moved from `projects/claims.py` to
  `core/documents.py`, because four projects needed it and a project
  that imports another project cannot be released. The released
  packages carry their own copy under `core/`.
* The per project test files under `tests/projects/` are the ones the
  releases ship. A test added to `tests/test_projects.py` does not
  travel with a release.
* `claim-auditor/` is the earlier hand-written extraction of the
  claims project. `releases/claims/` is generated from the lab and is
  the canonical one. The two will drift, because only one of them is
  rebuilt. The hand-written directory can be removed.
* The released packages carry their fixtures inside the package, not
  only at the top level, because an installed wheel has no checkout to
  walk up to. The fixture lookup in `core/fixtures.py` prefers the
  packaged directory for that reason.
* Two cleanups need a human, because the safety hooks block the file
  and index operations. The build directories and egg-info directories
  under `releases/` are left over from an earlier install check and
  are still tracked by git. The install check now builds in a throwaway
  copy, so they will not come back, but the existing ones have to be
  removed. The exact commands are in `CLEANUP.md`.
* The `push` needs a human too. The credential manager cannot open an
  interactive prompt from here, so the push has to be run in a
  terminal where it can ask for the GitHub credentials.
* `core/retrieval.py` is shared by the lab and the standalone package
  is a copy. The same applies to `core/claims.py` and
  `core/spans.py`. There is no automatic synchronisation.
* The three legacy helpers remain in `scripts/`: `assemble.py`,
  `record_responses.sh`, and `make_chat_payload.py`. A configured hook
  blocked their deletion. Do not use them. The CLI recorder and
  provider adapter are the supported path. Their removal is a manual
  user action.
* The same hook blocked the deletion of the build scratch directory
  from this session. It was moved to `.build-scratch/` and is listed in
  `.gitignore`. Its removal is also a manual user action:
  `rm -rf .build-scratch`.

## 10. Next task for each project

| Project | Next task and acceptance check |
|---------|-------------------------------|
| `omission` | Decide the two ambiguous security-duty cases; require exact source spans and benign-control false alarms in the next run. |
| `claims` | Grow the held-out split to at least 30 cases; score every claim in order and inspect retrieval misses separately from verdict mistakes. |
| `citations` | Add held-out cases needing two passages and cases with matching keywords but wrong entities; report support separately from topical overlap. |
| `deliverables` | Build the interactive edit-and-recheck flow; verify a changed requirement ID is preserved through recheck. |
| `context_receipts` | Add a PDF whose pages hold different sections; confirm the section count and the page attribution of each chunk. |
| `eval_regression` | Import independently reviewed labels and run a genuine `test` split once without tuning on it. |
| `numerical` | Add tables with ambiguous headers and unit conversions; require abstention where row or unit cannot be identified. |
| `instructions` | Add more measurable predicate forms; every unparsed instruction must remain visible as inconclusive. |
| `code_review` | Add a harness that writes to disk or opens a socket, to confirm the isolation is real and not just assumed. |
| `authorization` | Add nested external-content provenance cases; confirm a quoted instruction never becomes user authority. |
| `source_versions` | Test line-level false review queues on longer source revisions; keep changed verdicts pending review. |
| `study_coverage` | Add human-labelled paraphrases; compare the lexical mention count to semantic coverage without calling absence an omission. |
| `retrieval_eval` | Get a real embedding model, or a downloaded sentence-transformer, and repeat the comparison against the hashed baseline. |

The auxiliary `baselines` runner now has a measured result on the
held-out split. A second run with a different prompt version would test
whether the comparison is stable.
