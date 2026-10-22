# Projects

One document per project. Each says what the project answers,
how it works, the rules that keep it honest, and what it cannot
do.

| Project | Answers | Released as |
|---------|---------|--------------|
| [`authorization`](authorization.md) | Was the agent authorised to take this action? | `authorization_checker` |
| [`baselines`](baselines.md) | Do the careful prompts earn their keep, or would a simple judge do as well? | `naive_judge_baselines` |
| [`citations`](citations.md) | Does this cited passage actually support the sentence it is attached to? | `citation_auditor` |
| [`claims`](claims.md) | Which sentences in this answer are not supported by the source? | `claim_auditor` |
| [`code_review`](code_review.md) | Is this review comment technically true of the code? | `code_review_verifier` |
| [`context_receipts`](context_receipts.md) | Which file version and which text did the pipeline send? | `context_receipt` |
| [`deliverables`](deliverables.md) | Did the output meet every requirement in the request? | `deliverable_checker` |
| `eval_regression` | the cross project regression suite | not released |
| [`instructions`](instructions.md) | Did the agent follow the measurable repository instructions? | `instruction_checker` |
| [`numerical`](numerical.md) | Did the number come from the right row, column, year, and unit? | `numerical_verifier` |
| [`omission`](omission.md) | What did this summary leave out that my brief cared about? | `omission_auditor` |
| [`retrieval_eval`](retrieval_eval.md) | Does the ranker retrieve the passage that answers the query? | `retrieval_eval` |
| [`source_versions`](source_versions.md) | Which claims must be rechecked because the source they cite changed? | `source_version_tracker` |
| [`study_coverage`](study_coverage.md) | Which syllabus topics are explicitly mentioned in the material? | `study_coverage` |
