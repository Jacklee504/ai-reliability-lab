"""The written content for each project's release.

This is the single source of truth for what a project says about
itself. The release builder turns it into the package README, and the
lab writes the same text into its own documentation, so the two can
never drift apart.

Every field here is written by hand. The measured numbers are not
here on purpose: they change with every run, so the builder reads
them from the run files rather than from prose that would go stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectContent:
    """The written content for one project."""

    title: str
    summary: str
    question: str
    why: str
    stages: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)
    method: str = "deterministic"

    def bullets(self, items: list[str]) -> str:
        """A markdown bullet list."""
        return "\n".join("* %s" % item for item in items)


CONTENT: dict[str, ProjectContent] = {}


CONTENT["claims"] = ProjectContent(
    title="Claim auditor",
    summary="Check whether the claims in an AI answer are supported by the sources it was given.",
    question="Which sentences in this answer are not supported by the source?",
    why=(
        "An AI answer can mix supported, contradicted, and unsupported\n"
        "statements in the same sentence. A source can discuss exactly the\n"
        "right topic and still not answer the question. A tool that reports\n"
        "a contradiction in those cases is worse than no tool, because a\n"
        "false accusation is a claim the user cannot easily dismiss.\n"
        "\n"
        "So the tool is built around one rule: a missing passage is never a\n"
        "contradiction. When the evidence does not settle the question, the\n"
        "answer is `insufficient_evidence`, and that is reported separately\n"
        "from `contradicted` so the two can be counted apart."
    ),
    stages=[
        "**Claim extraction** splits the answer into atomic claims and keeps "
        "the qualifiers that change their meaning: dates, quantities, "
        "subject, scope, negation, and modality. Offsets into the original "
        "answer are preserved exactly.",
        "**Evidence retrieval** finds candidate source passages. The "
        "retrieval score is topical relevance and nothing more; a passage "
        "that shares every keyword can still fail to state the fact the "
        "claim needs.",
        "**Verdict classification** decides, per claim, whether the evidence "
        "supports, contradicts, or does not settle the claim.",
    ],
    rules=[
        "A verdict may only cite evidence that was actually retrieved.",
        "A universal claim such as all, every, or only is never "
        "contradicted by a passage about one example, and never supported by "
        "one either.",
        "A verdict with an empty evidence set cannot be `contradicted`.",
        "A passage about a different subject cannot carry a verdict, even "
        "when the model asserts that two differently named things are the "
        "same thing.",
    ],
    limits=[
        "The deterministic path cannot recognise a contradiction. It "
        "reports `supported` when a passage contains all of the claim's "
        "terms and abstains otherwise. A model is needed for the "
        "contradiction cases.",
        "The claim splitter works on English sentences. It is not a "
        "multilingual claim segmenter.",
        "The tool judges only against the sources it is given. It does not "
        "check whether those sources are current or correct.",
    ],
    method="hybrid",
)


CONTENT["citations"] = ProjectContent(
    title="Citation auditor",
    summary="Check whether a cited passage actually supports the sentence it is attached to.",
    question="Does this cited passage actually support the sentence it is attached to?",
    why=(
        "A citation that looks right and supports nothing is the most "
        "common failure in a written answer, and it is invisible to the "
        "reader. The passage mentions every keyword and simply does not "
        "state the fact.\n"
        "\n"
        "This tool separates two things that are usually conflated: the "
        "citation is *topically relevant*, and the citation *supports the "
        "claim*. A passage can be perfectly on topic and still fail, and "
        "that is `insufficient_evidence`, not `supported`."
    ),
    stages=[
        "**Passage alignment** attaches each cited passage to the sentence "
        "it is cited for.",
        "**Topical scoring** measures how much vocabulary the passage "
        "shares with the claim, reported separately from support.",
        "**Support checking** decides whether the passage states the claim, "
        "contradicts it, or leaves it open. Two passages may together "
        "support a claim that neither supports alone.",
    ],
    rules=[
        "Topical overlap is never reported as support.",
        "A citation that is absent is `insufficient_evidence`, not a "
        "contradiction. A missing citation is a failure mode, not a truth "
        "claim.",
        "Partial support is not full support.",
        "A citation about a different entity cannot support the claim.",
    ],
    limits=[
        "The judge works on the passages given to it. It cannot tell you "
        "that a better passage exists elsewhere in the document.",
        "The deterministic path cannot recognise a contradiction from term "
        "overlap.",
        "The tool does not check that the cited source is real, current, or "
        "says what it is claimed to say outside the quoted passage.",
    ],
    method="hybrid",
)


CONTENT["omission"] = ProjectContent(
    title="Omission auditor",
    summary="Find what a summary left out that the reader's stated brief cared about.",
    question="What did this summary leave out that my brief cared about?",
    why=(
        "A summary can contain only true statements and still be wrong, "
        "because what it left out may be the thing the reader needed. The "
        "problem is that an omission has no positive evidence: there is no "
        "sentence in the summary to point at.\n"
        "\n"
        "The tool needs a stated purpose to work from. What was left out is "
        "meaningless without one, and it is a different answer for every "
        "brief. The same summary against the same source produces different "
        "findings depending on what the summary was for, and that is the "
        "point rather than a limitation of the method."
    ),
    stages=[
        "**Segmentation** splits the source into candidate statements, each "
        "with the exact span it came from.",
        "**Alignment** matches source statements against the summary.",
        "**Relevance ranking** scores the unmatched statements against the "
        "stated purpose, so what is reported is what the brief cared about, "
        "not everything that was left out.",
    ],
    rules=[
        "The purpose is required. Without one the tool reports nothing "
        "rather than guessing what mattered.",
        "A finding carries the exact source text and its location, so a "
        "reader can check it.",
        "Benign omissions are not findings. A summary that leaves out "
        "irrelevant detail is not an error, and a detector that flags every "
        "dropped sentence is useless.",
    ],
    limits=[
        "Importance is judged against a stated purpose. A summary with no "
        "stated purpose cannot be audited for omissions.",
        "The tool finds omissions in the *source*. It does not know what the "
        "reader already knew.",
        "Two omission cases in the evaluation accept either outcome, because "
        "whether a governance duty counts as a security duty is a judgement "
        "the tool does not make for you.",
    ],
    method="hybrid",
)


CONTENT["deliverables"] = ProjectContent(
    title="Deliverable checker",
    summary="Check a generated deliverable against every requirement in the request that produced it.",
    question="Did the output meet every requirement in the request?",
    why=(
        "A generated deliverable can read fluently and still miss what was "
        "asked: a missing answer key, three questions where fifty were "
        "wanted, a format that was not the one requested. The fluency is "
        "exactly what hides the gap.\n"
        "\n"
        "The checker turns the request into explicit requirement records and "
        "checks them one at a time. The rule that makes it usable is that it "
        "never invents a requirement. A checker that failed a correct "
        "deliverable for a rule the user never gave would be worse than no "
        "checker at all."
    ),
    stages=[
        "**Requirement extraction** turns the request into records that say "
        "what is required and whether it can be checked mechanically or "
        "needs judgement.",
        "**Structural checking** verifies counts, headings, required "
        "fields, and named sections against the deliverable.",
        "**Reporting** shows every requirement, what was observed, and "
        "which ones still need a human.",
    ],
    rules=[
        "Every requirement traces back to something in the request. No "
        "requirement is invented.",
        "A requirement that needs judgement is reported as "
        "`cannot_verify`, never guessed at.",
        "A request with no extractable requirements produces "
        "`cannot_verify`, not `passed`. Reporting a pass for a vague request "
        "would be a false assurance.",
        "The extracted list is editable, and the requirement IDs survive a "
        "recheck, so a correction can be traced.",
    ],
    limits=[
        "Requirement extraction from free-form prose is the weakest link. A "
        "request that says make it good yields no requirements.",
        "The checks confirm structure, not quality. Fifty questions of "
        "nonsense passes the count check.",
        "Only English request phrasing is understood.",
    ],
)


CONTENT["context_receipts"] = ProjectContent(
    title="Context receipt",
    summary="Record exactly which file version and which text a pipeline submitted to a model.",
    question="Which file version and which text did the pipeline send?",
    why=(
        "A controlled pipeline can quietly send the wrong file, an older "
        "revision, or a document whose extraction half failed. The model's "
        "answer then looks reasonable and is grounded in something the user "
        "never chose.\n"
        "\n"
        "A receipt answers the question before the model is asked: which "
        "bytes were read, what text came out of them, which chunks were "
        "submitted, and what went wrong. It makes no model calls, so it is "
        "evidence rather than another inference."
    ),
    stages=[
        "**Ingestion** reads the file, recording its content hash, media "
        "type, extraction status, and any warnings.",
        "**Chunking** splits the text on statement boundaries, so a chunk "
        "never cuts an obligation in half.",
        "**Receipt** records the extracted characters, the page and section "
        "counts, the chunk identifiers, and the exact text that was "
        "submitted. Two receipts can be diffed.",
    ],
    rules=[
        "A file that could not be read still produces a receipt, and that "
        "receipt says the extraction failed rather than reporting an empty "
        "document.",
        "The receipt is reproducible: the same file gives the same hash, "
        "the same chunks, and the same identifiers.",
        "The receipt records what the pipeline submitted. It does not claim "
        "the model used it, which is not knowable from outside.",
    ],
    limits=[
        "The receipt shows what was submitted, not what the model read or "
        "reasoned over.",
        "PDF support needs `pypdf`. Without it the receipt reports that PDF "
        "extraction is unavailable rather than guessing.",
        "Only text extraction is supported. Scanned documents come back as "
        "empty with a warning.",
    ],
)


CONTENT["numerical"] = ProjectContent(
    title="Numerical verifier",
    summary="Check that a number in an answer came from the right row, column, year, and unit.",
    question="Did the number come from the right row, column, year, and unit?",
    why=(
        "A number in an answer is the most likely thing to be wrong and the "
        "hardest for a reader to check. The value is right, but it came from "
        "last year's column, or the wrong company, or the wrong metric, or "
        "it is off by a factor of a thousand.\n"
        "\n"
        "The verifier identifies the row, the column, and the unit from the "
        "claim itself, and it shows its arithmetic. A number that cannot be "
        "tied to a single cell is not asserted, because picking the nearest "
        "match is how a wrong number gets in."
    ),
    stages=[
        "**Table parsing** reads the table and labels its rows and columns.",
        "**Claim parsing** identifies the entity, the metric, the period, "
        "the value, and the unit from the claim text.",
        "**Verification** selects the cell, compares the values, and shows "
        "the arithmetic for derived claims such as a percentage change.",
    ],
    rules=[
        "A claim that does not identify a row or a year is reported as "
        "uncertain, not matched to the nearest cell.",
        "A derived claim must show the arithmetic that produced it.",
        "A scale word in a claim and no unit in the table is a mismatch the "
        "tool abstains on, because the table does not say what the numbers "
        "are.",
    ],
    limits=[
        "Only straightforward tables are read. Merged cells, footnotes, and "
        "multi-level headers are not interpreted.",
        "The verifier checks arithmetic and cell selection, not whether the "
        "source table is itself true.",
        "Percent changes are computed from the two stated values. It does "
        "not model compounding or fiscal calendars.",
    ],
)


CONTENT["instructions"] = ProjectContent(
    title="Instruction checker",
    summary="Replay a coding agent's output against the measurable instructions it was given.",
    question="Did the agent follow the measurable repository instructions?",
    why=(
        "A coding agent can finish a task and quietly break a rule it was "
        "given: a protected file was edited, a required test was never "
        "added, a linter was never run. The output looks finished, so the "
        "violation is only visible if you re-read the instructions against "
        "the diff.\n"
        "\n"
        "The checker replays the candidate output against the instructions "
        "and checks each one as a predicate. It checks the artifacts, not "
        "the attention, and it never claims that an agent read an "
        "instruction, only that the artifacts satisfy it."
    ),
    stages=[
        "**Parsing** turns each instruction into an explicit predicate that "
        "says what must be true of the final artifacts.",
        "**Evaluation** checks each predicate against the before and after "
        "file states, and against the agent trace for commands.",
        "**Reporting** lists what passed, what was violated, and what could "
        "not be checked.",
    ],
    rules=[
        "An instruction that cannot be turned into a predicate is reported "
        "as inconclusive, never as a pass.",
        "A satisfied predicate does not prove the instruction was read, only "
        "that the artifacts satisfy it.",
        "A file that was not in the starting state cannot be shown to be "
        "unchanged, so that check is inconclusive rather than a pass.",
    ],
    limits=[
        "Only measurable instructions are checked. Write it clearly is not a "
        "predicate.",
        "The checker reads the artifact state and the command trace. It "
        "cannot tell whether a command succeeded, only that it ran.",
        "It checks the final state, not the path taken to get there.",
    ],
)


CONTENT["code_review"] = ProjectContent(
    title="Code review verifier",
    summary="Check whether an AI review comment is technically true of the code it points at.",
    question="Is this review comment technically true of the code?",
    why=(
        "An AI review comment can sound precise while being false. It cites a "
        "real line number, quotes real code, and still misses a guard a few "
        "lines up, a caller elsewhere, or a test that covers the case.\n"
        "\n"
        "The verifier extracts the checkable assertion from the comment and "
        "looks for evidence. It reports what it can establish, and abstains "
        "on what it cannot, which is the honest answer for a comment about "
        "concurrency or a runtime failure."
    ),
    stages=[
        "**Assertion extraction** separates the checkable claim from the "
        "preamble and the advice.",
        "**Static checks** look for the guard, the bound, or the unbounded "
        "construct the comment says is missing.",
        "**Execution** runs the reviewer's own harness in a separate process "
        "with a timeout, when the comment is about runtime behaviour.",
    ],
    rules=[
        "A comment about a file that is not in the snapshot is unverified, "
        "not wrong and not right.",
        "A comment that cannot be parsed is unverified. Malformed code never "
        "supports an accusation.",
        "A harness that times out, fails to import, or is absent leaves the "
        "claim unverified. A broken harness can never manufacture evidence "
        "against a comment.",
        "Style is not a technical assertion and is never judged.",
    ],
    limits=[
        "The static checks cover a narrow set of structural claims: missing "
        "guards, unbounded loops, and the like.",
        "A concurrency claim cannot be settled by reading a function alone.",
        "The execution path runs the reviewer's harness, not the project's "
        "test suite, so it shows what that harness observed.",
    ],
)


CONTENT["authorization"] = ProjectContent(
    title="Authorization checker",
    summary="Check whether an agent was authorised to take the action it proposed to take.",
    question="Was the agent authorised to take this action?",
    why=(
        "An agent reading a web page can be told what to do by that page. A "
        "ticket that says email the report to this address, or a document "
        "that quotes the user giving permission, is untrusted content "
        "wearing the user's clothes.\n"
        "\n"
        "The checker evaluates a proposed action against what the user "
        "actually authorised, and it fails closed. It never acts on the "
        "action. It only answers whether the action was permitted, so the "
        "decision stays with the caller."
    ),
    stages=[
        "**Provenance** records where the instruction came from: the user, "
        "a trusted application policy, or untrusted content.",
        "**Scope** checks the action type, the target, and the data class "
        "against what was authorised.",
        "**Decision** returns allowed, denied, or needs clarification, with "
        "the rule that fired recorded in the trace.",
    ],
    rules=[
        "Untrusted content can never authorise an action, whatever it quotes "
        "or claims.",
        "Reading untrusted content is allowed. Only acting on it needs "
        "authority.",
        "An action in scope for the action type but out of scope for the "
        "target or the data class returns needs clarification, not a silent "
        "allow and not a blunt deny.",
        "An instruction whose provenance is unknown returns needs "
        "clarification rather than being assumed safe.",
    ],
    limits=[
        "The checker evaluates a proposed action. It does not observe what "
        "the agent actually did.",
        "The policy is a small fixed rule set. A real deployment needs its "
        "own organisation's rules.",
        "It judges the scope of an action, not whether the action was wise.",
    ],
)


CONTENT["source_versions"] = ProjectContent(
    title="Source version tracker",
    summary="Find the claims that need rechecking because the source they cited has changed.",
    question="Which claims must be rechecked because the source they cite changed?",
    why=(
        "An answer grounded in a document stays grounded only as long as the "
        "document is unchanged. Edit a clause and every claim that quoted "
        "that clause is now unverified, but nothing in the answer says so.\n"
        "\n"
        "The tracker finds the claims whose evidence overlaps a changed "
        "region and queues them for review. It does not reverse a verdict, "
        "because a changed source means the evidence has to be looked at "
        "again, not that the old answer was right or wrong."
    ),
    stages=[
        "**Diffing** finds the changed character ranges between the two "
        "versions of a source.",
        "**Overlap** tests each recorded evidence span against those ranges.",
        "**Queueing** lists the claims to revisit, and records why each one "
        "was queued.",
    ],
    rules=[
        "A changed source queues a claim for review. It never reverses the "
        "verdict.",
        "An insertion above an evidence span queues the claim, because the "
        "same offsets now cover different words.",
        "Evidence offsets that point outside the source, or a hash that does "
        "not match the source, fail closed into the queue.",
    ],
    limits=[
        "The tracker needs the recorded evidence offsets, so it only works "
        "for claims that were audited with this tool.",
        "It compares versions of the same document. It does not detect that "
        "the document itself was superseded by a different one.",
        "The queue is a work list, not a verdict.",
    ],
)


CONTENT["study_coverage"] = ProjectContent(
    title="Study coverage auditor",
    summary="Check which syllabus topics are explicitly mentioned in a set of study material.",
    question="Which syllabus topics are explicitly mentioned in the material?",
    why=(
        "Syllabus coverage is usually checked by eye, and by eye it is easy "
        "to accept a description of a topic as coverage of it. A student "
        "reading about how a function calls itself has read about "
        "recursion, but the word never appears.\n"
        "\n"
        "The tool proves only the narrow thing it can prove: a topic phrase "
        "appeared. It reports a mention or it reports an unverified topic, "
        "and it never says a topic was taught well, because a mention is not "
        "evidence of understanding."
    ),
    stages=[
        "**Topic definition** records each syllabus topic with its label and "
        "any accepted aliases.",
        "**Mention counting** finds the phrase occurrences, respecting word "
        "boundaries so a longer word does not count.",
        "**Reporting** lists the topics that were mentioned and the ones that "
        "were not, without judging either.",
    ],
    rules=[
        "A mention is a lexical observation, not a claim about teaching "
        "quality.",
        "A paraphrase is not a mention. The tool reports what it can prove.",
        "Repetition is counted, but the count is not a measure of emphasis.",
    ],
    limits=[
        "The tool measures lexical presence, not semantic coverage. A topic "
        "described in other words is reported as unverified.",
        "Aliases are supplied by the caller. A missing alias means a missed "
        "mention.",
        "It says nothing about whether the material teaches the topic well "
        "enough.",
    ],
)


CONTENT["retrieval_eval"] = ProjectContent(
    title="Retrieval comparison",
    summary="Measure whether a ranker retrieves the passage that actually answers the query.",
    question="Does the ranker retrieve the passage that answers the query?",
    why=(
        "A tool can retrieve the right passage and still judge it wrongly, "
        "and it can judge correctly while having retrieved the wrong passage. "
        "The two failures look the same from the outside unless retrieval is "
        "measured on its own.\n"
        "\n"
        "This project isolates the ranking stage. Each case supplies a query, "
        "a set of candidate passages, and the span that actually answers it, "
        "so the two rankers can be compared on the same question: did the "
        "right passage make the cut, and how high did it rank?"
    ),
    stages=[
        "**Candidate selection** takes the labelled candidate set and the "
        "span that answers the query.",
        "**Ranking** runs each method over the same candidates.",
        "**Scoring** reports recall at k and mean reciprocal rank for each "
        "method.",
    ],
    rules=[
        "A retrieval score is topical relevance, never evidence of "
        "entailment.",
        "The candidate set is supplied, so the measurement isolates ranking "
        "and says nothing about chunking or source selection.",
    ],
    limits=[
        "The embedding provider here is a deterministic hashed embedder, not "
        "a trained model. It has no synonymy, so it measures the mechanism "
        "rather than semantic retrieval.",
        "The candidate set is given, so the measurement does not cover the "
        "whole retrieval pipeline.",
    ],
)


CONTENT["baselines"] = ProjectContent(
    title="Naive judge baselines",
    summary="Stripped-down LLM judges, used to test whether the careful prompts earn their keep.",
    question="Do the careful prompts earn their keep, or would a simple judge do as well?",
    why=(
        "A careful prompt can flatter a method. The tool prompts are narrow, "
        "they name four outcomes, and they are wrapped in abstention rules "
        "that downgrade an unjustified contradiction. That is a lot of "
        "machinery, and machinery should be measured.\n"
        "\n"
        "These baselines are the stripped alternatives: the same model, the "
        "same cases, the same evidence, with the guidance removed. One is the "
        "most common way an LLM grader is wired up, a yes or no question. The "
        "other is the three way version, still without any guidance about "
        "when to accuse."
    ),
    stages=[
        "**Context building** gives each baseline exactly the evidence the "
        "tool saw, so the comparison is fair.",
        "**Prompting** sends the stripped question to the same model.",
        "**Scoring** counts the same outcomes, including the cases where a "
        "baseline accused where it should have abstained.",
    ],
    rules=[
        "A baseline that abstains where the tool abstains is not a failure. "
        "A baseline that accuses where the tool abstains is a false "
        "accusation and is counted as one.",
        "The baselines are not tuned. Nothing here was changed to make a "
        "baseline look worse.",
        "A binary grader has no way to express that the passage does not "
        "answer the question, so a no is mapped to `contradicted`. That is "
        "the point of the baseline.",
    ],
    limits=[
        "The comparison is against one model on one case set, so it says "
        "something about that configuration, not about models in general.",
        "The baselines reuse the claim and citation cases, so the comparison "
        "covers those two projects.",
    ],
    method="naive_llm_judge",
)
