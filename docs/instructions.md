# Instruction checker

Replay a coding agent's output against the measurable instructions it was given.

**Question it answers:** Did the agent follow the measurable repository instructions?

**Released as:** `instruction_checker`

## Why it exists

A coding agent can finish a task and quietly break a rule it was given: a protected file was edited, a required test was never added, a linter was never run. The output looks finished, so the violation is only visible if you re-read the instructions against the diff.

The checker replays the candidate output against the instructions and checks each one as a predicate. It checks the artifacts, not the attention, and it never claims that an agent read an instruction, only that the artifacts satisfy it.

## How it works

1. **Parsing** turns each instruction into an explicit predicate that says what must be true of the final artifacts.
2. **Evaluation** checks each predicate against the before and after file states, and against the agent trace for commands.
3. **Reporting** lists what passed, what was violated, and what could not be checked.

## The rules that keep it honest

* An instruction that cannot be turned into a predicate is reported as inconclusive, never as a pass.
* A satisfied predicate does not prove the instruction was read, only that the artifacts satisfy it.
* A file that was not in the starting state cannot be shown to be unchanged, so that check is inconclusive rather than a pass.

## Known limitations

* Only measurable instructions are checked. Write it clearly is not a predicate.
* The checker reads the artifact state and the command trace. It cannot tell whether a command succeeded, only that it ran.
* It checks the final state, not the path taken to get there.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval instructions --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release instructions --out releases
```
