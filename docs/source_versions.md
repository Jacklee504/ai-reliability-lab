# Source version tracker

Find the claims that need rechecking because the source they cited has changed.

**Question it answers:** Which claims must be rechecked because the source they cite changed?

**Released as:** `source_version_tracker`

## Why it exists

An answer grounded in a document stays grounded only as long as the document is unchanged. Edit a clause and every claim that quoted that clause is now unverified, but nothing in the answer says so.

The tracker finds the claims whose evidence overlaps a changed region and queues them for review. It does not reverse a verdict, because a changed source means the evidence has to be looked at again, not that the old answer was right or wrong.

## How it works

1. **Diffing** finds the changed character ranges between the two versions of a source.
2. **Overlap** tests each recorded evidence span against those ranges.
3. **Queueing** lists the claims to revisit, and records why each one was queued.

## The rules that keep it honest

* A changed source queues a claim for review. It never reverses the verdict.
* An insertion above an evidence span queues the claim, because the same offsets now cover different words.
* Evidence offsets that point outside the source, or a hash that does not match the source, fail closed into the queue.

## Known limitations

* The tracker needs the recorded evidence offsets, so it only works for claims that were audited with this tool.
* It compares versions of the same document. It does not detect that the document itself was superseded by a different one.
* The queue is a work list, not a verdict.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval source_versions --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release source_versions --out releases
```
