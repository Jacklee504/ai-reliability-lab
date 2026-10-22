# Context receipt

Record exactly which file version and which text a pipeline submitted to a model.

**Question it answers:** Which file version and which text did the pipeline send?

**Released as:** `context_receipt`

## Why it exists

A controlled pipeline can quietly send the wrong file, an older revision, or a document whose extraction half failed. The model's answer then looks reasonable and is grounded in something the user never chose.

A receipt answers the question before the model is asked: which bytes were read, what text came out of them, which chunks were submitted, and what went wrong. It makes no model calls, so it is evidence rather than another inference.

## How it works

1. **Ingestion** reads the file, recording its content hash, media type, extraction status, and any warnings.
2. **Chunking** splits the text on statement boundaries, so a chunk never cuts an obligation in half.
3. **Receipt** records the extracted characters, the page and section counts, the chunk identifiers, and the exact text that was submitted. Two receipts can be diffed.

## The rules that keep it honest

* A file that could not be read still produces a receipt, and that receipt says the extraction failed rather than reporting an empty document.
* The receipt is reproducible: the same file gives the same hash, the same chunks, and the same identifiers.
* The receipt records what the pipeline submitted. It does not claim the model used it, which is not knowable from outside.

## Known limitations

* The receipt shows what was submitted, not what the model read or reasoned over.
* PDF support needs `pypdf`. Without it the receipt reports that PDF extraction is unavailable rather than guessing.
* Only text extraction is supported. Scanned documents come back as empty with a warning.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval context_receipts --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release context_receipts --out releases
```
