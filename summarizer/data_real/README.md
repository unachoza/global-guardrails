# data_real

Drop point for real leaderboard exports pulled from the live app.

**Nothing in here is committed.** `../.gitignore` excludes `data_real/*.json`
because these are real user submissions — user-generated content does not belong
in a public repo. Export locally, run the pipeline, commit neither.

## Expected format

Identical to [`../data_sim/submissions.json`](../data_sim/submissions.json) — a
flat JSON array, highest score first:

```json
[
  { "submission": "Tell me when you don't know something instead of guessing.", "score": 1842 },
  { "submission": "Never fabricate citations or sources.", "score": 1733 }
]
```

- `submission` — the rule text as the user wrote it. Max 200 chars, matching
  `MAX_LENGTH` in `src/GuardrailForm.tsx`.
- `score` — any positive integer ranking metric. Net upvotes, a 0–100 rating, a
  Wilson bound: the pipeline's score floor is a *fraction of the top score*, so
  it behaves identically across scales. Swap the ranking formula upstream without
  touching this package.

Sort order is not required — the pipeline sorts internally.

## Usage

```bash
python -m guardrail_summarizer.cli data_real/export.json -o out/system_prompt.txt
```
