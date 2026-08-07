# Summarizer

Takes the ranked submission leaderboard and produces one system prompt a user can
download and paste into their own Claude setup.

```
[{"submission": str, "score": int}, ...]  ->  system prompt text
```

Self-contained: everything lives under `summarizer/`, nothing outside it is
touched, and the Vite build ignores it (`tsconfig.app.json` only includes `src`).

## Run

```bash
cd summarizer
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

./.venv/bin/python -m guardrail_summarizer.cli data_sim/submissions.json \
    -o out/system_prompt.txt \
    --manifest out/manifest.json
```

Tests run without an API key — the LLM stages are stubbed, so CI costs nothing:

```bash
./.venv/bin/python tests/test_pipeline.py
```

## Pipeline

```
normalize -> select -> screen -> cluster -> resolve -> prune -> compose -> validate
  (local)   (local)    (LLM)     (LLM)      (LLM)     (local)   (LLM)      (LLM)
```

| Stage | Does |
|---|---|
| `normalize` | Collapse whitespace, drop empties, **sum** exact duplicates |
| `select` | Score floor + top-K cap. Cost control only |
| `screen` | Drop injections, unsafe asks, non-rules. Judges validity, never popularity |
| `cluster` | Group submissions asking for the same thing; sum their scores |
| `resolve` | Reconcile rules that cannot all be satisfied at once |
| `prune` | Drop rules below a share of the top *rule*'s weight — consensus filter |
| `compose` | Write the prompt: grouped, weighted, plain language |
| `validate` | Every line traces to a rule; no contradictions; nothing injected survived |

### Why there are two weight filters

`select` compares raw submissions and exists to cap cost. `prune` compares
*clustered* rules and exists to enforce consensus. They cannot be collapsed into
one: before clustering, a lone novelty is indistinguishable from a real rule that
happens to have been phrased once, so any pre-cluster floor strict enough to
catch the novelty would also delete genuine single-phrasing rules.

This is not hypothetical. In the first live run against `data_sim`, *"Speak like
a pirate at all times"* (188) cleared the pre-cluster floor of 93 and landed in
the finished prompt. After clustering, the top rule weighed 3452 — making the
novelty 5.4%, plainly below consensus. `prune` is that second look.

It is a consensus threshold, not a taste filter: a joke rule that genuinely wins
the vote still ships. That is the community's call, not the pipeline's.

Cost is flat in the number of LLM calls (~5), not the number of submissions —
`select` caps what reaches the model and `screen`/`cluster` batch what remains. A
50,000-row leaderboard costs about what a 50-row one does.

## Three design decisions worth knowing about

**Clustering is the point, not an optimization.** Four people asking for the same
rule in four wordings is the strongest signal a voting platform produces. A
pipeline that emits four lines has thrown that signal away *and* burned the
user's context budget. Clusters sum their members' scores, so convergence
becomes weight.

**Conflict resolution is a product decision**, exposed as `Config.conflict_policy`.
Real submission sets contain rules that cannot both be followed — "keep it under
150 words" and "always explain thoroughly" both poll well. `"scope"` reconciles
by giving each behaviour a context; `"score"` lets the vote decide outright.
Pick one and defend it: shipping a self-contradicting prompt is the one outcome
that is always wrong.

**Injection defence is layered, because no single layer holds.** The score floor
is *not* one of the layers — in `data_sim/submissions.json` the injection row
scores 97 against a floor of 93, and clears it. The layers are: `screen`
classifies every submission before it influences anything; untrusted text is
HTML-escaped and wrapped in delimiters it cannot break out of, with every stage
prompt restating that submissions are data and never instructions; `validate`
re-reads the finished prompt for anything that survived. This pipeline compiles
user-generated text into a system prompt other people will run — treat it
accordingly.

## Data

| Folder | Contents |
|---|---|
| `data_sim/` | Simulated 30-row fixture. Committed. Deliberately contains duplicate clusters, two contradictions, an injection attempt, and low-signal noise |
| `data_real/` | Drop point for live exports. **Gitignored** — real user submissions don't belong in a public repo |

## Wiring it to the app

The frontend is a Vite SPA with no backend, and the Anthropic API key can never
reach the browser, so this needs a server process either way. The seam is one
function:

```python
from guardrail_summarizer import Config, build, normalize

result = build(normalize(rows), Config(target_words=400))
result.prompt        # the downloadable text
result.manifest()    # which submissions produced which rule
```

Wrap that in whatever you deploy — a FastAPI route, a serverless function, a
scheduled job that regenerates on leaderboard change.

**Cache the output keyed on a hash of the input rows.** `temperature` is removed
on Claude Opus 5 and Sonnet 5 (passing it returns a 400), so identical inputs are
not guaranteed to produce identical prompts. Caching gives you the stability, and
is cheaper than regenerating per download regardless.

`result.manifest()` is worth surfacing in the UI: *"this line came from 4
submissions, 6,200 votes."* Traceability is what makes a generated prompt
trustworthy.

## Layout

```
summarizer/
  guardrail_summarizer/
    config.py       tuning knobs
    schemas.py      typed contracts for every LLM stage
    prompts.py      stage prompts
    pipeline.py     stages + orchestration
    cli.py          command line entry point
  data_sim/         simulated input (committed)
  data_real/        live exports (gitignored)
  tests/            runs without an API key
```
