"""GlobalGuardrail summarizer.

Input:  [{"submission": str, "score": int}, ...]
Output: a system prompt a user can paste into their own Claude setup.

Stages, in order:

    normalize -> select -> screen -> cluster -> resolve -> compose -> validate
      (local)   (local)    (LLM)     (LLM)      (LLM)      (LLM)      (LLM)

Local stages are deterministic and free; run them in tests without an API key.
Cost is flat in the number of LLM calls, not the number of submissions: screening
and clustering batch, so a 5,000-row leaderboard costs about the same as a 50-row
one once top_k has done its work.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .config import Config
from .prompts import (
    CLUSTER_SYSTEM,
    COMPOSE_SYSTEM,
    CONFLICT_POLICY,
    CONFLICT_SYSTEM,
    SCREEN_SYSTEM,
    VALIDATE_SYSTEM,
)
from .schemas import (
    Cluster,
    ClusterResult,
    Conflict,
    ConflictResult,
    ScreenResult,
    Submission,
    ValidationResult,
)


class PipelineError(RuntimeError):
    pass


@dataclass
class Rule:
    """One deduplicated requirement, plus a trail back to what produced it."""

    text: str
    theme: str
    score: int
    sources: list[int] = field(default_factory=list)
    """Indices into the normalized submission list."""


@dataclass
class Result:
    prompt: str
    rules: list[Rule]
    submissions: list[Submission]
    rejected: list[tuple[int, str]]
    conflicts: list[Conflict]
    validation: ValidationResult | None
    stats: dict[str, Any]

    def manifest(self) -> dict[str, Any]:
        """Traceability record: which submissions produced which rule.

        Worth surfacing in the UI -- "this line came from 4 submissions, 6,200
        votes" is the thing that makes a generated prompt trustworthy.
        """
        return {
            "stats": self.stats,
            "rules": [
                {
                    "text": r.text,
                    "theme": r.theme,
                    "weight": r.score,
                    "sources": [
                        {"index": i, "submission": self.submissions[i].submission,
                         "score": self.submissions[i].score}
                        for i in r.sources
                    ],
                }
                for r in self.rules
            ],
            "rejected": [
                {"index": i, "submission": self.submissions[i].submission, "reason": why}
                for i, why in self.rejected
            ],
            "conflicts": [c.model_dump() for c in self.conflicts],
            "validation": self.validation.model_dump() if self.validation else None,
        }


# --------------------------------------------------------------------------- #
# LLM access
# --------------------------------------------------------------------------- #


class LLM:
    """Thin wrapper so stages take a dependency that can be faked in tests."""

    def __init__(self, client: Any = None, max_tokens: int = 16000) -> None:
        if client is None:
            import anthropic  # imported lazily: local stages work without the SDK

            client = anthropic.Anthropic()
        self.client = client
        self.max_tokens = max_tokens

    def parse(self, *, model: str, system: str, user: str, schema: type) -> Any:
        response = self.client.messages.parse(
            model=model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if response.stop_reason == "refusal":
            raise PipelineError(f"{model} declined the request during structured parsing")
        parsed = response.parsed_output
        if parsed is None:
            raise PipelineError(f"{model} returned no parseable output for {schema.__name__}")
        return parsed

    def text(self, *, model: str, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            raise PipelineError(f"{model} declined the request during composition")
        out = "".join(b.text for b in response.content if b.type == "text").strip()
        if not out:
            raise PipelineError(f"{model} returned an empty composition")
        return out


# --------------------------------------------------------------------------- #
# Local stages
# --------------------------------------------------------------------------- #


def load(path: str | Path) -> list[Submission]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise PipelineError("expected a JSON array of {submission, score} objects")
    return normalize(raw)


def normalize(raw: Iterable[dict[str, Any]]) -> list[Submission]:
    """Collapse whitespace, drop empties, and sum exact duplicates.

    Summing rather than deduping is the right aggregation here: two identical
    submissions really are two people asking for the same thing.
    """
    merged: dict[str, Submission] = {}
    order: list[str] = []
    for entry in raw:
        try:
            text = " ".join(str(entry["submission"]).split())
            score = int(entry["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError(f"malformed row {entry!r}: {exc}") from exc
        if not text:
            continue
        key = text.casefold()
        if key in merged:
            merged[key].score += score
        else:
            merged[key] = Submission(submission=text, score=score)
            order.append(key)
    return [merged[k] for k in order]


def select(subs: Sequence[Submission], cfg: Config) -> list[int]:
    """Return indices of the submissions worth spending tokens on, best first.

    The floor is a fraction of the top score rather than an absolute, so this
    behaves the same whether scores are vote counts, percentages, or Wilson bounds.
    """
    if not subs:
        return []
    ranked = sorted(range(len(subs)), key=lambda i: subs[i].score, reverse=True)
    top = subs[ranked[0]].score
    floor = max(
        cfg.min_score or 0,
        math.ceil(top * cfg.min_score_ratio) if cfg.min_score_ratio else 0,
    )
    kept = [i for i in ranked if subs[i].score >= floor]
    return kept[: cfg.top_k] if cfg.top_k else kept


def _chunk(items: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _wrap(texts: Sequence[str]) -> str:
    """Render untrusted text inside delimiters it cannot escape."""
    lines = ["<submissions>"]
    for i, t in enumerate(texts):
        lines.append(f'<item index="{i}">{html.escape(t, quote=False)}</item>')
    lines.append("</submissions>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM stages
# --------------------------------------------------------------------------- #


def screen(
    llm: LLM, subs: Sequence[Submission], indices: Sequence[int], cfg: Config
) -> tuple[list[int], list[tuple[int, str]]]:
    """Drop injections, unsafe asks, and non-rules. Popularity is not judged here."""
    kept: list[int] = []
    rejected: list[tuple[int, str]] = []
    for chunk in _chunk(indices, cfg.chunk_size):
        result: ScreenResult = llm.parse(
            model=cfg.screen_model,
            system=SCREEN_SYSTEM,
            user=_wrap([subs[i].submission for i in chunk]),
            schema=ScreenResult,
        )
        seen = set()
        for v in result.verdicts:
            if not 0 <= v.index < len(chunk) or v.index in seen:
                continue
            seen.add(v.index)
            original = chunk[v.index]
            if v.keep:
                kept.append(original)
            else:
                rejected.append((original, v.reason))
        # A submission the screener skipped is kept: silence must not delete
        # someone's contribution. The validate stage is the backstop.
        for local, original in enumerate(chunk):
            if local not in seen:
                kept.append(original)
    kept.sort(key=lambda i: subs[i].score, reverse=True)
    return kept, rejected


def _cluster_once(llm: LLM, texts: Sequence[str], cfg: Config) -> list[Cluster]:
    result: ClusterResult = llm.parse(
        model=cfg.work_model,
        system=CLUSTER_SYSTEM,
        user=_wrap(texts),
        schema=ClusterResult,
    )
    clusters = [
        Cluster(
            members=[m for m in c.members if 0 <= m < len(texts)],
            canonical=html.unescape(c.canonical).strip(),
            theme=html.unescape(c.theme).strip() or "general",
        )
        for c in result.clusters
    ]
    clusters = [c for c in clusters if c.members and c.canonical]

    # Anything the model forgot becomes its own cluster rather than vanishing.
    assigned = {m for c in clusters for m in c.members}
    for i, text in enumerate(texts):
        if i not in assigned:
            clusters.append(Cluster(members=[i], canonical=text, theme="general"))
    return clusters


def cluster(
    llm: LLM, subs: Sequence[Submission], indices: Sequence[int], cfg: Config
) -> list[Rule]:
    """Group submissions that ask for the same thing; sum their scores.

    This is the stage that makes the leaderboard mean something. Four people
    asking for the same rule in four wordings is a strong signal, and it should
    produce one weighty rule -- not four lines that each look like one opinion.

    Large inputs are clustered per chunk, then the canonical rules are clustered
    again so duplicates that landed in different chunks still merge.
    """
    chunks = _chunk(indices, cfg.chunk_size)
    prelim: list[Rule] = []
    for chunk_indices in chunks:
        for c in _cluster_once(llm, [subs[i].submission for i in chunk_indices], cfg):
            members = [chunk_indices[m] for m in c.members]
            prelim.append(
                Rule(
                    text=c.canonical,
                    theme=c.theme,
                    score=sum(subs[i].score for i in members),
                    sources=members,
                )
            )

    if len(chunks) > 1 and len(prelim) > 1:
        merged: list[Rule] = []
        for c in _cluster_once(llm, [r.text for r in prelim], cfg):
            group = [prelim[m] for m in c.members]
            merged.append(
                Rule(
                    text=c.canonical,
                    theme=c.theme,
                    score=sum(r.score for r in group),
                    sources=sorted({s for r in group for s in r.sources}),
                )
            )
        prelim = merged

    return sorted(prelim, key=lambda r: r.score, reverse=True)


def resolve(llm: LLM, rules: Sequence[Rule], cfg: Config) -> tuple[list[Rule], list[Conflict]]:
    """Reconcile rules that cannot all be satisfied at once.

    Which policy to apply is a product decision, not a technical one -- see
    Config.conflict_policy. Leaving a contradiction in the output is the one
    outcome that is always wrong.
    """
    if len(rules) < 2:
        return list(rules), []

    listing = "\n".join(
        f"{i}. (weight {r.score}) {r.text}" for i, r in enumerate(rules)
    )
    result: ConflictResult = llm.parse(
        model=cfg.work_model,
        system=CONFLICT_SYSTEM.format(policy_line=CONFLICT_POLICY[cfg.conflict_policy]),
        user=f"<rules>\n{listing}\n</rules>",
        schema=ConflictResult,
    )

    removed: set[int] = set()
    added: list[Rule] = []
    for c in result.conflicts:
        idxs = [i for i in c.rule_indices if 0 <= i < len(rules) and i not in removed]
        if len(idxs) < 2 or c.resolution == "keep_both":
            continue
        involved = [rules[i] for i in idxs]
        if c.resolution == "merge" and c.text.strip():
            added.append(
                Rule(
                    text=c.text.strip(),
                    theme=involved[0].theme,
                    score=sum(r.score for r in involved),
                    sources=sorted({s for r in involved for s in r.sources}),
                )
            )
            removed.update(idxs)
        elif c.resolution == "prefer_higher":
            winner = max(involved, key=lambda r: r.score)
            removed.update(i for i, r in zip(idxs, involved) if r is not winner)

    kept = [r for i, r in enumerate(rules) if i not in removed]
    return sorted(kept + added, key=lambda r: r.score, reverse=True), result.conflicts


def compose(llm: LLM, rules: Sequence[Rule], cfg: Config) -> str:
    if not rules:
        raise PipelineError("nothing survived screening; cannot compose a prompt")
    listing = "\n".join(
        f"- (weight {r.score}, theme: {r.theme}) {r.text}" for r in rules
    )
    return llm.text(
        model=cfg.work_model,
        system=COMPOSE_SYSTEM.format(target_words=cfg.target_words),
        user=f"<rules>\n{listing}\n</rules>",
    )


def validate(llm: LLM, prompt: str, rules: Sequence[Rule], cfg: Config) -> ValidationResult:
    listing = "\n".join(f"- {r.text}" for r in rules)
    return llm.parse(
        model=cfg.work_model,
        system=VALIDATE_SYSTEM,
        user=f"<rules>\n{listing}\n</rules>\n\n<generated_prompt>\n{prompt}\n</generated_prompt>",
        schema=ValidationResult,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def build(
    subs: Sequence[Submission],
    cfg: Config | None = None,
    llm: LLM | None = None,
    on_stage: Callable[[str, str], None] | None = None,
) -> Result:
    cfg = (cfg or Config()).validated()
    llm = llm or LLM(max_tokens=cfg.max_tokens)
    say = on_stage or (lambda stage, detail: None)

    selected = select(subs, cfg)
    say("select", f"{len(selected)} of {len(subs)} submissions above the floor")
    if not selected:
        raise PipelineError("no submissions cleared the score floor")

    kept, rejected = screen(llm, subs, selected, cfg)
    say("screen", f"{len(kept)} kept, {len(rejected)} rejected")
    if not kept:
        raise PipelineError("every selected submission was rejected during screening")

    rules = cluster(llm, subs, kept, cfg)
    say("cluster", f"{len(kept)} submissions -> {len(rules)} distinct rules")

    rules, conflicts = resolve(llm, rules, cfg)
    real = [c for c in conflicts if c.resolution != "keep_both"]
    say("resolve", f"{len(real)} conflict(s) resolved, {len(rules)} rules remain")

    prompt = compose(llm, rules, cfg)
    say("compose", f"{len(prompt.split())} words")

    report = validate(llm, prompt, rules, cfg)
    problems = (
        len(report.untraceable_claims)
        + len(report.contradictions)
        + len(report.leaked_instructions)
    )
    say("validate", "clean" if problems == 0 else f"{problems} issue(s) flagged")

    return Result(
        prompt=prompt,
        rules=rules,
        submissions=list(subs),
        rejected=rejected,
        conflicts=conflicts,
        validation=report,
        stats={
            "submissions_in": len(subs),
            "selected": len(selected),
            "screened_out": len(rejected),
            "rules_out": len(rules),
            "conflicts_resolved": len(real),
            "prompt_words": len(prompt.split()),
        },
    )


def build_from_file(path: str | Path, cfg: Config | None = None, **kwargs: Any) -> Result:
    return build(load(path), cfg, **kwargs)
