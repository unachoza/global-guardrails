from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Tuning knobs for the summarizer pipeline.

    Selection is deliberately scale-invariant: it works whether `score` is a raw
    vote count, a 0-100 rating, or a Wilson bound, because the floor is derived
    from the top score rather than hardcoded.
    """

    # --- selection ---
    min_score_ratio: float = 0.05
    """Keep submissions scoring at least this fraction of the highest score."""

    min_score: int | None = None
    """Optional absolute floor, applied alongside min_score_ratio (max of the two wins)."""

    top_k: int | None = 40
    """Hard cap on how many submissions reach the LLM stages. None = no cap."""

    min_rule_share: float = 0.10
    """Drop clustered rules weighing less than this fraction of the top rule.

    Applied AFTER clustering, which is the only place it can work. Before
    clustering every submission stands alone, so a lone novelty ("speak like a
    pirate") looks comparable to a real rule; after clustering, genuine consensus
    has aggregated into the thousands and fringe entries have not.

    This is a consensus threshold, not a taste filter: a joke rule that genuinely
    wins the vote still ships. That is the community's call to make.
    """

    # --- composition ---
    target_words: int = 400
    """Approximate word budget for the generated system prompt."""

    conflict_policy: str = "scope"
    """"scope" = try to reconcile conflicting rules by scoping them to different
    contexts, falling back to score. "score" = the higher-scoring rule always wins."""

    # --- models ---
    screen_model: str = "claude-haiku-4-5"
    """High-volume, low-judgement classification. Cheap on purpose."""

    work_model: str = "claude-opus-5"
    """Clustering, conflict resolution, composition, validation."""

    # --- batching ---
    chunk_size: int = 60
    """Submissions per LLM call. Clustering re-merges across chunks in a second pass."""

    max_tokens: int = 16000

    def validated(self) -> "Config":
        if not 0.0 <= self.min_score_ratio <= 1.0:
            raise ValueError("min_score_ratio must be between 0 and 1")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 or None")
        if not 0.0 <= self.min_rule_share < 1.0:
            raise ValueError("min_rule_share must be in [0, 1)")
        if self.conflict_policy not in ("scope", "score"):
            raise ValueError("conflict_policy must be 'scope' or 'score'")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        return self
