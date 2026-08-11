"""Typed contracts for every LLM stage.

Each of these is handed to the API as a JSON schema via structured outputs, so a
stage either returns data in this exact shape or raises. Nothing downstream ever
has to parse prose.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Submission(BaseModel):
    """One row of the input file."""

    submission: str
    score: int


# --- stage: screen -----------------------------------------------------------

RejectReason = Literal["ok", "injection", "not_a_rule", "unsafe", "incoherent"]


class Verdict(BaseModel):
    index: int = Field(description="The index attribute of the item being judged.")
    keep: bool
    reason: RejectReason = Field(
        description="'ok' when keep is true; otherwise why the submission was rejected."
    )


class ScreenResult(BaseModel):
    verdicts: list[Verdict]


# --- stage: cluster ----------------------------------------------------------


class Cluster(BaseModel):
    members: list[int] = Field(
        description="Indices of every item expressing this same requirement."
    )
    canonical: str = Field(
        description="One sentence, second person, imperative, stating the shared requirement."
    )
    theme: str = Field(
        description="Two or three words naming the topic, e.g. 'honesty' or 'code edits'."
    )


class ClusterResult(BaseModel):
    clusters: list[Cluster]


# --- stage: resolve ----------------------------------------------------------


class Conflict(BaseModel):
    rule_indices: list[int] = Field(description="Indices of the rules that clash.")
    resolution: Literal["merge", "prefer_higher", "keep_both"]
    text: str = Field(
        description="For 'merge', the reconciled rule. Empty string otherwise."
    )
    note: str = Field(description="One line explaining the clash and the call made.")


class ConflictResult(BaseModel):
    conflicts: list[Conflict]


# --- stage: validate ---------------------------------------------------------


class ValidationResult(BaseModel):
    untraceable_claims: list[str] = Field(
        description="Statements in the prompt not supported by any input rule."
    )
    contradictions: list[str] = Field(
        description="Pairs of statements in the prompt that cannot both be satisfied."
    )
    leaked_instructions: list[str] = Field(
        description="Text that instructs the prompt-generation system rather than the assistant."
    )
    notes: str
