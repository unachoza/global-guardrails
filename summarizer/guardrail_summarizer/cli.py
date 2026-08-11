"""Command line entry point.

    python -m guardrail_summarizer.cli data_sim/submissions.json -o out/system_prompt.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .pipeline import PipelineError, build_from_file


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="guardrail-summarizer",
        description="Turn ranked GlobalGuardrail submissions into a system prompt.",
    )
    ap.add_argument("input", nargs="?", default="data_sim/submissions.json",
                    help="JSON array of {submission, score} objects")
    ap.add_argument("-o", "--out", default="out/system_prompt.txt",
                    help="where to write the generated prompt")
    ap.add_argument("--manifest", default=None,
                    help="optional path for the traceability manifest (JSON)")
    ap.add_argument("--top-k", type=int, default=Config.top_k,
                    help="cap on submissions reaching the LLM stages")
    ap.add_argument("--min-score-ratio", type=float, default=Config.min_score_ratio,
                    help="pre-cluster floor as a fraction of the top score (0-1)")
    ap.add_argument("--min-rule-share", type=float, default=Config.min_rule_share,
                    help="post-cluster floor: drop rules below this share of the top rule")
    ap.add_argument("--words", type=int, default=Config.target_words,
                    help="approximate word budget for the prompt")
    ap.add_argument("--policy", choices=("scope", "score"), default=Config.conflict_policy,
                    help="how to resolve rules that conflict")
    ap.add_argument("--screen-model", default=Config.screen_model)
    ap.add_argument("--work-model", default=Config.work_model)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config(
        top_k=args.top_k,
        min_score_ratio=args.min_score_ratio,
        min_rule_share=args.min_rule_share,
        target_words=args.words,
        conflict_policy=args.policy,
        screen_model=args.screen_model,
        work_model=args.work_model,
    )

    def say(stage: str, detail: str) -> None:
        if not args.quiet:
            print(f"  {stage:<9} {detail}", file=sys.stderr)

    try:
        result = build_from_file(args.input, cfg, on_stage=say)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.prompt + "\n")

    if args.manifest:
        man = Path(args.manifest)
        man.parent.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps(result.manifest(), indent=2, ensure_ascii=False) + "\n")

    if not args.quiet:
        print(f"\nwrote {out}", file=sys.stderr)
        report = result.validation
        if report:
            issues = (
                report.untraceable_claims
                + report.contradictions
                + report.leaked_instructions
            )
            for issue in issues:
                print(f"  ! {issue}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
