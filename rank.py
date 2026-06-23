#!/usr/bin/env python3
"""Generate a Redrob top-100 candidate ranking CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from src.features import extract_features
from src.load import iter_candidates, write_audit, write_submission
from src.reasoning import build_reasoning
from src.scoring import score_candidate


def rank_candidates(candidates_path: str | Path, top_k: int = 100, audit_k: int = 200) -> List[Dict]:
    scored: List[Dict] = []
    for candidate in iter_candidates(candidates_path):
        features = extract_features(candidate)
        row = score_candidate(features)
        scored.append(row)

    scored.sort(key=lambda row: (-row["score"], row["candidate_id"]))

    for rank, row in enumerate(scored[: max(top_k, audit_k)], 1):
        features = row["_features"]
        row["rank"] = rank
        row["reasoning"] = build_reasoning(features, rank)

    return scored


def build_submission(candidates_path: str | Path, output_path: str | Path, audit_path: str | Path | None = None) -> None:
    scored = rank_candidates(candidates_path)
    top100 = scored[:100]
    write_submission(top100, output_path)
    if audit_path:
        audit_rows = []
        for row in scored[:200]:
            audit_row = dict(row)
            audit_row.pop("_features", None)
            audit_rows.append(audit_row)
        write_audit(audit_rows, audit_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--audit-out",
        default="artifacts/audit_top_200.csv",
        help="Optional audit CSV path for top 200 candidates. Use empty string to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = args.audit_out or None
    if audit_path:
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    build_submission(args.candidates, args.out, audit_path)


if __name__ == "__main__":
    main()
