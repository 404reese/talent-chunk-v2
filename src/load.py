"""Input and output helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


def iter_candidates(path: str | Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc


def write_submission(rows: Iterable[Dict], path: str | Path) -> None:
    fieldnames = ["candidate_id", "rank", "score", "reasoning"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                    "score": f"{row['score']:.6f}",
                    "reasoning": row["reasoning"],
                }
            )


def write_audit(rows: List[Dict], path: str | Path) -> None:
    if not rows:
        return
    keys = [
        "candidate_id",
        "rank",
        "score",
        "current_title",
        "years_experience",
        "location",
        "country",
        "role_depth",
        "production",
        "retrieval",
        "evaluation",
        "seniority",
        "product",
        "location_score",
        "behavior_modifier",
        "risk_penalty",
        "risk_flags",
        "positive_facts",
        "concern",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
