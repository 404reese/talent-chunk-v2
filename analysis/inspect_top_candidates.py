#!/usr/bin/env python3
"""Print a compact view of the audit file for manual review."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/audit_top_200.csv")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            print(
                f"{row['rank']:>3} {row['candidate_id']} {float(row['score']):6.2f} "
                f"{row['current_title'][:32]:32} {row['years_experience']:>4} "
                f"{row['location'][:24]:24} flags={row['risk_flags']}"
            )


if __name__ == "__main__":
    main()
