# Redrob Senior AI Engineer Ranker

This project builds a deterministic CPU-only ranker for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

The ranker selects the top 100 candidates from `candidates.jsonl` for the released Senior AI Engineer JD and writes a CSV with the exact required columns:

```text
candidate_id,rank,score,reasoning
```

## Approach

The model is a transparent hybrid heuristic ranker. It scores direct career evidence of production retrieval, search, ranking, recommendation, embeddings, vector infrastructure, Python, and evaluation systems. Career-history evidence is weighted above skill tags to avoid AI-keyword stuffing.

A separate behavioral modifier uses Redrob signals such as recent activity, open-to-work status, recruiter response rate, response time, notice period, verification, recruiter saves, GitHub activity, and interview completion. Risk penalties demote service-only careers, irrelevant current titles, pure non-target AI domains, keyword-heavy profiles without career support, stale candidates, and honeypot-like inconsistencies.

Reasoning is generated from candidate facts used in scoring. It does not call an LLM or external API.

## Setup

Python 3.10+ is sufficient. The ranking step uses only the Python standard library.

```bash
python --version
```

Optional for tests:

```bash
python -m pytest
```

## Reproduce Submission

Run this from the repository root:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

This also writes an audit file at:

```text
artifacts/audit_top_200.csv
```

To disable audit output:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv --audit-out ""
```

## Validate Output

```bash
python data/validate_submission.py ./submission.csv
```

Expected result:

```text
Submission is valid.
```

## Runtime Constraints

The ranking step is designed for:

- CPU only
- No network access
- No hosted LLM/API calls
- Under 16 GB RAM
- Under 5 minutes on the full 100K candidate file

## Important Files

- `rank.py`: CLI entry point.
- `src/features.py`: feature extraction, behavior modifier, and risk penalties.
- `src/scoring.py`: final score composition.
- `src/reasoning.py`: fact-based reasoning generation.
- `artifacts/audit_top_200.csv`: score breakdown for manual review.
- `submission_metadata.yaml`: metadata to mirror the portal submission.

## Suggested Final Checks

```bash
/usr/bin/time -v python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
python data/validate_submission.py ./submission.csv
wc -l submission.csv
head -5 submission.csv
```
