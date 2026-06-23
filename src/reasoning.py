"""Fact-based reasoning generation."""

from __future__ import annotations

from .features import CandidateFeatures


def _trim(sentence: str, limit: int = 500) -> str:
    sentence = " ".join(sentence.split())
    if len(sentence) <= limit:
        return sentence
    return sentence[: limit - 1].rsplit(" ", 1)[0] + "."


def build_reasoning(features: CandidateFeatures, rank: int) -> str:
    facts = features.positive_facts[:]
    profile = features.raw.get("profile", {})
    signals = features.raw.get("redrob_signals", {})

    opener = (
        f"{features.current_title} with {features.years_experience:.1f} years"
        if features.current_title
        else f"{features.years_experience:.1f} years of experience"
    )

    selected = []
    for fact in facts:
        if fact not in selected:
            selected.append(fact)
        if len(selected) == 2:
            break

    if not selected:
        selected.append("some adjacent engineering and platform signals")

    location = ", ".join(part for part in [profile.get("location"), profile.get("country")] if part)
    if location and rank <= 60:
        selected.append(f"located in {location}")

    first = f"{opener}; " + "; ".join(selected[:3]) + "."

    concern = features.concern
    if concern:
        second = f"Concern: {concern}."
    else:
        response = float(signals.get("recruiter_response_rate") or 0)
        notice = int(signals.get("notice_period_days") or 0)
        second = f"Hiring signals are workable with {response:.0%} recruiter response rate and {notice}-day notice."

    return _trim(f"{first} {second}", limit=500)
