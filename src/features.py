"""Feature extraction for candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple

from .config import (
    DISFAVORED_TITLES,
    PREFERRED_CITIES,
    PRODUCT_COMPANIES,
    PRODUCT_INDUSTRIES,
    REFERENCE_DATE,
    SERVICE_COMPANIES,
    SKILL_PROFICIENCY,
    TERM_GROUPS,
    TITLE_WEIGHTS,
)
from .normalize import clean_text, contains_phrase, count_phrases, join_text


@dataclass
class CandidateFeatures:
    candidate_id: str
    current_title: str
    years_experience: float
    location: str
    country: str
    current_company: str
    current_industry: str
    title_score: float
    seniority_score: float
    retrieval_career: float
    retrieval_skill: float
    vector_db_score: float
    llm_nlp_score: float
    production_score: float
    evaluation_score: float
    python_code_score: float
    product_company_score: float
    location_score: float
    behavior_modifier: float
    risk_penalty: float
    risk_flags: List[str] = field(default_factory=list)
    positive_facts: List[str] = field(default_factory=list)
    concern: str = ""
    raw: Dict = field(default_factory=dict)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _cap(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _term_score(text: str, group: str, cap_at: int = 6) -> Tuple[float, int]:
    count = count_phrases(text, TERM_GROUPS[group])
    return _cap(count / cap_at), count


def _skill_group_score(skills: List[Dict], group: str) -> Tuple[float, int, List[str]]:
    total = 0.0
    hits = 0
    names: List[str] = []
    for skill in skills:
        name = clean_text(skill.get("name", ""))
        if not name:
            continue
        if any(contains_phrase(name, phrase) for phrase in TERM_GROUPS[group]):
            hits += 1
            names.append(skill.get("name", ""))
            proficiency = SKILL_PROFICIENCY.get(clean_text(skill.get("proficiency", "")), 0.45)
            endorsements = min(float(skill.get("endorsements") or 0), 50.0) / 50.0
            duration = min(float(skill.get("duration_months") or 0), 48.0) / 48.0
            total += 0.55 * proficiency + 0.25 * duration + 0.20 * endorsements
    return _cap(total / 4.0), hits, names[:4]


def _title_score(title: str) -> float:
    text = clean_text(title)
    for phrase, score in TITLE_WEIGHTS.items():
        if phrase == text or contains_phrase(text, phrase):
            return score
    if "engineer" in text and ("ml" in text or "ai" in text or "machine learning" in text):
        return 0.82
    if "scientist" in text and "data" in text:
        return 0.72
    if "engineer" in text:
        return 0.38
    return 0.12


def _seniority_score(years: float, title: str) -> float:
    if 5.0 <= years <= 9.0:
        base = 1.0
    elif 4.0 <= years < 5.0 or 9.0 < years <= 10.5:
        base = 0.82
    elif 3.0 <= years < 4.0 or 10.5 < years <= 12.0:
        base = 0.58
    elif years < 3.0:
        base = 0.22
    else:
        base = 0.42
    title_text = clean_text(title)
    if any(word in title_text for word in ("senior", "lead", "staff", "principal")) and years >= 4.0:
        base = min(1.0, base + 0.08)
    return base


def _product_company_score(candidate: Dict) -> Tuple[float, bool, bool]:
    profile = candidate.get("profile", {})
    histories = candidate.get("career_history", [])
    companies = [profile.get("current_company", "")] + [h.get("company", "") for h in histories]
    industries = [profile.get("current_industry", "")] + [h.get("industry", "") for h in histories]
    clean_companies = [clean_text(company) for company in companies if company]
    clean_industries = [clean_text(industry) for industry in industries if industry]
    product_hits = sum(company in PRODUCT_COMPANIES for company in clean_companies)
    product_hits += sum(industry in PRODUCT_INDUSTRIES for industry in clean_industries)
    service_roles = sum(company in SERVICE_COMPANIES for company in clean_companies)
    service_only = bool(clean_companies) and service_roles == len(clean_companies)
    has_product = product_hits > 0
    score = _cap(product_hits / 4.0)
    if service_only:
        score *= 0.25
    return score, service_only, has_product


def _location_score(profile: Dict, signals: Dict) -> float:
    location = clean_text(profile.get("location", ""))
    country = clean_text(profile.get("country", ""))
    preferred_mode = clean_text(signals.get("preferred_work_mode", ""))
    relocate = bool(signals.get("willing_to_relocate"))
    score = 0.35
    if country == "india":
        score += 0.25
    if any(city in location for city in PREFERRED_CITIES):
        score += 0.25
    if relocate:
        score += 0.12
    if preferred_mode in {"hybrid", "onsite", "flexible"}:
        score += 0.08
    if country != "india" and not relocate:
        score -= 0.18
    if preferred_mode == "remote" and country != "india":
        score -= 0.08
    return _cap(score)


def _behavior_modifier(signals: Dict) -> Tuple[float, List[str]]:
    modifier = 1.0
    notes: List[str] = []

    last_active = _parse_date(signals.get("last_active_date"))
    if last_active:
        days = (REFERENCE_DATE - last_active).days
        if days <= 30:
            modifier += 0.08
            notes.append(f"active {days} days before reference date")
        elif days <= 90:
            modifier += 0.02
        else:
            modifier -= 0.10
            notes.append("stale platform activity")
    else:
        modifier -= 0.05

    if signals.get("open_to_work_flag"):
        modifier += 0.07
        notes.append("open to work")
    else:
        modifier -= 0.04

    response = float(signals.get("recruiter_response_rate") or 0)
    if response >= 0.70:
        modifier += 0.08
        notes.append(f"{response:.0%} recruiter response rate")
    elif response >= 0.45:
        modifier += 0.03
    elif response < 0.15:
        modifier -= 0.10
        notes.append("low recruiter response rate")

    response_time = float(signals.get("avg_response_time_hours") or 0)
    if 0 < response_time <= 48:
        modifier += 0.04
    elif response_time >= 168:
        modifier -= 0.05

    notice = int(signals.get("notice_period_days") or 0)
    if notice <= 30:
        modifier += 0.06
        notes.append("short notice period")
    elif notice >= 120:
        modifier -= 0.08
        notes.append(f"{notice}-day notice period")
    elif notice >= 90:
        modifier -= 0.04

    if signals.get("verified_email"):
        modifier += 0.015
    if signals.get("verified_phone"):
        modifier += 0.015
    if signals.get("linkedin_connected"):
        modifier += 0.015

    saved = min(int(signals.get("saved_by_recruiters_30d") or 0), 12)
    modifier += saved * 0.006
    github = float(signals.get("github_activity_score") or -1)
    if github >= 60:
        modifier += 0.04
        notes.append("strong GitHub activity")
    elif github >= 25:
        modifier += 0.02

    completion = float(signals.get("interview_completion_rate") or 0)
    if completion >= 0.8:
        modifier += 0.03
    elif completion < 0.35:
        modifier -= 0.05

    offer_accept = float(signals.get("offer_acceptance_rate") if signals.get("offer_acceptance_rate") is not None else -1)
    if offer_accept >= 0.75:
        modifier += 0.02
    elif offer_accept == 0:
        modifier -= 0.03

    return _cap(modifier, 0.70, 1.28), notes


def extract_features(candidate: Dict) -> CandidateFeatures:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    histories = candidate.get("career_history", [])

    title = profile.get("current_title", "")
    title_text = clean_text(title)
    current_company = profile.get("current_company", "")
    current_industry = profile.get("current_industry", "")
    years = float(profile.get("years_of_experience") or 0)

    career_parts = []
    recent_parts = []
    for index, history in enumerate(histories):
        text = join_text(
            [
                history.get("title"),
                history.get("company"),
                history.get("industry"),
                history.get("description"),
            ]
        )
        career_parts.append(text)
        if index <= 1 or history.get("is_current"):
            recent_parts.append(text)
    profile_text = join_text(
        [
            profile.get("headline"),
            profile.get("summary"),
            title,
            current_company,
            current_industry,
        ]
    )
    career_text = join_text(career_parts)
    recent_text = join_text(recent_parts)
    combined_text = join_text([profile_text, career_text])

    retrieval_career, retrieval_count = _term_score(join_text([career_text, profile_text]), "retrieval", cap_at=7)
    vector_db_score, vector_count = _term_score(combined_text, "vector_db", cap_at=4)
    llm_nlp_text_score, llm_count = _term_score(combined_text, "llm_nlp", cap_at=6)
    production_score, production_count = _term_score(join_text([career_text, recent_text]), "production", cap_at=8)
    evaluation_score, evaluation_count = _term_score(combined_text, "evaluation", cap_at=3)
    python_code_score, python_count = _term_score(combined_text, "python_code", cap_at=6)
    non_target_ai_score, non_target_ai_count = _term_score(combined_text, "non_target_ai", cap_at=4)

    retrieval_skill, retrieval_skill_hits, retrieval_skill_names = _skill_group_score(skills, "retrieval")
    vector_skill, vector_skill_hits, vector_skill_names = _skill_group_score(skills, "vector_db")
    llm_skill, llm_skill_hits, llm_skill_names = _skill_group_score(skills, "llm_nlp")
    eval_skill, _, _ = _skill_group_score(skills, "evaluation")
    python_skill, _, python_skill_names = _skill_group_score(skills, "python_code")
    non_target_skill, non_target_skill_hits, _ = _skill_group_score(skills, "non_target_ai")

    retrieval_skill = _cap(0.65 * retrieval_skill + 0.35 * vector_skill)
    vector_db_score = _cap(0.70 * vector_db_score + 0.30 * vector_skill)
    llm_nlp_score = _cap(0.70 * llm_nlp_text_score + 0.30 * llm_skill)
    evaluation_score = _cap(0.85 * evaluation_score + 0.15 * eval_skill)
    python_code_score = _cap(0.75 * python_code_score + 0.25 * python_skill)

    title_score = _title_score(title)
    seniority_score = _seniority_score(years, title)
    product_score, service_only, has_product = _product_company_score(candidate)
    location_score = _location_score(profile, signals)
    behavior_modifier, behavior_notes = _behavior_modifier(signals)

    risk_penalty = 0.0
    risk_flags: List[str] = []

    if title_text in DISFAVORED_TITLES and retrieval_career < 0.28:
        risk_penalty += 13.0
        risk_flags.append("irrelevant current title without retrieval evidence")
    elif title_text in DISFAVORED_TITLES:
        risk_penalty += 5.0
        risk_flags.append("non-target current title")

    ai_skill_hits = retrieval_skill_hits + vector_skill_hits + llm_skill_hits
    if ai_skill_hits >= 7 and retrieval_career < 0.25 and production_score < 0.35:
        risk_penalty += 18.0
        risk_flags.append("AI keyword-heavy skills without career support")
    elif ai_skill_hits >= 5 and retrieval_career < 0.20:
        risk_penalty += 10.0
        risk_flags.append("weak career support for AI skill list")

    if service_only:
        risk_penalty += 9.0
        risk_flags.append("service-company-only career")

    if non_target_ai_count + non_target_skill_hits >= 4 and retrieval_career < 0.30:
        risk_penalty += 6.0
        risk_flags.append("AI depth appears outside NLP/search")

    if years < 3.0:
        risk_penalty += 8.0
        risk_flags.append("below seniority band")
    elif years > 12.0:
        if title_score >= 0.70:
            risk_penalty += 5.0
            risk_flags.append("well above target seniority band")
        else:
            risk_penalty += 9.0
            risk_flags.append("over target band without clear hands-on AI title")
        if years > 15.0:
            risk_penalty += 4.0
            risk_flags.append("far above requested experience range")

    expert_zero = [
        skill.get("name", "")
        for skill in skills
        if clean_text(skill.get("proficiency")) == "expert" and int(skill.get("duration_months") or 0) == 0
    ]
    if expert_zero:
        risk_penalty += min(12.0, 4.0 + len(expert_zero) * 2.0)
        risk_flags.append("expert skill with zero duration")

    career_months = sum(int(history.get("duration_months") or 0) for history in histories)
    if career_months and abs((career_months / 12.0) - years) > 4.5:
        risk_penalty += 10.0
        risk_flags.append("career duration inconsistent with years of experience")

    last_active = _parse_date(signals.get("last_active_date"))
    response = float(signals.get("recruiter_response_rate") or 0)
    if last_active and (REFERENCE_DATE - last_active).days > 120 and response < 0.15:
        risk_penalty += 8.0
        risk_flags.append("stale and unlikely to respond")

    if int(signals.get("notice_period_days") or 0) >= 150 and not signals.get("open_to_work_flag"):
        risk_penalty += 4.0
        risk_flags.append("long notice and not open to work")

    country = clean_text(profile.get("country", ""))
    if country != "india":
        if signals.get("willing_to_relocate"):
            risk_penalty += 6.0
            risk_flags.append("outside India but willing to relocate")
        else:
            risk_penalty += 12.0
            risk_flags.append("outside India with weaker logistics fit")

    positive_facts: List[str] = []
    if title_score >= 0.85:
        positive_facts.append(f"{title} title directly matches senior AI/ML engineering")
    elif title_score >= 0.55:
        positive_facts.append(f"{title} background is adjacent to the role")
    if retrieval_count >= 3:
        positive_facts.append("career text shows retrieval/search/ranking evidence")
    elif retrieval_skill_names:
        positive_facts.append("skills include " + ", ".join(retrieval_skill_names[:3]))
    if vector_count + vector_skill_hits >= 2:
        names = vector_skill_names[:2] or ["vector/search infrastructure"]
        positive_facts.append("vector infrastructure signal: " + ", ".join(names))
    if evaluation_count >= 1:
        positive_facts.append("mentions ranking or retrieval evaluation")
    if production_count >= 4:
        positive_facts.append("career history emphasizes shipped production systems")
    if python_count >= 2 or python_skill_names:
        positive_facts.append("Python/ML engineering stack is present")
    if has_product:
        positive_facts.append("has product-company or product-industry context")
    if behavior_notes:
        positive_facts.extend(behavior_notes[:2])

    concern = ""
    if risk_flags:
        concern = risk_flags[0]
    elif int(signals.get("notice_period_days") or 0) >= 90:
        concern = f"{int(signals.get('notice_period_days') or 0)}-day notice period"
    elif not signals.get("open_to_work_flag"):
        concern = "not explicitly open to work"
    elif location_score < 0.45:
        concern = "location or relocation fit is weaker"
    elif evaluation_score < 0.2:
        concern = "limited explicit ranking-evaluation evidence"

    return CandidateFeatures(
        candidate_id=candidate.get("candidate_id", ""),
        current_title=title,
        years_experience=years,
        location=profile.get("location", ""),
        country=profile.get("country", ""),
        current_company=current_company,
        current_industry=current_industry,
        title_score=title_score,
        seniority_score=seniority_score,
        retrieval_career=retrieval_career,
        retrieval_skill=retrieval_skill,
        vector_db_score=vector_db_score,
        llm_nlp_score=llm_nlp_score,
        production_score=production_score,
        evaluation_score=evaluation_score,
        python_code_score=python_code_score,
        product_company_score=product_score,
        location_score=location_score,
        behavior_modifier=behavior_modifier,
        risk_penalty=risk_penalty,
        risk_flags=risk_flags,
        positive_facts=positive_facts,
        concern=concern,
        raw=candidate,
    )
