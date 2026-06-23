"""Candidate scoring."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from .config import COMPONENT_WEIGHTS
from .features import CandidateFeatures


def score_candidate(features: CandidateFeatures) -> Dict:
    role_depth = min(
        1.0,
        0.36 * features.title_score
        + 0.25 * features.retrieval_career
        + 0.16 * features.retrieval_skill
        + 0.10 * features.vector_db_score
        + 0.07 * features.llm_nlp_score
        + 0.06 * features.python_code_score,
    )
    production = min(
        1.0,
        0.58 * features.production_score
        + 0.20 * features.python_code_score
        + 0.12 * features.product_company_score
        + 0.10 * features.vector_db_score,
    )
    retrieval = min(
        1.0,
        0.48 * features.retrieval_career
        + 0.27 * features.retrieval_skill
        + 0.20 * features.vector_db_score
        + 0.05 * features.llm_nlp_score,
    )
    evaluation = features.evaluation_score
    seniority = features.seniority_score
    product = features.product_company_score
    location = features.location_score

    base = (
        COMPONENT_WEIGHTS["role_depth"] * role_depth
        + COMPONENT_WEIGHTS["production"] * production
        + COMPONENT_WEIGHTS["retrieval"] * retrieval
        + COMPONENT_WEIGHTS["evaluation"] * evaluation
        + COMPONENT_WEIGHTS["seniority"] * seniority
        + COMPONENT_WEIGHTS["product"] * product
        + COMPONENT_WEIGHTS["location"] * location
    )
    raw_score = 100.0 * base * features.behavior_modifier - features.risk_penalty
    score = max(0.0, raw_score)
    row = {
        "candidate_id": features.candidate_id,
        "score": score,
        "role_depth": role_depth,
        "production": production,
        "retrieval": retrieval,
        "evaluation": evaluation,
        "seniority": seniority,
        "product": product,
        "location_score": location,
        "behavior_modifier": features.behavior_modifier,
        "risk_penalty": features.risk_penalty,
        "risk_flags": "; ".join(features.risk_flags),
        "positive_facts": "; ".join(features.positive_facts[:5]),
        "concern": features.concern,
    }
    row.update(
        {
            "current_title": features.current_title,
            "years_experience": features.years_experience,
            "location": features.location,
            "country": features.country,
            "current_company": features.current_company,
            "current_industry": features.current_industry,
            "_features": features,
        }
    )
    return row
