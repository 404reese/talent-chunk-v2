from src.features import extract_features
from src.scoring import score_candidate


def _candidate(**overrides):
    base = {
        "candidate_id": "CAND_9999999",
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Senior ML Engineer",
            "summary": "Built semantic search, vector search, and ranking evaluation in production.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior Machine Learning Engineer",
            "current_company": "Pied Piper",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "Pied Piper",
                "title": "Senior Machine Learning Engineer",
                "start_date": "2023-01-01",
                "end_date": None,
                "duration_months": 40,
                "is_current": True,
                "industry": "Software",
                "company_size": "201-500",
                "description": "Shipped production semantic search, FAISS retrieval, NDCG evaluation, and A/B testing for real users using Python.",
            }
        ],
        "education": [],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 60},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
            {"name": "Semantic Search", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
        ],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-05-20",
            "open_to_work_flag": True,
            "profile_views_received_30d": 20,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 20,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 50,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20, "max": 35},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 70,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    base.update(overrides)
    return base


def test_strong_candidate_scores_high():
    features = extract_features(_candidate())
    row = score_candidate(features)
    assert row["score"] > 70
    assert row["risk_penalty"] == 0


def test_keyword_stuffer_is_penalized():
    candidate = _candidate()
    candidate["profile"]["current_title"] = "Marketing Manager"
    candidate["profile"]["summary"] = "Interested in AI tools."
    candidate["career_history"][0]["title"] = "Marketing Manager"
    candidate["career_history"][0]["description"] = "Managed campaigns and content calendars."
    features = extract_features(candidate)
    row = score_candidate(features)
    assert row["risk_penalty"] >= 10
