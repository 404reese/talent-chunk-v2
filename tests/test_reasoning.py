from src.features import extract_features
from src.reasoning import build_reasoning
from tests.test_scoring import _candidate


def test_reasoning_mentions_record_facts():
    features = extract_features(_candidate())
    reasoning = build_reasoning(features, 1)
    assert "Senior Machine Learning Engineer" in reasoning
    assert "7.0 years" in reasoning
    assert len(reasoning) <= 300
