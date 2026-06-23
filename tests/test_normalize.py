from src.normalize import clean_text, contains_phrase


def test_clean_text_stabilizes_whitespace_and_case():
    assert clean_text("  Senior   AI Engineer!! ") == "senior ai engineer"


def test_contains_phrase_uses_word_boundaries():
    assert contains_phrase("built semantic search in production", "semantic search")
    assert not contains_phrase("researched searchingly", "search")
