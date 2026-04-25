from src.preprocess import clean_text


def test_clean_text_basic_normalization():
    text = "WIN $1000 now!!! Visit http://example.com"
    cleaned = clean_text(text)
    assert "win" in cleaned
    assert "url" in cleaned
    assert "num" in cleaned
    assert "$" not in cleaned
