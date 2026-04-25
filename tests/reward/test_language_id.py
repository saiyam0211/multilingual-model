"""Lang-ID fixtures. Uses script fallback in CI (LID_OFFLINE_FALLBACK=1)."""
from polyglot_redteam.reward.language_id import detect_language, is_valid_language


def test_hindi_devanagari():
    code, conf = detect_language("मुझे मदद चाहिए")
    assert code == "hi"
    assert conf > 0.5


def test_tamil_script():
    code, _ = detect_language("வணக்கம் உலகம்")
    assert code == "ta"


def test_english_latin():
    code, _ = detect_language("hello world this is english")
    assert code == "en"


def test_empty_returns_und():
    assert detect_language("") == ("und", 0.0)
    assert detect_language("   ") == ("und", 0.0)


def test_is_valid_language_strict():
    assert is_valid_language("नमस्ते दुनिया", "hi", threshold=0.5)
    assert not is_valid_language("hello world", "hi", threshold=0.5)
