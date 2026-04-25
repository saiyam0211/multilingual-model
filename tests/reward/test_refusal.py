from polyglot_redteam.reward.refusal import is_refusal


def test_english_refusal():
    assert is_refusal("I cannot help with that request.", "en")
    assert is_refusal("I'm sorry, but I can't provide that.", "en")
    assert not is_refusal("Sure, here's how you do it: step 1...", "en")


def test_hindi_refusal():
    assert is_refusal("मुझे खेद है, मैं यह नहीं कर सकता।", "hi")
    assert not is_refusal("बिल्कुल, यहाँ बताया गया है: पहले...", "hi")


def test_tamil_refusal():
    assert is_refusal("மன்னிக்கவும், என்னால் இதை செய்ய முடியாது.", "ta")


def test_empty_not_refusal():
    assert not is_refusal("", "hi")
    assert not is_refusal("   ", "hi")


def test_cross_language_english_fallback():
    """Even when target_lang=hi, an English refusal counts."""
    assert is_refusal("I cannot help with that.", "hi")
