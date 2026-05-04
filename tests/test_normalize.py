from nepali_corpus.core.utils.normalize import normalize_text, make_dedup_key, detect_nepali


def test_normalize_text_collapses_whitespace():
    # Horizontal spaces collapse; single newline (word-wrap) becomes a space.
    text = "  यो   एउटा\nपरीक्षण   हो  "
    assert normalize_text(text) == "यो एउटा परीक्षण हो"


def test_normalize_text_preserves_paragraph_breaks():
    # Double newlines (paragraph separators built by the PDF extractor) survive.
    text = "पहिलो अनुच्छेद।\n\nदोस्रो अनुच्छेद।"
    result = normalize_text(text)
    assert "\n\n" in result
    assert result.startswith("पहिलो")
    assert result.endswith("अनुच्छेद।")


def test_normalize_text_caps_excess_blank_lines():
    # 3+ blank lines are reduced to exactly one blank line (two newlines).
    text = "पहिलो।\n\n\n\nदोस्रो।"
    result = normalize_text(text)
    assert result == "पहिलो।\n\nदोस्रो।"


def test_dedup_key_stable_for_equivalent_text():
    a = "नेपाल सरकार"
    b = "नेपाल   सरकार!!"
    assert make_dedup_key(a) == make_dedup_key(b)


def test_detect_nepali_ratio():
    nepali = "नेपाल सरकारको सूचना"
    english = "Government notice"
    assert detect_nepali(nepali, min_ratio=0.4)
    assert not detect_nepali(english, min_ratio=0.4)
