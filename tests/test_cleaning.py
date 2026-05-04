from nepali_corpus.core.models import RawRecord
from nepali_corpus.core.utils.normalize import normalize_record
from nepali_corpus.core.utils.cleaning import clean_text, is_nepali, min_length


def test_normalize_and_filter_helpers():
    rec = RawRecord(
        source_id="x",
        source_name="X",
        url="http://x",
        title="नेपाल सरकारको सूचना",
        language="ne",
    )
    doc = normalize_record(rec)
    assert doc is not None
    assert is_nepali(doc, min_ratio=0.4)
    assert not min_length(doc, min_chars=200)


# ---------------------------------------------------------------------------
# clean_text – PDF artifact removal
# ---------------------------------------------------------------------------

def test_clean_text_removes_page_numbers():
    text = "प्रथम अनुच्छेद।\n42\nदोस्रो अनुच्छेद।"
    result = clean_text(text)
    assert "42" not in result
    assert "प्रथम" in result
    assert "दोस्रो" in result


def test_clean_text_removes_soft_hyphen():
    # U+00AD (soft hyphen) must be stripped entirely
    text = "नेपाल\u00adको"
    assert "\u00ad" not in clean_text(text)


def test_clean_text_joins_hyphen_wrapped_devanagari():
    # Hard hyphen at end-of-line before Devanagari should join the word
    text = "विकास-\nकार्य"
    result = clean_text(text)
    assert "-" not in result
    assert "विकासकार्य" in result


def test_clean_text_removes_repeated_punct():
    text = "नेपाल ............ सरकार"
    result = clean_text(text)
    assert "...." not in result


def test_clean_text_fixes_ocr_period_as_danda():
    text = "नेपाल. सरकार"
    result = clean_text(text)
    assert "।" in result


def test_clean_text_normalises_curly_quotes():
    text = "\u2018नेपाल\u2019 सरकार"
    result = clean_text(text)
    assert "\u2018" not in result
    assert "\u2019" not in result
    assert "'" in result


def test_clean_text_removes_gov_ui_boilerplate():
    text = "नेपाल सरकार A A- A+ Share विदेश मन्त्रालय"
    result = clean_text(text)
    assert "A A-" not in result
    assert "नेपाल सरकार" in result
    assert "विदेश मन्त्रालय" in result
