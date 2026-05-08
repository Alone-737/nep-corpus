import pytest

from nepali_corpus.core.services.scrapers.pdf.detect import detect_pdf_type, PdfType, HAS_PYMUPDF
from nepali_corpus.core.services.scrapers.pdf.score import score_page, score_document
from nepali_corpus.core.services.scrapers.pdf.clean import (
    clean_pdf_text,
    detect_repeated_headers_footers,
    has_preeti_patterns,
    normalize_unicode,
    remove_page_numbers,
)



@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
def test_detect_unicode_pdf(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "नेपाल सरकारको सूचना")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = detect_pdf_type(pdf_bytes)
    assert result.pdf_type == PdfType.UNICODE
    assert result.page_count == 1
    assert len(result.scanned_pages) == 0


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
def test_detect_empty_pdf():
    import fitz

    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()

    result = detect_pdf_type(pdf_bytes)
    assert result.page_count == 1



def test_score_page_good_nepali():
    text = "नेपाल सरकारको विभागले यो सूचना जारी गरेको छ। " * 10
    ps = score_page(text, page_num=0)
    assert ps.quality_score > 0.3
    assert ps.devanagari_ratio > 0.5
    assert not ps.needs_ocr


def test_score_page_empty():
    ps = score_page("", page_num=0)
    assert ps.quality_score == 0.0
    assert ps.needs_ocr


def test_score_page_english_only():
    ps = score_page("This is English text only with no Devanagari content", page_num=0)
    assert ps.devanagari_ratio == 0.0
    assert ps.needs_ocr


def test_score_document_multi_page():
    pages = [
        "नेपाल सरकारको मन्त्रालयबाट जारी गरिएको सूचना",
        "",
        "यो अर्को पृष्ठ हो जसमा विवरण छ।",
    ]
    ds = score_document(pages, extraction_method="pymupdf")
    assert ds.extraction_method == "pymupdf"
    assert 1 in ds.pages_needing_ocr
    assert ds.total_chars > 0
    assert ds.overall_score > 0



def test_normalize_unicode_nfc():
    decomposed = "क\u093E"
    result = normalize_unicode(decomposed)
    assert result


def test_normalize_unicode_strips_bom():
    text = "\ufeffनेपाल"
    assert normalize_unicode(text) == "नेपाल"


def test_normalize_unicode_strips_zero_width():
    text = "ने\u200bपा\u200bल"
    assert normalize_unicode(text) == "नेपाल"


def test_remove_page_numbers():
    text = "प्रथम अनुच्छेद।\n42\nदोस्रो अनुच्छेद।"
    result = remove_page_numbers(text)
    assert "42" not in result
    assert "प्रथम" in result


def test_remove_nepali_page_numbers():
    text = "प्रथम।\n१२\nदोस्रो।"
    result = remove_page_numbers(text)
    assert "१२" not in result


def test_detect_repeated_headers_footers():
    pages = [
        "नेपाल सरकार\nमन्त्रालय\nयो पहिलो पृष्ठ हो।\nwww.example.gov.np",
        "नेपाल सरकार\nमन्त्रालय\nयो दोस्रो पृष्ठ हो।\nwww.example.gov.np",
        "नेपाल सरकार\nमन्त्रालय\nयो तेस्रो पृष्ठ हो।\nwww.example.gov.np",
        "नेपाल सरकार\nमन्त्रालय\nयो चौथो पृष्ठ हो।\nwww.example.gov.np",
    ]
    headers, footers = detect_repeated_headers_footers(pages, min_pages=3)
    assert "नेपाल सरकार" in headers
    assert "www.example.gov.np" in footers


def test_clean_pdf_text_full_pipeline():
    pages = [
        "नेपाल सरकार\nयो पहिलो पृष्ठ हो।\n1",
        "नेपाल सरकार\nयो दोस्रो पृष्ठ हो।\n2",
        "नेपाल सरकार\nयो तेस्रो पृष्ठ हो।\n3",
    ]
    result = clean_pdf_text(pages, remove_repeated=True)
    assert "पहिलो" in result
    assert "दोस्रो" in result
    assert "तेस्रो" in result


def test_has_preeti_patterns():
    assert has_preeti_patterns("g]kfn ;/sf/")
    assert not has_preeti_patterns("नेपाल सरकार")


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
def test_extract_text_from_pdf():
    import glob
    import fitz
    from nepali_corpus.core.services.scrapers.pdf.utils import _extract_text_from_pdf

    font_paths = glob.glob("/usr/share/fonts/**/NotoSans*Devanagari*Regular*.ttf", recursive=True)
    if not font_paths:
        font_paths = glob.glob("/usr/share/fonts/**/Noto*Devanagari*.ttf", recursive=True)
    if not font_paths:
        pytest.skip("No Devanagari system font found")

    doc = fitz.open()
    page = doc.new_page()
    font = fitz.Font(fontfile=font_paths[0])
    tw = fitz.TextWriter(page.rect)
    tw.append((72, 72), "नेपाल सरकारको सूचना विभागले जारी गरेको छ", font=font, fontsize=11)
    tw.write_text(page)
    pdf_bytes = doc.tobytes()
    doc.close()

    text = _extract_text_from_pdf(pdf_bytes)
    assert "नेपाल" in text
    assert len(text) > 10

