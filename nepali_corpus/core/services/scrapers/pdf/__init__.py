from .detect import PdfType, detect_pdf_type, HAS_PYMUPDF
from .clean import clean_pdf_text, normalize_unicode, has_preeti_patterns


def __getattr__(name):
    if name in ("PdfJob", "extract_pdfs"):
        from .extractor import PdfJob, extract_pdfs
        return {"PdfJob": PdfJob, "extract_pdfs": extract_pdfs}[name]
    if name == "_extract_text_from_pdf":
        from .utils import _extract_text_from_pdf
        return _extract_text_from_pdf
    if name == "_extract_pdf_metadata":
        from .utils import _extract_pdf_metadata
        return _extract_pdf_metadata
    if name in ("score_page", "score_document", "PageScore", "DocumentScore"):
        from . import score
        return getattr(score, name)
    if name in ("ocr_pages", "ocr_full_document"):
        from . import ocr
        return getattr(ocr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PdfJob", "extract_pdfs", "HAS_PYMUPDF",
    "_extract_text_from_pdf", "_extract_pdf_metadata",
    "PdfType", "detect_pdf_type",
    "score_page", "score_document", "PageScore", "DocumentScore",
    "ocr_pages", "ocr_full_document",
    "clean_pdf_text", "normalize_unicode", "has_preeti_patterns",
]
