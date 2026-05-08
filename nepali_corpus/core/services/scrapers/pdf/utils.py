from __future__ import annotations

import io
import logging
from typing import Dict, List, Tuple

from .detect import HAS_PYMUPDF

logger = logging.getLogger(__name__)

try:
    import fitz
except Exception:
    pass

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

_PDFPLUMBER_X_TOLERANCE = 5
_PDFPLUMBER_Y_TOLERANCE = 3


def _pymupdf_pages(pdf_bytes: bytes) -> List[str]:
    if not HAS_PYMUPDF:
        return []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text("text") or "" for page in doc]
        doc.close()
        return pages
    except Exception as exc:
        logger.warning("PyMuPDF extraction failed: %s", exc)
        return []


def _pdfplumber_pages(pdf_bytes: bytes) -> List[str]:
    if not HAS_PDFPLUMBER:
        return []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages: List[str] = []
            for page in pdf.pages:
                try:
                    text = page.extract_text(
                        x_tolerance=_PDFPLUMBER_X_TOLERANCE,
                        y_tolerance=_PDFPLUMBER_Y_TOLERANCE,
                    ) or ""
                    pages.append(text)
                except Exception:
                    pages.append("")
            return pages
    except Exception as exc:
        logger.debug("pdfplumber extraction failed: %s", exc)
        return []


def _pypdf_pages(pdf_bytes: bytes) -> List[str]:
    if not HAS_PYPDF:
        return []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        logger.debug("pypdf extraction failed: %s", exc)
        return []


def _pick_best_pages(
    candidates: List[Tuple[str, List[str]]],
) -> Tuple[List[str], str]:
    from nepali_corpus.core.utils.normalize import devanagari_ratio

    if not candidates:
        return [], ""

    max_pages = max(len(pages) for _, pages in candidates)
    best_pages: List[str] = []
    method_votes: Dict[str, int] = {}

    for page_idx in range(max_pages):
        best_text = ""
        best_ratio = -1.0
        best_method = ""

        for method, pages in candidates:
            if page_idx >= len(pages):
                continue
            text = pages[page_idx]
            if not text:
                continue
            ratio = devanagari_ratio(text)
            if ratio > best_ratio or (ratio == best_ratio and len(text) > len(best_text)):
                best_text = text
                best_ratio = ratio
                best_method = method

        best_pages.append(best_text)
        if best_method:
            method_votes[best_method] = method_votes.get(best_method, 0) + 1

    winner = max(method_votes, key=method_votes.get) if method_votes else ""
    return best_pages, winner


def _extract_pdf_metadata(pdf_bytes: bytes) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    if not HAS_PYMUPDF:
        return meta
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        meta["page_count"] = doc.page_count
        fitz_meta = doc.metadata or {}
        for key in ("title", "author", "subject", "creator", "producer"):
            value = fitz_meta.get(key, "").strip()
            if value:
                meta[key] = value
        doc.close()
    except Exception as exc:
        logger.debug("Could not read PDF metadata: %s", exc)
    return meta


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF extraction")

    from .detect import PdfType, detect_pdf_type
    from .score import score_document
    from .ocr import ocr_full_document, ocr_pages
    from .clean import clean_pdf_text

    detection = detect_pdf_type(pdf_bytes)
    logger.debug(
        "PDF type: %s (pages=%d, scanned=%d, legacy=%d, unicode=%d)",
        detection.pdf_type.value, detection.page_count,
        len(detection.scanned_pages), len(detection.legacy_pages),
        len(detection.unicode_pages),
    )

    if detection.pdf_type == PdfType.SCANNED:
        ocr_page_texts = ocr_full_document(pdf_bytes)
        if ocr_page_texts:
            return clean_pdf_text(ocr_page_texts)
        return ""

    candidates: List[Tuple[str, List[str]]] = []

    mu_pages = _pymupdf_pages(pdf_bytes)
    if mu_pages:
        candidates.append(("pymupdf", mu_pages))

    mu_score = score_document(mu_pages, extraction_method="pymupdf") if mu_pages else None
    needs_fallback = (
        not mu_pages
        or (mu_score and mu_score.overall_score < 0.3)
        or (mu_score and mu_score.avg_devanagari_ratio < 0.15)
    )

    if needs_fallback:
        pl_pages = _pdfplumber_pages(pdf_bytes)
        if pl_pages:
            candidates.append(("pdfplumber", pl_pages))
        py_pages = _pypdf_pages(pdf_bytes)
        if py_pages:
            candidates.append(("pypdf", py_pages))

    if not candidates:
        ocr_page_texts = ocr_full_document(pdf_bytes)
        if ocr_page_texts:
            return clean_pdf_text(ocr_page_texts)
        return ""

    best_pages, method = _pick_best_pages(candidates)

    doc_score = score_document(best_pages, extraction_method=method)
    if doc_score.pages_needing_ocr:
        logger.debug(
            "OCR needed for %d/%d pages (method=%s, score=%.2f)",
            len(doc_score.pages_needing_ocr), len(best_pages),
            method, doc_score.overall_score,
        )
        ocr_results = ocr_pages(pdf_bytes, doc_score.pages_needing_ocr)
        for page_num, ocr_text in ocr_results.items():
            if ocr_text and len(ocr_text.strip()) > len(best_pages[page_num].strip()):
                best_pages[page_num] = ocr_text

    return clean_pdf_text(best_pages)

