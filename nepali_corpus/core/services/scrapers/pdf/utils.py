from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import Dict, List, Tuple

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

logger = logging.getLogger(__name__)


_PDFPLUMBER_X_TOLERANCE = 5
_PDFPLUMBER_Y_TOLERANCE = 3


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


def _try_pdfplumber(pdf_bytes: bytes) -> Tuple[str, bool]:
    if not HAS_PDFPLUMBER:
        return "", False
    try:
        from nepali_corpus.core.utils.normalize import devanagari_ratio
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_texts: List[str] = []
            for page in pdf.pages:
                try:
                    page_texts.append(
                        page.extract_text(
                            x_tolerance=_PDFPLUMBER_X_TOLERANCE,
                            y_tolerance=_PDFPLUMBER_Y_TOLERANCE,
                        )
                        or ""
                    )
                except Exception:
                    page_texts.append("")
            text = "\n\n".join(page_texts).strip()
        good = bool(text) and (
            len(text) > 200 or devanagari_ratio(text) > 0.2
        )
        return text, good
    except Exception as exc:
        logger.debug("pdfplumber extraction failed: %s", exc)
        return "", False


def _try_pymupdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n\n".join(pages).strip()
    except Exception as exc:
        logger.warning("PyMuPDF extraction failed: %s", exc)
        return ""


def _try_ocrmypdf(pdf_bytes: bytes) -> str:
    try:
        import ocrmypdf  # noqa: F401
    except ImportError:
        return ""

    in_path = out_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as in_f:
            in_f.write(pdf_bytes)
            in_path = in_f.name

        out_fd, out_path = tempfile.mkstemp(suffix=".pdf")
        os.close(out_fd)

        ocrmypdf.ocr(
            in_path,
            out_path,
            language="nep+eng",
            force_ocr=False,
            skip_text=False,
        )

        if HAS_PDFPLUMBER:
            with pdfplumber.open(out_path) as pdf2:
                pages2 = []
                for p in pdf2.pages:
                    try:
                        pages2.append(
                            p.extract_text(
                                x_tolerance=_PDFPLUMBER_X_TOLERANCE,
                                y_tolerance=_PDFPLUMBER_Y_TOLERANCE,
                            )
                            or ""
                        )
                    except Exception:
                        pages2.append("")
                return "\n\n".join(pages2).strip()
        elif HAS_PYMUPDF:
            doc2 = fitz.open(out_path)
            pages2 = [p.get_text("text") for p in doc2]
            doc2.close()
            return "\n\n".join(pages2).strip()
    except Exception as exc:
        logger.debug("ocrmypdf OCR failed: %s", exc)
    finally:
        for path in (in_path, out_path):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
    return ""


def _try_tesseract(pdf_bytes: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except ImportError:
        logger.debug("pytesseract or PIL not installed – skipping image OCR fallback.")
        return ""

    if not HAS_PYMUPDF:
        return ""

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        ocr_pages: List[str] = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            img = ImageOps.grayscale(img)
            img = ImageOps.autocontrast(img)
            ocr_pages.append(pytesseract.image_to_string(img, lang="nep+eng"))
        doc.close()
        return "\n\n".join(ocr_pages).strip()
    except Exception as exc:
        logger.debug("Per-page tesseract OCR failed: %s", exc)
        return ""


def _pick_best(candidates: List[str]) -> str:
    from nepali_corpus.core.utils.normalize import devanagari_ratio

    best = ""
    best_ratio = -1.0
    for text in candidates:
        if not text:
            continue
        r = devanagari_ratio(text)
        if r > best_ratio or (r == best_ratio and len(text) > len(best)):
            best, best_ratio = text, r
    return best


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF extraction")

    from nepali_corpus.core.utils.normalize import devanagari_ratio

    try:
        pl_text, pl_good = _try_pdfplumber(pdf_bytes)
        if pl_good:
            return pl_text

        mu_text = _try_pymupdf(pdf_bytes)
        mu_ratio = devanagari_ratio(mu_text)

        if len(mu_text) >= 200 and mu_ratio >= 0.2:
            return _pick_best([pl_text, mu_text])

        logger.debug("Native extraction weak (chars=%d, ratio=%.2f); trying OCR.", len(mu_text), mu_ratio)
        ocr_text = _try_ocrmypdf(pdf_bytes)

        if not ocr_text:
            ocr_text = _try_tesseract(pdf_bytes)

        return _pick_best([pl_text, mu_text, ocr_text])

    except Exception as exc:
        logger.error("PDF extraction error: %s", exc)
        return ""
