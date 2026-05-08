from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    import fitz
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def _ocr_page_image(page, dpi: int = 300, lang: str = "nep+eng") -> str:
    if not HAS_TESSERACT or not HAS_PYMUPDF:
        return ""

    try:
        pix = page.get_pixmap(dpi=dpi)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.SHARPEN)

        text = pytesseract.image_to_string(img, lang=lang, config=r"--oem 3 --psm 6")
        return text.strip()
    except Exception as exc:
        logger.debug("OCR failed for page: %s", exc)
        return ""


def ocr_pages(
    pdf_bytes: bytes,
    page_numbers: List[int],
    dpi: int = 300,
    lang: str = "nep+eng",
) -> Dict[int, str]:
    if not HAS_PYMUPDF or not HAS_TESSERACT or not page_numbers:
        return {}

    results: Dict[int, str] = {}
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in page_numbers:
            if 0 <= page_num < doc.page_count:
                text = _ocr_page_image(doc[page_num], dpi=dpi, lang=lang)
                if text:
                    results[page_num] = text
        doc.close()
    except Exception as exc:
        logger.warning("Page-level OCR failed: %s", exc)

    return results


def ocr_full_document(
    pdf_bytes: bytes, dpi: int = 300, lang: str = "nep+eng",
) -> List[str]:
    if not HAS_PYMUPDF or not HAS_TESSERACT:
        return []

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [_ocr_page_image(page, dpi=dpi, lang=lang) for page in doc]
        doc.close()
        return pages
    except Exception as exc:
        logger.warning("Full document OCR failed: %s", exc)
        return []
