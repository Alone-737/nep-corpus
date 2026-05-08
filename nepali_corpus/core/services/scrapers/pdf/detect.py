from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

try:
    import fitz
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False


class PdfType(str, Enum):
    UNICODE = "unicode"
    SCANNED = "scanned"
    LEGACY_FONT = "legacy_font"
    MIXED = "mixed"
    EMPTY = "empty"


_LEGACY_FONT_PATTERNS = [
    re.compile(r"preeti", re.IGNORECASE),
    re.compile(r"kantipur", re.IGNORECASE),
    re.compile(r"sagarmatha", re.IGNORECASE),
    re.compile(r"himali", re.IGNORECASE),
    re.compile(r"himalb", re.IGNORECASE),
    re.compile(r"fontasy", re.IGNORECASE),
    re.compile(r"prachalit", re.IGNORECASE),
    re.compile(r"sabdatara", re.IGNORECASE),
    re.compile(r"shangrila", re.IGNORECASE),
    re.compile(r"pcs\s*nepali", re.IGNORECASE),
    re.compile(r"aakriti", re.IGNORECASE),
    re.compile(r"gauri", re.IGNORECASE),
    re.compile(r"rukmini", re.IGNORECASE),
    re.compile(r"kanchan", re.IGNORECASE),
]


@dataclass
class PageInfo:
    page_num: int
    has_text: bool = False
    text_length: int = 0
    has_images: bool = False
    image_area_ratio: float = 0.0
    fonts: Set[str] = field(default_factory=set)
    has_legacy_fonts: bool = False
    legacy_font_names: Set[str] = field(default_factory=set)
    is_scanned: bool = False


@dataclass
class PdfDetectionResult:
    pdf_type: PdfType
    page_count: int = 0
    pages: List[PageInfo] = field(default_factory=list)
    all_fonts: Set[str] = field(default_factory=set)
    legacy_fonts: Set[str] = field(default_factory=set)
    scanned_pages: List[int] = field(default_factory=list)
    unicode_pages: List[int] = field(default_factory=list)
    legacy_pages: List[int] = field(default_factory=list)
    empty_pages: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "pdf_type": self.pdf_type.value,
            "page_count": self.page_count,
            "all_fonts": sorted(self.all_fonts),
            "legacy_fonts": sorted(self.legacy_fonts),
            "scanned_page_count": len(self.scanned_pages),
            "unicode_page_count": len(self.unicode_pages),
            "legacy_page_count": len(self.legacy_pages),
            "empty_page_count": len(self.empty_pages),
        }


def _is_legacy_font(font_name: str) -> bool:
    for pattern in _LEGACY_FONT_PATTERNS:
        if pattern.search(font_name):
            return True
    return False


def _analyze_page(page, page_num: int) -> PageInfo:
    info = PageInfo(page_num=page_num)

    text = page.get_text("text") or ""
    info.text_length = len(text.strip())
    info.has_text = info.text_length > 10

    image_list = page.get_images(full=True)
    info.has_images = len(image_list) > 0

    if info.has_images and page.rect.width > 0 and page.rect.height > 0:
        page_area = page.rect.width * page.rect.height
        total_image_area = 0.0
        for img in image_list:
            try:
                xref = img[0]
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    for r in img_rects:
                        total_image_area += r.width * r.height
            except Exception:
                pass
        info.image_area_ratio = min(total_image_area / page_area, 1.0) if page_area > 0 else 0.0

    try:
        font_list = page.get_fonts(full=True)
        for font_info in font_list:
            font_name = font_info[3] if len(font_info) > 3 else ""
            if font_name:
                info.fonts.add(font_name)
                if _is_legacy_font(font_name):
                    info.has_legacy_fonts = True
                    info.legacy_font_names.add(font_name)
    except Exception:
        pass

    info.is_scanned = (
        info.has_images
        and info.image_area_ratio > 0.5
        and info.text_length < 50
    )

    return info


def detect_pdf_type(pdf_bytes: bytes) -> PdfDetectionResult:
    if not HAS_PYMUPDF:
        return PdfDetectionResult(pdf_type=PdfType.EMPTY)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.warning("Could not open PDF for type detection: %s", exc)
        return PdfDetectionResult(pdf_type=PdfType.EMPTY)

    result = PdfDetectionResult(pdf_type=PdfType.EMPTY, page_count=doc.page_count)

    for page_num in range(doc.page_count):
        page = doc[page_num]
        info = _analyze_page(page, page_num)
        result.pages.append(info)
        result.all_fonts.update(info.fonts)
        result.legacy_fonts.update(info.legacy_font_names)

        if info.is_scanned:
            result.scanned_pages.append(page_num)
        elif info.has_legacy_fonts:
            result.legacy_pages.append(page_num)
        elif info.has_text:
            result.unicode_pages.append(page_num)
        else:
            result.empty_pages.append(page_num)

    doc.close()

    total = result.page_count
    if total == 0:
        result.pdf_type = PdfType.EMPTY
    elif len(result.scanned_pages) == total:
        result.pdf_type = PdfType.SCANNED
    elif len(result.legacy_pages) == total:
        result.pdf_type = PdfType.LEGACY_FONT
    elif len(result.unicode_pages) == total:
        result.pdf_type = PdfType.UNICODE
    elif len(result.unicode_pages) == 0 and len(result.legacy_pages) == 0:
        result.pdf_type = PdfType.SCANNED
    elif len(result.scanned_pages) > 0 or len(result.legacy_pages) > 0:
        result.pdf_type = PdfType.MIXED
    else:
        result.pdf_type = PdfType.UNICODE

    return result
