from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from nepali_corpus.core.utils.normalize import devanagari_ratio

_NEPALI_MARKER_WORDS = [
    "नेपाल", "सरकार", "मन्त्रालय", "विभाग", "कार्यालय",
    "सूचना", "निर्णय", "अनुसार", "गरिएको", "जानकारी",
    "प्रदेश", "जिल्ला", "पालिका", "नगरपालिका", "गाउँपालिका",
    "ऐन", "नियम", "कानून", "आदेश", "परिपत्र",
    "बजेट", "योजना", "कार्यक्रम", "प्रतिवेदन", "विवरण",
]

_GARBLED_PATTERN = re.compile(r"[a-zA-Z]{5,}")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class PageScore:
    page_num: int
    text_length: int = 0
    devanagari_ratio: float = 0.0
    marker_word_count: int = 0
    garbled_ratio: float = 0.0
    control_char_count: int = 0
    quality_score: float = 0.0
    needs_ocr: bool = False

    def to_dict(self) -> Dict:
        return {
            "page_num": self.page_num,
            "text_length": self.text_length,
            "devanagari_ratio": round(self.devanagari_ratio, 3),
            "marker_word_count": self.marker_word_count,
            "quality_score": round(self.quality_score, 3),
            "needs_ocr": self.needs_ocr,
        }


@dataclass
class DocumentScore:
    page_scores: List[PageScore] = field(default_factory=list)
    overall_score: float = 0.0
    total_chars: int = 0
    avg_devanagari_ratio: float = 0.0
    pages_needing_ocr: List[int] = field(default_factory=list)
    extraction_method: str = ""

    def to_dict(self) -> Dict:
        return {
            "overall_score": round(self.overall_score, 3),
            "total_chars": self.total_chars,
            "avg_devanagari_ratio": round(self.avg_devanagari_ratio, 3),
            "pages_needing_ocr": self.pages_needing_ocr,
            "extraction_method": self.extraction_method,
            "page_count": len(self.page_scores),
        }


def score_page(text: str, page_num: int, min_devanagari: float = 0.15) -> PageScore:
    ps = PageScore(page_num=page_num)

    if not text or not text.strip():
        ps.needs_ocr = True
        return ps

    ps.text_length = len(text.strip())
    ps.devanagari_ratio = devanagari_ratio(text)

    for word in _NEPALI_MARKER_WORDS:
        if word in text:
            ps.marker_word_count += 1

    ascii_runs = _GARBLED_PATTERN.findall(text)
    total_garbled = sum(len(r) for r in ascii_runs)
    ps.garbled_ratio = total_garbled / max(len(text), 1)

    ps.control_char_count = len(_CONTROL_CHAR_RE.findall(text))

    score = 0.0
    score += min(ps.devanagari_ratio, 1.0) * 0.4
    score += min(ps.text_length / 500.0, 1.0) * 0.2
    score += min(ps.marker_word_count / 5.0, 1.0) * 0.2
    score -= ps.garbled_ratio * 0.3
    score -= min(ps.control_char_count / 10.0, 0.1)

    ps.quality_score = max(0.0, min(1.0, score))

    ps.needs_ocr = (
        ps.text_length < 30
        or ps.devanagari_ratio < min_devanagari
        or ps.quality_score < 0.15
    )

    return ps


def score_document(
    page_texts: List[str],
    extraction_method: str = "",
    min_devanagari: float = 0.15,
) -> DocumentScore:
    ds = DocumentScore(extraction_method=extraction_method)

    for i, text in enumerate(page_texts):
        ps = score_page(text, i, min_devanagari=min_devanagari)
        ds.page_scores.append(ps)
        ds.total_chars += ps.text_length
        if ps.needs_ocr:
            ds.pages_needing_ocr.append(i)

    if ds.page_scores:
        ds.avg_devanagari_ratio = (
            sum(ps.devanagari_ratio for ps in ds.page_scores) / len(ds.page_scores)
        )
        total_weight = 0.0
        weighted_score = 0.0
        for ps in ds.page_scores:
            weight = max(ps.text_length, 1)
            weighted_score += ps.quality_score * weight
            total_weight += weight
        ds.overall_score = weighted_score / max(total_weight, 1)

    return ds
