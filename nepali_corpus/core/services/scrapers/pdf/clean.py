from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List, Set, Tuple

_PAGE_NUM_RE = re.compile(
    r"(?m)^[ \t]*[-\u2013\u2014|]*[ \t]*[\d\u0966-\u096f]+[ \t]*[-\u2013\u2014|]*[ \t]*$"
)

_PAGE_OF_RE = re.compile(
    r"(?m)^[ \t]*(?:page|पृष्ठ)[ \t]*[\d\u0966-\u096f]+"
    r"[ \t]*(?:of|मध्ये|को)[ \t]*[\d\u0966-\u096f]+[ \t]*$",
    re.IGNORECASE,
)

_PREETI_INDICATORS = re.compile(
    r"(?:sf/jfno|;/sfn|;]gf|g]kfn|;/sf/|dfo{no|ljefu|sfof{no)"
)


def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def detect_repeated_headers_footers(
    page_texts: List[str],
    max_lines: int = 3,
    min_pages: int = 3,
    threshold: float = 0.6,
) -> Tuple[Set[str], Set[str]]:
    if len(page_texts) < min_pages:
        return set(), set()

    header_counter: Counter = Counter()
    footer_counter: Counter = Counter()

    for text in page_texts:
        if not text or not text.strip():
            continue
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue
        for ln in lines[:max_lines]:
            if len(ln) > 5:
                header_counter[ln] += 1
        for ln in lines[-max_lines:]:
            if len(ln) > 5:
                footer_counter[ln] += 1

    total_pages = len([t for t in page_texts if t and t.strip()])
    min_occurrences = max(min_pages, int(total_pages * threshold))

    headers = {line for line, count in header_counter.items() if count >= min_occurrences}
    footers = {line for line, count in footer_counter.items() if count >= min_occurrences}
    return headers, footers


def remove_headers_footers(
    page_texts: List[str], headers: Set[str], footers: Set[str],
) -> List[str]:
    if not headers and not footers:
        return page_texts
    cleaned = []
    for text in page_texts:
        if not text:
            cleaned.append("")
            continue
        lines = text.split("\n")
        filtered = [ln for ln in lines if ln.strip() not in headers and ln.strip() not in footers]
        cleaned.append("\n".join(filtered))
    return cleaned


def remove_page_numbers(text: str) -> str:
    text = _PAGE_NUM_RE.sub("", text)
    text = _PAGE_OF_RE.sub("", text)
    return text


def has_preeti_patterns(text: str) -> bool:
    if not text:
        return False
    return bool(_PREETI_INDICATORS.search(text))


def clean_pdf_text(page_texts: List[str], remove_repeated: bool = True) -> str:
    page_texts = [normalize_unicode(t) for t in page_texts]
    if remove_repeated and len(page_texts) > 2:
        headers, footers = detect_repeated_headers_footers(page_texts)
        page_texts = remove_headers_footers(page_texts, headers, footers)
    page_texts = [remove_page_numbers(t) for t in page_texts]
    text = "\n\n".join(t.strip() for t in page_texts if t.strip())
    return text.strip()
