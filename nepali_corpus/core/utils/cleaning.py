from __future__ import annotations

import re
from typing import Optional

from .normalize import detect_nepali, normalize_text
from ..models import NormalizedDocument

_PAGE_NUMBER_RE = re.compile(r"(?m)^[ \t]*[-|]*[ \t]*[\d\u0966-\u096f]+[ \t]*[-|]*[ \t]*$")
_SOFT_HYPHEN_RE = re.compile(r"\u00ad")
_HYPHEN_WRAP_RE = re.compile(r"-\n([\u0900-\u097F])")
_REPEATED_PUNCT_RE = re.compile(r"([!?.\-_=*#|]{4,})")
_QUOTE_NORM: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\u2018\u2019\u2032]"), "'"),
    (re.compile(r"[\u201c\u201d\u2033]"), '"'),
    (re.compile(r"\u2013"), "-"),
    (re.compile(r"\u2014"), "\u2014"),
]
_ASCII_PERIOD_AFTER_DEV_RE = re.compile(r"([\u0900-\u097F])\.(?= |$)", re.MULTILINE)
_GOV_UI_RE = re.compile(r"\bA\s+A[-\u2013]\s+A\+?\s*(?:Share)?\b", re.IGNORECASE)


def _strip_pdf_artifacts(text: str) -> str:
    text = _SOFT_HYPHEN_RE.sub("", text)
    text = _HYPHEN_WRAP_RE.sub(r"\1", text)
    text = _PAGE_NUMBER_RE.sub("", text)
    text = _REPEATED_PUNCT_RE.sub(" ", text)
    text = _GOV_UI_RE.sub("", text)
    for pattern, repl in _QUOTE_NORM:
        text = pattern.sub(repl, text)
    text = _ASCII_PERIOD_AFTER_DEV_RE.sub(r"\1\u0964", text)
    return text


def clean_text(text: str) -> str:
    text = _strip_pdf_artifacts(text)
    text = normalize_text(text)
    return text


def is_nepali(doc: NormalizedDocument, min_ratio: float = 0.4) -> bool:
    if doc.language == "ne":
        return True
    return detect_nepali(doc.text, min_ratio=min_ratio)


def min_length(doc: NormalizedDocument, min_chars: int = 200) -> bool:
    return len(doc.text) >= min_chars
