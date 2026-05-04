from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import aiohttp
from pydantic import ConfigDict

from .utils import HAS_PYMUPDF, _extract_pdf_metadata, _extract_text_from_pdf

from nepali_corpus.core.models import RawRecord
from nepali_corpus.core.models.base import CorpusEntity
from nepali_corpus.core.utils.cleaning import clean_text
from nepali_corpus.core.utils.normalize import detect_nepali

logger = logging.getLogger(__name__)


class PdfJob(CorpusEntity):
    model_config = ConfigDict(frozen=True)

    url: str
    category: Optional[str] = None


def _pdf_path(output_dir: Path, url: str) -> Path:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    return output_dir / f"{digest}.pdf"


async def _download_pdf(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Optional[bytes]:
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    return await resp.read()
                elif resp.status >= 500:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                else:
                    return None
        except (asyncio.TimeoutError, Exception) as exc:
            logger.debug("Download attempt %d failed for %s: %s", attempt + 1, url, exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay * (2 ** attempt))

    return None


async def _handle_job(
    job: PdfJob,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
    max_mb: int,
    min_chars: int,
    nepali_ratio: float,
    timeout: int,
    seen_url,
    mark_url,
) -> Optional[RawRecord]:
    async with semaphore:
        if await seen_url(job.url):
            return None
        await mark_url(job.url)

        pdf_bytes = await _download_pdf(session, job.url, timeout)
        if not pdf_bytes:
            logger.debug("Download failed or empty: %s", job.url)
            return None

        size_mb = len(pdf_bytes) / (1024 * 1024)
        if size_mb > max_mb:
            logger.debug("PDF too large (%.1f MB > %d MB): %s", size_mb, max_mb, job.url)
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = _pdf_path(output_dir, job.url)
        try:
            pdf_path.write_bytes(pdf_bytes)
        except Exception as exc:
            logger.warning("Could not save PDF to disk (%s): %s", exc, job.url)
            return None

        try:
            text = _extract_text_from_pdf(pdf_bytes)
        except Exception as exc:
            logger.warning("Text extraction raised: %s – url=%s", exc, job.url)
            return None

        text = clean_text(text)
        if len(text) < min_chars:
            logger.debug("Text too short (%d < %d): %s", len(text), min_chars, job.url)
            return None
        if not detect_nepali(text, min_ratio=nepali_ratio):
            logger.debug("Nepali ratio below threshold: %s", job.url)
            return None

        pdf_meta = _extract_pdf_metadata(pdf_bytes)
        raw_meta = {
            "content_type": "pdf",
            "pdf_path": str(pdf_path),
            "size_bytes": len(pdf_bytes),
            **pdf_meta,
        }

        return RawRecord(
            source_id=job.source_id,
            source_name=job.source_name,
            url=job.url,
            title=pdf_meta.get("title"),
            content=text,
            content_type="pdf",
            language="ne",
            published_at=None,
            category=job.category,
            fetched_at=datetime.utcnow().isoformat(),
            raw_meta=raw_meta,
        )


async def extract_pdfs(
    jobs: Iterable[PdfJob],
    *,
    output_dir: str,
    max_workers: int = 10,
    max_mb: int = 100,
    min_chars: int = 200,
    nepali_ratio: float = 0.35,
    timeout: int = 60,
    seen_url=None,
    mark_url=None,
) -> List[RawRecord]:
    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF extraction")

    if seen_url is None or mark_url is None:
        raise RuntimeError("seen_url and mark_url callbacks are required")

    output_path = Path(output_dir)
    semaphore = asyncio.Semaphore(max_workers)
    connector = aiohttp.TCPConnector(limit=max_workers, ssl=False)
    results: List[RawRecord] = []

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _handle_job(
                job,
                session,
                semaphore,
                output_path,
                max_mb,
                min_chars,
                nepali_ratio,
                timeout,
                seen_url,
                mark_url,
            )
            for job in jobs
        ]
        for coro in asyncio.as_completed(tasks):
            record = await coro
            if record:
                results.append(record)

    return results


__all__ = ["PdfJob", "extract_pdfs", "HAS_PYMUPDF"]
