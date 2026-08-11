#!/usr/bin/env python3
"""
Municipality Scraper for Nepal local government websites (नगरपालिका).

Nepal has 293 municipalities. Two dominant site families exist:

1. Drupal-based (e.g. vyasmun.gov.np, chandragirimun.gov.np):
   - listing:  /ne/content/suchana-darpan
   - item:     /ne/content/{slug}
   - date:     <meta name="dcterms.created" content="YYYY-MM-DD">
   - files:    /ne/system/files/{date}/{file}.pdf
   - pages:    ?page=N

2. Custom CMS (e.g. kirtipurmun.gov.np):
   - listing:  /announcements
   - item:     /announcements/detail/{slug}
   - pages:    ?page=N

Listing URL is discovered from the homepage when not configured:
links whose text matches (सूचना|suchana|notice|announcement) and whose
href matches (content|suchana|notice|announcement). Entries may override
via ``endpoints.notice_list``.

Every post gets province/district stamped from the registry entry, so the
resulting corpus records are Nepal-locatable (NepalLocatableMixin).

Usage:
    python municipality_scraper.py --list
    python municipality_scraper.py --all --pages 3
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import urllib3
import yaml
from bs4 import BeautifulSoup

try:
    from ...models import RawRecord
    from ...models.government_schemas import GovtPost, RegistryEntry
    from ...models.source_config import SourceConfig
    from .scraper_base import ScraperBase
    from ...utils.content_types import identify_content_type
except ImportError:  # pragma: no cover
    from nepali_corpus.core.models import RawRecord
    from nepali_corpus.core.models.government_schemas import GovtPost, RegistryEntry
    from nepali_corpus.core.models.source_config import SourceConfig
    from nepali_corpus.core.services.scrapers.scraper_base import ScraperBase
    from nepali_corpus.core.utils.content_types import identify_content_type

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Listing link discovery: text OR href must carry one of these.
LISTING_TEXT_KEYWORDS = re.compile(
    r"(suchana|suchna|सूचना|notice|announcement|soochana|sucana)", re.I
)
LISTING_HREF_KEYWORDS = re.compile(
    r"(suchana|सूचना|notice|announcement|content-list|sucana)", re.I
)

# Item link families — detected from href, no config needed.
# Negative lookahead excludes listing pages themselves (suchana-darpan etc.).
DRUPAL_ITEM = re.compile(r"/(?:ne|en)?/?content/(?!suchana-darpan/?$|sucana/?$|suchana/?$)[^/?#]+/?$")
# kirtipur-style: /announcements/{slug} (current) + /announcements/detail/{slug} (legacy).
# Bare /announcements (the listing) is excluded because it has no trailing segment.
ANNOUNCE_ITEM = re.compile(r"/announcements?/(?:detail/)?[^/?#]+/?$", re.I)
# kathmandu-style: /archives/notice/{slug-or-id}/ — notice-type/* category pages excluded.
ARCHIVES_ITEM = re.compile(r"/archives/notice/(?!type/)[^/?#]+/?$", re.I)

# Homepage links that LOOK like a notice listing but never are (procurement
# boards, tender feeds) — reading them as listing yields 0 items + an extra fetch.
LISTING_BLOCK_RE = re.compile(r"procurement|tender-notices|javascript:", re.I)

# Attachments / published dates.
ATTACHMENT_RE = re.compile(r"\.(pdf|doc|docx|xls|xlsx|jpg|jpeg|png|zip)$", re.I)
PUBLISH_DATE_META = re.compile(
    r"dcterms\.created|article:published_time|datePublished|date", re.I
)

NEPALI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Python `re` has no \p{} — use the explicit Devanagari block.
DEVANAGARI_RE = re.compile("[\u0900-\u097F]")

DEFAULT_MAX_PAGES = 3
DEFAULT_MAX_ITEMS = 80


def post_to_raw(post: GovtPost) -> RawRecord:
    """Convert a govt post to a RawRecord, mirroring govt_scraper.post_to_raw."""
    scraped_at = post.scraped_at
    if hasattr(scraped_at, "isoformat"):
        scraped_at = scraped_at.isoformat()
    raw_meta = {
        "has_attachment": post.has_attachment,
        "attachment_urls": post.attachment_urls,
        "province": post.province,
        "district": post.district,
    }
    if post.date_bs:
        raw_meta["date_bs"] = post.date_bs
    return RawRecord(
        source_id=post.source_id,
        source_name=post.source_name,
        url=post.url,
        title=post.title,
        language=post.language,
        published_at=post.date_ad.isoformat() if post.date_ad else None,
        date_bs=post.date_bs,
        province=post.province,
        district=post.district,
        category=post.category,
        content_type=identify_content_type(post.url),
        fetched_at=scraped_at,
        raw_meta=raw_meta,
    )


def _convert_nepali_digits(text: str) -> str:
    return text.translate(NEPALI_DIGITS)


def extract_bs_date(text: Optional[str]) -> Optional[str]:
    """Extract Bikram Sambat date from text (works with Nepali or Latin digits)."""
    if not text:
        return None
    text = _convert_nepali_digits(text)
    match = re.search(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return None


def parse_iso_date(text: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 date from `<meta content=...>`, tolerant of junk."""
    if not text:
        return None
    text = _convert_nepali_digits(text.strip())
    match = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if not match:
        return None
    year, month, day = (int(g) for g in match.groups())
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def _is_nepali_heading(text: Optional[str]) -> bool:
    return bool(text) and len(text.strip()) >= 6 and bool(DEVANAGARI_RE.search(text or ""))


TITLE_JUNK = {"मेनु", "होम", "खोज", "मुख्य", "समाचार"}

# Static site pages that masquerade as notices in drupal homepages
# (ward profiles, organization charts, contact pages, language switch).
# Exact-match set + pattern for the more positional nav titles.
NAV_TITLE_EXACT = {
    "नेपाली",
    "परिचय",
    "सम्पर्क",
    "संगठन",
    "संगठनात्मक स्वरुप",
    "संगठनात्मक स्वरूप",
    "संगठन संरचना",
    "नागरिक वडापत्र",
    "नगर पार्श्वचित्र",
    "संक्षिप्त परिचय",
    "प्रोफाईल",
    "प्रोफाइल",
    "डिजिटल प्रोफाईल",
    "महत्त्वपूर्ण लिङ्क",
    "महत्वपूर्ण स्थानहरु",
    "बजेट तथा कार्यक्रम",
    "बिस्तृत स्वरुप",
}
NAV_TITLE_RE = re.compile(
    r"(?:वडा (?:नं?\.? )?\d|वडा प्रोफाइल|वडा विवरण|\d+ नं?\.? वडा)"
    r"|(?:पार्श्वचित्र|पाश्वर्व चित्र|संगठनात्मक|वडापत्र|स्वरुप|स्वरूप|संरचना|परिचय|सम्पर्क|प्रोफाईल|प्रोफाइल|लिङ्क|स्थानहरु|सूचनाहरु|सुचनाहरु|नेपाली)$"
)


def _is_nav_title(title: str) -> bool:
    """True when a title looks like a static site page, not a notice."""
    t = title.strip()
    if t in NAV_TITLE_EXACT:
        return True
    if NAV_TITLE_RE.search(t):
        return True
    # "वडा २-ज्यामरुकोट" / "श्री वडा कार्यालय ..." style ward pages —
    # only when short; a long title like "वडा नं ५ मा विद्युत अवरोध..." is a notice.
    if len(t) <= 16 and re.match(r"^(वडा|श्री वडा कार्यालय|१|२|३|४|५|६|७|८|९|१०|११|१२|१३|१४|१५|१६|१७|१८|१९|२०|२१|२२|२३|२४|२५|२६|२७|२८|२९|३०)", t) and re.search(r"वडा", t):
        return True
    return False


def _clean_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" \t.,;:!?।()[]\"'")


def _extract_item_title(soup: BeautifulSoup, config, title_hint: Optional[str]) -> Optional[str]:
    """Strongest first: listing hint, then h1/h2, then meta/title tag."""
    if title_hint:
        return _clean_title(title_hint)

    for tag in ("h1", "h2", "h3"):
        for el in soup.find_all(tag):
            text = el.get_text(" ", strip=True)
            if not text or len(text) < 6 or not DEVANAGARI_RE.search(text):
                continue
            text = _clean_title(text)
            if text in TITLE_JUNK:
                continue
            if _is_brand_title(text, config):
                continue
            return text

    og = soup.find("meta", {"property": "og:title"}) or soup.find("meta", {"name": "twitter:title"})
    if og and og.get("content"):
        content = _clean_title(og["content"])
        if len(content) >= 6 and DEVANAGARI_RE.search(content):
            return content

    title_tag = soup.find("title")
    if title_tag:
        content = _clean_title(title_tag.get_text(" ", strip=True))
        if content and len(content) >= 6 and DEVANAGARI_RE.search(content):
            return content.split("|")[0].strip()
    return None


def _is_brand_title(text: str, config) -> bool:
    """True when a heading is just the municipality's own name (site brand)."""
    norm = re.sub(r"\s+", " ", text).strip()
    if not norm:
        return True
    candidates = []
    if getattr(config, "name_ne", None):
        candidates.append(config.name_ne)
    if getattr(config, "name", None):
        candidates.append(config.name)
    for cand in candidates:
        if cand and re.sub(r"\s+", " ", cand).strip() in norm:
            return True
    return False


def _guess_family(href: str) -> Optional[str]:
    if ANNOUNCE_ITEM.search(href):
        return "announcement"
    if ARCHIVES_ITEM.search(href):
        return "archives"
    if DRUPAL_ITEM.search(href):
        return "drupal"
    return None


class MunicipalityScraper(ScraperBase):
    """Generic scraper for Nepal municipality notice boards (two CMS families)."""

    def __init__(self, config, delay: float = 0.5):
        """``config`` — SourceConfig or anything with the same attributes
        (id, name, name_ne, url, endpoints, province, district, priority).
        """
        self.config = config
        self.source_id = getattr(config, "id", None) or getattr(config, "source_id", None)
        self.name = getattr(config, "name", None) or self.source_id
        self.name_ne = getattr(config, "name_ne", None)
        self.province = getattr(config, "province", None)
        self.district = getattr(config, "district", None)
        self.endpoints = getattr(config, "endpoints", None) or {}
        base_url = getattr(config, "url", None) or getattr(config, "base_url", None)
        self.language = getattr(config, "language", "ne")
        super().__init__(base_url or "", delay=delay, verify_ssl=False)
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,ne;q=0.8",
            }
        )

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Thin wrapper around the retried base fetch (mirrors MinistryScraper)."""
        return super().fetch_page(url)

    # ------------------------------------------------------------------
    # Listing discovery
    # ------------------------------------------------------------------

    def _resolve_listing_url(self, home_soup: Optional[BeautifulSoup]) -> Optional[str]:
        """Explicit endpoint override first, then homepage keyword discovery."""
        configured = self.endpoints.get("notice_list") or self.endpoints.get("notice")
        if configured:
            return urljoin(self.base_url + "/", configured)

        if home_soup is None:
            return None

        best: Optional[Tuple[int, int, str]] = None  # (score, text_len, url)
        seen = set()
        for a in home_soup.find_all("a", href=True):
            text = a.get_text(" ", strip=True)
            href = a["href"]
            if not href or href.startswith(("#", "javascript:", "mailto:")):
                continue
            text_match = bool(LISTING_TEXT_KEYWORDS.search(text))
            href_match = bool(LISTING_HREF_KEYWORDS.search(href))
            if not (text_match or href_match):
                continue
            # A link whose own href is an *item* (notice page) can never be the
            # listing — long notice titles used to win the scoring and redirect
            # the crawl into a single notice page (nepalgunj/budhanilkantha).
            canonical = href.split("?")[0]
            if _guess_family(canonical) is not None:
                continue
            if LISTING_BLOCK_RE.search(canonical):
                continue
            score = 2 if text_match and href_match else 1
            if href.startswith("http") and urlparse(href).netloc != urlparse(self.base_url).netloc:
                continue  # external link is not our listing
            url = urljoin(self.base_url + "/", href)
            norm = url.split("?")[0]
            if norm in seen:
                continue
            seen.add(norm)
            if best is None or (score, len(text), norm) > (best[0], best[1], best[2]):
                best = (score, len(text), norm)

        return best[2] if best else None

    # ------------------------------------------------------------------
    # Listing pages
    # ------------------------------------------------------------------

    def _parse_listing(self, soup: BeautifulSoup) -> List[Tuple[str, Optional[str]]]:
        """Return [(item_url, title_hint)] found on a listing page.

        URLs are canonicalized (query strings and fragments stripped) so the
        same item surfaced as ``/x`` and ``/x?lang=ne`` collapses to one.
        """
        items: List[Tuple[str, Optional[str]]] = []
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Canonicalize before family detection so ?page=0 / ?lang=ne
            # variants still match and collapse to one URL.
            canonical = href.split("?")[0].split("#")[0]
            family = _guess_family(canonical)
            if family is None:
                continue
            url = urljoin(self.base_url + "/", canonical)
            if url in seen:
                continue
            seen.add(url)
            text = a.get_text(" ", strip=True)
            # Drupal item links carry the full Nepali title in text; kirtipur
            # links carry only a slug — detect via Devanagari presence.
            title_hint = text if _is_nepali_heading(text) else None
            items.append((url, title_hint))
        return items

    def _next_page_url(self, soup: BeautifulSoup, current_url: str, page_num: int) -> Optional[str]:
        """Drupal and kirtipur both use ``?page=N``. Check rel=next/pagination links first."""
        for a in soup.find_all("a", href=True):
            rel = a.get("rel") or []
            if "next" in rel and a["href"]:
                return urljoin(current_url, a["href"])
        pagination = (
            soup.find("ul", class_=re.compile(r"pagination", re.I))
            or soup.find("div", class_=re.compile(r"pagination", re.I))
        )
        if pagination:
            next_page = pagination.find("a", string=str(page_num + 1))
            if next_page and next_page.get("href"):
                return urljoin(current_url, next_page["href"])

        # Synthesize ?page=N+1 preserving existing query params.
        parsed = urlparse(current_url)
        qs = parse_qs(parsed.query)
        qs["page"] = [str(page_num + 1)]
        return urljoin(
            current_url,
            parsed._replace(query=urlencode(qs, doseq=True)).geturl(),
        )

    # ------------------------------------------------------------------
    # Item pages
    # ------------------------------------------------------------------

    def _parse_item(self, url: str, title_hint: Optional[str]) -> Optional[GovtPost]:
        soup = self._fetch_page(url)
        if soup is None:
            return None

        # Category/list pages masquerade as items (e.g. "वडा विवरण" on vyas
        # homepages) and render a near-identical news carousel to real
        # notices, so raw item-link counts cannot separate them. One reliable
        # tell: real Drupal notices carry attachments or embedded dates,
        # category rows carry neither. Refuse link-dense Drupal pages with no
        # attachment and no date. (Announcement-family pages are exempt —
        # kirtipur's related-notices footer legitimately links 5+ items.)
        if _guess_family(url) == "drupal":
            nested = self._parse_listing(soup)
            if len(nested) >= 8:
                attachment_urls = [
                    urljoin(url, a["href"])
                    for a in soup.find_all("a", href=True)
                    if ATTACHMENT_RE.search(a["href"]) or "/system/files/" in a["href"]
                ]
                has_meta_date = any(
                    PUBLISH_DATE_META.search(m.get("name") or m.get("property") or "")
                    for m in soup.find_all("meta")
                )
                has_text_date = bool(extract_bs_date(soup.get_text(" ", strip=True)))
                if not attachment_urls and not has_meta_date and not has_text_date:
                    logger.debug("%s: skipping link-dense (category?) page %s", self.source_id, url)
                    return None

        title = _extract_item_title(soup, self.config, title_hint)
        if not title or len(title) < 5:
            return None
        if not DEVANAGARI_RE.search(title):
            return None
        if _is_nav_title(title):
            return None
        if len(title) > 200:
            title = title[:197] + "..."

        # Date: dcterms.created meta (Drupal) → datetime, else BS regex on text.
        date_ad: Optional[datetime] = None
        for meta in soup.find_all("meta"):
            name = (meta.get("name") or meta.get("property") or "").strip()
            if PUBLISH_DATE_META.search(name):
                date_ad = parse_iso_date(meta.get("content"))
                if date_ad:
                    break
        if date_ad is None:
            time_el = soup.find("time", {"datetime": True})
            if time_el:
                date_ad = parse_iso_date(time_el.get("datetime"))

        page_text = soup.get_text(" ", strip=True)
        date_bs = extract_bs_date(page_text)

        attachment_urls: List[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ATTACHMENT_RE.search(href) or "/system/files/" in href:
                full = urljoin(url, href)
                if full not in attachment_urls:
                    attachment_urls.append(full)

        # Content snippet for later enrichment hints.
        content_snippet = None
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"content|body", re.I))
            or soup.find("div", {"id": re.compile(r"content|body", re.I)})
        )
        if main:
            snippet = re.sub(r"\s+", " ", main.get_text(" ", strip=True)).strip()
            if len(snippet) >= 30:
                content_snippet = snippet[:500]

        post_id = hashlib.md5(f"{self.source_id}:{url}".encode()).hexdigest()[:12]
        return GovtPost(
            id=post_id,
            title=title,
            url=url,
            source_id=self.source_id,
            source_name=self.name,
            source_domain=self.base_url.replace("https://", "").replace("http://", ""),
            date_ad=date_ad,
            date_bs=date_bs,
            province=self.province,
            district=self.district,
            category="notice",
            language="ne",
            has_attachment=bool(attachment_urls),
            attachment_urls=attachment_urls,
            content_snippet=content_snippet,
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def scrape(self, max_pages: int = DEFAULT_MAX_PAGES, max_items: int = DEFAULT_MAX_ITEMS) -> List[GovtPost]:
        """Scrape the municipality's notice board end-to-end."""
        home_soup = self._fetch_page(self.base_url + "/")
        listing_url = self._resolve_listing_url(home_soup)
        if listing_url is None:
            # Some homepages ARE the listing — crawl the homepage directly.
            logger.info("%s: no listing discovered, crawling homepage", self.source_id)
            listing_url = self.base_url + "/"
            listing_soup = home_soup
        else:
            listing_soup = self._fetch_page(listing_url)
            if listing_soup is None:
                # Declared/discovered listing is dead (404, moved). Drupal
                # homepages embed recent notices directly — fall back to them.
                logger.warning(
                    "%s: listing %s unreachable, falling back to homepage",
                    self.source_id, listing_url,
                )
                listing_url = self.base_url + "/"
                listing_soup = home_soup
            elif len(self._parse_listing(listing_soup)) < 2:
                # Discovered listing page carries no items — it was likely
                # mis-picked (nav page, procurement board). Homepage itself
                # embeds recent notices; use it as the listing instead.
                logger.warning(
                    "%s: listing %s has no items, falling back to homepage",
                    self.source_id, listing_url,
                )
                listing_url = self.base_url + "/"
                listing_soup = home_soup

        all_items: List[Tuple[str, Optional[str]]] = []
        seen_urls: Dict[str, Optional[str]] = {}
        current_url = listing_url
        page_num = 1
        while listing_soup is not None and page_num <= max_pages and len(seen_urls) < max_items:
            for url, hint in self._parse_listing(listing_soup):
                if url not in seen_urls:
                    seen_urls[url] = hint
            next_url = self._next_page_url(listing_soup, current_url, page_num)
            if next_url is None or next_url == current_url or page_num >= max_pages:
                break
            current_url = next_url
            page_num += 1
            listing_soup = self._fetch_page(current_url)

        all_items = list(seen_urls.items())
        logger.info("%s: %d items found, fetching details", self.source_id, len(all_items))

        posts: List[GovtPost] = []
        for idx, (url, hint) in enumerate(all_items[:max_items]):
            try:
                post = self._parse_item(url, hint)
            except Exception as exc:  # keep site crawl alive on item errors
                logger.warning("%s item failed %s: %s", self.source_id, url, exc)
                post = None
            if post is not None:
                posts.append(post)
            if idx and idx % 20 == 0:
                logger.info("%s: %d/%d items parsed", self.source_id, idx, len(all_items))

        logger.info("%s: done, %d posts", self.source_id, len(posts))
        return posts


# ============ Registry wiring ============

def build_configs(entries):
    """Normalize entries (SourceConfig or RegistryEntry) into scraper configs."""
    configs: Dict[str, object] = {}
    for entry in entries:
        sid = getattr(entry, "id", None) or getattr(entry, "source_id", None)
        url = getattr(entry, "url", None) or getattr(entry, "base_url", None)
        if not sid or not url:
            continue
        configs[sid] = entry
    return configs


def fetch_raw_records(entries, pages: int = DEFAULT_MAX_PAGES, max_items: int = DEFAULT_MAX_ITEMS) -> List[RawRecord]:
    """Scrape a list of municipality entries → RawRecords."""
    records: List[RawRecord] = []
    for entry in entries:
        try:
            scraper = MunicipalityScraper(entry)
            posts = scraper.scrape(max_pages=max(1, pages), max_items=max_items)
            records.extend(post_to_raw(p) for p in posts)
        except Exception as exc:  # one dead site must not kill the batch
            logger.error("Municipality %s failed: %s", getattr(entry, "id", "?"), exc)
    return records


# ============ Config — single source of truth ============

DEFAULT_CONFIG_PATH = os.path.join("sources", "municipalities.yaml")

PROVINCES = {
    "Koshi", "Madhesh", "Bagmati", "Gandaki", "Lumbini", "Karnali", "Sudurpashchim",
}

DISTRICTS = {
    "Achham", "Arghakhanchi", "Baglung", "Baitadi", "Bajhang", "Bajura", "Banke",
    "Bara", "Bardiya", "Bhaktapur", "Bhojpur", "Chitwan", "Dadeldhura", "Dailekh",
    "Dang", "Darchula", "Dhading", "Dhankuta", "Dhanusa", "Dhanusha", "Dolakha",
    "Dolpa","Doti", "Gorkha", "Gulmi", "Humla", "Ilam", "Jajarkot", "Jhapa", "Jumla",
    "Kailali", "Kalikot", "Kanchanpur", "Kapilvastu", "Kaski", "Kathmandu",
    "Kavrepalanchok", "Khotang", "Lalitpur", "Lamjung", "Mahottari", "Makwanpur",
    "Manang", "Morang", "Mugu", "Mustang", "Myagdi", "Nawalparasi", "Nuwakot",
    "Okhaldhunga", "Palpa", "Panchthar", "Parbat", "Parsa", "Pyuthan", "Ramechhap",
    "Rasuwa", "Rautahat", "Rolpa", "Rukum", "Rupandehi", "Salyan", "Sankhuwasabha",
    "Saptari", "Sarlahi", "Sindhuli", "Sindhupalchok", "Siraha", "Solukhumbu",
    "Sunsari", "Surkhet", "Syangja", "Tanahun", "Taplejung", "Terhathum", "Udayapur",
}


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, SourceConfig]:
    """Load the municipality config YAML — the single source of truth.

    Fail loud on ANY violation: duplicate id, missing field, unknown
    province/district, bad URL, non-root endpoint. Never silently drop.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: config must be a YAML list of entries")

    configs: Dict[str, SourceConfig] = {}
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry must be a mapping, got {type(entry).__name__}")
        sid = entry.get("id")
        if not sid:
            raise ValueError(f"{path}: entry missing id")
        if sid in configs:
            raise ValueError(f"{path}: duplicate municipality id {sid!r}")
        for field in ("url", "province", "district", "name_ne"):
            if not entry.get(field):
                raise ValueError(f"{path}: municipality {sid!r} missing field {field!r}")
        url = entry["url"]
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"{path}: municipality {sid!r} url not http(s): {url!r}")
        if entry["province"] not in PROVINCES:
            raise ValueError(
                f"{path}: municipality {sid!r} unknown province {entry['province']!r} "
                f"(use one of {sorted(PROVINCES)})"
            )
        if entry["district"] not in DISTRICTS:
            raise ValueError(
                f"{path}: municipality {sid!r} unknown district {entry['district']!r}"
            )
        endpoints = entry.get("endpoints") or {}
        if not isinstance(endpoints, dict):
            raise ValueError(f"{path}: municipality {sid!r} endpoints must be a mapping")
        for key, val in endpoints.items():
            if not isinstance(val, str) or not val.startswith("/") or "://" in val:
                raise ValueError(
                    f"{path}: municipality {sid!r} endpoints.{key} must be a "
                    f"root-relative path, got {val!r}"
                )
        configs[sid] = SourceConfig(**entry)
    return configs


# ============ CLI ============

MUNICIPALITIES: Dict[str, SourceConfig] = {}


def get_scraper(municipality_id: str) -> MunicipalityScraper:
    if municipality_id not in MUNICIPALITIES:
        raise ValueError(f"Unknown municipality: {municipality_id}")
    return MunicipalityScraper(MUNICIPALITIES[municipality_id])


def main(argv: Optional[List[str]] = None) -> None:
    # Windows consoles default to cp1252 — Devanagari titles need UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Scrape Nepal municipality notice boards")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH,
        help=f"Config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--id", help="Municipality ID (e.g. vyas, kirtipur)")
    parser.add_argument("--all", action="store_true", help="Scrape all configured municipalities")
    parser.add_argument("--pages", "-p", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--list", "-l", action="store_true", help="List configured municipalities")
    parser.add_argument("--output", "-o", help="Write posts as JSONL (one RawRecord per line)")
    args = parser.parse_args(argv)

    try:
        configs = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    MUNICIPALITIES.update(configs)

    if args.list or (not args.id and not args.all):
        print("=" * 70)
        print(f"Nepal Municipality Scraper — {len(configs)} municipalities")
        print("=" * 70)
        for cid, cfg in configs.items():
            print(f"  {cid:16s} {cfg.name_ne:32s} {str(cfg.province or ''):10s} "
                  f"{str(cfg.district or ''):14s} {cfg.url}")
        print(f"\nconfig: {args.config}")
        return

    if args.id and args.id not in configs:
        parser.error(f"unknown municipality {args.id!r} — use --list")

    targets = [args.id] if args.id else list(configs)

    out_f = None
    if args.output:
        out_f = open(args.output, "w", encoding="utf-8")

    try:
        total = 0
        for mid in targets:
            try:
                scraper = get_scraper(mid)
                posts = scraper.scrape(max_pages=args.pages, max_items=args.max_items)
            except Exception as exc:  # one dead site must not kill the batch
                print(f"\n{mid}: FAILED: {exc}")
                continue
            total += len(posts)
            print(f"\n{mid}: {len(posts)} posts")
            for p in posts[:5]:
                print(f"  - {p.title[:70]}")
                if p.date_ad:
                    print(f"    AD: {p.date_ad.date()}  BS: {p.date_bs}")
            if out_f:
                for p in posts:
                    rec = post_to_raw(p)
                    if hasattr(rec, "model_dump"):
                        payload = rec.model_dump(mode="json")
                    else:
                        payload = vars(rec)
                    out_f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        print(f"\nTotal: {total} posts from {len(targets)} target(s)")
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()