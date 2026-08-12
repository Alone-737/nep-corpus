"""Tests for the municipality scraper (no network — mocked fetch_page)."""

from datetime import datetime

import pytest
from bs4 import BeautifulSoup

from nepali_corpus.core.models.source_config import SourceConfig
from nepali_corpus.core.services.scrapers import municipality_scraper as ms


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


# ----------------------------------------------------------------------
# Date helpers
# ----------------------------------------------------------------------

def test_extract_bs_date_latin_digits():
    assert ms.extract_bs_date("प्रकाशित मिति 2081-09-15") == "2081-09-15"
    assert ms.extract_bs_date("2081/9/1") == "2081-09-01"


def test_extract_bs_date_nepali_digits():
    assert ms.extract_bs_date("मिति २०८१-०९-१५") == "2081-09-15"
    assert ms.extract_bs_date("published 2081.09.15") is None  # dotted sep unsupported


def test_extract_bs_date_none():
    assert ms.extract_bs_date(None) is None
    assert ms.extract_bs_date("no date here") is None


def test_parse_iso_date():
    d = ms.parse_iso_date("  2026-08-10T00:00:00+05:45 ")
    assert d == datetime(2026, 8, 10)
    assert ms.parse_iso_date("2026-13-40") is None
    assert ms.parse_iso_date(None) is None


# ----------------------------------------------------------------------
# Family detection
# ----------------------------------------------------------------------

def test_guess_family():
    assert ms._guess_family("/ne/content/drawer-bridda-bhatta-5xja2c9p") == "drupal"
    assert ms._guess_family("/content/item-123") == "drupal"
    assert ms._guess_family("/en/content/item-456") == "drupal"
    # kirtipur current pattern (no /detail/) + legacy pattern
    assert ms._guess_family("/announcements/aavashayaka-satarakata-apanauna") == "announcement"
    assert ms._guess_family("/announcements/detail/notice-48") == "announcement"
    assert ms._guess_family("/announcements") is None  # the listing itself
    assert ms._guess_family("/announcements?page=2") is None  # pagination link
    assert ms._guess_family("/about-us") is None
    assert ms._guess_family("/ne/content/suchana-darpan") is None  # listing, not item
    assert ms._guess_family("/ne/content/suchana") is None  # listing, not item
    assert ms._guess_family("/ne/content/suchana-darpan?page=2") is None  # pagination link


# ----------------------------------------------------------------------
# Listing parsing
# ----------------------------------------------------------------------

def test_parse_listing_drupal_items_carry_title_hints():
    html = """
    <ul>
      <li><a href="/ne/content/drawer-bridda-bhatta-5xja2c9p">
         वृद्ध भत्ता कोषमा म्याद बितेका राहत फिर्ता गर्न वारे सूचना</a></li>
      <li><a href="/ne/content/other-notice-xyz">सूचना</a></li>
      <li><a href="/ne/content/suchana-darpan">सूचना दर्पण</a></li>
      <li><a href="/about">About</a></li>
    </ul>
    """
    items = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )._parse_listing(make_soup(html))
    urls = [u for u, _, _ in items]
    assert "https://vyasmun.gov.np/ne/content/drawer-bridda-bhatta-5xja2c9p" in urls
    assert "https://vyasmun.gov.np/ne/content/other-notice-xyz" in urls
    assert not any("suchana-darpan" in u for u in urls)  # listing link must not be an item
    long_title = [t for u, t, _ in items if u.endswith("5xja2c9p")][0]
    assert long_title.startswith("वृद्ध भत्ता")
    short = [t for u, t, _ in items if u.endswith("other-notice-xyz")][0]
    assert short is None  # "सूचना" too short to be a real title


def test_parse_listing_kirtipur_slug_links_no_hints():
    html = """
    <div class="item"><a href="/announcements/aavashayaka-satarakata">aavashayaka-satarakata</a></div>
    <div class="item"><a href="/announcements/detail/notice-48">notice-48</a></div>
    """
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="kirtipur", name="Kirtipur", url="https://kirtipurmun.gov.np")
    )
    items = scraper._parse_listing(make_soup(html))
    assert len(items) == 2
    assert all(t is None for _, t, _ in items)  # slug text is never a title hint


def test_parse_listing_normalizes_lang_query_duplicates():
    html = """
    <a href="/announcements/karayapalka-bthaka">karayapalka-bthaka</a>
    <a href="/announcements/karayapalka-bthaka?lang=ne">karayapalka-bthaka</a>
    <a href="/ne/content/item-one?page=0">पहिलो सूचना</a>
    """
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="kirtipur", name="Kirtipur", url="https://kirtipurmun.gov.np")
    )
    items = scraper._parse_listing(make_soup(html))
    urls = [u for u, _, _ in items]
    assert len(urls) == 2
    assert urls[0] == "https://kirtipurmun.gov.np/announcements/karayapalka-bthaka"
    assert urls[1] == "https://kirtipurmun.gov.np/ne/content/item-one"


def test_parse_listing_deduplicates_urls():
    html = """
    <a href="/ne/content/same-article">पहिलो सूचना</a>
    <a href="/ne/content/same-article">पहिलो सूचना</a>
    """
    items = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )._parse_listing(make_soup(html))
    assert len(items) == 1


def test_parse_listing_rejects_external_item_urls():
    html = """
    <a href="https://evil.example/ne/content/poison">
      सरकारी सूचना परीक्षण
    </a>
    """
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )
    assert scraper._parse_listing(make_soup(html)) == []


# ----------------------------------------------------------------------
# Listing URL discovery
# ----------------------------------------------------------------------

def test_resolve_listing_url_endpoint_override_wins():
    config = SourceConfig(
        id="kirtipur", name="Kirtipur", url="https://kirtipurmun.gov.np",
        endpoints={"notice_list": "/announcements"},
    )
    scraper = ms.MunicipalityScraper(config)
    url = scraper._resolve_listing_url(None)  # even with no homepage
    assert url == "https://kirtipurmun.gov.np/announcements"


def test_resolve_listing_url_homepage_discovery():
    html = """
    <nav>
      <a href="/">होम</a>
      <a href="/ne/content/suchana-darpan">सूचना दर्पण</a>
      <a href="/ne/content/news">समाचार</a>
      <a href="https://facebook.com/mun">Facebook</a>
    </nav>
    """
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )
    # Suchana-darpan: href + text both match -> score 2; news: text only -> score 1.
    assert scraper._resolve_listing_url(make_soup(html)) == \
        "https://vyasmun.gov.np/ne/content/suchana-darpan"


def test_resolve_listing_url_none_when_no_match():
    html = '<a href="/about">About us</a><a href="/ne">नेपाली</a>'
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )
    assert scraper._resolve_listing_url(make_soup(html)) is None


# ----------------------------------------------------------------------
# Pagination
# ----------------------------------------------------------------------

def test_next_page_url_synthesizes_page_param():
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="kirtipur", name="Kirtipur", url="https://kirtipurmun.gov.np")
    )
    url = scraper._next_page_url(make_soup("<html></html>"), "https://kirtipurmun.gov.np/announcements", 1)
    assert url == "https://kirtipurmun.gov.np/announcements?page=2"


def test_next_page_url_uses_zero_based_drupal_fallback():
    html = '<a href="/ne/content/notice-one">पहिलो सरकारी सूचना</a>'
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )
    url = scraper._next_page_url(
        make_soup(html), "https://vyasmun.gov.np/ne/content/suchana-darpan", 1
    )
    assert url == "https://vyasmun.gov.np/ne/content/suchana-darpan?page=1"


def test_next_page_url_keeps_existing_query_params():
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )
    url = scraper._next_page_url(
        make_soup("<html></html>"),
        "https://vyasmun.gov.np/ne/content/suchana-darpan?lang=ne", 2,
    )
    assert url == "https://vyasmun.gov.np/ne/content/suchana-darpan?lang=ne&page=3"


def test_next_page_url_honors_rel_next():
    html = '<a rel="next" href="/announcements?page=2">Next</a>'
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="kirtipur", name="Kirtipur", url="https://kirtipurmun.gov.np")
    )
    url = scraper._next_page_url(make_soup(html), "https://kirtipurmun.gov.np/announcements", 1)
    assert url == "https://kirtipurmun.gov.np/announcements?page=2"


def test_next_page_url_rejects_external_rel_next():
    html = '<a rel="next" href="https://evil.example/ne/content/page-2">Next</a>'
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )
    assert scraper._next_page_url(
        make_soup(html), "https://vyasmun.gov.np/ne/content/suchana-darpan", 1
    ) is None


# ----------------------------------------------------------------------
# Item parsing (mocked fetch)
# ----------------------------------------------------------------------

DRUPAL_ITEM_HTML = """
<html><head>
  <meta name="dcterms.created" content="2026-08-10T00:00:00+05:45">
  <title>वृद्ध भत्ता कोषमा फिर्ता गर्न वारे सूचना | व्यास नगरपालिका</title>
</head><body>
  <h1>वृद्ध भत्ता कोषमा म्याद बितेका राहत फिर्ता गर्न वारे सूचना</h1>
  <article>
    व्यास नगरपालिका, नगर कार्यपालिकाको कार्यालयबाट म्याद बितेका राहत रकम फिर्ता गर्नुपर्नेछ ।
    यो सूचनाको विस्तृत विवरण संलग्न PDF मा हेर्न सकिनेछ ।
  </article>
  <a href="/ne/system/files/2026-08/FINAL%20NOTICE.pdf">FINAL NOTICE.pdf</a>
  <a href="/ne/content/drawer-bridda-bhatta-5xja2c9p">वृद्ध भत्ता कोषमा ...</a>
</body></html>
"""


def test_parse_item_drupal(monkeypatch):
    scraper = ms.MunicipalityScraper(
        SourceConfig(
            id="vyas", name="Vyas", name_ne="व्यास नगरपालिका",
            url="https://vyasmun.gov.np", province="Gandaki", district="Tanahun",
        )
    )
    monkeypatch.setattr(scraper, "_fetch_page", lambda url: make_soup(DRUPAL_ITEM_HTML))

    post = scraper._parse_item("https://vyasmun.gov.np/ne/content/drawer-bridda-bhatta-5xja2c9p", None)
    assert post is not None
    assert post.title == "वृद्ध भत्ता कोषमा म्याद बितेका राहत फिर्ता गर्न वारे सूचना"
    assert post.date_ad == datetime(2026, 8, 10)
    # BS date comes from page text only (no AD->BS converter in pipeline yet);
    # Drupal pages carry AD in meta, so date_bs stays None unless text has it.
    assert post.date_bs is None
    assert post.province == "Gandaki"
    assert post.district == "Tanahun"
    assert post.category == "notice"
    assert post.attachment_urls == [
        "https://vyasmun.gov.np/ne/system/files/2026-08/FINAL%20NOTICE.pdf"
    ]
    assert post.has_attachment is True
    assert post.content_snippet and "व्यास नगरपालिका" in post.content_snippet


def test_parse_item_uses_title_hint_when_no_h1(monkeypatch):
    html = """
    <html><head><meta name="dcterms.created" content="2026-08-01"></head>
    <body><div>सूचना विवरण, प्रकाशित मिति २०८१-०४-१६</div></body></html>
    """
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )
    monkeypatch.setattr(scraper, "_fetch_page", lambda url: make_soup(html))
    post = scraper._parse_item(
        "https://vyasmun.gov.np/ne/content/x",
        "फर्म दरखास्तपछि शिक्षक करार सम्बन्धी अन्तिम नतिजा सूचना",
    )
    assert post is not None
    assert post.title == "फर्म दरखास्तपछि शिक्षक करार सम्बन्धी अन्तिम नतिजा सूचना"
    assert post.date_ad == datetime(2026, 8, 1)
    assert post.date_bs == "2081-04-16"


def test_parse_item_returns_none_on_fetch_failure(monkeypatch):
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )
    monkeypatch.setattr(scraper, "_fetch_page", lambda url: None)
    assert scraper._parse_item("https://vyasmun.gov.np/ne/content/x", None) is None


def test_parse_item_rejects_english_only_pages(monkeypatch):
    html = "<html><head><title>Home</title></head><body><h1>Welcome</h1></body></html>"
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )
    monkeypatch.setattr(scraper, "_fetch_page", lambda url: make_soup(html))
    post = scraper._parse_item("https://vyasmun.gov.np/en/content/english-page", "English page")
    assert post is None


# ----------------------------------------------------------------------
# End-to-end (mocked page sequence)
# ----------------------------------------------------------------------

HOME = """
<html><body>
  <nav><a href="/ne/content/suchana-darpan">सूचना दर्पण</a></nav>
</body></html>
"""
LISTING_P1 = """
<html><body>
  <ul>
    <li><a href="/ne/content/item-one">पहिलो सूचना: आवेदन सम्बन्धी</a></li>
    <li><a href="/ne/content/item-two">दोस्रो सूचना: दरखास्त सम्बन्धी</a></li>
  </ul>
  <div class="pagination"><a href="/ne/content/suchana-darpan?page=2">2</a></div>
</body></html>
"""
LISTING_P2 = """
<html><body>
  <ul>
    <li><a href="/ne/content/item-two">दोस्रो सूचना: दरखास्त सम्बन्धी</a></li>
    <li><a href="/ne/content/item-three">तेस्रो सूचना: नतिजा प्रकाशन सम्बन्धी</a></li>
  </ul>
</body></html>
"""
ITEM = """
<html><head><meta name="dcterms.created" content="2026-08-01"></head>
<body><h1>आवेदन सम्बन्धी सूचना</h1><article>विस्तृत विवरण यहाँ छ ।</article></body></html>
"""


def test_scrape_end_to_end(monkeypatch):
    config = SourceConfig(
        id="vyas", name="Vyas", url="https://vyasmun.gov.np",
        province="Gandaki", district="Tanahun",
    )
    scraper = ms.MunicipalityScraper(config)

    pages = {
        "https://vyasmun.gov.np/": make_soup(HOME),
        "https://vyasmun.gov.np/ne/content/suchana-darpan": make_soup(LISTING_P1),
        "https://vyasmun.gov.np/ne/content/suchana-darpan?page=2": make_soup(LISTING_P2),
    }
    item_pages = {
        "/ne/content/item-one": make_soup(ITEM),
        "/ne/content/item-two": make_soup(
            '<html><body><h1>दोस्रो सूचना: दरखास्त सम्बन्धी</h1><article>विवरण</article></body></html>'
        ),
        "/ne/content/item-three": make_soup(
            '<html><body><h1>तेस्रो सूचना: नतिजा प्रकाशन सम्बन्धी</h1><article>विवरण</article></body></html>'
        ),
    }

    def fake_fetch(url):
        if url in pages:
            return pages[url]
        for prefix, soup in item_pages.items():
            if prefix in url:
                return soup
        return None

    monkeypatch.setattr(scraper, "_fetch_page", fake_fetch)

    posts = scraper.scrape(max_pages=2, max_items=20)
    assert len(posts) == 3  # dedup across pages: item-two appears on both
    urls = sorted(p.url for p in posts)
    assert urls[0].endswith("item-one")
    assert urls[1].endswith("item-three")
    assert all(p.province == "Gandaki" for p in posts)
    assert all(p.district == "Tanahun" for p in posts)
    assert all(p.category == "notice" for p in posts)


def test_scrape_falls_back_when_listing_dead(monkeypatch):
    """Configured listing returns 404/None → homepage (which embeds items) is crawled."""
    config = SourceConfig(
        id="vyas", name="Vyas", url="https://vyasmun.gov.np",
        endpoints={"notice_list": "/ne/content/suchana-darpan"},
    )
    scraper = ms.MunicipalityScraper(config)

    home = """
    <html><body>
      <ul>
        <li><a href="/ne/content/item-one">पहिलो सूचना: आवेदन सम्बन्धी</a></li>
        <li><a href="/ne/content/item-two">दोस्रो सूचना: दरखास्त सम्बन्धी</a></li>
      </ul>
    </body></html>
    """
    item = '<html><head><meta name="dcterms.created" content="2026-08-01"></head><body><h1>सूचना शीर्षक</h1><article>विवरण</article></body></html>'

    def fake_fetch(url):
        if url == "https://vyasmun.gov.np/":
            return make_soup(home)
        if "/suchana-darpan" in url:
            return None  # dead listing
        return make_soup(item)

    monkeypatch.setattr(scraper, "_fetch_page", fake_fetch)
    posts = scraper.scrape(max_pages=1, max_items=20)
    assert len(posts) == 2  # both homepage items survived despite dead listing
    titles = sorted(p.title for p in posts)
    # listing link text (hint) wins over the generic h1; द < प in code points
    assert titles == ["दोस्रो सूचना: दरखास्त सम्बन्धी", "पहिलो सूचना: आवेदन सम्बन्धी"]


def test_scrape_survives_dead_items(monkeypatch):
    config = SourceConfig(
        id="vyas", name="Vyas", url="https://vyasmun.gov.np",
        province="Gandaki", district="Tanahun",
    )
    scraper = ms.MunicipalityScraper(config)

    pages = {
        "https://vyasmun.gov.np/": make_soup(HOME),
        "https://vyasmun.gov.np/ne/content/suchana-darpan": make_soup(LISTING_P1),
    }

    def fake_fetch(url):
        if url in pages:
            return pages[url]
        if "item-two" in url:
            return None
        return make_soup(ITEM)

    monkeypatch.setattr(scraper, "_fetch_page", fake_fetch)
    posts = scraper.scrape(max_pages=1, max_items=20)
    assert len(posts) == 1
    assert posts[0].url.endswith("item-one")


def test_parse_item_uses_second_heading_when_h1_is_brand(monkeypatch):
    """kirtipur: h1 is the office name, the notice title sits in h2."""
    html = """
    <html><head><title>Kirtipur Municipality</title></head>
    <body>
      <h1>कीर्तिपुर नगरपालिका नगर कार्यपालिकाको कार्यालय</h1>
      <h2>मेनु</h2>
      <h2>आवश्यक सतर्कता अपनाउनु हुन अनुरोध।</h2>
      <div>विवरण पाठ यहाँ छ।</div>
    </body></html>
    """
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="kirtipur", name="Kirtipur Municipality", name_ne="कीर्तिपुर नगरपालिका",
                     url="https://kirtipurmun.gov.np")
    )
    monkeypatch.setattr(scraper, "_fetch_page", lambda url: make_soup(html))
    post = scraper._parse_item("https://kirtipurmun.gov.np/announcements/aavashayaka-satarakata", None)
    assert post is not None
    assert post.title == "आवश्यक सतर्कता अपनाउनु हुन अनुरोध"  # trailing danda trimmed


def test_parse_item_keeps_notice_title_containing_municipality_name(monkeypatch):
    html = """
    <html><body>
      <h1>काठमाडौं महानगरपालिकाको अत्यन्त जरुरी सूचना</h1>
      <article>यो काठमाडौं महानगरपालिकाबाट प्रकाशित विस्तृत सार्वजनिक सूचना हो।</article>
    </body></html>
    """
    scraper = ms.MunicipalityScraper(
        SourceConfig(
            id="kathmandu",
            name="Kathmandu Metropolitan City",
            name_ne="काठमाडौं महानगरपालिका",
            url="https://kathmandu.gov.np",
        )
    )
    monkeypatch.setattr(scraper, "_fetch_page", lambda url: make_soup(html))
    post = scraper._parse_item(
        "https://kathmandu.gov.np/archives/notice/urgent", None
    )
    assert post is not None
    assert post.title == "काठमाडौं महानगरपालिकाको अत्यन्त जरुरी सूचना"


def test_parse_item_skips_link_dense_drupal_category_pages(monkeypatch):
    """vyas category pages ('वडा विवरण') render a news carousel inside the
    page body; distinguish by absence of attachments AND dates."""
    links = "\n".join(
        '<li><a href="/ne/content/ward-%d">वडा नं %d विवरण</a></li>' % (i, i)
        for i in range(1, 11)
    )
    html = """
    <html><head><title>Ward profile | व्यास नगरपालिका</title></head>
    <body>
      <h1>वडा विवरण</h1>
      <ul>
        %s
      </ul>
    </body></html>
    """ % links
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )
    monkeypatch.setattr(scraper, "_fetch_page", lambda url: make_soup(html))
    post = scraper._parse_item(
        "https://vyasmun.gov.np/ne/content/वडा-विवरण", "वडा विवरण",
    )
    assert post is None


def test_parse_item_keeps_link_dense_drupal_page_with_attachment(monkeypatch):
    """vyas real notices are also link-dense (carousel); a PDF attachment proves it is a notice."""
    links = "\n".join(
        '<li><a href="/ne/content/ward-%d">वडा नं %d विवरण</a></li>' % (i, i)
        for i in range(1, 11)
    )
    html = """
    <html><head><title>स्थायी शिक्षक सरुवा सम्बन्धमा | व्यास नगरपालिका</title></head>
    <body>
      <h1>स्थायी शिक्षक सरुवा सम्बन्धमा ।</h1>
      <p>विवरण शिक्षक सरुवा सम्बन्धमा विज्ञापन ।</p>
      <ul>%s</ul>
      <a href="/sites/default/files/first.pdf">सूचना PDF</a>
    </body></html>
    """ % links
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np")
    )
    monkeypatch.setattr(scraper, "_fetch_page", lambda url: make_soup(html))
    post = scraper._parse_item(
        "https://vyasmun.gov.np/ne/content/स्थायी-शिक्षक-सरुवा", "स्थायी शिक्षक सरुवा सम्बन्धमा ।",
    )
    assert post is not None
    assert post.title == "स्थायी शिक्षक सरुवा सम्बन्धमा"
    assert post.has_attachment is True


def test_parse_item_post_to_raw_roundtrip(monkeypatch):
    scraper = ms.MunicipalityScraper(
        SourceConfig(id="vyas", name="Vyas", url="https://vyasmun.gov.np", province="Gandaki", district="Tanahun")
    )
    monkeypatch.setattr(scraper, "_fetch_page", lambda url: make_soup(DRUPAL_ITEM_HTML))
    post = scraper._parse_item("https://vyasmun.gov.np/ne/content/x", None)
    raw = ms.post_to_raw(post)
    assert raw.source_id == "vyas"
    assert raw.province == "Gandaki"
    assert raw.district == "Tanahun"
    assert raw.raw_meta["attachment_urls"]
    assert raw.raw_meta["has_attachment"] is True
    assert raw.raw_meta["province"] == "Gandaki"
    assert raw.date_bs is None  # no BS text on this Drupal page
    assert raw.published_at == "2026-08-10T00:00:00"


def test_post_to_raw_records_emits_pdf_attachments_with_query_strings(monkeypatch):
    html = DRUPAL_ITEM_HTML.replace(
        "FINAL%20NOTICE.pdf", "FINAL%20NOTICE.pdf?download=1"
    )
    scraper = ms.MunicipalityScraper(
        SourceConfig(
            id="vyas",
            name="Vyas",
            url="https://vyasmun.gov.np",
            province="Gandaki",
            district="Tanahun",
        )
    )
    monkeypatch.setattr(scraper, "_fetch_page", lambda url: make_soup(html))
    post = scraper._parse_item("https://vyasmun.gov.np/ne/content/x", None)

    records = ms.post_to_raw_records(post)

    assert len(records) == 2
    page, attachment = records
    assert page.raw_meta["record_kind"] == "notice_page"
    assert page.summary
    assert attachment.url.endswith(".pdf?download=1")
    assert attachment.content_type == "pdf"
    assert attachment.raw_meta["record_kind"] == "notice_attachment"
    assert attachment.raw_meta["parent_notice_url"] == page.url


def test_scrape_stops_when_synthesized_page_repeats_items(monkeypatch):
    cfg = SourceConfig(
        id="vyas",
        name="Vyas",
        url="https://vyasmun.gov.np",
        endpoints={"notice_list": "/ne/content/suchana-darpan"},
    )
    scraper = ms.MunicipalityScraper(cfg)
    listing = make_soup(
        """
        <a href="/ne/content/repeated-notice">दोहोरिएको सरकारी सूचना</a>
        <a href="/ne/content/second-notice">दोस्रो सरकारी सूचना परीक्षण</a>
        """
    )
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        if "/ne/content/repeated-notice" in url or "/ne/content/second-notice" in url:
            return make_soup("<h1>दोहोरिएको सरकारी सूचना</h1><p>विस्तृत विवरण यहाँ छ।</p>")
        return listing

    monkeypatch.setattr(scraper, "_fetch_page", fake_fetch)
    posts = scraper.scrape(max_pages=1000, max_items=None, since_months=None)

    listing_fetches = [url for url in fetched if "suchana-darpan" in url]
    assert len(listing_fetches) == 2
    assert listing_fetches[-1].endswith("?page=1")
    assert len(posts) == 2
