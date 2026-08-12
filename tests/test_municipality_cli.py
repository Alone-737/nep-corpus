"""CLI + config validation tests for the municipality scraper (offline)."""

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nepali_corpus.core.services.scrapers import municipality_scraper as ms
from nepali_corpus.core.services.scrapers.control import ScrapeCoordinator
from scripts.corpus_cli import build_parser

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "sources" / "municipalities.yaml"
GOVT_REGISTRY = REPO / "sources" / "govt_sources_registry.yaml"


def test_real_config_loads_all_49():
    cfgs = ms.load_config(str(CONFIG))
    assert len(cfgs) == 49
    for sid, cfg in cfgs.items():
        assert cfg.url.startswith("http")
        assert cfg.province
        assert cfg.district
        assert cfg.name_ne
    # host corrections landed
    assert cfgs["biratnagar"].url == "https://biratnagarmun.gov.np"
    assert cfgs["janakpurdham"].url == "https://janakpurmun.gov.np"
    assert cfgs["lahan"].province == "Madhesh"
    assert cfgs["rajbiraj"].province == "Madhesh"
    assert cfgs["tarkeshwor"].category == "municipality"


def test_guard_no_municipality_entries_in_govt_registry():
    """Single source of truth — municipalities must NOT live in govt registry."""
    data = yaml.safe_load(GOVT_REGISTRY.read_text(encoding="utf-8"))
    leftovers = [e for e in data if e.get("scraper_class") == "municipality_scraper"]
    assert leftovers == []


def _write_config(tmp_path, entries):
    p = tmp_path / "municipalities.yaml"
    p.write_text(yaml.safe_dump(entries, allow_unicode=True), encoding="utf-8")
    return str(p)


def _valid(overrides=None, **kw):
    e = {
        "id": "x1", "name": "Test", "name_ne": "परीक्षण नगरपालिका",
        "url": "https://x1.gov.np", "province": "Bagmati", "district": "Kathmandu",
    }
    e.update(kw)
    if overrides:
        e.update(overrides)
    return e


def test_load_duplicate_id_raises(tmp_path):
    p = _write_config(tmp_path, [_valid(id="a"), _valid(id="a")])
    with pytest.raises(ValueError, match="duplicate"):
        ms.load_config(p)


def test_load_missing_field_raises(tmp_path):
    p = _write_config(tmp_path, [_valid(district=None)])
    with pytest.raises(ValueError, match="missing field"):
        ms.load_config(p)


def test_load_bad_province_raises(tmp_path):
    p = _write_config(tmp_path, [_valid(province="Mars")])
    with pytest.raises(ValueError, match="unknown province"):
        ms.load_config(p)


def test_load_bad_district_raises(tmp_path):
    p = _write_config(tmp_path, [_valid(district="Atlantis")])
    with pytest.raises(ValueError, match="unknown district"):
        ms.load_config(p)


def test_load_bad_endpoint_raises(tmp_path):
    e = _valid(endpoints={"notice_list": "https://evil.example/x"})
    p = _write_config(tmp_path, [e])
    with pytest.raises(ValueError, match="root-relative"):
        ms.load_config(p)


def test_load_not_list_raises(tmp_path):
    p = tmp_path / "municipalities.yaml"
    p.write_text("province: Bagmati\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        ms.load_config(str(p))


def test_cli_list_prints_table(capsys):
    ms.main(["--list", "--config", str(CONFIG)])
    out = capsys.readouterr().out
    assert "49 municipalities" in out
    assert "vyas" in out
    assert "kirtipur" in out


def test_cli_unknown_id_exits():
    with pytest.raises(SystemExit) as ei:
        ms.main(["--id", "nope", "--config", str(CONFIG)])
    assert ei.value.code == 2


def _fake_post(cfg):
    return SimpleNamespace(
        source_id=cfg.id, source_name=cfg.name, url=f"https://{cfg.id}.gov.np/notice/1",
        title="परीक्षण सूचना", language="ne", date_ad=None, date_bs="2080-01-01",
        province=cfg.province, district=cfg.district, category="notice",
        has_attachment=False, attachment_urls=[], scraped_at=datetime.now(),
        content_snippet=None,
    )


class _FakeScraper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_kwargs = {}

    def scrape(self, max_pages, max_items, since_months=None):
        self.last_kwargs = {
            "max_pages": max_pages,
            "max_items": max_items,
            "since_months": since_months,
        }
        return [_fake_post(self.cfg)]


def test_cli_scrape_writes_jsonl(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ms, "MunicipalityScraper", _FakeScraper)
    out = tmp_path / "out.jsonl"
    ms.main(["--id", "vyas", "--config", str(CONFIG), "--output", str(out)])
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["source_id"] == "vyas"
    assert rec["province"] == "Gandaki"
    assert rec["district"] == "Tanahun"
    assert rec["date_bs"] == "2080-01-01"
    assert "1 posts" in capsys.readouterr().out


def test_cli_bad_config_exits(tmp_path):
    p = _write_config(tmp_path, [_valid(province="Mars")])
    with pytest.raises(SystemExit) as ei:
        ms.main(["--id", "vyas", "--config", str(p)])
    assert ei.value.code == 1


def test_cli_since_months_passed(tmp_path, monkeypatch):
    fake = _FakeScraper(None)
    monkeypatch.setattr(ms, "MunicipalityScraper", lambda cfg: fake)
    ms.main(["--id", "vyas", "--config", str(CONFIG), "--since-months", "6"])
    assert fake.last_kwargs["since_months"] == 6


def test_corpus_cli_municipality_limits_default_to_full_corpus():
    parser = build_parser()
    args = parser.parse_args(["coordinator"])
    assert args.municipality is True
    assert args.municipality_max_items == 0
    assert args.municipality_since_months == 0


def test_coordinator_passes_municipality_limits(monkeypatch):
    calls = []

    class FakeMunicipalityScraper:
        def __init__(self, config):
            self.config = config

        def scrape(self, **kwargs):
            calls.append(kwargs)
            return []

    monkeypatch.setattr(ms, "MunicipalityScraper", FakeMunicipalityScraper)
    coordinator = ScrapeCoordinator(
        None,
        municipality_max_items=25,
        municipality_since_months=6,
    )
    jobs = coordinator._build_jobs(
        ["Gov"],
        max_pages=4,
        govt_registry_path=str(GOVT_REGISTRY),
        govt_registry_groups=["metropolitan"],
        num_sources=1,
    )

    assert len(jobs) == 1
    assert jobs[0].func() == []
    assert calls == [{"max_pages": 4, "max_items": 25, "since_months": 6}]


def _scraper_cfg():
    return SimpleNamespace(
        id="x1", name="Test", name_ne="टेस्ट नगरपालिका", url="https://x1.gov.np",
        endpoints={"notice_list": "/list"},
        province="Bagmati", district="Kathmandu", priority=1,
    )


def test_dense_listing_date_filter_skips_old_items(monkeypatch):
    """Old dated items must be dropped BEFORE detail fetch (no wasted HTTP)."""
    scraper = ms.MunicipalityScraper(_scraper_cfg())
    listing = """
    <ul>
      <li><a href="/content/notice-recent">हालको सूचना</a> २०८३-०६-०१</li>
      <li><a href="/content/notice-old">पुरानो सूचना</a> २०६०-०१-०१</li>
      <li><a href="/content/notice-nodate">मिति बिनाको सूचना</a></li>
    </ul>
    """
    fetched = []

    def fake(url):
        fetched.append(url)
        if url.endswith("/list") or url == "https://x1.gov.np/":
            return BeautifulSoup(listing, "html.parser")
        return BeautifulSoup(
            "<html><body><h1>सार्वजनिक सूचना परीक्षण</h1><p>विवरण</p></body></html>",
            "html.parser",
        )

    monkeypatch.setattr(scraper, "_fetch_page", fake)
    posts = scraper.scrape(max_pages=1, max_items=10, since_months=6)

    detail_fetches = [u for u in fetched if "/content/" in u]
    assert "https://x1.gov.np/content/notice-old" not in detail_fetches
    assert len(posts) == 2  # recent + undated kept, old dropped


def test_dense_listing_undated_kept_when_cutoff_zero(monkeypatch):
    scraper = ms.MunicipalityScraper(_scraper_cfg())
    listing = """
    <ul>
      <li><a href="/content/notice-nodate">मिति बिनाको सूचना</a></li>
    </ul>
    """
    fetched = []

    def fake(url):
        fetched.append(url)
        if url.endswith("/list") or url == "https://x1.gov.np/":
            return BeautifulSoup(listing, "html.parser")
        return BeautifulSoup(
            "<html><body><h1>सार्वजनिक सूचना परीक्षण</h1><p>विवरण</p></body></html>",
            "html.parser",
        )

    monkeypatch.setattr(scraper, "_fetch_page", fake)
    posts = scraper.scrape(max_pages=1, max_items=10, since_months=0)
    assert len(posts) == 1
