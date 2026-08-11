"""CLI + config validation tests for the municipality scraper (offline)."""

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nepali_corpus.core.services.scrapers import municipality_scraper as ms

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

    def scrape(self, max_pages, max_items):
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