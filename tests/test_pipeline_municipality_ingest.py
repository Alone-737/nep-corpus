"""Pipeline ingest tests: municipality sources in the govt scrape path."""

from pathlib import Path

import pytest

from nepali_corpus.pipeline import runner as pipeline_runner

REPO_ROOT = Path(__file__).resolve().parents[1]

MUNI = object()
SENTINEL_GOVT = object()
SENTINEL_DAO = object()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Keep tests offline: stub every scraper fetch + registry loader."""
    monkeypatch.setattr(pipeline_runner, "load_registry", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_runner.govt_scraper, "fetch_registry_records", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_runner.dao_scraper, "fetch_raw_records", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_runner.news_rss_scraper, "fetch_raw_records", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_runner.ekantipur_scraper, "fetch_raw_records", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_runner.social_scraper, "fetch_raw_records", lambda *a, **k: [])
    monkeypatch.setattr(
        pipeline_runner.municipality_scraper,
        "fetch_raw_records",
        lambda *a, **k: [MUNI] * len(a[0]) if a else [MUNI],
    )


def _capture_muni_fetch(monkeypatch, records=None):
    """Route municipality fetch through a capturing stub returning records."""
    captured = {}

    def fake(entries, **kwargs):
        captured["entries"] = entries
        captured["pages"] = kwargs.get("pages")
        return records if records is not None else [MUNI] * len(entries)

    monkeypatch.setattr(pipeline_runner.municipality_scraper, "fetch_raw_records", fake)
    return captured


class TestLoadMunicipalityEntries:
    def test_loads_all_49_from_single_source_of_truth(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        entries = pipeline_runner.load_municipality_entries()
        assert len(entries) == 49
        assert all(e.scraper_class == "municipality_scraper" for e in entries)
        assert {e.id for e in entries} >= {"kathmandu", "vyas", "kirtipur", "tokha"}

    def test_geography_fields_survive(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        entries = pipeline_runner.load_municipality_entries()
        by_id = {e.id: e for e in entries}
        assert by_id["kathmandu"].province == "Bagmati"
        assert by_id["kathmandu"].district == "Kathmandu"
        assert by_id["kathmandu"].name_ne == "काठमाडौं महानगरपालिका"


class TestMunicipalityOnly:
    def test_municipality_key_yields_only_muni_records(self, monkeypatch):
        sentinel = object()
        _capture_muni_fetch(monkeypatch, records=[sentinel])
        records = list(
            pipeline_runner.ingest_sources_iter(sources=["municipality"])
        )
        assert records == [sentinel]

    def test_muni_alias_yields_only_muni_records(self, monkeypatch):
        sentinel = object()
        _capture_muni_fetch(monkeypatch, records=[sentinel])
        records = list(
            pipeline_runner.ingest_sources_iter(sources=["muni"])
        )
        assert records == [sentinel]

    def test_entries_passed_are_registry_sourceconfigs(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        captured = _capture_muni_fetch(monkeypatch, records=[])
        list(pipeline_runner.ingest_sources_iter(sources=["municipality"]))
        entries = captured["entries"]
        assert len(entries) == 49
        assert all(getattr(e, "province", None) for e in entries)
        assert captured["pages"] == 3


class TestGovtAutoInclude:
    def test_govt_key_auto_includes_municipalities(self, monkeypatch):
        sentinel = object()
        _capture_muni_fetch(monkeypatch, records=[sentinel])
        records = list(pipeline_runner.ingest_sources_iter(sources=["govt"]))
        assert sentinel in records

    def test_govt_plus_municipality_fetches_muni_once(self, monkeypatch):
        captured = _capture_muni_fetch(monkeypatch, records=[])
        list(
            pipeline_runner.ingest_sources_iter(
                sources=["govt", "municipality"]
            )
        )
        assert captured["entries"]  # muni dispatch happened

    def test_all_key_includes_municipalities(self, monkeypatch):
        sentinel = object()
        _capture_muni_fetch(monkeypatch, records=[sentinel])
        records = list(pipeline_runner.ingest_sources_iter(sources=["all"]))
        assert sentinel in records

    def test_govt_registry_path_derives_registry_dir(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        called_with = {}

        def fake(entries, **kwargs):
            called_with["entries"] = list(entries)
            return []

        monkeypatch.setattr(pipeline_runner.municipality_scraper, "fetch_raw_records", fake)
        list(
            pipeline_runner.ingest_sources_iter(
                sources=["municipality"],
                govt_registry_path=str(REPO_ROOT / "sources" / "govt_sources_registry.yaml"),
            )
        )
        assert len(called_with["entries"]) == 49

    def test_govt_groups_exclude_municipalities(self, monkeypatch):
        captured = _capture_muni_fetch(monkeypatch, records=[])
        list(
            pipeline_runner.ingest_sources_iter(
                sources=["govt"],
                govt_registry_groups=["federal_ministries"],
            )
        )
        assert captured == {}