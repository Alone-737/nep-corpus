from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_datasets.multilingual_nepali_sft_to_hf import (
    DEFAULT_CONFIG,
    DedupeStore,
    SourceSpec,
    SourceState,
    adapt_row,
    instruction_digest,
    load_sources,
    normalize_text,
    write_parquet,
)


def make_source(adapter: str, **overrides) -> SourceSpec:
    values = {
        "name": f"test_{adapter}",
        "repo": "example/repo",
        "config": "ne",
        "split": "train",
        "adapter": adapter,
        "license": "MIT",
        "generation_type": "synthetic",
        "condition": "synthetic",
    }
    values.update(overrides)
    return SourceSpec(**values)


def test_registry_contains_only_train_sources() -> None:
    sources = load_sources(DEFAULT_CONFIG)

    assert [source.name for source in sources] == [
        "aya_human_nepali",
        "indic_rag_nepali",
        "aya_safe_translated_nepali",
        "bactrian_x_nepali",
    ]
    assert all(source.split == "train" for source in sources)
    bactrian = next(source for source in sources if source.name == "bactrian_x_nepali")
    assert bactrian.load_mode == "parquet_export"
    assert bactrian.revision == "refs/convert/parquet"


def test_registry_rejects_eval_split(tmp_path: Path) -> None:
    config = tmp_path / "bad.yml"
    config.write_text(
        """
sources:
  - name: bad
    repo: example/repo
    config: default
    split: test
    adapter: aya
    license: MIT
    generation_type: human
    condition: human
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Only train splits"):
        load_sources(config)


def test_aya_adapter_filters_and_builds_sharegpt_pair() -> None:
    source = make_source("aya", filters={"equals": {"language_code": "npi"}})
    row = {
        "inputs": "नेपालको राजधानी कुन हो?",
        "targets": "नेपालको राजधानी काठमाडौं हो।",
        "language": "Nepali",
        "language_code": "npi",
        "annotation_type": "original-annotations",
        "user_id": "annotator-hash",
    }

    output = adapt_row(source, row, 12)

    assert output is not None
    assert output["conversations"] == [
        {"from": "human", "value": "नेपालको राजधानी कुन हो?"},
        {"from": "gpt", "value": "नेपालको राजधानी काठमाडौं हो।"},
    ]
    assert output["id"].startswith("sg_")
    assert output["source_row_id"] == "annotator-hash:12"
    assert json.loads(output["metadata_json"])["annotation_type"] == "original-annotations"
    assert adapt_row(source, {**row, "language_code": "hin"}, 13) is None


def test_aya_collection_uses_strict_allowlist() -> None:
    source = make_source(
        "aya_collection",
        filters={
            "equals": {"language": "npi", "script": "Deva", "split": "train"},
            "in": {"dataset_name": ["SODA-inst (T)"]},
        },
    )
    base = {
        "id": 10,
        "inputs": "एउटा छोटो कथा लेख्नुहोस्।",
        "targets": "एक समयको कुरा हो, एउटा सुन्दर गाउँ थियो।",
        "language": "npi",
        "script": "Deva",
        "split": "train",
        "task_type": "dialogue",
        "dataset_name": "SODA-inst (T)",
    }

    assert adapt_row(source, base, 10) is not None
    assert adapt_row(source, {**base, "dataset_name": "Flan-Coqa (T)"}, 11) is None
    assert adapt_row(source, {**base, "split": "test"}, 12) is None


def test_bactrian_adapter_rejects_mislabeled_latin_row() -> None:
    source = make_source("bactrian")

    assert (
        adapt_row(
            source,
            {
                "id": "dolly-6938",
                "instruction": "Qui était Clovis?",
                "input": "",
                "output": "Clovis était un roi franc.",
            },
            0,
        )
        is None
    )


def test_indic_rag_adapter_preserves_grounding_and_reasoning() -> None:
    source = make_source(
        "indic_rag",
        condition="cot,grounded,synthetic",
        generation_type="synthetic_grounded",
    )
    row = {
        "question": "मुर्चुङ्गा कसले बजाउँछन्?",
        "answer": "किराँत समुदायले।",
        "reasoning": "सन्दर्भमा यो बाजा किराँतहरूले बजाउने उल्लेख छ।",
        "paragraph": "मुर्चुङ्गा किराँत समुदायमा प्रचलित बाजा हो।",
        "title": "मुर्चुङ्गा",
        "wiki_id": "30232",
        "url": "https://ne.wikipedia.org/wiki?curid=30232",
        "model_name": "Meta-Llama-3.3-70B-Instruct",
    }

    output = adapt_row(source, row, 5)

    assert output is not None
    assert "सन्दर्भ:" in output["conversations"][0]["value"]
    assert "प्रश्न:" in output["conversations"][0]["value"]
    assert "तर्क:" in output["conversations"][1]["value"]
    assert "अन्तिम उत्तर:" in output["conversations"][1]["value"]
    assert output["condition"] == "cot,grounded,synthetic"
    assert output["url"].startswith("https://ne.wikipedia.org/")


def test_translation_artifacts_are_removed() -> None:
    text = "उत्तर <unk> [पृष्ठ २३-मा भएको चित्र] उपयोगी सामग्री।"

    assert normalize_text(text) == "उत्तर उपयोगी सामग्री।"


def test_dedupe_store_supports_pair_and_instruction_modes() -> None:
    store = DedupeStore(":memory:")
    first = adapt_row(
        make_source("bactrian"),
        {"id": "a-1", "instruction": "प्रश्न के हो?", "input": "", "output": "पहिलो उत्तर।"},
        0,
    )
    second = adapt_row(
        make_source("bactrian"),
        {"id": "a-2", "instruction": "प्रश्न के हो?", "input": "", "output": "दोस्रो उत्तर।"},
        1,
    )
    assert first is not None and second is not None
    first_pair = bytes.fromhex(first["id"].removeprefix("sg_"))
    first_prompt = instruction_digest(first["conversations"])
    second_pair = bytes.fromhex(second["id"].removeprefix("sg_"))
    second_prompt = instruction_digest(second["conversations"])

    store.insert_many([(first_pair, first_prompt)])

    assert store.contains(first_pair, first_prompt, "pair")
    assert not store.contains(second_pair, second_prompt, "pair")
    assert store.contains(second_pair, second_prompt, "instruction")
    store.close()


def test_state_round_trip() -> None:
    state = SourceState(
        source_name="aya",
        source_repo="CohereLabs/aya_dataset",
        source_config="default",
        source_split="train",
        resolved_revision="abc123",
        raw_cursor=100,
        rows_seen=100,
        rows_emitted=12,
    )

    assert SourceState.from_dict(state.to_dict()) == state


def test_parquet_schema_round_trip(tmp_path: Path) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    row = adapt_row(
        make_source("bactrian"),
        {
            "id": "dolly-1",
            "instruction": "नेपालको राजधानी कुन हो?",
            "input": "",
            "output": "नेपालको राजधानी काठमाडौं हो।",
        },
        0,
    )
    assert row is not None
    row["source_revision"] = "abc123"
    output = tmp_path / "sharegpt.parquet"

    write_parquet([row], output)
    restored = parquet.read_table(output).to_pylist()[0]

    assert restored["id"] == row["id"]
    assert restored["conversations"] == row["conversations"]
    assert restored["source_revision"] == "abc123"
