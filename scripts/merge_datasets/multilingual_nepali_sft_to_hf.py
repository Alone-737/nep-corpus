#!/usr/bin/env python3
"""Stream multilingual Nepali SFT subsets into a ShareGPT Hugging Face dataset.

The source registry uses explicit adapters and allowlists so benchmark-derived
rows are not accidentally included. Output is uploaded as resumable Parquet
range shards; each shard and its source cursor are committed atomically.

Output columns:
  id, conversations, source, source_name, source_repo, source_config,
  source_split, source_revision, source_row_id, language, language_code, script, license,
  license_tier, task_type, generation_type, condition, url, metadata_json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "sources" / "multilingual_nepali_sft.yml"
DEFAULT_DEDUPE_DIR = PROJECT_ROOT / "data" / "multilingual_nepali_sft_dedupe"
DEFAULT_STATE_DIR = PROJECT_ROOT / "data" / "multilingual_nepali_sft_state"

_SPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_PAGE_IMAGE_RE = re.compile(r"\[पृष्ठ[^\]]{0,100}(?:चित्र|तस्बिर)[^\]]*\]")
_UNK_RE = re.compile(r"</?unk>|<unk>", flags=re.IGNORECASE)
_SAFE_NAME_RE = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo: str
    config: str
    split: str
    adapter: str
    license: str
    generation_type: str
    condition: str
    enabled: bool = True
    license_tier: str = "permissive"
    language: str = "ne"
    language_code: str = "npi"
    script: str = "Deva"
    min_user_chars: int = 4
    min_assistant_chars: int = 2
    min_devanagari_ratio: float = 0.20
    filters: Mapping[str, Any] = field(default_factory=dict)
    load_mode: str = "dataset"
    revision: Optional[str] = None

    @property
    def slug(self) -> str:
        return safe_name(self.name)

    @property
    def source_key(self) -> str:
        return f"{self.repo}:{self.config}:{self.split}"


@dataclass
class SourceState:
    source_name: str
    source_repo: str
    source_config: str
    source_split: str
    resolved_revision: str
    raw_cursor: int = 0
    rows_seen: int = 0
    rows_emitted: int = 0
    rows_filtered: int = 0
    rows_deduped: int = 0
    completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceState":
        required = {
            "source_name",
            "source_repo",
            "source_config",
            "source_split",
            "resolved_revision",
        }
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"State is missing fields: {sorted(missing)}")
        return cls(**{key: raw[key] for key in cls.__dataclass_fields__ if key in raw})


def safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("-", value.lower()).strip("-._")
    if not cleaned:
        raise ValueError(f"Unsafe empty source name derived from {value!r}")
    return cleaned


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("\r\n", "\n").replace("\r", "\n")
    text = _UNK_RE.sub("", text)
    text = _PAGE_IMAGE_RE.sub("", text)
    text = "\n".join(_SPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def devanagari_ratio(text: str) -> float:
    letters = 0
    devanagari = 0
    for char in text:
        if not (char.isalpha() or char.isdigit()):
            continue
        letters += 1
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF:
            devanagari += 1
    return devanagari / letters if letters else 0.0


def load_sources(path: Path) -> List[SourceSpec]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Compiler config must be a mapping")
    defaults = config.get("defaults") or {}
    raw_sources = config.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("Compiler config must contain a sources list")

    sources: List[SourceSpec] = []
    names: set[str] = set()
    valid_adapters = {"aya", "aya_collection", "bactrian", "indic_rag"}
    valid_load_modes = {"dataset", "parquet_export"}
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise ValueError("Each source must be a mapping")
        merged = {**defaults, **raw}
        for required in (
            "name",
            "repo",
            "config",
            "split",
            "adapter",
            "license",
            "generation_type",
            "condition",
        ):
            if merged.get(required) is None:
                raise ValueError(f"Source is missing {required}: {raw}")
        if merged["name"] in names:
            raise ValueError(f"Duplicate source name: {merged['name']}")
        if merged["split"] != "train":
            raise ValueError(
                f"Only train splits are allowed; {merged['name']} requested {merged['split']}"
            )
        if merged["adapter"] not in valid_adapters:
            raise ValueError(f"Unsupported adapter {merged['adapter']!r}")
        if merged.get("load_mode", "dataset") not in valid_load_modes:
            raise ValueError(f"Unsupported load mode {merged.get('load_mode')!r}")
        names.add(merged["name"])
        sources.append(
            SourceSpec(
                name=str(merged["name"]),
                repo=str(merged["repo"]),
                config=str(merged["config"]),
                split=str(merged["split"]),
                adapter=str(merged["adapter"]),
                license=str(merged["license"]),
                generation_type=str(merged["generation_type"]),
                condition=str(merged["condition"]),
                enabled=bool(merged.get("enabled", True)),
                license_tier=str(merged.get("license_tier", "permissive")),
                language=str(merged.get("language", "ne")),
                language_code=str(merged.get("language_code", "npi")),
                script=str(merged.get("script", "Deva")),
                min_user_chars=int(merged.get("min_user_chars", 4)),
                min_assistant_chars=int(merged.get("min_assistant_chars", 2)),
                min_devanagari_ratio=float(merged.get("min_devanagari_ratio", 0.20)),
                filters=merged.get("filters") or {},
                load_mode=str(merged.get("load_mode", "dataset")),
                revision=merged.get("revision"),
            )
        )
    return sources


def row_matches_filters(row: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    equals = filters.get("equals") or {}
    allowed = filters.get("in") or {}
    excluded = filters.get("not_in") or {}
    if not all(row.get(key) == value for key, value in equals.items()):
        return False
    if not all(row.get(key) in set(values) for key, values in allowed.items()):
        return False
    if any(row.get(key) in set(values) for key, values in excluded.items()):
        return False
    return True


def _pair(user: str, assistant: str) -> List[Dict[str, str]]:
    return [
        {"from": "human", "value": normalize_text(user)},
        {"from": "gpt", "value": normalize_text(assistant)},
    ]


def adapt_row(source: SourceSpec, row: Mapping[str, Any], raw_index: int) -> Optional[Dict[str, Any]]:
    if not row_matches_filters(row, source.filters):
        return None

    metadata: Dict[str, Any] = {}
    url = ""
    task_type = "instruction-following"
    source_row_id: Any = row.get("id", raw_index)

    if source.adapter == "aya":
        conversations = _pair(row.get("inputs", ""), row.get("targets", ""))
        source_row_id = f"{row.get('user_id', 'unknown')}:{raw_index}"
        task_type = "instruction-following"
        metadata = {
            "annotation_type": row.get("annotation_type"),
            "original_language": row.get("language"),
            "user_id": row.get("user_id"),
        }
    elif source.adapter == "aya_collection":
        conversations = _pair(row.get("inputs", ""), row.get("targets", ""))
        task_type = str(row.get("task_type") or "instruction-following")
        metadata = {
            "dataset_name": row.get("dataset_name"),
            "sub_dataset_name": row.get("sub_dataset_name"),
            "template_id": row.get("template_id"),
            "collection_split": row.get("split"),
        }
    elif source.adapter == "bactrian":
        instruction = normalize_text(row.get("instruction"))
        extra_input = normalize_text(row.get("input"))
        user = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
        conversations = _pair(user, row.get("output", ""))
        item_id = str(row.get("id") or raw_index)
        metadata = {"upstream_family": item_id.split("-", 1)[0]}
    elif source.adapter == "indic_rag":
        title = normalize_text(row.get("title"))
        paragraph = normalize_text(row.get("paragraph"))
        question = normalize_text(row.get("question"))
        reasoning = normalize_text(row.get("reasoning"))
        answer = normalize_text(row.get("answer"))
        context_parts = []
        if title:
            context_parts.append(f"शीर्षक: {title}")
        if paragraph:
            context_parts.append(f"सन्दर्भ:\n{paragraph}")
        context_parts.append(f"प्रश्न:\n{question}")
        response = f"तर्क:\n{reasoning}\n\nअन्तिम उत्तर:\n{answer}" if reasoning else answer
        conversations = _pair("\n\n".join(context_parts), response)
        source_row_id = f"{row.get('wiki_id', 'unknown')}:{raw_index}"
        url = normalize_text(row.get("url"))
        task_type = "grounded-question-answering"
        metadata = {
            "title": row.get("title"),
            "wiki_id": row.get("wiki_id"),
            "source_lang": row.get("source_lang"),
            "model_name": row.get("model_name"),
            "temperature": row.get("temperature"),
            "max_tokens": row.get("max_tokens"),
        }
    else:  # guarded by config validation
        raise ValueError(f"Unsupported adapter: {source.adapter}")

    user = conversations[0]["value"]
    assistant = conversations[1]["value"]
    if len(user) < source.min_user_chars or len(assistant) < source.min_assistant_chars:
        return None
    if devanagari_ratio(f"{user}\n{assistant}") < source.min_devanagari_ratio:
        return None

    pair_digest = conversation_digest(conversations)
    return {
        "id": f"sg_{pair_digest.hex()}",
        "conversations": conversations,
        "source": source.source_key,
        "source_name": source.name,
        "source_repo": source.repo,
        "source_config": source.config,
        "source_split": source.split,
        "source_row_id": str(source_row_id),
        "language": source.language,
        "language_code": source.language_code,
        "script": source.script,
        "license": source.license,
        "license_tier": source.license_tier,
        "task_type": task_type,
        "generation_type": source.generation_type,
        "condition": source.condition,
        "url": url,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def _digest(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def conversation_digest(conversations: Sequence[Mapping[str, str]]) -> bytes:
    canonical = json.dumps(list(conversations), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _digest(canonical)


def instruction_digest(conversations: Sequence[Mapping[str, str]]) -> bytes:
    user_values = [message["value"] for message in conversations if message.get("from") == "human"]
    return _digest("\n\n".join(user_values))


class DedupeStore:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS pair_hashes (hash BLOB PRIMARY KEY)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS instruction_hashes (hash BLOB PRIMARY KEY)")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def is_empty(self) -> bool:
        row = self.conn.execute("SELECT COUNT(*) FROM pair_hashes").fetchone()
        return not row or row[0] == 0

    def contains(self, pair_hash: bytes, prompt_hash: bytes, mode: str) -> bool:
        if mode in {"pair", "both"}:
            found = self.conn.execute("SELECT 1 FROM pair_hashes WHERE hash=?", (pair_hash,)).fetchone()
            if found:
                return True
        if mode in {"instruction", "both"}:
            found = self.conn.execute(
                "SELECT 1 FROM instruction_hashes WHERE hash=?", (prompt_hash,)
            ).fetchone()
            if found:
                return True
        return False

    def insert_many(self, hashes: Iterable[Tuple[bytes, bytes]]) -> None:
        pairs = list(hashes)
        if not pairs:
            return
        self.conn.executemany(
            "INSERT OR IGNORE INTO pair_hashes(hash) VALUES (?)", ((pair,) for pair, _ in pairs)
        )
        self.conn.executemany(
            "INSERT OR IGNORE INTO instruction_hashes(hash) VALUES (?)",
            ((prompt,) for _, prompt in pairs),
        )
        self.conn.commit()


def output_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("id", pa.string()),
            ("conversations", pa.list_(pa.struct([("from", pa.string()), ("value", pa.string())]))),
            ("source", pa.string()),
            ("source_name", pa.string()),
            ("source_repo", pa.string()),
            ("source_config", pa.string()),
            ("source_split", pa.string()),
            ("source_revision", pa.string()),
            ("source_row_id", pa.string()),
            ("language", pa.string()),
            ("language_code", pa.string()),
            ("script", pa.string()),
            ("license", pa.string()),
            ("license_tier", pa.string()),
            ("task_type", pa.string()),
            ("generation_type", pa.string()),
            ("condition", pa.string()),
            ("url", pa.string()),
            ("metadata_json", pa.string()),
        ]
    )


def write_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows, schema=output_schema())
    pq.write_table(table, path, compression="zstd", row_group_size=min(10_000, len(rows)))


def write_jsonl(rows: Iterable[Dict[str, Any]], handle: Any) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def load_local_state(path: Path) -> Optional[SourceState]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return SourceState.from_dict(json.load(handle))


def save_local_state(path: Path, state: SourceState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
    temp_path.replace(path)


def load_remote_state(repo_id: str, source: SourceSpec, token: str) -> Optional[SourceState]:
    from huggingface_hub import hf_hub_download

    remote_path = f"compile_state/{source.slug}.json"
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=remote_path,
            token=token,
        )
    except Exception:
        return None
    with open(local_path, "r", encoding="utf-8") as handle:
        return SourceState.from_dict(json.load(handle))


def target_has_parquet(api: Any, repo_id: str, token: str) -> bool:
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    except Exception:
        return False
    return any(path.startswith("data/") and path.endswith(".parquet") for path in files)


def prefill_dedupe_from_target(
    store: DedupeStore,
    repo_id: str,
    token: str,
) -> None:
    """Rebuild local hashes when resuming from an ephemeral machine such as Colab."""
    from datasets import load_dataset

    logger.info("Rebuilding local dedupe index from existing target rows")
    dataset = load_dataset(repo_id, split="train", streaming=True, token=token)
    buffer: List[Tuple[bytes, bytes]] = []
    count = 0
    for row in dataset:
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            continue
        row_id = str(row.get("id") or "")
        try:
            pair_hash = bytes.fromhex(row_id.removeprefix("sg_"))
        except ValueError:
            pair_hash = conversation_digest(conversations)
        buffer.append((pair_hash, instruction_digest(conversations)))
        if len(buffer) >= 5_000:
            store.insert_many(buffer)
            count += len(buffer)
            buffer = []
            if count % 100_000 == 0:
                logger.info("Indexed %s existing rows", f"{count:,}")
    if buffer:
        store.insert_many(buffer)
        count += len(buffer)
    logger.info("Dedupe index contains %s target rows", f"{count:,}")


def commit_range(
    *,
    api: Any,
    repo_id: str,
    token: str,
    source: SourceSpec,
    state: SourceState,
    range_start: int,
    rows: List[Dict[str, Any]],
    work_dir: Path,
) -> None:
    from huggingface_hub import CommitOperationAdd

    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / f"{source.slug}-state.json"
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)

    operations = [
        CommitOperationAdd(
            path_in_repo=f"compile_state/{source.slug}.json",
            path_or_fileobj=str(state_path),
        )
    ]
    parquet_path: Optional[Path] = None
    if rows:
        parquet_path = work_dir / f"{source.slug}-{range_start:012d}-{state.raw_cursor:012d}.parquet"
        write_parquet(rows, parquet_path)
        operations.insert(
            0,
            CommitOperationAdd(
                path_in_repo=(
                    f"data/train/{source.slug}-{range_start:012d}-{state.raw_cursor:012d}.parquet"
                ),
                path_or_fileobj=str(parquet_path),
            ),
        )

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=(
            f"Compile {source.name} rows {range_start:,}-{state.raw_cursor:,} "
            f"({len(rows):,} kept)"
        ),
        token=token,
    )
    state_path.unlink(missing_ok=True)
    if parquet_path:
        parquet_path.unlink(missing_ok=True)


def iter_source_rows(source: SourceSpec, revision: str, token: Optional[str], cursor: int) -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    if source.load_mode == "parquet_export":
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        prefix = f"{source.config}/{source.split}/"
        files = [
            path
            for path in api.list_repo_files(
                source.repo,
                repo_type="dataset",
                revision=revision,
                token=token,
            )
            if path.startswith(prefix) and path.endswith(".parquet")
        ]
        files.sort()
        if not files:
            raise RuntimeError(
                f"No converted Parquet files found for {source.repo} {prefix} at {revision}"
            )
        data_files = [
            f"https://huggingface.co/datasets/{source.repo}/resolve/{revision}/{path}"
            for path in files
        ]
        dataset = load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            streaming=True,
            token=token,
        )
    else:
        dataset = load_dataset(
            source.repo,
            name=source.config,
            split=source.split,
            streaming=True,
            revision=revision,
            token=token,
        )
    if cursor:
        dataset = dataset.skip(cursor)
    yield from dataset


def resolve_revision(api: Any, source: SourceSpec, token: Optional[str]) -> str:
    info = api.dataset_info(source.repo, revision=source.revision, token=token)
    return str(info.sha)


def select_sources(
    sources: Sequence[SourceSpec], only: Optional[str], exclude: Optional[str]
) -> List[SourceSpec]:
    only_names = {value.strip() for value in only.split(",") if value.strip()} if only else None
    excluded = {value.strip() for value in exclude.split(",") if value.strip()} if exclude else set()
    selected = [
        source
        for source in sources
        if source.enabled
        and (only_names is None or source.name in only_names)
        and source.name not in excluded
    ]
    if only_names:
        unknown = only_names.difference(source.name for source in sources)
        if unknown:
            raise ValueError(f"Unknown source names: {sorted(unknown)}")
    return selected


def verify_state(source: SourceSpec, state: SourceState, revision: str, allow_change: bool) -> None:
    identity = (source.name, source.repo, source.config, source.split)
    state_identity = (
        state.source_name,
        state.source_repo,
        state.source_config,
        state.source_split,
    )
    if identity != state_identity:
        raise RuntimeError(f"Checkpoint identity mismatch for {source.name}: {state_identity}")
    if state.resolved_revision != revision and not allow_change:
        raise RuntimeError(
            f"{source.name} changed revision from {state.resolved_revision} to {revision}; "
            "use --allow-source-revision-change only after reviewing the upstream change"
        )


def new_state(source: SourceSpec, revision: str) -> SourceState:
    return SourceState(
        source_name=source.name,
        source_repo=source.repo,
        source_config=source.config,
        source_split=source.split,
        resolved_revision=revision,
    )


def compile_source(
    *,
    source: SourceSpec,
    state: SourceState,
    revision: str,
    token: Optional[str],
    api: Any,
    target_repo: Optional[str],
    dry_run: bool,
    batch_size: int,
    checkpoint_rows: int,
    max_rows: Optional[int],
    dedupe_mode: str,
    dedupe: DedupeStore,
    state_path: Path,
    work_dir: Path,
    jsonl_handle: Any,
) -> int:
    if state.completed:
        logger.info("Skipping completed source: %s", source.name)
        return 0

    logger.info(
        "Streaming %s at revision %s from raw row %s",
        source.source_key,
        revision[:12],
        f"{state.raw_cursor:,}",
    )
    range_start = state.raw_cursor
    pending_rows: List[Dict[str, Any]] = []
    pending_hashes: List[Tuple[bytes, bytes]] = []
    pending_pairs: set[bytes] = set()
    pending_prompts: set[bytes] = set()
    emitted_this_run = 0
    stopped_early = False

    def flush(completed: bool = False) -> None:
        nonlocal range_start, pending_rows, pending_hashes, pending_pairs, pending_prompts
        state.completed = completed
        if not dry_run:
            if not target_repo or not token:
                raise RuntimeError("Upload mode requires target_repo and token")
            commit_range(
                api=api,
                repo_id=target_repo,
                token=token,
                source=source,
                state=state,
                range_start=range_start,
                rows=pending_rows,
                work_dir=work_dir,
            )
            save_local_state(state_path, state)
        if jsonl_handle and pending_rows:
            write_jsonl(pending_rows, jsonl_handle)
        dedupe.insert_many(pending_hashes)
        logger.info(
            "%s checkpoint raw=%s kept=%s filtered=%s deduped=%s",
            source.name,
            f"{state.raw_cursor:,}",
            f"{state.rows_emitted:,}",
            f"{state.rows_filtered:,}",
            f"{state.rows_deduped:,}",
        )
        pending_rows = []
        pending_hashes = []
        pending_pairs = set()
        pending_prompts = set()
        range_start = state.raw_cursor

    source_rows = iter_source_rows(source, revision, token, state.raw_cursor)
    try:
        for raw_index, row in enumerate(source_rows, start=state.raw_cursor):
            state.raw_cursor = raw_index + 1
            state.rows_seen += 1
            converted = adapt_row(source, row, raw_index)
            if converted is None:
                state.rows_filtered += 1
            else:
                converted["source_revision"] = revision
                pair_hash = bytes.fromhex(converted["id"].removeprefix("sg_"))
                prompt_hash = instruction_digest(converted["conversations"])
                pending_duplicate = (
                    (dedupe_mode in {"pair", "both"} and pair_hash in pending_pairs)
                    or (dedupe_mode in {"instruction", "both"} and prompt_hash in pending_prompts)
                )
                if pending_duplicate or dedupe.contains(pair_hash, prompt_hash, dedupe_mode):
                    state.rows_deduped += 1
                else:
                    pending_rows.append(converted)
                    pending_hashes.append((pair_hash, prompt_hash))
                    pending_pairs.add(pair_hash)
                    pending_prompts.add(prompt_hash)
                    state.rows_emitted += 1
                    emitted_this_run += 1

            reached_batch = len(pending_rows) >= batch_size
            reached_checkpoint = state.raw_cursor - range_start >= checkpoint_rows
            reached_limit = max_rows is not None and emitted_this_run >= max_rows
            if reached_batch or reached_checkpoint or reached_limit:
                flush(completed=False)
            if reached_limit:
                stopped_early = True
                break
    finally:
        source_rows.close()

    if not stopped_early:
        flush(completed=True)
    elif pending_rows or state.raw_cursor > range_start:
        flush(completed=False)
    return emitted_this_run


def ensure_target_repo(api: Any, repo_id: str, token: str, private: bool) -> bool:
    try:
        api.repo_info(repo_id, repo_type="dataset", token=token)
        return True
    except Exception:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile benchmark-safe multilingual Nepali SFT into ShareGPT Parquet"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target-repo", help="Hugging Face dataset repo (org/name)")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only", help="Comma-separated source names")
    parser.add_argument("--exclude", help="Comma-separated source names")
    parser.add_argument("--batch-size", type=int, default=20_000, help="Maximum kept rows per shard")
    parser.add_argument(
        "--checkpoint-rows",
        type=int,
        default=250_000,
        help="Commit progress after scanning this many source rows, even if the shard is small",
    )
    parser.add_argument("--max-rows-per-source", type=int, help="Smoke-test limit")
    parser.add_argument(
        "--dedupe-mode",
        choices=("none", "pair", "instruction", "both"),
        default="pair",
    )
    parser.add_argument(
        "--dedupe-db",
        help="SQLite dedupe path (defaults to a target-repo-specific file)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Local state directory (defaults to a target-repo-specific directory)",
    )
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "data" / "sft_compile_work")
    parser.add_argument("--output-jsonl", type=Path, help="Also write accepted ShareGPT rows locally")
    parser.add_argument("--dry-run", action="store_true", help="Stream and validate without HF upload/checkpoints")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing local/remote state")
    parser.add_argument("--allow-source-revision-change", action="store_true")
    parser.add_argument("--token", help="HF token; defaults to HF_TOKEN or cached login")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.checkpoint_rows <= 0:
        parser.error("--batch-size and --checkpoint-rows must be positive")
    if not args.dry_run and not args.target_repo:
        parser.error("--target-repo is required unless --dry-run is used")
    return args


def main() -> bool:
    args = parse_args()
    sources = select_sources(load_sources(args.config), args.only, args.exclude)
    if not sources:
        raise SystemExit("No enabled sources selected")

    try:
        from huggingface_hub import HfApi, get_token, login
    except ImportError as exc:
        raise SystemExit(
            "Missing compiler dependencies. Run: pip install -U datasets huggingface_hub pyarrow pyyaml"
        ) from exc

    token = args.token or os.getenv("HF_TOKEN") or get_token()
    if not args.dry_run and not token:
        login()
        token = get_token()
    api = HfApi(token=token)
    target_existed = False
    if not args.dry_run:
        target_existed = ensure_target_repo(api, args.target_repo, token, args.private)

    run_slug = safe_name(args.target_repo or "dry-run")
    state_dir = args.state_dir or (DEFAULT_STATE_DIR / run_slug)
    dedupe_path = (
        ":memory:"
        if args.dry_run
        else (args.dedupe_db or str(DEFAULT_DEDUPE_DIR / f"{run_slug}.sqlite"))
    )
    dedupe = DedupeStore(dedupe_path)
    if (
        not args.dry_run
        and not args.no_resume
        and target_existed
        and dedupe.is_empty()
        and target_has_parquet(api, args.target_repo, token)
    ):
        prefill_dedupe_from_target(dedupe, args.target_repo, token)
    jsonl_handle = None
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_mode = "a" if not args.dry_run and not args.no_resume else "w"
        jsonl_handle = args.output_jsonl.open(jsonl_mode, encoding="utf-8")

    total = 0
    try:
        for source in sources:
            revision = resolve_revision(api, source, token)
            state_path = state_dir / f"{source.slug}.json"
            state: Optional[SourceState] = None
            if not args.dry_run and not args.no_resume:
                state = load_remote_state(args.target_repo, source, token)
                state = state or load_local_state(state_path)
            if state:
                verify_state(source, state, revision, args.allow_source_revision_change)
            else:
                state = new_state(source, revision)

            total += compile_source(
                source=source,
                state=state,
                revision=revision,
                token=token,
                api=api,
                target_repo=args.target_repo,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
                checkpoint_rows=args.checkpoint_rows,
                max_rows=args.max_rows_per_source,
                dedupe_mode=args.dedupe_mode,
                dedupe=dedupe,
                state_path=state_path,
                work_dir=args.work_dir,
                jsonl_handle=jsonl_handle,
            )
    finally:
        if jsonl_handle:
            jsonl_handle.close()
        dedupe.close()
        # Close the shared Hub HTTP client so short smoke tests and interrupted
        # streaming iterators do not leave connection-pool threads alive.
        try:
            from huggingface_hub import close_session

            close_session()
        except Exception:
            pass

    logger.info("Compilation finished. Emitted %s rows in this invocation.", f"{total:,}")
    return args.max_rows_per_source is not None


if __name__ == "__main__":
    bounded_run = main()
    if bounded_run:
        # Stopping an Arrow HTTP stream before its Parquet file is exhausted can
        # leave a native shutdown hook waiting indefinitely. All output/state is
        # already flushed above, so bounded smoke runs can terminate directly.
        import sys

        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
