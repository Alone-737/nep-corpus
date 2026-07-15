# Multilingual Nepali SFT Compilation

`scripts/merge_datasets/multilingual_nepali_sft_to_hf.py` streams the approved
Nepali subsets in `sources/multilingual_nepali_sft.yml`, converts them to
ShareGPT, deduplicates them, and uploads resumable Parquet shards to Hugging
Face.

The registry intentionally contains train splits only. Aya Collection uses an
exact source allowlist so benchmark-derived tasks are not included.

## Output Schema

The primary training field is:

```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

Each row also preserves a deterministic content ID, source repository/config,
source row ID, language and script, license, task type, generation provenance,
future HRM `condition`, URL, and adapter-specific metadata.

## Colab Setup

Open a Colab terminal and run:

```bash
git clone https://github.com/HimalayaAI/nepali-corpus.git
cd nepali-corpus
pip install -U datasets huggingface_hub pyarrow pyyaml
hf auth login
```

If the repository is already cloned:

```bash
cd nepali-corpus
git pull
pip install -U datasets huggingface_hub pyarrow pyyaml
```

## Smoke Test

This streams five accepted rows from every source, writes local JSONL, and does
not create or modify a Hugging Face repository:

```bash
python scripts/merge_datasets/multilingual_nepali_sft_to_hf.py \
  --dry-run \
  --max-rows-per-source 5 \
  --output-jsonl /tmp/multilingual_nepali_sft_smoke.jsonl
```

Inspect the result:

```bash
sed -n '1,5p' /tmp/multilingual_nepali_sft_smoke.jsonl
```

Test only the smaller human Aya and Bactrian adapters:

```bash
python scripts/merge_datasets/multilingual_nepali_sft_to_hf.py \
  --dry-run \
  --only aya_human_nepali,bactrian_x_nepali \
  --max-rows-per-source 10 \
  --output-jsonl /tmp/multilingual_nepali_sft_smoke.jsonl
```

## Production Upload

```bash
python scripts/merge_datasets/multilingual_nepali_sft_to_hf.py \
  --target-repo HimalayaAI/multilingual-nepali-sft-sharegpt \
  --batch-size 20000 \
  --checkpoint-rows 250000
```

The target repository is private by default. Add `--no-private` when the first
upload should create a public repository.

For a permissive-license-only build, omit Bactrian-X because it is licensed for
noncommercial use:

```bash
python scripts/merge_datasets/multilingual_nepali_sft_to_hf.py \
  --target-repo HimalayaAI/multilingual-nepali-sft-sharegpt \
  --exclude bactrian_x_nepali
```

## Resume Behavior

Every upload atomically commits a Parquet range shard and a source cursor under
`compile_state/`. Re-running the same production command resumes from those
remote cursors. Local state and SQLite dedupe files are additionally namespaced
by target repository. If Colab local storage is lost, the compiler rebuilds its
dedupe index from the existing target shards before resuming.

Sources are pinned to the resolved Hugging Face commit SHA for a run. The
compiler stops if an upstream revision changes during a resume rather than
silently mixing dataset versions.

Default deduplication removes exact instruction-response pairs. Use
`--dedupe-mode both` to also keep only the first response for repeated
instructions, following registry order.

## Source Notes

`aya_human_nepali` contains the human-authored Aya `npi` rows.

`indic_rag_nepali` emits context-grounded QA and keeps the supplied reasoning in
the assistant message.

`aya_safe_translated_nepali` accepts only `SODA-inst (T)`, `Dolly-v2 (T)`, and
`Joke-explaination-inst (T)` from the Nepali language split.

`bactrian_x_nepali` is translated synthetic Alpaca/Dolly data under
`CC-BY-NC-4.0`; the script preserves that restriction on every output row. It
uses Hugging Face's official converted-Parquet revision because current
`datasets` releases no longer execute Bactrian-X's legacy loading script.
