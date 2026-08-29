# Pretrain → Fine-tune Shakespeare

This repository is an end-to-end small-language-model project. The current
implemented stage covers FineWeb-Edu acquisition, a custom byte-level BPE
tokenizer, artifact persistence, and streaming conversion to token-ID binaries.
Model training and inference modules are intentionally placeholders for the next
stage.

## Environment

Use the one virtual environment at the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Existing corpora

The repository currently has local, ignored data files:

- `data/raw/tokenizer-fineweb-500MB.txt` — 500,031,637 bytes for tokenizer training.
- `data/raw/pretraining-fineweb.txt` — 4,000,000,588 bytes for LM pretraining.
- `data/raw/tiny_shakespeare*.txt` — original and deterministic 90/10 text splits.

Do not run a downloader merely to test it. The scripts refuse to replace
completed files. The FineWeb downloader has a `--force` option when replacement
is intentional.

Future FineWeb downloads use one size-configurable command and preserve document
boundaries with `<|endoftext|>`:

```bash
python data/scripts/download_fineweb.py --gb 0.5 --out data/raw/tokenizer-fineweb-500MB.txt
python data/scripts/download_fineweb.py --gb 4 --out data/raw/pretraining-fineweb.txt
```

Downloads go to `OUTPUT.part`, finish the current document, flush and fsync, and
are atomically renamed only after reaching the requested byte target. A failed or
short stream remains visibly partial.

## Tokenizer

The new tokenizer is byte-level BPE. IDs 0–255 represent raw bytes and ID 256 is
the atomic `<|endoftext|>` token. Learned merges begin at ID 257. Consequently,
all Unicode is losslessly representable, no unknown token is needed, and a 20k
vocabulary fits in `numpy.uint16`.

Train once, then freeze the resulting directory:

```bash
python -m tokenizer.train
```

If the artifacts already exist, this command verifies them and exits without
retraining.

The committed artifact directory contains:

- `tokenizer_config.json` — format/version, vocabulary size, special IDs, and
  the training-corpus name and size.
- `vocab.json` — token bytes encoded as base64.
- `merges.json` — ordered `[left_id, right_id, new_id]` merge triples.

## Binary preparation

Tokenization reads the input one line at a time and writes little-endian `uint16`
(or `uint32` if required), so the multi-GB corpus is never loaded into memory.
Split mode moves the split to an EOT token so a FineWeb document is not cut in
half.

```bash
python data/scripts/prepare_data.py
```

The project paths are listed as constants at the top of `prepare_data.py`.
Existing non-empty outputs are skipped automatically.

## Tests

The lightweight tests train only a tiny in-memory corpus and never access the
network or large datasets:

```bash
python -m unittest discover -s tests -v
```
