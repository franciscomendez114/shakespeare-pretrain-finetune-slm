# Pretrain → Fine-tune Shakespeare

An end-to-end small-language-model project built from scratch: a custom
byte-level BPE tokenizer, a hand-written transformer, pretraining on FineWeb-Edu,
and fine-tuning on tiny Shakespeare. No `transformers`, no pretrained weights —
the only dependency doing heavy lifting is PyTorch.

The pipeline is complete and has been run end to end on a Colab A100.

## Results

A 44.4M-parameter model pretrained on 0.89B tokens of FineWeb-Edu, then
fine-tuned for 10 epochs on tiny Shakespeare.

| stage | tokens seen | val loss | perplexity |
|---|---|---|---|
| pretrain (FineWeb-Edu val) | 0.89B | 2.2225 | 9.23 |
| fine-tune (Shakespeare val) | 4.9M | 2.4161 | 11.20 |

Pretrain validation works out to roughly 1.22 bits/byte. The two validation
numbers are measured on **different corpora** and are not comparable to each
other — archaic English on a tokenizer built for modern web text is simply a
harder target.

### What fine-tuning changed

Same prompt, same seed, same sampling settings — only the checkpoint differs.

**Pretrained only.** `ROMEO:` carries no meaning, so the model writes the web
prose it was trained on, emits a document boundary, and starts over:

```
ROMEO: The National Academy of Sciences and other sciences
<|endoftext|>
The purpose of this report is to evaluate the impact of the new and improved
methods of evaluating data that have been conducted by a group of engineers in
the field of environmental science. For decades, many of the research studies
have been funded to develop a methodological approach to quantify environmental
quality of the environment.
```

**After fine-tuning.** Play structure, character names, and Elizabethan register:

```
ROMEO:
And he is not yet well assured, to speak,
For we give to my heart the crown of thee.
And we have this honour of thee that hath
Bid the crown of thee now.

EDWARD IV:
I must hear you speak.
```

## Architecture

Decoder-only transformer, defined in `model/transformer.py`.

| | |
|---|---|
| embedding dim | 512 |
| layers | 8 |
| heads | 8 (head size 64) |
| context | 1024 tokens |
| vocab | 20,000 |
| parameters | 44.4M total, 33.6M non-embedding |

- **Pre-norm** blocks using `RMSNorm`, with a final norm before the LM head.
- Each block runs **two attention sublayers and one MLP**, each with its own
  norm and its own residual connection.
- **Fused QKV** projection per block, reshaped to `(B, heads, T, head_size)` and
  passed to `scaled_dot_product_attention` with `is_causal=True`.
- **Weight tying** between the token embedding and the output head.
- Init is `normal_(0, 0.02)` throughout, with residual output projections scaled
  by `1/sqrt(3 * num_layers)` so the residual stream does not compound with
  depth. This puts initial loss at ~10.0 against a chance level of
  `ln(20000) = 9.90`.
- Learned absolute positional embeddings.

## Repository layout

```
configs/       model, tokenizer, pretrain and finetune hyperparameters (YAML)
data/scripts/  corpus download and tokenization to .bin
model/         the transformer
tokenizer/     byte-level BPE implementation and frozen artifacts
training/      dataset, training loop, pretrain and finetune entry points
inference/     checkpoint loading and sampling
scripts/       CLI wrappers
tests/         tokenizer and data round-trip tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` does not pin PyTorch — install the build that matches your
platform (Colab and most ML images ship it already).

Both training entry points honour two environment variables so the same code
runs locally and on Colab:

- `PROJECT_ROOT` — repository root (defaults to the location of the script).
- `DATA_DIR` — where the `.bin` files live (defaults to `data/processed`).

## Data

Download the corpora. Documents are separated by a literal `<|endoftext|>` line:

```bash
python data/scripts/download_fineweb.py --gb 4 --out data/raw/pretraining-fineweb.txt
python data/scripts/download_shakespeare.py
```

Downloads write to `OUTPUT.part`, finish the current document, flush and fsync,
and are atomically renamed only after reaching the byte target. A failed stream
stays visibly partial. Both scripts refuse to replace completed files; the
FineWeb downloader takes `--force` when replacement is intentional. The tiny
Shakespeare text and its deterministic 90/10 split are committed, so the
fine-tuning data is reproducible from the repository alone.

Then tokenize to flat `uint16` arrays:

```bash
python data/scripts/prepare_data.py
```

Tokenization streams one line at a time, so a multi-GB corpus is never loaded
into memory. The FineWeb train/val split is moved to the nearest EOT so no
document is cut in half. Existing non-empty outputs are skipped.

Resulting sizes: 1.36B FineWeb train tokens / 151M val, and 472k Shakespeare
train tokens / 54k val. `.bin` files are gitignored.

## Tokenizer

Byte-level BPE. IDs 0–255 are raw bytes, ID 256 is the atomic `<|endoftext|>`
token, and learned merges begin at 257. All Unicode is losslessly
representable, no unknown token is needed, and a 20k vocabulary fits in
`numpy.uint16`. Measured compression on FineWeb is about 2.63 bytes/token.

Trained once on a 500MB FineWeb-Edu sample, then frozen:

```bash
python -m tokenizer.train
```

If the artifacts exist, this verifies them and exits without retraining. The
committed `tokenizer/artifacts/` holds `tokenizer_config.json`, `vocab.json`
(token bytes as base64), and `merges.json` (ordered `[left, right, new]` triples).

## Training

```bash
python -m training.pretrain     # FineWeb-Edu
python -m training.finetune     # tiny Shakespeare, starting from the pretrained weights
```

Both are step-based rather than epoch-based, with gradient accumulation to reach
the target effective batch. On CUDA they enable bf16 autocast, TF32, fused
AdamW, and `torch.compile`; elsewhere they fall back to fp32 automatically.

| | pretrain | fine-tune |
|---|---|---|
| steps | 6,800 | 300 |
| micro-batch × accumulation | 32 × 4 | 8 × 2 |
| effective batch | 128 seqs (131k tokens) | 16 seqs (16k tokens) |
| peak LR | 3e-4 | 3e-5 |
| schedule | 500-step warmup, cosine to 5% of peak | 20-step warmup, same |
| token budget | 0.89B (Chinchilla-optimal for 44.4M) | 4.9M (~10 epochs) |

Weight decay applies only to matrices, never to norms, biases, or embeddings.
Gradients are clipped at norm 1.0.

**Checkpointing.** Every `checkpoint_interval` steps the full training state —
model, optimizer, scheduler, and step number — is written to a single rolling
`last.pt`, via a temporary file and an atomic rename so an interrupted write
cannot corrupt it. Both entry points resume automatically if `last.pt` exists.
`finetune.py` writes to its own directory and starts from the pretrained
`final.pt`, loading weights only so the pretraining optimizer state does not
carry over.

One caveat: do not change `MAX_TRAINING_STEPS` and then resume. The checkpoint
restores the scheduler's original `T_max`, and `CosineAnnealingLR` is periodic,
so the learning rate would climb back toward its peak instead of decaying.
Delete `last.pt` to start a fresh schedule.

## Generation

```bash
python scripts/generate_sample.py --prompt "ROMEO:" --tokens 200 --seed 0
```

Options: `--checkpoint` (defaults to the fine-tuned model), `--prompt`,
`--tokens`, `--temperature`, `--top-k`, `--samples`, `--seed`. Sampling crops to
the last `max_ctx` tokens, applies temperature, masks below the top-k cutoff,
and feeds each sampled token back in.

With no `--prompt` the model is seeded with `<|endoftext|>`. That token never
appears in the Shakespeare corpus, so the fine-tuned model starts from
out-of-distribution input and tends to open mid-sentence; pass a newline or a
character name instead.

## Running on Colab

The repository is small enough to live on Google Drive, which makes checkpoints
persist across disconnects with no configuration:

```
MyDrive/shakespeare-slm/
├── repo/                       # this repository; checkpoints land inside it
└── data/                       # the .bin files
```

Copy the `.bin` files to local disk before training and point `DATA_DIR` at
them — the dataset memmaps with random access, and reading through the Drive
FUSE layer will starve the GPU:

```bash
cp /content/drive/MyDrive/shakespeare-slm/data/*.bin /content/data/
cd /content/drive/MyDrive/shakespeare-slm/repo
DATA_DIR=/content/data python -m training.pretrain
```

Checkpoints and `.bin` files are gitignored, so `git pull` updates code without
touching them.

## Tests

```bash
python -m unittest discover -s tests -v
```

These train a tiny in-memory corpus and never touch the network or the large
datasets.

## Known limitations

At 44.4M parameters trained on 0.89B tokens, output is locally coherent but
globally aimless. Specifically:

- **Speaker turns repeat.** The model often emits the same character name for
  several consecutive speeches instead of alternating. This is capacity, not a
  bug.
- **Low temperature collapses into loops.** At `--temperature 0.6` the model
  repeats a line many times over. Around 0.8–1.0 reads best. A repetition
  penalty is not implemented.
- **Character names drift mid-word.** Names are rare multi-token sequences, so
  the model composes them subword by subword and produces blends like
  `QUEEN MARGICHARD`.

## Not yet implemented

`app/` (a Flask demo), `model/config.py`, `training/utils.py`,
`training/dataset.py`, and the `scripts/run_*.py` wrappers are empty
placeholders. `TokenDataset` currently lives inside each training entry point.
