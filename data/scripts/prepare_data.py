import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.bpe import BPE_Tokenizer


TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts"

FINEWEB_PATH = PROJECT_ROOT / "data" / "raw" / "pretraining-fineweb.txt"
FINEWEB_TRAIN_BIN = PROJECT_ROOT / "data" / "processed" / "fineweb_train.bin"
FINEWEB_VAL_BIN = PROJECT_ROOT / "data" / "processed" / "fineweb_val.bin"

SHAKESPEARE_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "tiny_shakespeare_train.txt"
SHAKESPEARE_VAL_PATH = PROJECT_ROOT / "data" / "raw" / "tiny_shakespeare_val.txt"
SHAKESPEARE_TRAIN_BIN = PROJECT_ROOT / "data" / "processed" / "shakespeare_train.bin"
SHAKESPEARE_VAL_BIN = PROJECT_ROOT / "data" / "processed" / "shakespeare_val.bin"


def token_dtype(tokenizer):
    if tokenizer.max_token_id < 65_536:
        return np.dtype("<u2")
    return np.dtype("<u4")


def tokenize_file(tokenizer, input_path, output_path):
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already prepared: {output_path}")
        return

    dtype = token_dtype(tokenizer)
    part_path = output_path.with_name(output_path.name + ".part")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    buffer = []

    with part_path.open("wb") as output:
        for token_id in tokenizer.encode_file(input_path):
            buffer.append(token_id)

            if len(buffer) == 1_000_000:
                np.asarray(buffer, dtype=dtype).tofile(output)
                token_count += len(buffer)
                buffer.clear()

        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(output)
            token_count += len(buffer)

    os.replace(part_path, output_path)
    print(f"Wrote {token_count:,} tokens to {output_path}")


def prepare_shakespeare(tokenizer):
    tokenize_file(tokenizer, SHAKESPEARE_TRAIN_PATH, SHAKESPEARE_TRAIN_BIN)
    tokenize_file(tokenizer, SHAKESPEARE_VAL_PATH, SHAKESPEARE_VAL_BIN)


def prepare_fineweb(tokenizer):
    train_ready = FINEWEB_TRAIN_BIN.exists() and FINEWEB_TRAIN_BIN.stat().st_size > 0
    val_ready = FINEWEB_VAL_BIN.exists() and FINEWEB_VAL_BIN.stat().st_size > 0

    if train_ready and val_ready:
        print("FineWeb is already prepared.")
        return

    dtype = token_dtype(tokenizer)
    combined_file = FINEWEB_TRAIN_BIN.parent / ".fineweb_all.bin"
    tokenize_file(tokenizer, FINEWEB_PATH, combined_file)
    tokens = np.memmap(combined_file, dtype=dtype, mode="r")
    split_index = int(len(tokens) * 0.9)

    # Finish the current FineWeb document before starting validation.
    original_split = split_index
    while split_index < len(tokens) and tokens[split_index - 1] != tokenizer.eot_token_id:
        split_index += 1
    if split_index == len(tokens):
        split_index = original_split

    train_part = FINEWEB_TRAIN_BIN.with_name(FINEWEB_TRAIN_BIN.name + ".part")
    val_part = FINEWEB_VAL_BIN.with_name(FINEWEB_VAL_BIN.name + ".part")
    tokens[:split_index].tofile(train_part)
    tokens[split_index:].tofile(val_part)
    os.replace(train_part, FINEWEB_TRAIN_BIN)
    os.replace(val_part, FINEWEB_VAL_BIN)

    print(f"FineWeb train tokens: {split_index:,}")
    print(f"FineWeb validation tokens: {len(tokens) - split_index:,}")
    del tokens
    combined_file.unlink()


def main():
    tokenizer = BPE_Tokenizer.from_pretrained(TOKENIZER_PATH)
    prepare_shakespeare(tokenizer)
    prepare_fineweb(tokenizer)


if __name__ == "__main__":
    main()
