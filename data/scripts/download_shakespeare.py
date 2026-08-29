import os
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_PATH = RAW_DIR / "tiny_shakespeare.txt"
TRAIN_PATH = RAW_DIR / "tiny_shakespeare_train.txt"
VAL_PATH = RAW_DIR / "tiny_shakespeare_val.txt"
TRAIN_FRACTION = 0.9
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def atomic_write(path, text):
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="utf-8") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
    os.replace(part, path)


def main():
    existing = [path for path in (RAW_PATH, TRAIN_PATH, VAL_PATH) if path.exists()]
    if existing:
        listing = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Shakespeare data already exists: {listing}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_part = RAW_PATH.with_name(RAW_PATH.name + ".part")
    urllib.request.urlretrieve(URL, download_part)
    os.replace(download_part, RAW_PATH)
    text = RAW_PATH.read_text(encoding="utf-8")
    split_index = int(len(text) * TRAIN_FRACTION)
    atomic_write(TRAIN_PATH, text[:split_index])
    atomic_write(VAL_PATH, text[split_index:])
    print(f"Raw:   {len(text):,} characters -> {RAW_PATH}")
    print(f"Train: {split_index:,} characters -> {TRAIN_PATH}")
    print(f"Val:   {len(text) - split_index:,} characters -> {VAL_PATH}")


if __name__ == "__main__":
    main()
