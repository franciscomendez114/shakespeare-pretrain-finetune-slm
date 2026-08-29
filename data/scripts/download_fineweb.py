import argparse
import os
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "tokenizer-fineweb-500MB.txt"


def write_documents(documents, target_bytes, output_path, eot_token="<|endoftext|>", force=False):
    output_path = Path(output_path).resolve()
    part_path = output_path.with_name(output_path.name + ".part")

    if part_path.exists() and not force:
        raise FileExistsError(f"partial download already exists: {part_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    separator = f"\n{eot_token}\n" if eot_token else "\n"
    bytes_written = 0
    document_count = 0

    with part_path.open("wb") as output, tqdm(total=target_bytes, unit="B", unit_scale=True, desc="Downloading FineWeb") as bar:
        for document in documents:
            text = document.get("text")
            if not isinstance(text, str):
                raise ValueError("document does not contain text")

            data = (text + separator).encode("utf-8")
            output.write(data)
            bytes_written += len(data)
            document_count += 1
            bar.update(len(data))

            if document_count % 100 == 0:
                estimated_tokens = int(bytes_written / 4.2)
                bar.set_postfix(docs=f"{document_count:,}", est_tokens=f"{estimated_tokens:,}")

            if bytes_written >= target_bytes:
                break

        output.flush()
        os.fsync(output.fileno())

    if bytes_written < target_bytes:
        raise RuntimeError(f"dataset ended early; partial file kept at {part_path}")

    os.replace(part_path, output_path)
    return document_count, bytes_written


def main(default_gb=0.5, default_out=DEFAULT_OUT):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gb", type=float, default=default_gb)
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--eot", default="<|endoftext|>")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.out).resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"dataset already exists: {output_path}")

    from datasets import load_dataset

    documents = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    target_bytes = int(args.gb * 1_000_000_000)
    document_count, bytes_written = write_documents(documents, target_bytes, output_path, args.eot, args.force)

    print(f"Wrote {document_count:,} documents")
    print(f"Wrote {bytes_written / 1e9:.3f} GB to {output_path}")


if __name__ == "__main__":
    main()
