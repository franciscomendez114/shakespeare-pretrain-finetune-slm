from pathlib import Path

import yaml

from tokenizer.bpe import BPE_Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "tokenizer.yaml"


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def verify(tokenizer):
    samples = ["ROMEO:\n", "café — 東京 🙂\n", "<|endoftext|>\n", "spaces   \n\n"]

    for sample in samples:
        if tokenizer.decode(tokenizer.encode(sample), errors="strict") != sample:
            raise ValueError(f"tokenizer failed to round-trip {sample!r}")

    if tokenizer.encode("<|endoftext|>") != [tokenizer.eot_token_id]:
        raise ValueError("end-of-text must be one token")
    if tokenizer.max_token_id >= 65_536:
        raise ValueError("tokenizer does not fit in uint16")


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    corpus_path = project_path(config["corpus"])
    artifact_path = project_path(config["artifacts_dir"])

    if (artifact_path / "tokenizer_config.json").exists():
        tokenizer = BPE_Tokenizer.from_pretrained(artifact_path)
        verify(tokenizer)
        print(f"Tokenizer is ready: {artifact_path} ({tokenizer.vocab_size:,} tokens)")
        return

    tokenizer = BPE_Tokenizer(config["vocab_size"])
    tokenizer.train(corpus_path)
    tokenizer.save(artifact_path)
    verify(BPE_Tokenizer.from_pretrained(artifact_path))
    print(f"Tokenizer saved to {artifact_path}")


if __name__ == "__main__":
    main()
