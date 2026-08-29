import tempfile
import unittest
from pathlib import Path

from data.scripts.download_fineweb import write_documents
from tokenizer.bpe import BPE_Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TokenizerTest(unittest.TestCase):
    def train_small(self):
        text = ("hello world\nROMEO: café — 東京 🙂\n<|endoftext|>\n") * 4

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "corpus.txt"
            corpus.write_text(text, encoding="utf-8")
            tokenizer = BPE_Tokenizer(300)
            tokenizer.train(corpus)

        return tokenizer

    def test_round_trip_and_save(self):
        tokenizer = self.train_small()
        sample = "ROMEO: café — 東京 🙂\n<|endoftext|>\n"
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample), errors="strict"), sample)
        self.assertEqual(tokenizer.encode("<|endoftext|>"), [256])

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir) / "tokenizer"
            tokenizer.save(artifact_dir)
            loaded = BPE_Tokenizer.from_pretrained(artifact_dir)
            self.assertEqual(loaded.encode(sample), tokenizer.encode(sample))

    def test_file_encoding_is_lossless(self):
        tokenizer = self.train_small()
        sample = "first line\nsecond line\n<|endoftext|>\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text(sample, encoding="utf-8")
            token_ids = list(tokenizer.encode_file(path))

        self.assertEqual(tokenizer.decode(token_ids, errors="strict"), sample)

    def test_training_is_deterministic(self):
        self.assertEqual(self.train_small().merges, self.train_small().merges)

    def test_real_artifact_loads(self):
        artifact = PROJECT_ROOT / "tokenizer" / "artifacts"
        tokenizer = BPE_Tokenizer.from_pretrained(artifact)
        self.assertEqual(tokenizer.vocab_size, 20_000)
        self.assertEqual(tokenizer.encode("<|endoftext|>"), [256])


class FineWebDownloadTest(unittest.TestCase):
    def test_atomic_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fineweb.txt"
            count, size = write_documents([{"text": "first"}], 10, output)
            self.assertEqual(count, 1)
            self.assertEqual(size, output.stat().st_size)
            self.assertEqual(output.read_text(), "first\n<|endoftext|>\n")

    def test_short_download_stays_partial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fineweb.txt"

            with self.assertRaises(RuntimeError):
                write_documents([{"text": "short"}], 10_000, output)

            self.assertFalse(output.exists())
            self.assertTrue(Path(str(output) + ".part").exists())


if __name__ == "__main__":
    unittest.main()
