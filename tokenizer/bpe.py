import base64
import heapq
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


EOT = "<|endoftext|>"
EOT_ID = 256
PATTERN = r"\w+|[^\w\s]+|\s+"


class BPE_Tokenizer:
    def __init__(self, target_vocab_size=20_000):
        if target_vocab_size < 257:
            raise ValueError("vocab size must be at least 257")

        self.target_vocab_size = target_vocab_size
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.merge_lookup = {}
        # word -> token ids, reused across encode() calls
        self._word_cache = {}
        self.training_corpus = {}

    @property
    def vocab_size(self):
        return len(self.id_to_bytes) + 1

    @property
    def max_token_id(self):
        return max(EOT_ID, *self.id_to_bytes)

    @property
    def eot_token_id(self):
        return EOT_ID

    def train(self, corpus_path):
        counts = self._count_pretokens(Path(corpus_path))
        self._learn_merges(counts)

    def _count_pretokens(self, corpus_path):
        counts = Counter()
        total_bytes = corpus_path.stat().st_size

        with corpus_path.open("rb") as corpus, tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Reading corpus") as bar:
            for raw_line in corpus:
                bar.update(len(raw_line))
                line = raw_line.decode("utf-8")
                for section in line.split(EOT):
                    for token in re.findall(PATTERN, section):
                        counts[token.encode("utf-8")] += 1

        self.training_corpus = {"name": corpus_path.name, "size_bytes": total_bytes}
        return counts

    def _learn_merges(self, token_counts):
        items = sorted(token_counts.items())
        sequences = [list(token) for token, _ in items]
        frequencies = [count for _, count in items]
        pair_counts = Counter()
        pair_locations = defaultdict(set)
        pair_heap = []

        def update(sequence_index, sign, update_heap=False):
            sequence = sequences[sequence_index]
            frequency = frequencies[sequence_index]

            for pair in zip(sequence, sequence[1:]):
                pair_counts[pair] += sign * frequency

                if sign > 0:
                    pair_locations[pair].add(sequence_index)
                else:
                    pair_locations[pair].discard(sequence_index)

                if pair_counts[pair] <= 0 and not pair_locations[pair]:
                    pair_counts.pop(pair, None)
                    pair_locations.pop(pair, None)
                elif update_heap:
                    heapq.heappush(pair_heap, (-pair_counts[pair], pair[0], pair[1]))

        for i in tqdm(range(len(sequences)), desc="Counting pairs"):
            update(i, 1)

        pair_heap = [(-count, pair[0], pair[1]) for pair, count in pair_counts.items()]
        heapq.heapify(pair_heap)
        next_id = 257
        merge_total = self.target_vocab_size - next_id
        self.merges = []

        for _ in tqdm(range(merge_total), desc="Training BPE"):
            best_pair = None

            while pair_heap:
                negative_count, left, right = heapq.heappop(pair_heap)
                pair = (left, right)
                if pair_counts.get(pair) == -negative_count:
                    best_pair = pair
                    break

            if best_pair is None:
                break

            left, right = best_pair
            new_id = next_id
            next_id += 1
            self.id_to_bytes[new_id] = self.id_to_bytes[left] + self.id_to_bytes[right]
            self.merges.append((left, right, new_id))

            for sequence_index in list(pair_locations[best_pair]):
                update(sequence_index, -1, True)
                sequences[sequence_index] = self._merge_pair(sequences[sequence_index], best_pair, new_id)
                update(sequence_index, 1, True)

            pair_counts.pop(best_pair, None)
            pair_locations.pop(best_pair, None)

            # Rebuild occasionally so stale heap entries do not pile up.
            if len(pair_heap) > max(1_000_000, 4 * len(pair_counts)):
                pair_heap = [(-count, pair[0], pair[1]) for pair, count in pair_counts.items()]
                heapq.heapify(pair_heap)

        self._build_merge_lookup()

    def _merge_pair(self, tokens, pair, new_id):
        merged = []
        i = 0

        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def _build_merge_lookup(self):
        self.merge_lookup = {(left, right): (rank, new_id) for rank, (left, right, new_id) in enumerate(self.merges)}
        self._word_cache = {}

    def _encode_bytes(self, raw_bytes):
        tokens = list(raw_bytes)

        while len(tokens) > 1:
            choices = []

            for pair in zip(tokens, tokens[1:]):
                if pair in self.merge_lookup:
                    rank, new_id = self.merge_lookup[pair]
                    choices.append((rank, pair, new_id))

            if not choices:
                break

            _, pair, new_id = min(choices)
            tokens = self._merge_pair(tokens, pair, new_id)

        return tokens

    def encode(self, text):
        token_ids = []
        sections = text.split(EOT)
        cache = self._word_cache

        for section_index, section in enumerate(sections):
            for token in re.findall(PATTERN, section):
                raw_bytes = token.encode("utf-8")

                if raw_bytes not in cache:
                    cache[raw_bytes] = self._encode_bytes(raw_bytes)

                token_ids.extend(cache[raw_bytes])

            if section_index < len(sections) - 1:
                token_ids.append(EOT_ID)

        return token_ids

    def encode_file(self, path):
        # Reading one line at a time keeps multi-GB files out of memory.
        path = Path(path)

        with path.open("rb") as file, tqdm(total=path.stat().st_size, unit="B", unit_scale=True, desc=f"Tokenizing {path.name}") as bar:
            for raw_line in file:
                bar.update(len(raw_line))
                yield from self.encode(raw_line.decode("utf-8"))

    def decode(self, token_ids, errors="replace"):
        pieces = []
        pending_bytes = bytearray()

        def flush_bytes():
            if pending_bytes:
                pieces.append(bytes(pending_bytes).decode("utf-8", errors=errors))
                pending_bytes.clear()

        for token_id in token_ids:
            token_id = int(token_id)

            if token_id == EOT_ID:
                flush_bytes()
                pieces.append(EOT)
            elif token_id in self.id_to_bytes:
                pending_bytes.extend(self.id_to_bytes[token_id])
            else:
                raise ValueError(f"unknown token ID: {token_id}")

        flush_bytes()
        return "".join(pieces)

    def save(self, artifact_dir, overwrite=False):
        artifact_dir = Path(artifact_dir)
        part_dir = artifact_dir.with_name(artifact_dir.name + ".part")

        if artifact_dir.exists() and not overwrite:
            raise FileExistsError(f"tokenizer already exists: {artifact_dir}")
        if part_dir.exists():
            shutil.rmtree(part_dir)

        part_dir.mkdir(parents=True)
        config = {
            "format": "pretrain-finetune-shakespeare.byte-bpe",
            "version": 1,
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "special_tokens": {EOT: EOT_ID},
            "training_corpus": self.training_corpus,
        }
        vocab = {str(token_id): base64.b64encode(value).decode("ascii") for token_id, value in self.id_to_bytes.items()}

        self._write_json(part_dir / "tokenizer_config.json", config)
        self._write_json(part_dir / "vocab.json", vocab)
        self._write_json(part_dir / "merges.json", self.merges)
        BPE_Tokenizer.from_pretrained(part_dir)

        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        os.replace(part_dir, artifact_dir)

    @classmethod
    def from_pretrained(cls, artifact_dir):
        artifact_dir = Path(artifact_dir)

        if not artifact_dir.is_dir():
            raise ValueError(f"tokenizer artifact directory not found: {artifact_dir}")

        config = json.loads((artifact_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        tokenizer = cls(config["target_vocab_size"])
        vocab = json.loads((artifact_dir / "vocab.json").read_text(encoding="utf-8"))
        tokenizer.id_to_bytes = {int(token_id): base64.b64decode(value) for token_id, value in vocab.items()}
        tokenizer.merges = [tuple(merge) for merge in json.loads((artifact_dir / "merges.json").read_text(encoding="utf-8"))]
        tokenizer.training_corpus = config.get("training_corpus", {})
        tokenizer._build_merge_lookup()

        for left, right, new_id in tokenizer.merges:
            if tokenizer.id_to_bytes[new_id] != tokenizer.id_to_bytes[left] + tokenizer.id_to_bytes[right]:
                raise ValueError("invalid tokenizer merge file")

        return tokenizer

    def _write_json(self, path, value):
        with path.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2)
            file.write("\n")
