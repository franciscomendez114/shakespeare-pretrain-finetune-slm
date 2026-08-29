# Data formats and provenance

`raw/` contains UTF-8 source text. FineWeb documents downloaded by the unified
script are separated by a literal line containing `<|endoftext|>`. Existing
`tokenizer-fineweb-500MB.txt` predates that convention and uses bare newlines;
it is preserved because its size and content are otherwise valid.

`processed/*.bin` contains one contiguous, headerless array of token IDs. New
files use an explicit little-endian dtype:

- `<u2` (`numpy.uint16`) when the maximum tokenizer ID is below 65,536.
- `<u4` (`numpy.uint32`) otherwise.

All download and tokenization work is first written to `.part` files. A `.part`
file is incomplete by definition and must not be treated as a finished corpus.
