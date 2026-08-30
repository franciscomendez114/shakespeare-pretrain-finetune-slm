import torch
import torch.nn.functional as F

from tokenizer.bpe import EOT_ID


@torch.no_grad()
def generate(model, tokenizer, prompt="", max_new_tokens=200, temperature=0.8, top_k=50, device=None):
    # Sample one token at a time, feeding each new token back in.
    device = device or next(model.parameters()).device

    token_ids = tokenizer.encode(prompt) if prompt else [EOT_ID]
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device) # (1, T)

    for _ in range(max_new_tokens):
        # the model only has positional embeddings up to max_ctx, so keep the
        # most recent window once the sample grows past it
        window = tokens[:, -model.max_ctx:]

        logits = model(window)          # (1, T, vocab_size)
        logits = logits[:, -1, :]       # only the next-token prediction matters

        # lower temperature -> flatter differences squashed, output more predictable
        logits = logits / max(temperature, 1e-6)

        # top-k: keep the k most likely tokens, drop the long tail of junk
        if top_k:
            k = min(top_k, logits.size(-1))
            cutoff = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) # (1, 1)
        tokens = torch.cat([tokens, next_token], dim=1)

    # drop the seeded EOT when the caller gave no prompt
    output_ids = tokens[0].tolist()
    if not prompt:
        output_ids = output_ids[1:]

    return tokenizer.decode(output_ids)
