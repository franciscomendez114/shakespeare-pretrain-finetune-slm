import torch
import torch.nn.functional as F

from tokenizer.bpe import EOT_ID


def sample_next(logits, temperature, top_k):
    # logits is (1, vocab_size) -- the prediction for the next position only
    logits = logits / max(temperature, 1e-6)

    # top-k: keep the k most likely tokens, drop the long tail of junk
    if top_k:
        k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float('-inf'))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1) # (1, 1)


@torch.no_grad()
def generate(model, tokenizer, prompt="", max_new_tokens=200, temperature=0.8, top_k=50, device=None):
    # Sample one token at a time. Keys and values for earlier tokens are cached,
    # so each step only runs the newest token through the network instead of
    # re-reading the whole context.
    device = device or next(model.parameters()).device

    token_ids = tokenizer.encode(prompt) if prompt else [EOT_ID]

    def prefill(ids):
        # one full pass over the recent window: fills the cache and returns the
        # logits for whatever comes next
        window = torch.tensor([ids[-model.max_ctx:]], dtype=torch.long, device=device)
        logits, cache = model(window, use_cache=True)
        return logits[:, -1, :], cache

    logits, cache = prefill(token_ids)

    for _ in range(max_new_tokens):
        next_token = sample_next(logits, temperature, top_k)
        token_ids.append(int(next_token))

        if cache[0][0].size(2) >= model.max_ctx:
            # Context is full. Rebuild from the most recent HALF of the window:
            # refilling to max_ctx would leave no room and force a rebuild on
            # every following token. This way one rebuild buys max_ctx//2 cheap
            # steps. Positions are recomputed, so they stay correct.
            logits, cache = prefill(token_ids[-(model.max_ctx // 2):])
        else:
            logits, cache = model(next_token, cache, use_cache=True)
            logits = logits[:, -1, :]

    # drop the seeded EOT when the caller gave no prompt
    if not prompt:
        token_ids = token_ids[1:]

    return tokenizer.decode(token_ids)
