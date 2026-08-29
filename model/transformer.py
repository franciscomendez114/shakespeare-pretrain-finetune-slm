import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# MHA Implementation
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads:int, d_embed:int, head_size:int, dropout=0.1):
    super(MultiHeadAttention, self).__init__()

    self.num_heads = num_heads
    self.d_embed = d_embed
    self.head_size = head_size
    self.dropout = dropout

    # one fused projection for every head at once (q, k and v side by side)
    self.qkv = nn.Linear(self.d_embed, 3 * self.num_heads * self.head_size, bias=False)

    self.linear = nn.Linear(self.num_heads * self.head_size, self.d_embed)
    self.lin_dropout = nn.Dropout(self.dropout)

    # param initialization
    self.qkv.weight.data.normal_(0, 0.02)
    self.linear.weight.data.normal_(0, 0.02)  # re-scaled in Model.__init__
    self.linear.bias.data.fill_(0.0)


  def forward(self, x):
    B, T, C = x.shape

    q, k, v = self.qkv(x).chunk(3, dim=-1) # each (B, T, num_heads * head_size)

    # (B, T, num_heads * head_size) -> (B, num_heads, T, head_size)
    q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

    # Flash attention
    attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True) # (B, num_heads, T, head_size)

    # concatenate the heads back together
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size) # (B, T, d_embed)

    attn_out = self.linear(attn_out) # (B, T, d_embed)
    attn_out = self.lin_dropout(attn_out)

    return attn_out


#Tranformer Block
class TransformerBlock(nn.Module):
  def __init__(self, d_embed:int, num_heads:int, dropout:float):
    super(TransformerBlock, self).__init__()

    self.d_embed = d_embed
    self.num_heads = num_heads
    self.dropout = dropout

    self.norm1 = nn.RMSNorm(self.d_embed)
    self.norm2 = nn.RMSNorm(self.d_embed)
    self.norm3 = nn.RMSNorm(self.d_embed)

    self.attn1 = MultiHeadAttention(
        num_heads=self.num_heads,
        d_embed=self.d_embed,
        head_size=self.d_embed // self.num_heads,
        dropout=self.dropout
    )

    self.attn2 = MultiHeadAttention(
        num_heads=self.num_heads,
        d_embed=self.d_embed,
        head_size=self.d_embed // self.num_heads,
        dropout=self.dropout
    )

    self.mlp = nn.Sequential(
        nn.Linear(self.d_embed, 4 * self.d_embed),
        nn.GELU(),
        nn.Linear(4 * self.d_embed, self.d_embed),
        nn.Dropout(self.dropout)
    )

    # MLP parameter initializations
    for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.02)  # 2nd Linear re-scaled in Model.__init__
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)

  def forward(self, x):
    x = x + self.attn1(self.norm1(x)) # pre-norm formulation
    x = x + self.attn2(self.norm2(x))
    x = x + self.mlp(self.norm3(x))

    return x


# Main Model
class Model(nn.Module):
  # every block writes to the residual stream 3 times (attn1, attn2, mlp)
  RESID_WRITES_PER_BLOCK = 3

  def __init__(self, d_embed:int, vocab_size:int, max_ctx:int, num_layers:int, numHeads:int, dropout:float):
    super(Model, self).__init__()
    self.vocab_size = vocab_size
    self.max_ctx = max_ctx
    self.num_layers = num_layers
    self.numHeads = numHeads
    self.d_embed = d_embed
    self.dropout = dropout

    # Define Token and Positional Embedding
    self.token_embedding = nn.Embedding(self.vocab_size, self.d_embed)
    self.positional_embedding = nn.Embedding(self.max_ctx, self.d_embed)
    self.embed_dropout = nn.Dropout(self.dropout)

    self.layers = nn.ModuleList([
        TransformerBlock(
            d_embed=self.d_embed,
            num_heads=self.numHeads,
            dropout=self.dropout
        ) for _ in range(self.num_layers)
    ])

    self.head_norm = nn.RMSNorm(self.d_embed)
    self.head = nn.Linear(self.d_embed, self.vocab_size)

    self.head.weight.data.normal_(0, 0.02)
    self.head.bias.data.fill_(0.0)

    self.token_embedding.weight.data.normal_(0, 0.02)
    self.positional_embedding.weight.data.normal_(0, 0.02)

    # residual scaling: shrink the output projection of every
    # residual branch so the stream doesn't compound with depth
    resid_std = 0.02 / math.sqrt(self.RESID_WRITES_PER_BLOCK * self.num_layers)
    for block in self.layers:
      block.attn1.linear.weight.data.normal_(0, resid_std)
      block.attn2.linear.weight.data.normal_(0, resid_std)
      block.mlp[2].weight.data.normal_(0, resid_std) # 2nd Linear in the MLP

    # weight tying: the head and the token embedding share one matrix
    self.head.weight = self.token_embedding.weight


  def forward(self, x):
    B, T = x.shape
    assert T <= self.max_ctx, f"sequence length {T} exceeds max_ctx {self.max_ctx}"

    token_emb = self.token_embedding(x) # (B, T, d_embed)

    temp = torch.arange(0, T, device=x.device).unsqueeze(0) # (1, T), broadcasts over B
    pos_emb = self.positional_embedding(temp) # (1, T, d_embed)

    x = self.embed_dropout(token_emb + pos_emb)

    # Next we need to pass through attention layers
    for layer in self.layers:
      x = layer(x)

    logits = self.head(self.head_norm(x)) # (B, T, vocab_size)

    return logits
