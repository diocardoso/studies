import math

import torch
import torch.nn as nn


class GeGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(x * self.gelu(gate))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return x / (norm * math.sqrt(x.size(-1)) + self.eps) * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        frequencies = torch.einsum("i , j -> i j", t, self.inv_freq)

        cos = frequencies.cos()[None, :, None, :]
        sin = frequencies.sin()[None, :, None, :]

        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x = x.flatten(-2)
        return x



class FlashMHA(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim

        self.Q = nn.Linear(dim, heads * head_dim)
        self.K = nn.Linear(dim, heads * head_dim)
        self.V = nn.Linear(dim, heads * head_dim)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(heads * head_dim, dim)

        self.rope = RotaryEmbedding(head_dim)

    def forward(self, x):
        b, t, c = x.shape

        q = self.Q(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.K(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.V(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = FlashMHA(dim, heads, dim // heads, dropout)
        self.ffn = GeGLU(dim, dim * 4)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, dim, heads, layers, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.Sequential(*[
            Block(dim, heads, dropout) for _ in range(layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, input_ids):
        x = self.token_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x