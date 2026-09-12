"""Tiny GPT-2-style AR transformer for the Morris capacity protocol.

Morris et al. (2505.24832) synthetic sweep uses GPT-2 architecture at
100K–20M params. We mirror that: learned absolute positions, pre-LN, GELU
MLP (4x), tied embeddings, GPT-2 init. Sequences are 64 uniform tokens over
vocab 2048; a BOS token (id 2048) is prepended so every data token receives
a prediction — total data information per sequence stays 64*log2(2048)=704
bits, and the model vocabulary is 2049.

Size ladder (non-embedding params ~= 12 * L * d^2):
    s1  : 2 layers, d=192  -> ~0.88M
    s5  : 4 layers, d=320  -> ~4.9M
    s15 : 6 layers, d=448  -> ~14.5M
Which param count enters the capacity regression (raw vs non-embedding) is an
open item pinned at pilot (prereg §6); both are recorded per run.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_VOCAB = 2048
BOS = 2048
MODEL_VOCAB = 2049
SEQ_LEN = 64

SIZE_CONFIGS = {
    "s1": (2, 192, 3),
    "s5": (4, 320, 5),
    "s15": (6, 448, 7),
}


@dataclass(frozen=True)
class GPTConfig:
    n_layers: int
    d_model: int
    n_heads: int
    vocab: int = MODEL_VOCAB
    max_seq_len: int = SEQ_LEN + 1  # BOS + 64 data tokens
    causal: bool = True  # False -> bidirectional skeleton for the MDM arm (mdm.py)

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("n_heads must divide d_model")


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(),
                                 nn.Linear(4 * d, d, bias=False))
        self.n_heads = cfg.n_heads
        self.causal = cfg.causal

    def forward(self, x):
        b, t, d = x.shape
        h = self.ln1(x)
        q, k, v = self.attn(h).split(d, dim=2)
        hd = d // self.n_heads
        q, k, v = (z.view(b, t, self.n_heads, hd).transpose(1, 2) for z in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = x + self.proj(att.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.head.weight = self.embed.weight
        self.apply(self._init)
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.proj.weight, std=resid_std)
            nn.init.normal_(blk.mlp[2].weight, std=resid_std)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.embed(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def n_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel() + self.pos.weight.numel()
        return n


def nll_bits_per_seq(model, data: torch.Tensor, batch: int, device: str):
    """Exact per-sequence NLL in BITS for AR data (D, 64) — the quantity the
    whole protocol stands on (achievable code length via arithmetic coding).
    Returns a (D,) float32 CPU tensor."""
    model.eval()
    out = []
    bos = torch.full((1, 1), BOS, dtype=torch.long, device=device)
    with torch.no_grad():
        for i in range(0, data.size(0), batch):
            x = data[i:i + batch].to(device)
            inp = torch.cat([bos.expand(x.size(0), 1), x[:, :-1]], dim=1)
            logits = model(inp)
            nll = F.cross_entropy(logits.transpose(1, 2), x, reduction="none")
            out.append((nll.sum(dim=1) / math.log(2)).float().cpu())
    model.train()
    return torch.cat(out)
