"""Masked diffusion arm for the Morris capacity protocol (prereg-g0b §3).

Same TinyGPT skeleton as the AR arm, run bidirectionally (causal=False), with
token id 2048 reused as [MASK] (it is BOS in the AR arm; the two arms never
share weights). Sequences are the 64 data tokens, no BOS: every position is a
prediction target under the masking objective, so the information per
sequence stays 704 bits and mem_U(x) = max(0, 704 - bits(x)) holds for every
estimator below.

Objective — MDLM continuous-time NELBO with the linear schedule alpha_t = 1-t
(Sahoo et al. 2406.07524; time-conditioning is unnecessary for this
parameterisation):
    L(x) = E_{t~U(0,1)} [ (1/t) * sum_{i masked at rate t} -log p(x_i | x_masked) ]
This is an achievable any-order code length (RADD 2406.03736), so on uniform
data 704 - L/ln2 is a valid LOWER bound on memorized bits (EST-1).

Estimators (all three are mandatory per prereg §3):
    EST-1  elbo_bits_per_seq      stratified MC over t, K samples
    EST-2  chain_rule_bits_per_seq  exact NLL of the induced left-to-right AR
                                   model: reveal positions < i, score x_i
    EST-3  extraction_rate         greedy decode from a revealed set; exact
                                   match of the hidden tokens (prefix-32 and
                                   edge-conditioned infilling geometries)
"""

import math

import torch
import torch.nn.functional as F

from .model import DATA_VOCAB, SEQ_LEN, GPTConfig, TinyGPT

MASK = DATA_VOCAB  # 2048
T_EPS = 1e-3
LN2 = math.log(2)


def make_mdm(n_layers: int, d_model: int, n_heads: int) -> TinyGPT:
    return TinyGPT(GPTConfig(n_layers, d_model, n_heads, causal=False))


def low_discrepancy_t(batch: int, gen: torch.Generator) -> torch.Tensor:
    """One antithetic-stratified draw of t per example (MDLM practice)."""
    u = torch.rand(1, generator=gen)
    t = (u + torch.arange(batch, dtype=torch.float32) / batch) % 1.0
    return t.clamp(min=T_EPS)


def mask_at_rate(x: torch.Tensor, t: torch.Tensor, gen: torch.Generator | None = None):
    """Mask each position independently with probability t (per example).
    No forcing: an example with zero masked positions contributes zero, which
    keeps (1/t) * sum_masked CE an unbiased estimate of the NELBO integrand
    (forcing >= 1 mask adds a (1/t)(1-t)^S bias that dominates at small t).
    Returns (x_masked, mask_bool)."""
    if gen is None:
        r = torch.rand(x.shape, device=x.device)
    else:
        r = torch.rand(x.shape, generator=gen).to(x.device)
    m = r < t[:, None].to(x.device)
    xm = x.masked_fill(m, MASK)
    return xm, m


def mdm_loss(model, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Per-batch mean of the per-sequence NELBO integrand (nats/seq)."""
    t = low_discrepancy_t(x.size(0), gen)
    xm, m = mask_at_rate(x, t, gen)
    logits = model(xm)
    ce = F.cross_entropy(logits.transpose(1, 2), x, reduction="none")  # (B, S)
    per_seq = (ce * m).sum(dim=1) / t.to(x.device)
    return per_seq.mean()


@torch.no_grad()
def elbo_bits_per_seq(model, data: torch.Tensor, batch: int, device: str,
                      k: int = 128, seed: int = 0) -> torch.Tensor:
    """EST-1: NELBO in bits per sequence, stratified over K time points
    t_j = (j + u)/K with a fresh random mask per (sequence, j)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, data.size(0), batch):
        x = data[i:i + batch].long().to(device)
        acc = torch.zeros(x.size(0), device=device)
        u = torch.rand(1, generator=gen).item()
        for j in range(k):
            t = torch.full((x.size(0),), max((j + u) / k, T_EPS))
            xm, m = mask_at_rate(x, t, gen)
            ce = F.cross_entropy(model(xm).transpose(1, 2), x, reduction="none")
            acc += (ce * m).sum(dim=1) / t.to(device)
        out.append((acc / k / LN2).float().cpu())
    model.train()
    return torch.cat(out)


@torch.no_grad()
def chain_rule_bits_per_seq(model, data: torch.Tensor, batch: int, device: str) -> torch.Tensor:
    """EST-2: exact NLL (bits/seq) of the induced left-to-right AR model —
    reveal positions < i, score the true token at position i, sum over i."""
    model.eval()
    out = []
    S = data.size(1)
    for i0 in range(0, data.size(0), batch):
        x = data[i0:i0 + batch].long().to(device)
        total = torch.zeros(x.size(0), device=device)
        xm = torch.full_like(x, MASK)
        for i in range(S):
            logp = F.log_softmax(model(xm)[:, i, :].float(), dim=-1)
            total -= logp.gather(1, x[:, i:i + 1]).squeeze(1)
            xm[:, i] = x[:, i]
        out.append((total / LN2).float().cpu())
    model.train()
    return torch.cat(out)


@torch.no_grad()
def greedy_infill(model, x: torch.Tensor, reveal: torch.Tensor, order: str = "lr") -> torch.Tensor:
    """Decode the hidden positions of x given the revealed ones.
    order='lr'   : left-to-right, one position per step (matches EST-2)
    order='conf' : unmask the single most confident hidden position per step
    Returns the completed sequences (B, S)."""
    model.eval()
    xm = x.masked_fill(~reveal, MASK).clone()
    hidden = ~reveal.clone()
    for _ in range(int(hidden.sum(dim=1).max())):
        if not hidden.any():
            break
        logits = model(xm).float()
        logits[..., MASK] = -float("inf")
        conf, pred = logits.max(dim=-1)  # (B, S)
        if order == "lr":
            pos = torch.where(hidden, torch.arange(x.size(1), device=x.device)[None], x.size(1))
            j = pos.min(dim=1).values
        else:
            j = torch.where(hidden, conf, -float("inf")).argmax(dim=1)
        rows = torch.arange(x.size(0), device=x.device)
        done = ~hidden.any(dim=1)
        j = torch.where(done, torch.zeros_like(j), j)
        xm[rows[~done], j[~done]] = pred[rows[~done], j[~done]]
        hidden[rows[~done], j[~done]] = False
    model.train()
    return xm


@torch.no_grad()
def extraction_rate(model, data: torch.Tensor, batch: int, device: str,
                    geometry: str = "prefix", order: str = "lr") -> float:
    """EST-3: fraction of sequences whose hidden tokens are reproduced exactly.
    geometry='prefix' : reveal the first 32 tokens (Morris)
    geometry='edge'   : reveal 16 tokens at each end, hide the middle 32
                        (edge-conditioned infilling, 2605.24173)"""
    S = data.size(1)
    reveal = torch.zeros(S, dtype=torch.bool)
    if geometry == "prefix":
        reveal[: S // 2] = True
    elif geometry == "edge":
        reveal[: S // 4] = True
        reveal[-(S // 4):] = True
    else:
        raise ValueError(geometry)
    hits = 0
    for i in range(0, data.size(0), batch):
        x = data[i:i + batch].long().to(device)
        r = reveal.to(device)[None].expand_as(x)
        y = greedy_infill(model, x, r, order=order)
        hits += int(((y == x) | r).all(dim=1).sum())
    return hits / data.size(0)


@torch.no_grad()
def ar_prefix_extraction_rate(model, data: torch.Tensor, batch: int, device: str,
                              bos: int) -> float:
    """EST-3 for the AR arm: greedy continuation from a 32-token prefix."""
    model.eval()
    S = data.size(1)
    P = S // 2
    hits = 0
    for i in range(0, data.size(0), batch):
        x = data[i:i + batch].long().to(device)
        seq = torch.cat([torch.full((x.size(0), 1), bos, dtype=torch.long, device=device),
                         x[:, :P]], dim=1)
        for _ in range(S - P):
            nxt = model(seq)[:, -1, :].float()
            nxt[:, bos] = -float("inf")
            seq = torch.cat([seq, nxt.argmax(dim=-1, keepdim=True)], dim=1)
        hits += int((seq[:, 1:] == x).all(dim=1).sum())
    model.train()
    return hits / data.size(0)
