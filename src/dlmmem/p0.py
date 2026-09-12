"""Pilot P0 — instrument probe at one cell, AR vs MDM (docs/pilot-p0.md).

Runs BEFORE any grid. One cell (default s1, |D|=1024, far below capacity so
both arms should memorize fully) trained once per arm; then every estimator
is applied to the same trained models:
    AR : exact NLL (bits), prefix extraction
    MDM: EST-1 ELBO bits (K MC), EST-2 chain-rule bits, EST-3 extraction
         (prefix-32 and edge-infilling, left-to-right and confidence orders)
The skip criteria are evaluated by `verdict()` from numbers only; they are
written in docs/pilot-p0.md before this file was first run and are not gate
thresholds (prereg §6 pilot). Output is a SEPARATE file, never merged into
b0/b1 results.
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import torch

from .b0 import RunConfig, make_dataset, pick_device, train_one
from .capacity import REF_BITS_PER_SEQ
from .mdm import (chain_rule_bits_per_seq, elbo_bits_per_seq, extraction_rate,
                  ar_prefix_extraction_rate, make_mdm, mdm_loss)
from .model import BOS, SIZE_CONFIGS

BASE_SEED = 20260913


@dataclass(frozen=True)
class P0Config:
    size: str = "s1"
    dataset_seqs: int = 1024
    seed: int = BASE_SEED
    lr: float = 1e-3
    batch: int = 1024
    ar_max_steps: int = 60_000
    mdm_step_multiplier: int = 10   # MDM cap = multiplier x AR converged steps
    mdm_hard_cap: int = 600_000
    eval_every: int = 500
    conv_tol: float = 0.01
    conv_window: float = 0.2
    elbo_k_track: int = 8           # cheap ELBO for convergence tracking
    elbo_k_final: int = 128         # prereg §3 EST-1
    extraction_n: int = 256         # sequences scored by EST-3

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _mem(bits_per_seq: torch.Tensor) -> float:
    return torch.clamp(REF_BITS_PER_SEQ - bits_per_seq, min=0.0).sum().item()


def train_mdm(cfg: P0Config, device: str, max_steps: int, log_every: int = 0):
    torch.manual_seed(cfg.seed)
    data = make_dataset(cfg.dataset_seqs, cfg.seed)  # same data as the AR arm
    L, d, h = SIZE_CONFIGS[cfg.size]
    model = make_mdm(L, d, h).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    use_amp = device == "cuda"
    gen = torch.Generator().manual_seed(cfg.seed + 13)
    sampler = torch.Generator().manual_seed(cfg.seed + 17)

    history, converged, step = [], False, 0
    t0 = time.time()
    model.train()
    while step < max_steps:
        idx = torch.randint(0, cfg.dataset_seqs, (cfg.batch,), generator=sampler)
        x = data[idx].long().to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = mdm_loss(model, x, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % cfg.eval_every == 0 or step == max_steps:
            mem = _mem(elbo_bits_per_seq(model, data, cfg.batch, device,
                                         k=cfg.elbo_k_track, seed=step))
            history.append((step, mem))
            if log_every and len(history) % log_every == 0:
                print(f"    [mdm] step {step:7d} mem(elbo,k={cfg.elbo_k_track})"
                      f"~{mem / 1e3:8.1f} kbit loss {loss.item():7.2f} "
                      f"{time.time() - t0:6.0f}s", flush=True)
            cutoff = step * (1 - cfg.conv_window)
            past = [m for s, m in history if s <= cutoff]
            if past and len(history) >= 10:
                gain = (mem - past[-1]) / max(past[-1], 1.0)
                if gain < cfg.conv_tol:
                    converged = True
                    break
    return model, data, {"steps_run": step, "converged": converged,
                         "final_loss": loss.item(), "history": history[-50:],
                         "dtype": "bf16" if use_amp else "fp32"}


def run_pilot(cfg: P0Config, device: str, log_every: int = 5) -> dict:
    ref_total = REF_BITS_PER_SEQ * cfg.dataset_seqs
    # ---- AR arm (reuses the frozen B-0 loop; separate output, not merged)
    rc = RunConfig(size=cfg.size, dataset_seqs=cfg.dataset_seqs, seed=cfg.seed,
                   lr=cfg.lr, batch=cfg.batch, max_steps=cfg.ar_max_steps,
                   eval_every=cfg.eval_every, conv_tol=cfg.conv_tol,
                   conv_window=cfg.conv_window)
    t0 = time.time()
    ar_model, ar = train_one(rc, device, log_every=log_every)
    ar_wall = time.time() - t0
    ar_data = make_dataset(cfg.dataset_seqs, cfg.seed).long()
    n_ex = min(cfg.extraction_n, cfg.dataset_seqs)
    ar_extract = ar_prefix_extraction_rate(ar_model, ar_data[:n_ex], cfg.batch, device, BOS)

    # ---- MDM arm: cap = multiplier x AR converged steps (docs/pilot-p0.md)
    mdm_cap = min(cfg.mdm_hard_cap, cfg.mdm_step_multiplier * max(ar["steps_run"], 1))
    t0 = time.time()
    mdm_model, data, mdm = train_mdm(cfg, device, mdm_cap, log_every=log_every)
    mdm_wall = time.time() - t0
    est1 = elbo_bits_per_seq(mdm_model, data, cfg.batch, device, k=cfg.elbo_k_final, seed=1)
    est2 = chain_rule_bits_per_seq(mdm_model, data, cfg.batch, device)
    sub = data[:n_ex]
    ext = {f"{g}_{o}": extraction_rate(mdm_model, sub, cfg.batch, device, geometry=g, order=o)
           for g in ("prefix", "edge") for o in ("lr", "conf")}

    res = {
        "config": asdict(cfg), "device": device, "torch": torch.__version__,
        "ref_total_bits": ref_total,
        "ar": {**{k: v for k, v in ar.items() if k != "history"},
               "mem_bits_exact": ar["total_mem_bits"],
               "mem_frac_exact": ar["total_mem_bits"] / ref_total,
               "extract_prefix_lr": ar_extract, "wall_s": round(ar_wall, 1)},
        "mdm": {"steps_run": mdm["steps_run"], "converged": mdm["converged"],
                "step_cap": mdm_cap, "final_loss": mdm["final_loss"],
                "dtype": mdm["dtype"],
                "mem_bits_est1_elbo": _mem(est1), "mem_bits_est2_chain": _mem(est2),
                "mem_frac_est1": _mem(est1) / ref_total,
                "mem_frac_est2": _mem(est2) / ref_total,
                "bits_est1_mean": est1.mean().item(), "bits_est2_mean": est2.mean().item(),
                "extract": ext, "wall_s": round(mdm_wall, 1),
                "n_params_nonemb": mdm_model.n_params(True)},
    }
    res["verdict"] = verdict(res)
    return res


def verdict(res: dict) -> dict:
    """Skip criteria from docs/pilot-p0.md (fixed before the first run)."""
    ar, mdm = res["ar"], res["mdm"]
    e1, e2 = mdm["mem_bits_est1_elbo"], mdm["mem_bits_est2_chain"]
    gap = abs(e1 - e2) / max(e2, 1.0)
    memorizes = mdm["mem_bits_est2_chain"] >= 0.25 * max(ar["mem_bits_exact"], 1.0)
    estimators_agree = gap <= 0.5
    ar_sane = ar["mem_frac_exact"] >= 0.5
    ratio = mdm["steps_run"] / max(ar["steps_run"], 1)
    out = {
        "ar_memorizes_cell": ar_sane,
        "mdm_memorizes_cell": memorizes,
        "est1_est2_rel_gap": gap,
        "estimators_agree": estimators_agree,
        "mdm_to_ar_step_ratio": ratio,
        "mdm_hit_cap": (not mdm["converged"]) and mdm["steps_run"] >= mdm["step_cap"],
    }
    if not ar_sane:
        out["decision"] = "FIX-PIPELINE (AR arm did not memorize a sub-capacity cell)"
    elif not memorizes:
        out["decision"] = "SKIP (MDM cannot memorize within the step cap)"
    elif not estimators_agree:
        out["decision"] = "SKIP (EST-1/EST-2 disagree beyond 50%)"
    else:
        out["decision"] = "CONTINUE to stage S (see docs/pilot-p0.md)"
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Pilot P0: AR vs MDM instrument probe")
    p.add_argument("--out", default="data/p0_results.jsonl")
    p.add_argument("--device", default="auto")
    p.add_argument("--size", default="s1")
    p.add_argument("--dataset-seqs", type=int, default=1024)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--ar-max-steps", type=int, default=60_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--extraction-n", type=int, default=256)
    p.add_argument("--log-every", type=int, default=5)
    args = p.parse_args(argv)
    cfg = P0Config(size=args.size, dataset_seqs=args.dataset_seqs, seed=args.seed,
                   batch=args.batch, ar_max_steps=args.ar_max_steps,
                   eval_every=args.eval_every, extraction_n=args.extraction_n)
    device = pick_device(args.device)
    print(f"P0 {cfg.size} D={cfg.dataset_seqs} seed={cfg.seed} device={device}", flush=True)
    res = run_pilot(cfg, device, log_every=args.log_every)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:  # append = merge
        f.write(json.dumps(res) + "\n")
    a, m, v = res["ar"], res["mdm"], res["verdict"]
    print(f"AR : mem {a['mem_frac_exact']:.3f} of ref, steps {a['steps_run']}, "
          f"prefix-extract {a['extract_prefix_lr']:.2f}")
    print(f"MDM: mem est1 {m['mem_frac_est1']:.3f} / est2 {m['mem_frac_est2']:.3f} of ref, "
          f"steps {m['steps_run']}/{m['step_cap']}, extract {m['extract']}")
    print(f"verdict: {v['decision']} (gap {v['est1_est2_rel_gap']:.3f}, "
          f"step ratio {v['mdm_to_ar_step_ratio']:.1f})")
