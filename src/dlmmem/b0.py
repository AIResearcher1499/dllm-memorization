"""Stage B-0 runner — Morris capacity protocol on AR models (prereg-g0b §2).

Uniform random data (V=2048, S=64, ref 704 bits/seq), GPT-2 skeleton at
three sizes, |D| grid log-spaced around the expected plateau alpha*N/704
(alpha=3.6 prior), Adam, batch 2048, bf16 on CUDA. Steps are capped by the
registered convergence criterion: total unintended memorization gain < 1%
over the trailing 20% of steps, hard cap 1e6.

Gate B-0 (frozen): fitted alpha in [3.0, 4.2] bits/param with a visible
plateau at >= 2 of 3 sizes. This stage can only kill the pipeline, not the
idea. Result files are MERGE-only; the full-field resume key prevents silent
parameter collisions.
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import torch

from .capacity import REF_BITS_PER_SEQ, has_plateau, plateau_alpha
from .model import (BOS, DATA_VOCAB, SEQ_LEN, SIZE_CONFIGS, GPTConfig,
                    TinyGPT, nll_bits_per_seq)

BASE_SEED = 20260825

# |D| grids: ~{0.25, 1, 4, 16} x expected plateau (3.6*N_nonemb/704 seqs),
# rounded to powers of two so the top two points sit firmly past capacity.
D_GRID = {
    "s1": (1024, 4096, 16384, 65536),
    "s5": (8192, 32768, 131072, 524288),
    "s15": (16384, 65536, 262144, 1048576),
}
SEEDS = (0, 1)


@dataclass(frozen=True)
class RunConfig:
    size: str
    dataset_seqs: int
    seed: int
    lr: float = 1e-3
    batch: int = 2048
    max_steps: int = 1_000_000
    eval_every: int = 2000
    conv_tol: float = 0.01
    conv_window: float = 0.2
    track_subsample: int = 16384

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def build_plan(sizes=("s1", "s5", "s15"), seeds=SEEDS, **overrides):
    return [RunConfig(size=s, dataset_seqs=d, seed=BASE_SEED + se, **overrides)
            for s in sizes for d in D_GRID[s] for se in seeds]


def pick_device(arg: str = "auto") -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_dataset(num: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # int16 storage: 1M seqs x 64 tokens = 128MB instead of 512MB
    return torch.randint(0, DATA_VOCAB, (num, SEQ_LEN), generator=g).to(torch.int16)


def _total_mem_bits(model, data, batch, device, scale=1.0):
    nll = nll_bits_per_seq(model, data.long(), batch, device)
    mem = torch.clamp(REF_BITS_PER_SEQ - nll, min=0.0)
    return mem.sum().item() * scale


def train_one(rc: RunConfig, device: str, log_every: int = 0):
    torch.manual_seed(rc.seed)
    data = make_dataset(rc.dataset_seqs, rc.seed)
    sub_n = min(rc.dataset_seqs, rc.track_subsample)
    sub_idx = torch.randperm(rc.dataset_seqs,
                             generator=torch.Generator().manual_seed(rc.seed + 7))[:sub_n]
    sub = data[sub_idx]
    scale = rc.dataset_seqs / sub_n

    L, d, h = SIZE_CONFIGS[rc.size]
    model = TinyGPT(GPTConfig(L, d, h)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=rc.lr)
    use_amp = device == "cuda"
    gen = torch.Generator().manual_seed(rc.seed + 13)
    bos_col = torch.full((rc.batch, 1), BOS, dtype=torch.long, device=device)

    history, converged = [], False
    t0 = time.time()
    model.train()
    step = 0
    while step < rc.max_steps:
        idx = torch.randint(0, rc.dataset_seqs, (rc.batch,), generator=gen)
        x = data[idx].long().to(device)
        inp = torch.cat([bos_col[:x.size(0)], x[:, :-1]], dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(inp)
            loss = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % rc.eval_every == 0 or step == rc.max_steps:
            mem = _total_mem_bits(model, sub, rc.batch, device, scale)
            history.append((step, mem))
            if log_every and (len(history) % log_every == 0):
                print(f"    step {step:8d} mem~{mem / 1e6:9.3f} Mbit "
                      f"loss {loss.item():6.3f} {time.time() - t0:7.0f}s",
                      flush=True)
            # registered criterion: gain < conv_tol over trailing conv_window
            cutoff = step * (1 - rc.conv_window)
            past = [m for s, m in history if s <= cutoff]
            if past and len(history) >= 10:
                gain = (mem - past[-1]) / max(past[-1], 1.0)
                if gain < rc.conv_tol:
                    converged = True
                    break
    final_mem = _total_mem_bits(model, data, rc.batch, device)
    return model, {
        "steps_run": step, "converged": converged,
        "total_mem_bits": final_mem,
        "mem_bits_per_param_nonemb": final_mem / model.n_params(True),
        "n_params_nonemb": model.n_params(True),
        "n_params_raw": model.n_params(False),
        "final_loss": loss.item(),
        "history": history[-50:],
        "dtype": "bf16" if use_amp else "fp32",
    }


def load_done_keys(path: str):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.dumps(json.loads(line)["config"], sort_keys=True))
    return done


def run(plan, out_path: str, device: str, log_every: int = 0):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = load_done_keys(out_path)
    todo = [rc for rc in plan if rc.key() not in done]
    print(f"plan={len(plan)} done={len(done)} todo={len(todo)} device={device}",
          flush=True)
    for i, rc in enumerate(todo):
        if log_every:
            print(f"[{i + 1}/{len(todo)}] START {rc.size} D={rc.dataset_seqs} "
                  f"seed={rc.seed}", flush=True)
        t0 = time.time()
        _, res = train_one(rc, device, log_every=log_every)
        rec = {"config": asdict(rc), **res,
               "wall_s": round(time.time() - t0, 1), "device": device,
               "torch": torch.__version__}
        with open(out_path, "a", encoding="utf-8") as f:  # append = merge
            f.write(json.dumps(rec) + "\n")
        print(f"[{i + 1}/{len(todo)}] {rc.size} D={rc.dataset_seqs} "
              f"seed={rc.seed} mem={res['total_mem_bits'] / 1e6:.3f}Mbit "
              f"({res['mem_bits_per_param_nonemb']:.2f} b/param) "
              f"steps={res['steps_run']} conv={res['converged']} "
              f"{rec['wall_s']}s", flush=True)


def analyse(path: str, params_axis: str = "nonemb"):
    with open(path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    pkey = {"nonemb": "n_params_nonemb", "raw": "n_params_raw"}[params_axis]

    per_size = {}
    for size in D_GRID:
        pts = []
        n_params = None
        for d in D_GRID[size]:
            vals = [r["total_mem_bits"] for r in records
                    if r["config"]["size"] == size
                    and r["config"]["dataset_seqs"] == d]
            if vals:
                pts.append((d, sum(vals) / len(vals)))
                n_params = next(r[pkey] for r in records
                                if r["config"]["size"] == size)
        if pts:
            per_size[size] = {
                "n_params": n_params,
                "mem_by_D": pts,
                "plateau": has_plateau(pts),
                "capacity_bits": max(m for _, m in pts),
            }
    plateaued = [s for s, v in per_size.items() if v["plateau"]]
    alpha = None
    if len(plateaued) >= 2:
        alpha = plateau_alpha([(per_size[s]["n_params"],
                                per_size[s]["capacity_bits"])
                               for s in plateaued])
    gate = alpha is not None and 3.0 <= alpha <= 4.2 and len(plateaued) >= 2
    summary = {"params_axis": params_axis, "alpha_bits_per_param": alpha,
               "sizes_with_plateau": plateaued, "gate_b0": gate,
               "per_size": per_size}
    print(json.dumps({k: v for k, v in summary.items() if k != "per_size"},
                     indent=2))
    for s, v in per_size.items():
        curve = " ".join(f"D={d}:{m / 1e6:.2f}M" for d, m in v["mem_by_D"])
        print(f"  {s:4s} N={v['n_params'] / 1e6:.2f}M "
              f"plateau={v['plateau']} cap={v['capacity_bits'] / 1e6:.2f}Mbit | {curve}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage B-0 Morris capacity (AR)")
    p.add_argument("--out", default="data/b0_results.jsonl")
    p.add_argument("--device", default="auto")
    p.add_argument("--sizes", default="s1,s5,s15")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--max-steps", type=int, default=1_000_000)
    p.add_argument("--pilot", action="store_true",
                   help="one small cell only, SEPARATE output file")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", default="", help="'i/n' split across GPUs")
    p.add_argument("--analyse", action="store_true")
    p.add_argument("--params-axis", default="nonemb", choices=("nonemb", "raw"))
    p.add_argument("--log-every", type=int, default=5,
                   help="progress line every N eval points (0 = silent)")
    args = p.parse_args(argv)

    if args.analyse:
        analyse(args.out, args.params_axis)
        return

    if args.pilot:
        plan = build_plan(sizes=("s1",), seeds=(0,),
                          max_steps=min(args.max_steps, 60_000))
        plan = [rc for rc in plan if rc.dataset_seqs == 4096]
        out = args.out if args.out != "data/b0_results.jsonl" else "data/b0_pilot.jsonl"
    else:
        seeds = tuple(int(s) for s in args.seeds.split(","))
        plan = build_plan(sizes=tuple(args.sizes.split(",")), seeds=seeds,
                          max_steps=args.max_steps)
        out = args.out
    # cheap cells first: small models and small datasets converge fastest
    plan.sort(key=lambda rc: (SIZE_CONFIGS[rc.size][1], rc.dataset_seqs))
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        plan = plan[i::n]
    if args.limit:
        plan = plan[:args.limit]
    if args.dry_run:
        print(f"{len(plan)} runs -> {out}")
        for rc in plan[:5]:
            print(" ", rc.key())
        return
    run(plan, out, pick_device(args.device), log_every=args.log_every)


if __name__ == "__main__":
    main()
