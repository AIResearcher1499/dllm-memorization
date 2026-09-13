# Pilot P0 — instrument probe before any grid (spec written 2026-09-13)

Purpose: answer the prereg's only KILL question — *does the instrument work?* — with two
small runs before spending anything on the B-0 / B-1 grids. P0 is the "one pilot cell"
required by `prereg-g0b.md` §6 (S=64 masking/padding conventions for the MDM arm). Its
numbers **do not** feed any gate threshold; the gates stay as frozen.

## Design

One cell, both arms, identical data:

| | AR arm | MDM arm |
|---|---|---|
| Skeleton | TinyGPT s1 (2L, d192, ≈0.88M non-emb), causal | same skeleton, bidirectional (`causal=False`), id 2048 = [MASK] |
| Data | 1,024 uniform random sequences, V=2048, S=64 (704 bits/seq; 0.72 Mbit total ≈ 23% of the 3.6·N prior capacity — both arms *should* memorize fully) | same sequences (same seed) |
| Objective | next-token CE with BOS | MDLM continuous-time NELBO, linear schedule, low-discrepancy t, no mask forcing |
| Optimizer | Adam 1e-3, batch 1024, bf16 on CUDA | same |
| Stop | registered convergence criterion (gain < 1% over trailing 20%), cap 60k | same criterion, cap = **10 × AR converged steps** (hard cap 600k) |
| Estimators | exact NLL bits; prefix-32 greedy extraction | **EST-1** ELBO bits (K=128 stratified MC) · **EST-2** chain-rule exact NLL (left-to-right) · **EST-3** extraction: prefix-32 and edge-16/16 infilling, left-to-right and confidence orders |

Command: `uv run dlmmem p0 --device cuda` (defaults = this table; output
`data/p0_results.jsonl`, append-only, never merged into b0/b1 files).

## Skip criteria (fixed here and in `p0.verdict()` before the first pilot run)

Evaluated in order; the first that fires decides.

1. **FIX-PIPELINE** — AR arm memorizes < 50% of the cell's 704·|D| bits. The frozen
   B-0 loop failed on a sub-capacity cell → bug, not a finding. Fix, re-run P0.
2. **SKIP (MDM cannot memorize)** — MDM EST-2 memorized bits < 25% of the AR arm's memorized
   bits after 10× the AR's converged steps. The masking objective does not reach a
   memorization plateau at a scale where AR does → the capacity protocol cannot be
   transferred with this recipe; abandon B without running a grid.
3. **SKIP (estimators disagree)** — |EST-1 − EST-2| / EST-2 > 0.5 on memorized bits. The
   ELBO lower bound and the induced-AR exact NLL disagree beyond any usable contrast → the
   comparability repair that is half the paper does not hold; abandon.
4. **CONTINUE** → stage S (`brief-2026-09-13.md` §5 ladder: 2 sizes × 3 |D| × 1 seed AR
   smoke, then mini-G0), all on the 2×A6000 box at $0.

Also recorded, not gated: MDM/AR step ratio to plateau (E1 expectation: MDM slower),
extraction rates per geometry/order for both arms (EST-3 sanity: a memorized cell should
extract near 100%), wall-clock per 1k steps (cost input for stage S).

## Why these thresholds

- 25% and 0.5 are deliberately loose: P0 is a *kill* probe for gross instrument failure,
  not a measurement. Fine contrasts belong to the gated grids with seeds.
- 10× AR steps: MDLM trains ~50% of positions per step and SMDM reports ~16× compute for
  equal val loss on text; 10× on pure memorization is generous but bounded.
- The cell sits well below capacity so "cannot memorize" cannot be blamed on capacity.

## Pipeline smoke (disclosed; not the pilot)

2026-09-13, Mac MPS, D=128, batch 128, eval every 50, 64 sequences scored by EST-3 —
to verify the code path only: AR memorized 0.989 of ref (500 steps); MDM EST-1 0.954 /
EST-2 0.956 (gap 0.002), 500 steps; extraction 1.00 on all four geometries; verdict path
CONTINUE. Estimator noise at init (untrained model, CE ≈ 7.6 nats): EST-1 per-sequence
std ≈ ±30 bits at K=128, mean unbiased (703.7 vs 704.0); noise vanishes as CE → 0.
The pilot cell (D=1024, K=128, full extraction set) has **not** been run.

## P0 run (2026-09-13, Mac M4 Pro, MPS, fp32)

`uv run dlmmem p0 --device mps` — one cell, defaults as above; record appended to
`data/p0_results.jsonl`. Wall-clock: AR 1,286 s (5,000 steps; converged by the registered
criterion), MDM 1,738 s (6,500 steps; converged; cap was 50,000).

| | AR arm | MDM arm |
|---|---|---|
| memorized / 0.72 Mbit ref | **0.985** (exact NLL) | **EST-1 0.885 · EST-2 0.891** (rel. gap 0.007) |
| bits per non-emb param | 0.80 | 0.72 (EST-2) |
| steps to plateau | 5,000 | 6,500 (ratio 1.3×) |
| extraction (EST-3) | prefix-32 L→R 1.00 | prefix-32 L→R 0.91 · conf 1.00; edge-16/16 L→R 0.95 · conf 1.00 |

**Verdict (`p0.verdict`): CONTINUE to stage S.** (1) AR memorizes the sub-capacity cell →
pipeline sane; (2) MDM memorizes 90 % of the AR arm's bits within the cap → the masking
objective reaches a plateau where AR does; (3) EST-1/EST-2 agree to 0.7 % → the ELBO
bound and the induced-AR exact NLL are interchangeable at this scale.

Noted, not gated: the MDM plateau sits ≈ 10 % below AR at identical N and |D| (0.885 vs
0.985 of ref; 0.72 vs 0.80 bits/param) — the first hint of the contrast the grids will
measure with seeds; confidence-order extraction recovers what left-to-right misses. P0 was
run on MPS in fp32 (the prereg's bf16 applies to the grids on CUDA).

## After P0

CONTINUE → record the S=64 masking/padding conventions and the params axis in
`data/b0_config_lock.json` (prereg §6) and start stage S. Any SKIP → bury this repo
under `../failed_ideas/` with `data/p0_results.jsonl` and this file as the verdict.
