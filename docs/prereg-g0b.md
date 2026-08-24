# Pre-registration — Gate G0-B (FROZEN 2026-08-24)

FROZEN as of the commit that adds it. After any `data/b0_*.jsonl` or `data/b1_*.jsonl`
exists, gate sections may not be altered; dated amendments only.
`tests/test_prereg_frozen.py` guards the content hash.

## 1. Question

Can the Morris et al. (2505.24832) memorization-capacity measurement be transferred to masked
diffusion LMs with estimator agreement good enough to support an AR-vs-MDM capacity/rate
comparison?

## 2. Stage B-0 — replicate Morris on AR (pipeline sanity)

- Data: uniform random token sequences, vocab V=2048, seq len S=64
  (reference code length analytic: 64·log2(2048) = 704 bits/seq).
- Models: GPT-2 skeleton, target non-embedding sizes ≈ {1M, 5M, 15M}. The regression axis
  (raw vs non-embedding params) is fixed at pilot to MATCH Morris's code exactly and recorded
  in `data/config_lock.json` before the grid.
- |D| grid: 4 points log-spaced per size, bracketing α·N/704 sequences (α=3.6 prior).
- Training: Adam, bf16, batch 2048; steps capped by convergence criterion — total unintended
  memorization gain < 1% over the trailing 20% of steps, hard cap 1e6 steps (registered
  deviation from Morris's flat 1e6).
- Seeds: 2 per cell (escalation to 3 permitted only if the gate margin is within the spread).
- Memorization per sequence: max(0, 704 − NLL_model_bits); capacity = plateau of total
  memorization across |D|; α = slope of plateau vs params.

**Gate B-0:** fitted α ∈ [3.0, 4.2] bits/param with a visible plateau (memorization within 10%
of its max at ≥ 2 consecutive |D| points) at ≥ 2 of 3 sizes. FAIL → estimator/pipeline is
broken; fix before any MDM run. B-0 cannot kill the idea, only the implementation.

## 3. Stage B-1 — matched MDM + control

- Arms at identical sizes/data/skeleton: (i) MDM — MDLM continuous-time NELBO, linear masking
  schedule, low-discrepancy t-sampler; (ii) AR + token-dropout input-masking augmentation
  (2510.04071 control) at the middle size only.
- MDM step cap: 2e6 (registered — expectation E1 below).
- Estimators per trained MDM, all three mandatory:
  - EST-1 ELBO bits (128 MC samples/sequence) — achievable-compression lower bound.
  - EST-2 chain-rule exact NLL, fixed left-to-right order.
  - EST-3 extraction: greedy prefix completion (32-token prefix, exact suffix match) AND
    edge-conditioned infilling masks (2605.24173 geometries).
- Registered expectations (hypotheses, not assumptions):
  - E1: MDM approaches its plateau MORE SLOWLY (only ~50% of positions trained per step).
  - E2: direction of the capacity contrast is NOT predicted; higher, lower, equal all live.

**Gate G0-B (GO/KILL):** GO iff ALL of
(a) MDM memorization curves plateau within the step cap at ≥ 1 size;
(b) EST-1 and EST-2 order the |D| cells consistently AND their capacity estimates agree
    within 2× the pooled seed spread;
(c) EST-3 ranking does not contradict the EST-1/EST-2 ranking across |D| cells.
KILL iff the EST-1/EST-2 gap or seed noise exceeds the AR-vs-MDM contrast at every size
(instrument dominates phenomenon). "Plateau not reached" at all sizes with curves still
rising → record as RATE-LIMITED (distinct outcome; triggers a step-cap amendment decision,
not silent continuation).

## 4. Analysis discipline

- No rank-correlation gates. All contrasts reported as effect size vs pooled seed spread.
- Capacity (plateau height) and rate (steps to plateau at fixed |D|) are SEPARATE quantities;
  no claim may conflate them.
- Result files keyed by (arm, size, D, seed, estimator); MERGE only, never overwrite.
- Divergence rule: loss spike > 2× trailing median unrecovered in 500 steps → exclude, rerun
  seed+10, log in `data/exclusions.jsonl`.

## 5. Foreknowledge declaration

Known at freeze: α ≈ 3.64 bits/param for GPT-family AR (bf16 3.51±0.1); MDM ≈ 16× compute for
equal val loss on real text (SMDM) — not obviously applicable to pure memorization; 2510.04071
attributes MDM data-reuse gains to input corruption; 2604.26841 shows a sharp
memorization→generalization transition in uniform-state diffusion. We have NO prior data on
MDM capacity; the direction is genuinely unknown to us.

## 6. Open items to resolve at pilot (before grid, logged in config_lock)

Morris code: synthetic-run LR/warmup and the exact param-count axis; MDLM repo: eval MC sample
count; masking/padding conventions for S=64 (MDLM recipe is S=1024) — one pilot cell required.

## 7. Budget cap

B-0 ≤ 12 card-days; B-1 ≤ 20 card-days. Exceeding 2× a cap → stop and re-plan; trim lever is
dropping to 2 sizes, never loosening the gate.
