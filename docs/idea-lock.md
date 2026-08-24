# Idea lock — memorization capacity of masked diffusion LMs (locked 2026-08-24)

## One-sentence claim (candidate)

Measure, for the first time, the bits-per-parameter memorization capacity (Morris et al.,
arXiv:2505.24832, ICML 2026 oral) of masked diffusion language models against matched
autoregressive models — separating capacity (plateau height) from memorization rate (steps to
plateau) — and, along the way, repair the AR-NLL-vs-MDM-ELBO comparability flaw that the
field's flagship comparisons (arXiv:2507.15857) left unaddressed.

## Why this is ours to take

- Nobody has run the capacity protocol on any text diffusion model (scoop check 2026-08-24).
  Closest: 2604.26841 (uniform-state diffusion associative memory, no capacity, no MDLM);
  2605.24173 (infilling extraction, fine-tuned 7–8B only); MIA line 2601.20125 / 2607.16207
  (fine-tuned checkpoints only).
- The intersection joins the two most-rewarded ICML 2026 threads (dLLM science — Outstanding
  Paper; memorization science — oral), and the protocol's native hardware class (≤20M-param
  synthetic sweeps, 500K–1.5B text runs) fits 2×A6000 exactly.
- Stakes: MDMs' data-reuse advantage (~100 epochs vs ~4 for AR, 2507.15857) is unpriced if it
  is partly a memorization/extraction advantage — a privacy-relevant result either way.

## The ELBO crux and its resolution (core methodological contribution)

Morris capacity is likelihood-based; MDMs yield an ELBO, not exact NLL. Resolution:
1. RADD (2406.03736): absorbing-diffusion ELBO = expected any-order AR code length → on
   uniform random data (nothing to generalize) the ELBO is an ACHIEVABLE compression length,
   so ELBO-based memorized bits is a valid LOWER bound on capacity.
2. Chain-rule fixed-order decoding gives an EXACT NLL of the induced AR model (SMDM practice;
   128 MC samples for ELBO eval).
3. Non-likelihood cross-check: extraction (greedy prefix per Morris; edge-conditioned infilling
   masks per 2605.24173, which extract up to 3× more from DLMs than prefix attacks).
Report capacity under all three; their agreement/disagreement is itself a result the field
needs (2507.15857 compared AR NLL to raw MDM ELBO with no tightening).

## Registered confound control

2510.04071 (Ubiquant) argues MDM data-reuse gains are largely token-dropout-style input
corruption, reproducible in AR. G0 therefore carries an AR + input-masking-augmentation arm;
any capacity difference must be attributed objective-vs-corruption before being claimed.

## Scope guards / kill history

- This is a controlled-training science paper, not an attack paper and not an eval paper.
  MIA/extraction are instruments, not the headline.
- FAST LANE: 4 adjacent papers Jan–Jul 2026. If G0-B passes, an arXiv preprint of the core
  capacity result ships within ~6 weeks of GO. Monitor arXiv for "memorization capacity
  diffusion" monthly.
- Both directions of the AR-vs-MDM contrast are findings; the only kill is instrument failure
  (estimator gap / seed noise swamping any plausible contrast).

## Venue & schedule

Primary plan: arXiv-early + NeurIPS 2027, OR swap into the ICML 2027 slot if G0-A dies or
underwhelms (decision after both gates). Full design context:
`../../literature_review/notes/topics/g0b-dllm-memorization-design-2026-08-24.md`.
