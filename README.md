# dllm-memorization

First measurement of bits-per-parameter memorization capacity (Morris et al. 2505.24832) for
masked diffusion LMs vs matched AR models, including the ELBO-vs-exact-NLL comparability
analysis the field's AR-vs-MDM comparisons skipped. Target: arXiv-early + NeurIPS 2027, or the
ICML 2027 slot depending on the sibling gate (`../kv-allocation`).

> **Status 2026-09-13:** no data yet. Novelty re-checked (capacity cell still empty;
> extraction/MIA side densifying — see `docs/brief-2026-09-13.md`). MDM arm and the three
> estimators implemented (`src/dlmmem/mdm.py`); pilot P0 specified (`docs/pilot-p0.md`),
> pipeline smoke passed on MPS; the pilot cell itself has not been run. Cheapest path:
> P0 → stage S → mini-G0, all $0 on the 2×A6000 box; cloud only buys calendar.

Read in order:
1. `docs/idea-lock.md` — locked claim, ELBO-crux resolution, scoop map.
2. `docs/prereg-g0b.md` — FROZEN gate design and thresholds (guarded by tests).
3. `docs/brief-2026-09-13.md` — what B (and the C tokenizer axis) is, novelty status, cost ladder.
4. `docs/pilot-p0.md` — the $0 instrument probe to run first, with skip criteria.

Layout: `src/dlmmem/` experiment code · `tests/` (no GPU needed) ·
`data/` gate outputs (merge-only, never overwrite).

Managed with uv (pattern from the fertility-precision repo):
```bash
uv sync
uv run dlmmem doctor    # import + GPU check
uv run pytest -q        # 29 tests: capacity math, prereg guard, MDM estimators, P0 verdict
uv run dlmmem p0        # pilot P0 (docs/pilot-p0.md) -> data/p0_results.jsonl
```
