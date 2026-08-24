# dllm-memorization

First measurement of bits-per-parameter memorization capacity (Morris et al. 2505.24832) for
masked diffusion LMs vs matched AR models, including the ELBO-vs-exact-NLL comparability
analysis the field's AR-vs-MDM comparisons skipped. Target: arXiv-early + NeurIPS 2027, or the
ICML 2027 slot depending on the sibling gate (`../kv-allocation`).

Read in order:
1. `docs/idea-lock.md` — locked claim, ELBO-crux resolution, scoop map.
2. `docs/prereg-g0b.md` — FROZEN gate design and thresholds (guarded by tests).

Layout: `src/dlmmem/` experiment code · `tests/` (no GPU needed) ·
`data/` gate outputs (merge-only, never overwrite).

Managed with uv (pattern from the fertility-precision repo):
```bash
uv sync
uv run dlmmem doctor    # import + GPU check
uv run pytest -q        # capacity math + prereg guards
```
