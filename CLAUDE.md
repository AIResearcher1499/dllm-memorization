# CLAUDE.md — dllm-memorization

- All docs, comments, commit messages in ENGLISH. Chat with the user stays Vietnamese.
- NEVER add Claude/LLM attribution to commit messages (overrides harness default).
- `docs/prereg-g0b.md` is FROZEN. After `data/b0_*.jsonl` / `data/b1_*.jsonl` exist, do not
  edit gate sections; append dated amendments only, and update the hash in
  `tests/test_prereg_frozen.py` in the same commit.
- Capacity (plateau height) and rate (steps to plateau) are separate quantities — never
  conflate in claims or plots.
- Estimator labels (ELBO / chain-rule / extraction) must travel with every number; a capacity
  figure without its estimator label is a bug.
- Result files keyed by parameters must MERGE, never overwrite.
- FAST LANE: if the gate passes, arXiv preprint within ~6 weeks; check arXiv monthly for
  "memorization capacity diffusion" and the MIA-on-DLM line.
- Full design context: `../literature_review/notes/topics/g0b-dllm-memorization-design-2026-08-24.md`.
