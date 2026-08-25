# B-0 runbook (2×A6000 box)

Setup once:
```bash
cd ~/working_dir && git clone https://github.com/AIResearcher1499/dllm-memorization && cd dllm-memorization
uv sync && uv run dlmmem doctor && uv run pytest -q   # 17 tests green before any run
```

## Order of operations

1. **Pilot** (one cell, s1/D=4096, ≤60k steps, SEPARATE file — resolves the
   prereg §6 open items before the grid):
   ```bash
   uv run dlmmem b0 --pilot 2>&1 | tee -a logs/b0_pilot.log
   ```
   Read from the pilot: (a) does memorization plateau cleanly under the
   convergence criterion; (b) wall-clock per 10k steps (budget input);
   (c) record LR/params-axis decisions in `data/b0_config_lock.json`.
2. **Grid** (24 runs = 3 sizes × 4 |D| × 2 seeds; cheap-first ordering built in):
   ```bash
   CUDA_VISIBLE_DEVICES=0 uv run dlmmem b0 --shard 0/2 \
     --out data/b0_results_gpu0.jsonl 2>&1 | tee -a logs/b0_gpu0.log
   CUDA_VISIBLE_DEVICES=1 uv run dlmmem b0 --shard 1/2 \
     --out data/b0_results_gpu1.jsonl 2>&1 | tee -a logs/b0_gpu1.log
   ```
   Restart-safe (full-field resume key). Budget cap: prereg §7, ≤12 card-days.
3. **Gate**:
   ```bash
   cat data/b0_results_gpu*.jsonl > data/b0_results.jsonl
   uv run dlmmem b0 --analyse --out data/b0_results.jsonl
   ```
   GO = alpha in [3.0, 4.2] bits/param + plateau at ≥2 of 3 sizes
   (frozen, docs/prereg-g0b.md §2). B-0 failing kills the PIPELINE, not the
   idea — fix the estimator before touching any MDM run.

## Notes

- bf16 on CUDA (Morris: bf16 gives 3.51±0.1); dtype recorded per run.
- Large-D cells (s15/D=1M) hold a 128MB int16 dataset in RAM and run the
  full-set NLL pass once at the end; the in-training tracker uses a fixed
  16k-sequence subsample, scaled.
- This stage shares the box with the kv-allocation A-0 grid: run B-0 only on
  a free GPU, or after A-0 finishes. Check `pwd` + `git remote -v` before
  touching any data file — both repos have similar layouts.
