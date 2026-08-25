import json
import os
import sys
import tempfile
import unittest
from dataclasses import fields, replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402

from dlmmem.b0 import (  # noqa: E402
    BASE_SEED, D_GRID, RunConfig, analyse, build_plan, load_done_keys,
    make_dataset,
)
from dlmmem.model import (  # noqa: E402
    DATA_VOCAB, SIZE_CONFIGS, GPTConfig, TinyGPT, nll_bits_per_seq,
)


class TestModelSizes(unittest.TestCase):
    def test_nonemb_params_near_targets(self):
        targets = {"s1": 0.885e6, "s5": 4.92e6, "s15": 14.45e6}
        for name, (L, d, h) in SIZE_CONFIGS.items():
            n = TinyGPT(GPTConfig(L, d, h)).n_params(True)
            self.assertLess(abs(n - targets[name]) / targets[name], 0.10, name)

    def test_untrained_nll_is_near_chance(self):
        torch.manual_seed(0)
        m = TinyGPT(GPTConfig(2, 64, 2))
        data = torch.randint(0, DATA_VOCAB, (8, 64))
        nll = nll_bits_per_seq(m, data, batch=4, device="cpu")
        self.assertEqual(nll.shape, (8,))
        # GPT-2 init gives near-uniform logits -> ~64*log2(2049) ~= 704 bits
        self.assertTrue(((nll > 650) & (nll < 760)).all(), nll)


class TestPlanAndData(unittest.TestCase):
    def test_resume_key_covers_every_field(self):
        rc = RunConfig("s1", 1024, BASE_SEED)
        bumps = {int: lambda v: v + 1, float: lambda v: v * 2,
                 str: lambda v: v + "x"}
        for f in fields(RunConfig):
            old = getattr(rc, f.name)
            mutated = replace(rc, **{f.name: bumps[type(old)](old)})
            self.assertNotEqual(rc.key(), mutated.key(),
                                f"field {f.name} missing from resume key")

    def test_plan_counts(self):
        plan = build_plan()
        self.assertEqual(len(plan), 3 * 4 * 2)
        self.assertEqual(len({rc.key() for rc in plan}), len(plan))

    def test_dataset_deterministic_int16(self):
        a = make_dataset(64, 5)
        b = make_dataset(64, 5)
        self.assertEqual(a.dtype, torch.int16)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue((a.long() >= 0).all() and (a.long() < DATA_VOCAB).all())
        self.assertFalse(torch.equal(a, make_dataset(64, 6)))

    def test_merge_semantics(self):
        plan = build_plan()[:3]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "b0.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"config": json.loads(plan[0].key())}) + "\n")
            done = load_done_keys(path)
            todo = [rc for rc in plan if rc.key() not in done]
            self.assertEqual(len(todo), 2)


class TestAnalyse(unittest.TestCase):
    def _write(self, recs):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "b0.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        return path

    def _recs(self, alpha):
        # synthetic plateau curves: mem rises then saturates at alpha*N
        n_params = {"s1": 0.885e6, "s5": 4.92e6, "s15": 14.45e6}
        recs = []
        for size, ds in D_GRID.items():
            cap = alpha * n_params[size]
            for d in ds:
                mem = min(cap, d * 704 * 0.98)
                for seed in (BASE_SEED, BASE_SEED + 1):
                    recs.append({
                        "config": {"size": size, "dataset_seqs": d, "seed": seed},
                        "total_mem_bits": mem,
                        "n_params_nonemb": n_params[size],
                        "n_params_raw": n_params[size] * 1.3,
                    })
        return recs

    def test_gate_passes_at_morris_alpha(self):
        s = analyse(self._write(self._recs(3.6)))
        self.assertEqual(sorted(s["sizes_with_plateau"]), ["s1", "s15", "s5"])
        self.assertAlmostEqual(s["alpha_bits_per_param"], 3.6, places=2)
        self.assertTrue(s["gate_b0"])

    def test_gate_fails_at_low_alpha(self):
        s = analyse(self._write(self._recs(2.0)))
        self.assertAlmostEqual(s["alpha_bits_per_param"], 2.0, places=2)
        self.assertFalse(s["gate_b0"])


class TestTinyTraining(unittest.TestCase):
    def test_small_model_memorizes_tiny_dataset(self):
        from dlmmem.b0 import train_one
        rc = RunConfig("s1", 64, BASE_SEED, batch=64, max_steps=400,
                       eval_every=50, track_subsample=64)
        _, res = train_one(rc, "cpu")
        # 64 seqs = 45k bits vs ~0.9M-param model: should memorize most of it
        self.assertGreater(res["total_mem_bits"], 0.5 * 64 * 704)
        self.assertLessEqual(res["steps_run"], 400)


if __name__ == "__main__":
    unittest.main()
