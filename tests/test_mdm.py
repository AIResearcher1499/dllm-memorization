import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402

from dlmmem.mdm import (  # noqa: E402
    MASK, chain_rule_bits_per_seq, elbo_bits_per_seq, extraction_rate,
    low_discrepancy_t, make_mdm, mask_at_rate, mdm_loss,
)
from dlmmem.model import DATA_VOCAB, SEQ_LEN  # noqa: E402
from dlmmem.p0 import verdict  # noqa: E402


class TestMasking(unittest.TestCase):
    def test_low_discrepancy_covers_unit_interval(self):
        t = low_discrepancy_t(64, torch.Generator().manual_seed(0))
        self.assertEqual(t.shape, (64,))
        self.assertTrue((t > 0).all() and (t <= 1).all())
        self.assertLess(t.min(), 0.05)
        self.assertGreater(t.max(), 0.95)

    def test_mask_rate_unbiased(self):
        x = torch.randint(0, DATA_VOCAB, (256, SEQ_LEN))
        t = torch.full((256,), 0.5)
        xm, m = mask_at_rate(x, t, torch.Generator().manual_seed(1))
        self.assertTrue((xm[m] == MASK).all())
        self.assertTrue((xm[~m] == x[~m]).all())
        self.assertLess(abs(m.float().mean().item() - 0.5), 0.05)
        # no forcing: at tiny t most examples have zero masked positions
        t0 = torch.full((256,), 1e-3)
        _, m0 = mask_at_rate(x, t0, torch.Generator().manual_seed(2))
        self.assertLess(m0.any(dim=1).float().mean().item(), 0.2)


class TestEstimatorsAtInit(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = make_mdm(2, 64, 2)
        self.data = torch.randint(0, DATA_VOCAB, (8, SEQ_LEN))

    def test_bidirectional_skeleton(self):
        self.assertFalse(self.model.cfg.causal)
        # a change at the last position must reach the first logit (no causal mask)
        x = self.data[:1].clone()
        y = x.clone()
        y[0, -1] = (y[0, -1] + 1) % DATA_VOCAB
        self.model.eval()
        with torch.no_grad():
            d = (self.model(x)[0, 0] - self.model(y)[0, 0]).abs().max()
        self.assertGreater(d.item(), 0.0)

    def test_elbo_unbiased_at_init(self):
        # per-sequence MC noise is large while CE is large (~+-30 bits at K=128,
        # vanishing once a sequence is memorized); the MEAN must sit on 704.
        b = elbo_bits_per_seq(self.model, self.data, batch=4, device="cpu", k=64)
        self.assertEqual(b.shape, (8,))
        self.assertLess(abs(b.mean().item() - 704.0) / 704.0, 0.03, b.mean())

    def test_chain_rule_near_reference_at_init(self):
        b = chain_rule_bits_per_seq(self.model, self.data, batch=4, device="cpu")
        self.assertEqual(b.shape, (8,))
        self.assertTrue(((b > 640) & (b < 780)).all(), b)

    def test_extraction_zero_at_init(self):
        for g in ("prefix", "edge"):
            for o in ("lr", "conf"):
                r = extraction_rate(self.model, self.data, 4, "cpu", geometry=g, order=o)
                self.assertEqual(r, 0.0, (g, o))

    def test_loss_is_finite_and_positive(self):
        gen = torch.Generator().manual_seed(3)
        loss = mdm_loss(self.model, self.data, gen)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)


class TestMDMMemorizesTinyCell(unittest.TestCase):
    """The pipeline test that matters: a tiny MDM swallows 16 sequences, and
    EST-1/EST-2/EST-3 all report it. CPU-only, ~1 min."""

    def test_memorize_16_sequences(self):
        torch.manual_seed(0)
        model = make_mdm(2, 64, 2)
        data = torch.randint(0, DATA_VOCAB, (16, SEQ_LEN))
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        gen = torch.Generator().manual_seed(5)
        for step in range(1500):
            loss = mdm_loss(model, data, gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
        e1 = elbo_bits_per_seq(model, data, 16, "cpu", k=32).mean().item()
        e2 = chain_rule_bits_per_seq(model, data, 16, "cpu").mean().item()
        self.assertLess(e2, 352, f"chain-rule bits still {e2:.0f} of 704")
        self.assertLess(e1, 352, f"elbo bits still {e1:.0f} of 704")
        self.assertLess(abs(e1 - e2) / max(704 - e2, 1), 0.5)
        r = extraction_rate(model, data, 16, "cpu", geometry="prefix", order="lr")
        self.assertGreater(r, 0.5, f"prefix extraction only {r:.2f}")


class TestVerdictLogic(unittest.TestCase):
    def _res(self, ar_mem, mdm1, mdm2, ar_steps=100, mdm_steps=500, cap=1000, conv=True):
        return {"ar": {"mem_bits_exact": ar_mem, "mem_frac_exact": ar_mem / 704000,
                       "steps_run": ar_steps},
                "mdm": {"mem_bits_est1_elbo": mdm1, "mem_bits_est2_chain": mdm2,
                        "steps_run": mdm_steps, "step_cap": cap, "converged": conv}}

    def test_continue(self):
        v = verdict(self._res(700000, 600000, 650000))
        self.assertTrue(v["decision"].startswith("CONTINUE"))

    def test_skip_when_mdm_cannot_memorize(self):
        v = verdict(self._res(700000, 100000, 120000, conv=False, mdm_steps=1000))
        self.assertTrue(v["decision"].startswith("SKIP (MDM cannot"))

    def test_skip_when_estimators_disagree(self):
        v = verdict(self._res(700000, 200000, 650000))
        self.assertTrue(v["decision"].startswith("SKIP (EST-1"))

    def test_fix_pipeline_when_ar_fails(self):
        v = verdict(self._res(100000, 600000, 650000))
        self.assertTrue(v["decision"].startswith("FIX-PIPELINE"))


if __name__ == "__main__":
    unittest.main()
