import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from dlmmem.capacity import (  # noqa: E402
    REF_BITS_PER_SEQ,
    has_plateau,
    memorized_bits,
    plateau_alpha,
    total_memorized_bits,
)


class TestCapacityMath(unittest.TestCase):
    def test_ref_bits(self):
        self.assertAlmostEqual(REF_BITS_PER_SEQ, 64 * math.log2(2048))
        self.assertAlmostEqual(REF_BITS_PER_SEQ, 704.0)

    def test_memorized_bits_perfect_and_none(self):
        # Fully memorized sequence: NLL -> 0 => saves all 704 bits.
        self.assertAlmostEqual(memorized_bits(0.0), 704.0)
        # Model at chance (NLL == reference): zero memorization.
        self.assertAlmostEqual(memorized_bits(704.0), 0.0)
        # Worse than reference is clipped at zero, never negative.
        self.assertAlmostEqual(memorized_bits(900.0), 0.0)

    def test_total(self):
        self.assertAlmostEqual(total_memorized_bits([0.0, 704.0, 352.0]), 704.0 + 352.0)

    def test_alpha_recovers_known_slope(self):
        # Synthetic: capacity = 3.6 bits/param exactly.
        pts = [(1_000_000, 3.6e6), (5_000_000, 1.8e7), (15_000_000, 5.4e7)]
        self.assertAlmostEqual(plateau_alpha(pts), 3.6, places=6)

    def test_plateau_detection(self):
        rising = [(10, 100.0), (100, 400.0), (1000, 900.0), (10000, 1500.0)]
        self.assertFalse(has_plateau(rising))
        plateaued = [(10, 100.0), (100, 950.0), (1000, 1000.0), (10000, 980.0)]
        self.assertTrue(has_plateau(plateaued))


if __name__ == "__main__":
    unittest.main()
