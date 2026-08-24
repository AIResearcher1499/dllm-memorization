"""Guards against silent edits to the frozen pre-registration (see kv-allocation twin)."""

import glob
import hashlib
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(ROOT, "docs", "prereg-g0b.md")

# sha256 of docs/prereg-g0b.md at freeze time (2026-08-24).
PREREG_SHA256 = "d68d1900787608e3a229f7c4ce63b60f041a2cbf39d31b10bf86d8375123bf58"


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestPreregFrozen(unittest.TestCase):
    def test_prereg_hash_matches(self):
        self.assertEqual(
            _sha256(PREREG),
            PREREG_SHA256,
            "docs/prereg-g0b.md changed. If data/b0_*|b1_* files exist, this "
            "edit is prohibited (append a dated amendment instead and only "
            "then update PREREG_SHA256 in the same commit).",
        )

    def test_thresholds_present_verbatim(self):
        with open(PREREG, encoding="utf-8") as f:
            text = f.read()
        for phrase in (
            "α ∈ [3.0, 4.2]",
            "agree\n    within 2× the pooled seed spread",
            "higher, lower, equal all live",
            "No rank-correlation gates",
        ):
            self.assertIn(phrase, text)

    def test_data_files_imply_hash_lock(self):
        produced = glob.glob(os.path.join(ROOT, "data", "b0_*.jsonl")) + glob.glob(
            os.path.join(ROOT, "data", "b1_*.jsonl")
        )
        if produced:
            self.assertNotEqual(
                PREREG_SHA256, "__FILL_AT_COMMIT__",
                "Gate data exists but the prereg hash was never locked.",
            )


if __name__ == "__main__":
    unittest.main()
