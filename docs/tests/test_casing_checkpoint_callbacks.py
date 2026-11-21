"""The debug runner avoids repeatedly overwriting last.ckpt on Windows."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_casing_overfit_no_last import build_callbacks


class CasingCheckpointTests(unittest.TestCase):
    def test_callbacks_save_best_without_last(self):
        with tempfile.TemporaryDirectory() as directory:
            callbacks = build_callbacks(Path(directory))
            self.assertEqual(len(callbacks), 2)
            self.assertTrue(all(callback.save_last is False for callback in callbacks))
            self.assertEqual([callback.monitor for callback in callbacks],
                             ["mean_iou", "macro_f1"])


if __name__ == "__main__":
    unittest.main()
