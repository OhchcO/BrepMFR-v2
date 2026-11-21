"""Unique parent-face aggregation for overlapping diagnostic crops."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_casing_crops import aggregate_pair_logits, aggregate_probabilities


class CasingCropEvaluationTests(unittest.TestCase):
    def test_overlap_is_averaged_per_parent_face(self):
        combined, counts = aggregate_probabilities(
            3, 2,
            [
                ([0, 1], [[0.9, 0.1], [0.2, 0.8]]),
                ([1, 2], [[0.6, 0.4], [0.1, 0.9]]),
            ],
        )
        np.testing.assert_allclose(combined, [
            [0.9, 0.1], [0.4, 0.6], [0.1, 0.9],
        ])
        self.assertEqual(counts.tolist(), [1, 2, 1])

    def test_missing_parent_face_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "uncovered"):
            aggregate_probabilities(2, 2, [([0], [[0.9, 0.1]])])

    def test_pair_logits_average_only_pairs_seen_together(self):
        scores, observed = aggregate_pair_logits(3, [
            ([0, 1], [[0, 1], [1, 0]]),
            ([0, 1], [[0, 3], [3, 0]]),
            ([1, 2], [[0, -2], [-2, 0]]),
        ])
        self.assertEqual(float(scores[0, 1]), 2.0)
        self.assertEqual(float(scores[1, 2]), -2.0)
        self.assertFalse(bool(observed[0, 2]))


if __name__ == "__main__":
    unittest.main()
