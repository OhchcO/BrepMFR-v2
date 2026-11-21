"""Checks for the one-parent casing pipeline's label contract."""

import sys
import unittest
from pathlib import Path

import numpy as np
import dgl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_casing_overfit import (
    build_instance_matrix, map_semantic_codes, plan_covering_crops,
)


class CasingOverfitPreparationTests(unittest.TestCase):
    def test_semantic_codes_become_fixed_contiguous_indices(self):
        mapped = map_semantic_codes([0, 101, 103, 204, 402, 601])
        self.assertEqual(mapped.tolist(), [0, 1, 3, 14, 22, 28])

    def test_unrecognized_code_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "999"):
            map_semantic_codes([0, 999])

    def test_instance_matrix_excludes_background(self):
        matrix = build_instance_matrix(np.array([-1, 1, 1, 2], dtype=np.int32))
        self.assertEqual(matrix.dtype, np.int8)
        self.assertEqual(matrix.tolist(), [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ])

    def test_planned_crops_cover_every_parent_face(self):
        graph = dgl.graph((
            torch.tensor([0, 1, 2, 3, 4, 5, 6]),
            torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        ), num_nodes=8)
        plan = plan_covering_crops(graph, [0, 0, 0, 0, 103, 103, 103, 103], 3)
        self.assertEqual(set().union(*(set(nodes) for _, nodes in plan)), set(range(8)))
        self.assertEqual({name for name, _ in plan[:2]},
                         {"class_000", "class_103"})

    def test_covering_plan_avoids_severely_redundant_chain_crops(self):
        graph = dgl.graph((
            torch.arange(29), torch.arange(1, 30),
        ), num_nodes=30)
        plan = plan_covering_crops(graph, [0] * 30, 5)
        self.assertLessEqual(len(plan), 8)

    def test_covering_plan_packs_leaf_faces_with_shared_context(self):
        graph = dgl.graph((
            torch.zeros(29, dtype=torch.int64), torch.arange(1, 30),
        ), num_nodes=30)
        plan = plan_covering_crops(graph, [0] * 30, 5)
        self.assertEqual(set().union(*(set(nodes) for _, nodes in plan)),
                         set(range(30)))
        self.assertLessEqual(len(plan), 8)


if __name__ == "__main__":
    unittest.main()
