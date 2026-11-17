from __future__ import annotations

import unittest

import numpy as np

from app.wireframe import build_polygon_wireframe_positions
from app.mesh_visualization import connected_component_mesh_data, face_component_colors

class MeshPreviewPolygonTests(unittest.TestCase):
    def test_connected_face_overlay_generates_stable_random_fills_for_disconnected_islands(self):
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]],
            dtype=np.float32,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        first = face_component_colors(vertices, faces, seed=1729)
        second = face_component_colors(vertices, faces, seed=1729)
        self.assertEqual(first.shape, (2, 4))
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first[0], first[1]))
        expanded_v, expanded_f, expanded_c = connected_component_mesh_data(
            vertices, faces, seed=1729
        )
        self.assertEqual(expanded_v.shape, (6, 3))
        self.assertEqual(expanded_f.shape, (2, 3))
        self.assertEqual(expanded_c.shape, (6, 4))
        self.assertTrue(np.all(expanded_c[:, 3] == 1.0))

    def test_quad_wireframe_has_no_internal_diagonal(self):
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            dtype=np.float32,
        )
        positions = build_polygon_wireframe_positions(
            vertices,
            np.empty((0, 3), dtype=np.int64),
            np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        )
        self.assertEqual(positions.shape, (8, 3))
        pairs = positions.reshape(-1, 2, 3)
        diagonal = {tuple(vertices[0]), tuple(vertices[2])}
        self.assertFalse(any({tuple(pair[0]), tuple(pair[1])} == diagonal for pair in pairs))

if __name__ == "__main__":
    unittest.main()
