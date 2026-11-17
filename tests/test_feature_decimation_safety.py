from __future__ import annotations

import unittest

import numpy as np

from app.feature_decimation import _count_components, decimate_mesh_li_nan
from app.pipeline import _decimate_mesh_arrays

def make_grid(side: int, offset: float = 0.0):
    x, y = np.meshgrid(np.linspace(0.0, 1.0, side), np.linspace(0.0, 1.0, side))
    vertices = np.column_stack((x.reshape(-1) + offset, y.reshape(-1), np.zeros(side * side)))
    faces = []
    for row in range(side - 1):
        for col in range(side - 1):
            a = row * side + col
            b, c, d = a + 1, a + side, a + side + 1
            faces.extend(((a, b, d), (a, d, c)))
    return vertices, np.asarray(faces, dtype=np.int64)

class FeatureDecimationSafetyTests(unittest.TestCase):
    def test_component_counter_keeps_nonmanifold_edge_faces_in_one_island(self):
        faces = np.asarray(((0, 1, 2), (1, 0, 3), (0, 1, 4)), dtype=np.int64)
        self.assertEqual(_count_components(faces), 1)

    def test_component_counter_connects_faces_that_touch_at_one_vertex(self):
        faces = np.asarray(((0, 1, 2), (0, 3, 4), (5, 6, 7)), dtype=np.int64)
        self.assertEqual(_count_components(faces), 2)

    def test_shared_output_face_budget_uses_the_safe_qem_core(self):
        vertices, faces = make_grid(20)
        reduced_vertices, reduced_faces = _decimate_mesh_arrays(vertices, faces, 160)
        self.assertLessEqual(len(reduced_faces), 160)
        self.assertLess(len(reduced_faces), len(faces))
        self.assertGreaterEqual(len(reduced_vertices), 4)

    def test_qem_reports_stage_timings_and_preserves_safety_invariants(self):
        vertices, faces = make_grid(14)
        result = decimate_mesh_li_nan(
            vertices, faces, None, target_vertex_count=120,
            allow_boundary_collapses=False, collapse_intersection_guard=False,
            visualization_stride=10000,
        )
        timing_values = (
            result.stats.topology_build_seconds,
            result.stats.component_check_seconds,
            result.stats.quadric_setup_seconds,
            result.stats.initial_queue_seconds,
            result.stats.collapse_seconds,
            result.stats.final_validation_seconds,
        )
        self.assertTrue(all(np.isfinite(value) and value >= 0.0 for value in timing_values))
        self.assertGreater(sum(timing_values), 0.0)
        self.assertIn(result.stats.adaptive_qem_tier, {"small", "medium", "large", "xlarge"})
        self.assertGreater(result.stats.estimated_topology_bytes, 0)
        self.assertGreater(result.stats.estimated_queue_entry_bytes, 0)
        self.assertGreater(result.stats.adaptive_queue_compact_min_entries, 0)
        self.assertEqual(result.stats.initial_component_count, result.stats.final_component_count)
        self.assertEqual(result.stats.initial_boundary_edge_count, result.stats.final_boundary_edge_count)

    def test_vectorized_initial_vertex_quadrics_match_incident_face_reference(self):
        from app.feature_decimation import (
            _build_topology,
            _face_plane_quadrics,
            _initial_vertex_quadrics,
            _plane_quadric,
        )

        vertices, faces = make_grid(9)
        vertex_faces, _neighbors, _edge_faces = _build_topology(faces, len(vertices))
        face_quadrics = _face_plane_quadrics(vertices, faces)
        labels = np.full(len(faces), -1, dtype=np.int64)
        labels[:len(labels) // 2] = 3
        region_quadric = _plane_quadric(np.asarray([0.0, 0.0, 1.0, 0.0]))
        actual = _initial_vertex_quadrics(
            len(vertices), faces, labels, face_quadrics, {3: region_quadric}
        )
        expected = np.zeros_like(actual)
        for vertex_id, incident in enumerate(vertex_faces):
            if not incident:
                continue
            incident_labels = {int(labels[face_id]) for face_id in incident}
            if len(incident_labels) == 1:
                region_id = next(iter(incident_labels))
                if region_id >= 0 and region_id in {3}:
                    expected[vertex_id] = region_quadric
                    continue
            for face_id in incident:
                expected[vertex_id] += face_quadrics[face_id]
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_qem_scalar_cost_matches_homogeneous_quadric_equation(self):
        from app.feature_decimation import _plane_quadric, _quadric_cost

        rng = np.random.default_rng(42)
        planes = rng.normal(size=(7, 4))
        quadric = sum((_plane_quadric(plane) for plane in planes), np.zeros((4, 4)))
        for position in rng.normal(size=(20, 3)):
            homogeneous = np.append(position, 1.0)
            expected = max(float(homogeneous @ quadric @ homogeneous), 0.0)
            self.assertAlmostEqual(_quadric_cost(quadric, position), expected, places=10)

    def test_qem_edge_placement_matches_direct_segment_candidates(self):
        from app.feature_decimation import _optimal_edge_placement, _plane_quadric

        rng = np.random.default_rng(7)
        plane_coefficients = rng.normal(size=(9, 4))
        quadric = sum((_plane_quadric(plane) for plane in plane_coefficients), np.zeros((4, 4)))
        vertices = rng.normal(size=(2, 3))
        quadrics = np.stack((quadric * 0.35, quadric * 0.65))
        first, second = 0, 1
        p0, p1 = vertices[first], vertices[second]
        direction = p1 - p0
        direction_h = np.append(direction, 0.0)
        p0_h = np.append(p0, 1.0)
        denominator = float(direction_h @ quadric @ direction_h)
        numerator = float(direction_h @ quadric @ p0_h)
        candidates = [0.0, 0.5, 1.0]
        if np.isfinite(denominator) and denominator > 1e-18:
            candidates.append(float(np.clip(-numerator / denominator, 0.0, 1.0)))
        expected = []
        for t in candidates:
            point = p0 + t * direction
            h = np.append(point, 1.0)
            expected.append((max(float(h @ quadric @ h), 0.0), point))
        expected_cost, expected_position = min(expected, key=lambda item: item[0])
        actual_cost, actual_position = _optimal_edge_placement(vertices, quadrics, first, second)
        self.assertAlmostEqual(actual_cost, expected_cost, places=9)
        np.testing.assert_allclose(actual_position, expected_position, rtol=1e-10, atol=1e-10)

    def test_li_nan_qem_keeps_component_count_on_multiple_open_islands(self):
        first_vertices, first_faces = make_grid(12)
        second_vertices, second_faces = make_grid(12, offset=3.0)
        vertices = np.vstack((first_vertices, second_vertices))
        faces = np.vstack((first_faces, second_faces + len(first_vertices)))
        initial_components = _count_components(faces)
        result = decimate_mesh_li_nan(
            vertices, faces, None, target_vertex_count=90,
            allow_boundary_collapses=False, collapse_intersection_guard=False,
            visualization_stride=10000,
        )
        self.assertEqual(initial_components, 2)
        self.assertEqual(_count_components(result.faces), initial_components)
        self.assertEqual(result.stats.initial_component_count, initial_components)
        self.assertEqual(result.stats.final_component_count, initial_components)
        self.assertGreaterEqual(len(result.vertices), 6)
        self.assertGreater(result.stats.successful_collapses, 0)

if __name__ == "__main__":
    unittest.main()
