import unittest

import numpy as np

from app.feature_preserving import filter_mesh_li_nan
from app.feature_decimation import decimate_mesh_li_nan
from app.planar_regions import extract_planar_regions_li_nan
from app.preview_visuals import active_face_ids_from_vertex_motion, face_edge_segments

def make_noisy_grid(size=6, amplitude=0.02):
    vertices = np.asarray([
        (x / (size - 1), y / (size - 1), amplitude * np.sin(x * 1.7) * np.cos(y * 1.2))
        for y in range(size) for x in range(size)
    ], dtype=np.float64)
    faces = []
    for y in range(size - 1):
        for x in range(size - 1):
            a = y * size + x
            b, c, d = a + 1, a + size, a + size + 1
            faces.extend(((a, b, d), (a, d, c)))
    return vertices, np.asarray(faces, dtype=np.int64)

class RealtimeAlgorithmDrawingTests(unittest.TestCase):
    def test_normal_filter_emits_one_geometry_frame_per_iteration(self):
        vertices, faces = make_noisy_grid()
        locked = np.asarray((0, 5, 30, 35), dtype=np.int64)
        frames = []
        output_vertices, output_faces, stats = filter_mesh_li_nan(
            vertices,
            faces,
            normal_angle_degrees=30.0,
            iterations=4,
            locked_vertices=locked,
            preview_cb=lambda *args: frames.append(args),
        )
        self.assertEqual(stats.completed_iterations, 4)
        self.assertEqual(len(frames), 4)
        self.assertTrue(np.array_equal(output_faces, faces))
        self.assertEqual(frames[0][3].shape, faces.shape)
        self.assertEqual(frames[0][4].shape, (len(faces), 3))
        self.assertTrue(np.isfinite(frames[0][4]).all())
        self.assertTrue(all(float(frame[5]) >= 0.0 for frame in frames))
        np.testing.assert_allclose(output_vertices[locked], vertices[locked], atol=1e-12)
        self.assertGreater(float(np.max(np.linalg.norm(output_vertices - vertices, axis=1))), 0.0)

    def test_active_face_selection_tracks_the_vertices_moved_in_the_current_iteration(self):
        previous, faces = make_noisy_grid(size=4, amplitude=0.0)
        current = previous.copy()
        current[5, 2] += 0.25
        active = active_face_ids_from_vertex_motion(previous, current, faces, max_faces=2)
        self.assertGreater(len(active), 0)
        self.assertLessEqual(len(active), 2)
        self.assertTrue(np.all(np.isin(active, np.flatnonzero(np.any(faces == 5, axis=1)))))

    def test_li_nan_planar_region_extraction_emits_live_growth_faces(self):
        vertices, faces = make_noisy_grid(size=6, amplitude=0.001)
        frames = []
        result = extract_planar_regions_li_nan(
            vertices,
            faces,
            small_region_area_threshold=0.0001,
            fitting_threshold=0.1,
            normal_angle_degrees=30.0,
            visualization_cb=lambda *args: frames.append(args),
        )
        self.assertEqual(result.stats.input_faces, len(faces))
        self.assertGreater(len(frames), 0)
        labels, active_faces, done, total, phase = frames[0]
        self.assertEqual(labels.shape, (len(faces),))
        self.assertGreater(len(active_faces), 0)
        self.assertGreater(done, 0)
        self.assertEqual(total, len(faces))
        self.assertIn(phase, {"COARSE REGION GROWTH", "COARSE REGION GROWTH COMPLETE", "REFINED REGION PLANE FIT", "CONTOUR EDGE EXTRACTION"})

    def test_planar_angle_changes_the_normal_filter_vertex_solution(self):
        vertices, faces = make_noisy_grid()
        low_angle, _, _ = filter_mesh_li_nan(
            vertices, faces, normal_angle_degrees=10.0, iterations=3
        )
        high_angle, _, _ = filter_mesh_li_nan(
            vertices, faces, normal_angle_degrees=60.0, iterations=3
        )
        self.assertGreater(
            float(np.max(np.linalg.norm(low_angle - high_angle, axis=1))),
            1e-5,
        )

    def test_qem_emits_live_mesh_and_recent_edge_segments(self):
        vertices, faces = make_noisy_grid(size=5, amplitude=0.03)
        frames = []
        result = decimate_mesh_li_nan(
            vertices,
            faces,
            None,
            target_vertex_count=12,
            visualization_cb=lambda *args: frames.append(args),
            visualization_stride=1,
            allow_boundary_collapses=True,
        )
        self.assertGreater(result.stats.successful_collapses, 0)
        self.assertGreater(len(frames), 0)
        current_vertices, current_faces, region_ids, edge_segments, done, total, vertices_left = frames[0]
        self.assertEqual(current_vertices.ndim, 2)
        self.assertEqual(current_faces.ndim, 2)
        self.assertEqual(current_faces.shape[1], 3)
        self.assertEqual(region_ids.shape, (len(current_faces),))
        self.assertTrue(np.any(region_ids == -2))
        self.assertEqual(edge_segments.ndim, 3)
        self.assertEqual(edge_segments.shape[1:], (2, 3))
        self.assertGreater(done, 0)
        self.assertGreater(total, 0)
        self.assertGreater(vertices_left, 0)

    def test_active_faces_are_rendered_as_unique_edge_segments(self):
        vertices, faces = make_noisy_grid(size=4, amplitude=0.0)
        selected_faces = np.asarray((0, 1, 4), dtype=np.int64)
        segments = face_edge_segments(vertices, faces, selected_faces)
        self.assertEqual(segments.ndim, 3)
        self.assertEqual(segments.shape[1:], (2, 3))
        self.assertGreater(len(segments), 0)
        self.assertTrue(np.isfinite(segments).all())
        edges = np.sort(np.round(segments, 8), axis=1).reshape(-1, 6)
        self.assertEqual(len(edges), len(np.unique(edges, axis=0)))

    def test_face_edge_segments_ignores_invalid_face_indices(self):
        vertices, faces = make_noisy_grid(size=3, amplitude=0.0)
        segments = face_edge_segments(vertices, faces, np.asarray((-1, len(faces), 0)))
        self.assertEqual(segments.shape, (3, 2, 3))

if __name__ == "__main__":
    unittest.main()
