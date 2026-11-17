from __future__ import annotations

import heapq

import numpy as np
import pytest

import app.feature_decimation as feature_decimation
import app.planar_regions as planar_regions
from app.feature_decimation import _compact_qem_heap, decimate_mesh_li_nan
from app.planar_regions import (
    PlanarRegionExtractionError,
    _build_edge_incidence,
    _face_geometry,
    _fit_plane_from_points,
    _refine_regions,
    extract_planar_regions_li_nan,
)

def make_grid(side: int, amplitude: float = 0.0):
    x, y = np.meshgrid(np.linspace(0.0, 1.0, side), np.linspace(0.0, 1.0, side))
    z = amplitude * np.sin(4.0 * x) * np.cos(3.0 * y)
    vertices = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    faces = []
    for row in range(side - 1):
        for column in range(side - 1):
            a = row * side + column
            b, c, d = a + 1, a + side, a + side + 1
            faces.extend(((a, b, d), (a, d, c)))
    return vertices.astype(np.float64), np.asarray(faces, dtype=np.int64)

def test_large_plane_fit_uses_covariance_eigensolve_and_matches_svd(monkeypatch):
    rng = np.random.default_rng(20261114)
    xy = rng.normal(size=(6000, 2)) * 0.35
    z = 0.17 * xy[:, 0] - 0.23 * xy[:, 1] + rng.normal(scale=2e-5, size=len(xy))
    points = np.column_stack((xy, z)) + np.asarray((12_000.0, -8_000.0, 300.0))
    centered = points - np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    expected_normal = vh[-1].copy()
    reference = np.asarray((-0.17, 0.23, 1.0), dtype=np.float64)
    if np.dot(expected_normal, reference) < 0.0:
        expected_normal *= -1.0

    original_svd = np.linalg.svd
    calls = []

    def tracked_svd(*args, **kwargs):
        calls.append(1)
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", tracked_svd)
    fitted = _fit_plane_from_points(points, reference)
    assert calls == []
    np.testing.assert_allclose(fitted.normal, expected_normal, rtol=1e-10, atol=1e-10)
    assert fitted.rms_distance < 5e-5
    assert fitted.point_count == len(points)

def test_ill_conditioned_large_plane_fit_falls_back_to_original_svd():
    x = np.linspace(-1.0, 1.0, 600)
    points = np.column_stack((x, 1e-15 * x, np.zeros_like(x)))
    with pytest.raises(PlanarRegionExtractionError, match="collinear"):
        _fit_plane_from_points(points, np.asarray((0.0, 0.0, 1.0)))

def test_region_refinement_plane_cache_reduces_repeated_fit_calls(monkeypatch):
    vertices, faces = make_grid(8)
    face_normals, _centers, face_areas = _face_geometry(vertices, faces)
    edge_to_faces, _neighbors = _build_edge_incidence(faces)
    coarse_groups = [np.asarray((face_id,), dtype=np.int64) for face_id in range(len(faces))]
    calls = 0
    original = planar_regions._fit_plane_from_points

    def counted_fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(planar_regions, "_fit_plane_from_points", counted_fit)
    groups, ungrouped = _refine_regions(
        vertices,
        faces,
        face_normals,
        face_areas,
        coarse_groups,
        edge_to_faces,
        small_region_area_threshold=0.02,
        fitting_threshold=10.0,
        normal_angle_degrees=89.9,
    )
    assert len(groups) == 1
    assert not ungrouped
    assert len(next(iter(groups.values()))) == len(faces)

    assert calls < 280

def test_planar_extraction_keeps_region_labels_and_contours_deterministic():
    vertices, faces = make_grid(30, amplitude=0.02)
    options = dict(
        small_region_area_threshold=0.00015,
        fitting_threshold=0.001,
        normal_angle_degrees=8.0,
    )
    first = extract_planar_regions_li_nan(vertices, faces, **options)
    second = extract_planar_regions_li_nan(vertices, faces, **options)
    np.testing.assert_array_equal(first.face_region_ids, second.face_region_ids)
    assert first.region_adjacency == second.region_adjacency
    assert len(first.contours) == len(second.contours)
    assert len(first.regions) == len(second.regions)
    for a, b in zip(first.regions, second.regions):
        np.testing.assert_array_equal(a.face_indices, b.face_indices)
        np.testing.assert_array_equal(a.vertex_indices, b.vertex_indices)
        np.testing.assert_allclose(a.plane.normal, b.plane.normal, rtol=0.0, atol=1e-13)
        assert abs(a.plane.rms_distance - b.plane.rms_distance) <= 1e-13

def _run_qem(vertices, faces):
    return decimate_mesh_li_nan(
        vertices,
        faces,
        None,
        target_vertex_count=max(8, int(len(vertices) * 0.45)),
        allow_boundary_collapses=True,
        visualization_stride=100_000,
    )

def test_qem_cached_unchanged_edge_costs_match_full_recompute(monkeypatch):
    vertices, faces = make_grid(18, amplitude=0.02)
    monkeypatch.setattr(feature_decimation, "_QEM_REUSE_UNCHANGED_EDGE_COSTS", True)
    cached = _run_qem(vertices, faces)
    monkeypatch.setattr(feature_decimation, "_QEM_REUSE_UNCHANGED_EDGE_COSTS", False)
    recomputed = _run_qem(vertices, faces)
    np.testing.assert_array_equal(cached.faces, recomputed.faces)
    np.testing.assert_array_equal(cached.vertices, recomputed.vertices)
    assert cached.stats.successful_collapses == recomputed.stats.successful_collapses
    assert cached.stats.queue_cached_cost_requeues > 0

def test_qem_heap_compaction_preserves_live_priority_order():

    heap = [
        (0.2, 1, (0, 1), 2),
        (0.1, 2, (1, 2), 1),
        (0.3, 3, (2, 3), 1),
        (0.05, 4, (3, 4), 5),
    ]
    heapq.heapify(heap)
    queued_generation = {(0, 1): 2, (1, 2): 4, (2, 3): 1, (3, 4): 5}
    edge_faces = {(0, 1): {0}, (1, 2): {1}, (2, 3): {2}}
    active = np.ones(5, dtype=bool)
    expected = sorted(
        item for item in heap
        if queued_generation.get(item[2]) == item[3]
        and item[2] in edge_faces
    )
    removed = _compact_qem_heap(heap, queued_generation, edge_faces, active)
    assert removed == 2
    actual = [heapq.heappop(heap) for _ in range(len(heap))]
    assert actual == expected
    assert all(len(item) == 4 and isinstance(item[2], tuple) for item in actual)

def test_qem_priority_queue_and_scratch_reuse_preserve_v0295_output_hash():
    import hashlib

    vertices, faces = make_grid(24, amplitude=0.018)
    result = _run_qem(vertices, faces)
    digest = hashlib.sha256(
        result.vertices.tobytes()
        + result.faces.tobytes()
        + result.original_vertex_indices.tobytes()
        + result.original_face_indices.tobytes()
    ).hexdigest()
    assert digest == "7b9d0b078da4027d2c8225920732a19ebce0d04786dd1095d5d8e8f507293af7"
    assert result.stats.successful_collapses == 317

def test_parallel_qem_candidate_evaluation_preserves_exact_greedy_output(monkeypatch):
    vertices, faces = make_grid(18, amplitude=0.018)
    options = dict(
        target_vertex_count=int(len(vertices) * 0.52),
        allow_boundary_collapses=True,
        visualization_stride=100_000,
    )
    monkeypatch.setattr(feature_decimation, "_QEM_PARALLEL_MIN_CANDIDATES", 10**9)
    serial = decimate_mesh_li_nan(vertices, faces, None, **options)

    monkeypatch.setattr(feature_decimation, "_QEM_PARALLEL_MIN_CANDIDATES", 128)
    parallel = decimate_mesh_li_nan(vertices, faces, None, **options)
    np.testing.assert_array_equal(parallel.vertices, serial.vertices)
    np.testing.assert_array_equal(parallel.faces, serial.faces)
    np.testing.assert_array_equal(parallel.original_vertex_indices, serial.original_vertex_indices)
    np.testing.assert_array_equal(parallel.original_face_indices, serial.original_face_indices)
    assert parallel.stats.successful_collapses == serial.stats.successful_collapses
    assert parallel.stats.candidate_evaluation_batches == 1
    assert parallel.stats.candidate_evaluation_workers >= 1
    assert parallel.stats.candidate_evaluation_backend.startswith("CPU-")
    assert parallel.stats.candidate_geometry_checks > 0
    assert parallel.stats.candidate_geometry_seconds > 0.0

def test_vectorized_qem_candidate_cost_and_geometry_match_scalar_reference():
    from app.feature_decimation import (
        _build_topology,
        _collapse_geometry_is_safe,
        _qem_candidate_costs_vectorized,
        _qem_candidate_geometry_safe_vectorized,
    )

    vertices, faces = make_grid(9, amplitude=0.017)
    quadrics = feature_decimation._initial_vertex_quadrics(
        len(vertices), faces, np.full(len(faces), -1, dtype=np.int64),
        feature_decimation._face_plane_quadrics(vertices, faces), {},
    )
    _vertex_faces, _vertex_neighbors, edge_faces = _build_topology(faces, len(vertices))
    edges = np.asarray(sorted(edge_faces), dtype=np.int64)
    costs, positions = _qem_candidate_costs_vectorized(vertices, quadrics, edges)
    for index, (first, second) in enumerate(edges):
        reference_cost, reference_position = feature_decimation._optimal_edge_placement(
            vertices, quadrics, int(first), int(second)
        )
        assert costs[index] == reference_cost
        np.testing.assert_array_equal(positions[index], reference_position)

    active_faces = np.ones(len(faces), dtype=bool)
    vertex_faces, _neighbors, _edge_faces = _build_topology(faces, len(vertices))
    area_epsilon = 1e-14
    batch_safe = _qem_candidate_geometry_safe_vectorized(
        vertices, faces, active_faces, vertex_faces, edges, positions, area_epsilon
    )
    scalar_safe = np.asarray([
        _collapse_geometry_is_safe(
            vertices, faces, active_faces, vertex_faces,
            int(first), int(second), positions[index], area_epsilon,
        )
        for index, (first, second) in enumerate(edges)
    ], dtype=bool)
    np.testing.assert_array_equal(batch_safe, scalar_safe)

def test_qem_priority_queue_records_do_not_retain_numpy_placement_payloads(monkeypatch):
    vertices, faces = make_grid(10, amplitude=0.012)
    captured = []
    original_push = heapq.heappush

    def tracked_push(heap, item):
        assert len(item) == 4
        assert isinstance(item[2], tuple) and len(item[2]) == 2
        assert all(not isinstance(value, np.ndarray) for value in item)
        captured.append(item)
        original_push(heap, item)

    monkeypatch.setattr(feature_decimation.heapq, "heappush", tracked_push)
    result = _run_qem(vertices, faces)
    assert captured
    assert result.stats.successful_collapses > 0

def test_qem_reuses_candidate_neighborhood_scratch_sets(monkeypatch):
    vertices, faces = make_grid(16, amplitude=0.015)
    affected_ids = []
    generated_ids = []
    geometry_ids = []
    original_topology_check = feature_decimation._topology_rejection_reason
    original_geometry_check = feature_decimation._collapse_geometry_is_safe

    def topology_spy(*args, **kwargs):
        affected_ids.append(kwargs["affected_face_ids"])
        generated_ids.append(kwargs["generated_face_keys"])
        return original_topology_check(*args, **kwargs)

    def geometry_spy(*args, **kwargs):
        geometry_ids.append(kwargs["affected_face_ids"])
        return original_geometry_check(*args, **kwargs)

    monkeypatch.setattr(feature_decimation, "_topology_rejection_reason", topology_spy)
    monkeypatch.setattr(feature_decimation, "_collapse_geometry_is_safe", geometry_spy)
    result = _run_qem(vertices, faces)
    assert result.stats.successful_collapses > 0
    assert len(affected_ids) > 1
    assert len(generated_ids) == len(affected_ids)
    assert len(geometry_ids) > 0
    assert all(value is affected_ids[0] for value in affected_ids)
    assert all(value is affected_ids[0] for value in geometry_ids)
    assert all(value is generated_ids[0] for value in generated_ids)

def test_qem_forced_heap_compaction_does_not_change_result(monkeypatch):
    vertices, faces = make_grid(20, amplitude=0.02)
    monkeypatch.setattr(feature_decimation, "_QEM_HEAP_COMPACT_MIN_SIZE", 1)
    monkeypatch.setattr(feature_decimation, "_QEM_HEAP_COMPACT_STALE_FACTOR", 0.2)
    compacted = _run_qem(vertices, faces)
    assert compacted.stats.queue_compaction_count > 0
    assert compacted.stats.queue_stale_entries_removed > 0

    monkeypatch.setattr(feature_decimation, "_QEM_HEAP_COMPACT_MIN_SIZE", 10**9)
    monkeypatch.setattr(feature_decimation, "_QEM_HEAP_COMPACT_STALE_FACTOR", 1e9)
    uncompacted = _run_qem(vertices, faces)
    np.testing.assert_array_equal(compacted.faces, uncompacted.faces)
    np.testing.assert_array_equal(compacted.vertices, uncompacted.vertices)
    assert compacted.stats.successful_collapses == uncompacted.stats.successful_collapses

def _legacy_set_topology(faces, vertex_count):
    """Reference set-based one-ring topology builder for exact QEM parity."""
    vertex_faces = [set() for _ in range(vertex_count)]
    vertex_neighbors = [set() for _ in range(vertex_count)]
    edge_faces = {}
    for face_id, tri in enumerate(np.asarray(faces, dtype=np.int64)):
        a, b, c = map(int, tri)
        for vertex_id in (a, b, c):
            vertex_faces[vertex_id].add(face_id)
        for first, second in ((a, b), (b, c), (c, a)):
            edge = feature_decimation._edge_key(first, second)
            edge_faces.setdefault(edge, set()).add(face_id)
            vertex_neighbors[first].add(second)
            vertex_neighbors[second].add(first)
    return vertex_faces, vertex_neighbors, edge_faces

def _incidence_ids_for_test(value):
    if value is None:
        return set()
    if isinstance(value, (int, np.integer)):
        return {int(value)}
    return set(value)

def test_compact_topology_matches_legacy_vertex_and_edge_incidence():
    faces = np.asarray((
        (0, 1, 2),
        (1, 0, 3),
        (0, 1, 4),
        (0, 1, 2),
        (5, 6, 7),
    ), dtype=np.int64)
    expected = _legacy_set_topology(faces, 8)
    actual = feature_decimation._build_topology(faces, 8)
    for vertex_id in range(8):
        assert set(actual[0][vertex_id]) == expected[0][vertex_id]
        assert set(actual[1][vertex_id]) == expected[1][vertex_id]
    assert {key: _incidence_ids_for_test(value) for key, value in actual[2].items()} == expected[2]
    assert all(isinstance(key, tuple) and len(key) == 2 for key in actual[2])
    assert all(isinstance(value, (int, tuple)) for value in actual[2].values())

def test_duplicate_collapse_result_is_rejected_by_link_condition_without_face_key_map():

    faces = np.asarray((
        (0, 1, 2),
        (0, 1, 5),
        (0, 3, 4),
        (1, 3, 4),
    ), dtype=np.int64)
    vertex_faces, vertex_neighbors, edge_faces = feature_decimation._build_topology(faces, 6)
    reason = feature_decimation._topology_rejection_reason(
        faces,
        np.ones(len(faces), dtype=bool),
        vertex_faces,
        vertex_neighbors,
        edge_faces,
        0,
        1,
        allow_boundary=True,
    )
    assert reason == "link"

def test_compact_edge_incidence_materializes_as_native_set_on_first_edit():
    mapping = {(2, 7): (11, 12), (3, 8): 14}
    materialized = feature_decimation._mutable_incidence_set(mapping, (2, 7))
    assert materialized == {11, 12}
    assert mapping[(2, 7)] is materialized
    materialized.discard(12)
    materialized.add(13)
    assert mapping[(2, 7)] == {11, 13}

    new_edge = feature_decimation._mutable_incidence_set(mapping, (4, 9))
    new_edge.add(15)
    assert mapping[(4, 9)] == {15}
    assert mapping[(3, 8)] == 14

def test_compact_topology_qem_output_matches_legacy_set_topology(monkeypatch):
    vertices, faces = make_grid(24, amplitude=0.018)
    compact = _run_qem(vertices, faces)
    monkeypatch.setattr(feature_decimation, "_build_topology", _legacy_set_topology)
    reference = _run_qem(vertices, faces)
    np.testing.assert_array_equal(compact.vertices, reference.vertices)
    np.testing.assert_array_equal(compact.faces, reference.faces)
    np.testing.assert_array_equal(compact.face_region_ids, reference.face_region_ids)
    np.testing.assert_array_equal(compact.original_vertex_indices, reference.original_vertex_indices)
    np.testing.assert_array_equal(compact.original_face_indices, reference.original_face_indices)
    assert compact.stats.successful_collapses == reference.stats.successful_collapses
    assert compact.stats.initial_boundary_edge_count == reference.stats.initial_boundary_edge_count
    assert compact.stats.final_boundary_edge_count == reference.stats.final_boundary_edge_count
    assert compact.stats.final_non_manifold_edge_count == reference.stats.final_non_manifold_edge_count
    assert (
        compact.stats.topology_compact_edge_entries
        + compact.stats.topology_materialized_edge_entries
        == compact.stats.topology_compact_map_entries
    )

def test_adaptive_qem_policy_scales_queue_pressure_for_mesh_complexity():
    small = feature_decimation._adaptive_qem_policy(2_000, 4_000, 1_000)
    medium = feature_decimation._adaptive_qem_policy(12_000, 24_000, 8_000)
    large = feature_decimation._adaptive_qem_policy(80_000, 160_000, 50_000)
    xlarge = feature_decimation._adaptive_qem_policy(300_000, 600_000, 200_000)

    assert (small.tier, medium.tier, large.tier, xlarge.tier) == (
        "small", "medium", "large", "xlarge"
    )
    assert small.compact_min_entries < medium.compact_min_entries < large.compact_min_entries < xlarge.compact_min_entries
    assert small.stale_ratio >= medium.stale_ratio > large.stale_ratio > xlarge.stale_ratio
    assert small.stale_memory_budget_bytes < medium.stale_memory_budget_bytes < large.stale_memory_budget_bytes < xlarge.stale_memory_budget_bytes

def test_adaptive_qem_heap_compaction_can_be_triggered_by_memory_pressure():
    from app.feature_decimation import AdaptiveQEMPolicy, _should_compact_adaptive_heap

    policy = AdaptiveQEMPolicy(
        tier="test",
        compact_min_entries=10,
        stale_ratio=100.0,
        stale_divisor=1,
        stale_floor=2,
        stale_memory_budget_bytes=1_000,
        max_progress_callbacks=10,
    )
    should_compact, memory_triggered = _should_compact_adaptive_heap(
        heap_size=100,
        live_entries=80,
        stale_entries=20,
        policy=policy,
        entry_bytes=128,
    )
    assert should_compact is True
    assert memory_triggered is True

def test_adaptive_qem_progress_callback_is_throttled_and_flushes_final_state():
    vertices, faces = make_grid(28, amplitude=0.01)
    target = 100
    callbacks = []
    result = decimate_mesh_li_nan(
        vertices,
        faces,
        None,
        target_vertex_count=target,
        allow_boundary_collapses=True,
        collapse_intersection_guard=False,
        visualization_stride=100_000,
        progress_cb=lambda done, total, current: callbacks.append((done, total, current)),
    )
    assert result.stats.adaptive_progress_callback_stride > 1
    assert result.stats.progress_callback_count == len(callbacks)
    assert len(callbacks) <= 512
    assert callbacks
    assert callbacks[-1][0] == result.stats.successful_collapses
    assert callbacks[-1][2] == result.stats.output_vertex_count or callbacks[-1][2] >= result.stats.output_vertex_count

def test_adaptive_and_fixed_qem_policies_keep_exact_output_arrays(monkeypatch):
    vertices, faces = make_grid(72, amplitude=0.018)
    kwargs = dict(
        target_vertex_count=int(len(vertices) * 0.45),
        allow_boundary_collapses=True,
        collapse_intersection_guard=False,
        visualization_stride=100_000,
    )
    monkeypatch.setattr(feature_decimation, "_QEM_ADAPTIVE_POLICY_ENABLED", False)
    fixed = decimate_mesh_li_nan(vertices, faces, None, **kwargs)
    monkeypatch.setattr(feature_decimation, "_QEM_ADAPTIVE_POLICY_ENABLED", True)
    adaptive = decimate_mesh_li_nan(vertices, faces, None, **kwargs)
    np.testing.assert_array_equal(adaptive.vertices, fixed.vertices)
    np.testing.assert_array_equal(adaptive.faces, fixed.faces)
    np.testing.assert_array_equal(adaptive.face_region_ids, fixed.face_region_ids)
    assert adaptive.stats.successful_collapses == fixed.stats.successful_collapses
    assert adaptive.stats.adaptive_qem_tier == "medium"

def test_memory_budget_compaction_preserves_exact_qem_result(monkeypatch):
    from app.feature_decimation import AdaptiveQEMPolicy

    vertices, faces = make_grid(16, amplitude=0.014)
    kwargs = dict(
        target_vertex_count=150,
        allow_boundary_collapses=True,
        collapse_intersection_guard=False,
        visualization_stride=100_000,
    )
    monkeypatch.setattr(feature_decimation, "_QEM_ADAPTIVE_POLICY_ENABLED", False)
    reference = decimate_mesh_li_nan(vertices, faces, None, **kwargs)
    forced_memory_policy = AdaptiveQEMPolicy(
        tier="memory-test",
        compact_min_entries=1,
        stale_ratio=1_000.0,
        stale_divisor=1,
        stale_floor=1,
        stale_memory_budget_bytes=1,
        max_progress_callbacks=512,
    )
    monkeypatch.setattr(
        feature_decimation,
        "_adaptive_qem_policy",
        lambda *_args: forced_memory_policy,
    )
    compacted = decimate_mesh_li_nan(vertices, faces, None, **kwargs)
    assert compacted.stats.queue_memory_compaction_count > 0
    assert compacted.stats.queue_compaction_count > 0
    np.testing.assert_array_equal(compacted.vertices, reference.vertices)
    np.testing.assert_array_equal(compacted.faces, reference.faces)
    np.testing.assert_array_equal(compacted.face_region_ids, reference.face_region_ids)
