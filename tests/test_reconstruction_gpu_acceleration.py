import numpy as np

from app.feature_preserving import (
    _average_unique_edge_length,
    _bilateral_filter_face_normals,
    _csr_offsets_from_sorted_ids,
    _face_geometry,
    _topological_face_neighborhoods,
    _update_vertices,
    filter_mesh_li_nan,
)
from app.gpu_compute import _filter_mesh_step_xp
from app.planar_regions import _build_edge_incidence, extract_planar_regions_li_nan

def make_grid(size=32, amplitude=0.025):
    xy = np.stack(
        np.meshgrid(np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, size)),
        axis=-1,
    ).reshape(-1, 2)
    z = amplitude * np.sin(xy[:, 0] * 11.0) * np.cos(xy[:, 1] * 9.0)
    vertices = np.column_stack((xy, z)).astype(np.float64)
    faces = []
    for y in range(size - 1):
        for x in range(size - 1):
            a = y * size + x
            b, c, d = a + 1, a + size, a + size + 1
            faces.extend(((a, b, d), (a, d, c)))
    return vertices, np.asarray(faces, dtype=np.int64)

def reference_face_neighborhoods(vertex_count, faces):
    vertex_faces = [[] for _ in range(vertex_count)]
    for face_id, triangle in enumerate(faces):
        for vertex_id in triangle:
            vertex_faces[int(vertex_id)].append(face_id)
    pairs = []
    for face_id, triangle in enumerate(faces):
        candidates = set()
        for vertex_id in triangle:
            candidates.update(vertex_faces[int(vertex_id)])
        pairs.extend((face_id, neighbor) for neighbor in sorted(candidates))
    return np.asarray(pairs, dtype=np.int64)

def reference_edge_incidence(faces):
    edge_to_faces = {}
    for face_id, triangle in enumerate(faces):
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = tuple(sorted((int(first), int(second))))
            edge_to_faces.setdefault(edge, []).append(face_id)
    neighbors = [set() for _ in range(len(faces))]
    for incident in edge_to_faces.values():
        unique = sorted(set(incident))
        for offset, first in enumerate(unique):
            for second in unique[offset + 1:]:
                neighbors[first].add(second)
                neighbors[second].add(first)
    return edge_to_faces, [sorted(item) for item in neighbors]

class NumpyComputeManager:
    """Exercises the same array-backend contract without requiring a GPU host."""

    backend_name = "CUDA"

    def __init__(self):
        self.backend = object()
        self.filter_gpu_seconds_total = 0.0
        self.filter_cpu_reference_seconds = 0.0
        self.filter_fallback_reason = ""
        self.disabled = False
        self.batches = 0

    def filter_mesh_step(self, *args):
        if self.disabled:
            return None
        self.batches += 1
        return _filter_mesh_step_xp(np, *args, dtype=np.float64)

    def disable_filter_kernel(self, reason):
        self.disabled = True
        self.filter_fallback_reason = str(reason)

    def record_filter_cpu_reference(self, elapsed):
        self.filter_cpu_reference_seconds += max(0.0, float(elapsed))

def test_qem_gpu_candidate_array_kernel_matches_exact_cpu_batch_and_manager_guard():
    from app.feature_decimation import (
        _build_topology,
        _face_plane_quadrics,
        _initial_vertex_quadrics,
        _qem_candidate_costs_vectorized,
    )
    from app.gpu_compute import GPUComputeManager, _qem_candidate_batch_xp

    vertices, faces = make_grid(size=9, amplitude=0.017)
    quadrics = _initial_vertex_quadrics(
        len(vertices),
        faces,
        np.full(len(faces), -1, dtype=np.int64),
        _face_plane_quadrics(vertices, faces),
        {},
    )
    _vertex_faces, _vertex_neighbors, edge_faces = _build_topology(faces, len(vertices))
    edges = np.asarray(sorted(edge_faces), dtype=np.int64)
    cpu_costs, cpu_positions = _qem_candidate_costs_vectorized(vertices, quadrics, edges)
    array_costs, array_positions = _qem_candidate_batch_xp(np, vertices, quadrics, edges)
    np.testing.assert_array_equal(array_costs, cpu_costs)
    np.testing.assert_array_equal(array_positions, cpu_positions)

    class ExactArrayBackend:
        precision = "float64"

        def qem_candidate_batch(self, source_vertices, source_quadrics, source_edges):
            return _qem_candidate_batch_xp(
                np, source_vertices, source_quadrics, source_edges
            )

    manager = GPUComputeManager.__new__(GPUComputeManager)
    manager.backend = ExactArrayBackend()
    manager.backend_name = "CUDA"
    manager.precision = "float64"
    manager._qem_candidate_disabled = False
    manager._qem_candidate_parity_verified = False
    manager.qem_candidate_fallback_reason = ""
    manager.qem_candidate_seconds_total = 0.0
    manager.qem_candidate_batches = 0
    checked = manager.qem_candidate_batch(vertices, quadrics, edges)
    assert checked is not None
    np.testing.assert_array_equal(checked[0], cpu_costs)
    np.testing.assert_array_equal(checked[1], cpu_positions)
    assert manager._qem_candidate_parity_verified is True
    assert manager.qem_candidate_batches == 1

    class InexactArrayBackend(ExactArrayBackend):
        def qem_candidate_batch(self, source_vertices, source_quadrics, source_edges):
            costs, positions = super().qem_candidate_batch(
                source_vertices, source_quadrics, source_edges
            )
            costs = costs.copy()
            costs[-1] += 1.0
            return costs, positions

    manager.backend = InexactArrayBackend()
    manager._qem_candidate_disabled = False
    manager._qem_candidate_parity_verified = False
    assert manager.qem_candidate_batch(vertices, quadrics, edges) is None
    assert manager._qem_candidate_disabled is True
    assert "bit-exact parity" in manager.qem_candidate_fallback_reason

def test_qem_opencl_and_hip_kernel_sources_include_candidate_placement_entrypoint():
    from app.gpu_compute import (
        _OPENCL_QEM_CANDIDATE_TEMPLATE,
        _hip_kernel_source,
        _hip_qem_kernel_source,
    )

    opencl_source = _OPENCL_QEM_CANDIDATE_TEMPLATE.format(
        fp64_pragma="", real_type="double"
    )
    hip_base_source = _hip_kernel_source(include_qem=False)
    hip_qem_source = _hip_qem_kernel_source()
    hip_combined_source = _hip_kernel_source()
    assert "__kernel void qem_candidate_placements" in opencl_source
    assert "qem_candidate_placements" not in hip_base_source
    assert 'extern "C" __global__ void qem_candidate_placements' in hip_qem_source
    assert 'extern "C" __global__ void qem_candidate_placements' in hip_combined_source
    for hip_source in (hip_base_source, hip_qem_source, hip_combined_source):
        assert "__global const" not in hip_source
        assert "__private " not in hip_source
        assert "get_global_id" not in hip_source

def test_sparse_face_incidence_matches_exact_previous_one_ring_definition():
    vertices, faces = make_grid(size=7)
    query_ids, neighbor_ids, vertex_face_ids, incident_face_ids = _topological_face_neighborhoods(
        len(vertices), faces
    )
    actual = np.column_stack((query_ids, neighbor_ids))
    expected = reference_face_neighborhoods(len(vertices), faces)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(vertex_face_ids, faces.reshape(-1))
    np.testing.assert_array_equal(incident_face_ids, np.repeat(np.arange(len(faces)), 3))

def test_vectorized_planar_edge_incidence_matches_nonmanifold_reference():

    faces = np.asarray(
        ((0, 1, 2), (1, 0, 3), (0, 1, 4), (2, 3, 5)),
        dtype=np.int64,
    )
    actual_edges, actual_neighbors = _build_edge_incidence(faces)
    expected_edges, expected_neighbors = reference_edge_incidence(faces)
    assert actual_edges == expected_edges
    assert actual_neighbors == expected_neighbors

def test_array_backend_filter_step_matches_numpy_reference_equations():
    vertices, faces = make_grid(size=12)
    query_ids, neighbor_ids, vertex_face_ids, incident_face_ids = _topological_face_neighborhoods(
        len(vertices), faces
    )
    vertex_order = np.argsort(vertex_face_ids, kind="stable")
    vertex_offsets = _csr_offsets_from_sorted_ids(
        vertex_face_ids[vertex_order], len(vertices)
    )
    vertex_incident_faces = incident_face_ids[vertex_order]
    face_offsets = _csr_offsets_from_sorted_ids(query_ids, len(faces))
    normals, centers, areas = _face_geometry(vertices, faces)
    sigma = _average_unique_edge_length(vertices, faces)
    theta = np.deg2rad(20.0)

    expected_normals = _bilateral_filter_face_normals(
        normals, centers, areas, query_ids, neighbor_ids, sigma, theta
    )
    expected_vertices, expected_displacement = _update_vertices(
        vertices, centers, normals, expected_normals, vertex_face_ids, incident_face_ids
    )
    actual = _filter_mesh_step_xp(
        np, vertices, normals, centers, areas, face_offsets, neighbor_ids,
        vertex_offsets, vertex_incident_faces, sigma, theta, dtype=np.float64,
    )

    np.testing.assert_allclose(actual[0], expected_normals, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected_vertices, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[2], expected_displacement, rtol=1e-12, atol=1e-12)

def test_li_nan_filter_uses_accelerated_batches_only_after_cpu_parity_check():
    vertices, faces = make_grid(size=32)
    manager = NumpyComputeManager()
    accelerated_vertices, accelerated_faces, accelerated_stats = filter_mesh_li_nan(
        vertices, faces, normal_angle_degrees=20.0, iterations=2, compute_manager=manager
    )
    cpu_vertices, cpu_faces, cpu_stats = filter_mesh_li_nan(
        vertices, faces, normal_angle_degrees=20.0, iterations=2
    )
    assert manager.batches == 2
    assert not manager.disabled
    assert accelerated_stats.filter_compute_backend == "CUDA"
    assert accelerated_stats.filter_gpu_batches == 2
    assert accelerated_stats.filter_gpu_seconds >= 0.0
    assert accelerated_stats.filter_cpu_reference_seconds > 0.0
    np.testing.assert_array_equal(accelerated_faces, faces)
    np.testing.assert_array_equal(cpu_faces, faces)
    np.testing.assert_allclose(accelerated_vertices, cpu_vertices, rtol=1e-12, atol=1e-12)
    assert accelerated_stats.completed_iterations == cpu_stats.completed_iterations == 2

def test_planar_extraction_reports_phase_timings():
    vertices, faces = make_grid(size=8, amplitude=0.001)
    result = extract_planar_regions_li_nan(
        vertices,
        faces,
        small_region_area_threshold=0.0001,
        fitting_threshold=0.05,
        normal_angle_degrees=30.0,
    )
    timings = (
        result.stats.edge_incidence_seconds,
        result.stats.coarse_region_seconds,
        result.stats.refinement_seconds,
        result.stats.plane_fit_seconds,
        result.stats.contour_seconds,
    )
    assert all(np.isfinite(value) and value >= 0.0 for value in timings)
    assert sum(timings) > 0.0

def test_small_mesh_keeps_cpu_path_to_avoid_gpu_transfer_overhead():
    vertices, faces = make_grid(size=8)
    manager = NumpyComputeManager()
    _output_vertices, _output_faces, stats = filter_mesh_li_nan(
        vertices, faces, iterations=1, compute_manager=manager
    )
    assert manager.batches == 0
    assert stats.filter_compute_backend == "CPU"
    assert "below the 12,000 entry GPU threshold" in stats.filter_fallback_reason

def test_gpu_filter_parity_failure_disables_kernel_and_reuses_cpu_reference():
    vertices, faces = make_grid(size=32)
    cpu_vertices, cpu_faces, _cpu_stats = filter_mesh_li_nan(vertices, faces, iterations=1)

    class BadComputeManager(NumpyComputeManager):
        def filter_mesh_step(self, *args):
            self.batches += 1
            normals, updated, displacement = _filter_mesh_step_xp(np, *args, dtype=np.float64)
            normals = normals.copy()
            normals[0, 0] += 0.1
            return normals, updated, displacement

    manager = BadComputeManager()
    output_vertices, output_faces, stats = filter_mesh_li_nan(
        vertices, faces, iterations=1, compute_manager=manager
    )
    assert manager.disabled
    assert stats.filter_compute_backend == "CPU"
    assert "failed the first-iteration CPU parity check" in stats.filter_fallback_reason
    np.testing.assert_array_equal(output_faces, cpu_faces)
    np.testing.assert_allclose(output_vertices, cpu_vertices, rtol=1e-12, atol=1e-12)

def test_large_kdtree_candidate_batches_use_parallel_workers():
    from app.texture_baking import _query_nearest_face_candidates

    class FakeTree:
        def __init__(self):
            self.workers = []

        def query(self, points, k, workers=None):
            self.workers.append(workers)
            return np.zeros(len(points)), np.zeros(len(points), dtype=np.int64)

    tree = FakeTree()
    _query_nearest_face_candidates(tree, np.zeros((128, 3)), 8)
    _query_nearest_face_candidates(tree, np.zeros((4096, 3)), 8)
    assert tree.workers == [1, -1]

def test_kdtree_query_falls_back_for_older_scipy_worker_signature():
    from app.texture_baking import _query_nearest_face_candidates

    class OlderTree:
        def __init__(self):
            self.calls = 0

        def query(self, points, k):
            self.calls += 1
            return np.zeros(len(points)), np.zeros(len(points), dtype=np.int64)

    tree = OlderTree()
    _query_nearest_face_candidates(tree, np.zeros((16, 3)), 4)
    assert tree.calls == 1

def test_opencl_filter_reuses_cached_topology_buffers_across_iterations():
    from types import SimpleNamespace

    from app.gpu_compute import _OpenCLBackend

    class FakeFlags:
        READ_ONLY = 1
        COPY_HOST_PTR = 2
        WRITE_ONLY = 4

    class FakeCL:
        mem_flags = FakeFlags()

        def __init__(self):
            self.buffers = []

        def Buffer(self, context, flags, size=None, hostbuf=None):
            result = SimpleNamespace(context=context, flags=flags, size=size, hostbuf=hostbuf)
            self.buffers.append(result)
            return result

        @staticmethod
        def enqueue_copy(queue, destination, source, is_blocking=True):
            return None

    class FakeProgram:
        def __init__(self):
            self.calls = []

        def filter_face_normals_csr(self, *args):
            self.calls.append(("normals", args))

        def update_filter_vertices_csr(self, *args):
            self.calls.append(("vertices", args))

    backend = _OpenCLBackend.__new__(_OpenCLBackend)
    backend.cl = FakeCL()
    backend.dtype = np.float64
    backend.context = object()
    backend.queue = object()
    backend.program = FakeProgram()
    backend._filter_topology_key = None
    backend._filter_topology_buffers = None

    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normals = np.asarray([[0.0, 0.0, 1.0]])
    centers = np.asarray([[1.0 / 3.0, 1.0 / 3.0, 0.0]])
    areas = np.asarray([0.5])
    f_offsets = np.asarray([0, 1], dtype=np.int64)
    f_neighbors = np.asarray([0], dtype=np.int64)
    v_offsets = np.asarray([0, 1, 2, 3], dtype=np.int64)
    v_incident = np.asarray([0, 0, 0], dtype=np.int64)

    first = backend.filter_mesh_step(
        vertices, normals, centers, areas,
        f_offsets, f_neighbors, v_offsets, v_incident, 1.0, 0.5,
    )
    cached_topology = backend._filter_topology_buffers
    first_buffer_count = len(backend.cl.buffers)
    second = backend.filter_mesh_step(
        vertices, normals, centers, areas,
        f_offsets, f_neighbors, v_offsets, v_incident, 1.0, 0.5,
    )

    assert backend._filter_topology_buffers is cached_topology
    assert len(backend.cl.buffers) - first_buffer_count == 7
    assert len(backend.program.calls) == 4
    assert len(first) == len(second) == 3

def test_gpu_manager_reports_reconstruction_filter_batches_as_gpu_active():
    from types import SimpleNamespace

    from app.gpu_compute import GPUComputeManager

    manager = SimpleNamespace(
        selection="auto",
        strict_gpu_only=False,
        performance_fallback=False,
        backend_name="OpenCL",
        device_name="AMD Radeon",
        vendor="AMD",
        memory_bytes=8 * 1024**3,
        precision="float64",
        status="GPU · verified",
        activation_verified=True,
        gpu_batch_count=0,
        filter_gpu_batches=3,
        backend=object(),
        fallback_reason="",
        cpu_reference_seconds=0.0,
        gpu_benchmark_seconds=0.0,
        speedup_ratio=0.0,
        speedup_gain_percent=0.0,
        time_saved_percent=0.0,
        gpu_compute_seconds_total=0.0,
        gpu_intersection_tests_total=0,
        gpu_closest_point_tests_total=0,
        gpu_tests_per_second=0.0,
        filter_backend_name="OpenCL",
        filter_gpu_seconds_total=0.25,
        filter_cpu_reference_seconds=0.05,
        filter_fallback_reason="",
        benchmark_valid=False,
        benchmark_detail="Reconstruction filter stage",
    )

    info = GPUComputeManager.info(manager)
    assert info["gpu_active"] is True
    assert info["reconstruction_filter_active"] is True
    assert info["reconstruction_filter_gpu_batches"] == 3
    assert info["reconstruction_qem_candidate_active"] is False

    manager.filter_gpu_batches = 0
    manager.qem_candidate_batches = 2
    assert GPUComputeManager.info(manager)["gpu_active"] is True
    assert GPUComputeManager.info(manager)["reconstruction_qem_candidate_active"] is True

def test_planar_growth_preview_does_not_change_region_results():
    vertices, faces = make_grid(size=12, amplitude=0.0001)
    base = extract_planar_regions_li_nan(
        vertices, faces, small_region_area_threshold=0.0,
        fitting_threshold=0.0, normal_angle_degrees=25.0,
    )
    frames = []
    preview = extract_planar_regions_li_nan(
        vertices, faces, small_region_area_threshold=0.0,
        fitting_threshold=0.0, normal_angle_degrees=25.0,
        visualization_cb=lambda labels, active, done, total, phase: frames.append(
            (labels.copy(), active.copy(), done, total, phase)
        ),
    )
    np.testing.assert_array_equal(base.face_region_ids, preview.face_region_ids)
    assert base.region_adjacency == preview.region_adjacency
    assert len(base.contours) == len(preview.contours)
    assert frames
    assert all(np.asarray(frame[1]).ndim == 1 for frame in frames)
