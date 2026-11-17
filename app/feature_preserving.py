from __future__ import annotations

"""Feature-preserving mesh filtering from Li & Nan (2021), Section 3.1.

This module intentionally implements only the paper's first phase:
face-normal bilateral filtering followed by vertex-position updates.
Planar-region extraction and hierarchical edge-collapse decimation are
implemented in later pipeline stages, not approximated here.

The optional isotropic-neighborhood bootstrap mentioned for highly crumpled
initial meshes is not enabled automatically: the paper does not prescribe a
numerical range threshold or an objective crumpled-mesh detector.
"""

from dataclasses import dataclass
import time
from typing import Callable, Optional

import numpy as np

class FeaturePreservingFilterError(ValueError):
    """Raised when a mesh cannot be safely processed by the paper-based filter."""

@dataclass(frozen=True)
class MeshFilterStats:
    input_vertices: int
    input_faces: int
    average_edge_length: float
    normal_angle_degrees: float
    requested_iterations: int
    completed_iterations: int
    final_max_vertex_displacement: float
    topology_build_seconds: float = 0.0
    filtering_seconds: float = 0.0
    filter_compute_backend: str = "CPU"
    filter_gpu_batches: int = 0
    filter_gpu_seconds: float = 0.0
    filter_cpu_reference_seconds: float = 0.0
    filter_fallback_reason: str = ""

def _validate_mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    v = np.asarray(vertices, dtype=np.float64)
    f_raw = np.asarray(faces)

    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3:
        raise FeaturePreservingFilterError("vertices must be a non-empty (N, 3) array.")
    if f_raw.ndim != 2 or f_raw.shape[1] != 3 or len(f_raw) < 1:
        raise FeaturePreservingFilterError("faces must be a non-empty triangulated (M, 3) array.")
    if not np.issubdtype(f_raw.dtype, np.integer):
        if not np.isfinite(f_raw).all() or not np.equal(f_raw, np.floor(f_raw)).all():
            raise FeaturePreservingFilterError("face indices must be integers.")
    f = np.asarray(f_raw, dtype=np.int64)
    if not np.isfinite(v).all():
        raise FeaturePreservingFilterError("vertices contain NaN or infinite coordinates.")
    if np.any(f < 0) or np.any(f >= len(v)):
        raise FeaturePreservingFilterError("face indices are outside the vertex array.")
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])):
        raise FeaturePreservingFilterError("faces with repeated vertex indices are not supported.")

    diagonal = float(np.linalg.norm(np.ptp(v, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise FeaturePreservingFilterError("mesh has zero or invalid spatial extent.")

    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    area_epsilon = max(diagonal * diagonal, 1.0) * 1e-14
    if np.any(~np.isfinite(double_area)) or np.any(double_area <= area_epsilon):
        bad = int(np.count_nonzero((~np.isfinite(double_area)) | (double_area <= area_epsilon)))
        raise FeaturePreservingFilterError(
            f"mesh contains {bad:,} degenerate or near-degenerate triangles; "
            "clean or inspect them before filtering."
        )
    return v.copy(), f.copy(), diagonal

def _face_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    normals = cross / double_area[:, None]
    centers = np.mean(tri, axis=1)
    areas = 0.5 * double_area
    return normals, centers, areas

def _average_unique_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    unique_edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(vertices[unique_edges[:, 0]] - vertices[unique_edges[:, 1]], axis=1)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    if len(lengths) == 0:
        raise FeaturePreservingFilterError("could not estimate the average mesh edge length.")
    return float(np.mean(lengths))

def _topological_face_neighborhoods(
    vertex_count: int,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact one-ring face neighborhoods with sparse incidence algebra.

    Li & Nan define N_fi as every face sharing at least one vertex with f_i,
    including f_i itself. The face/vertex incidence matrix B is sparse with
    exactly three non-zeros per row. The non-zero pattern of B @ B.T is exactly
    that neighborhood relation. SciPy performs the sparse product in compiled
    code, avoiding Python list/set expansion on large meshes.

    The returned face-pair arrays are CSR ordered (query face, then ascending
    neighbor face), matching the deterministic ordering of the previous
    Python set implementation. A small pure-Python fallback is retained for
    environments where scipy.sparse is unavailable.
    """
    face_array = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    face_count = int(len(face_array))
    vertex_face_ids = face_array.reshape(-1).copy()
    incident_face_ids = np.repeat(np.arange(face_count, dtype=np.int64), 3)
    if face_count == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, vertex_face_ids, incident_face_ids

    try:
        from scipy.sparse import csr_matrix

        rows = np.repeat(np.arange(face_count, dtype=np.int64), 3)
        incidence = csr_matrix(
            (np.ones(len(rows), dtype=np.int32), (rows, vertex_face_ids)),
            shape=(face_count, int(vertex_count)),
            dtype=np.int32,
        )
        adjacency = (incidence @ incidence.transpose()).tocsr()
        adjacency.eliminate_zeros()
        adjacency.sort_indices()
        row_counts = np.diff(adjacency.indptr).astype(np.int64, copy=False)
        query_ids = np.repeat(np.arange(face_count, dtype=np.int64), row_counts)
        neighbor_ids = np.asarray(adjacency.indices, dtype=np.int64)
        return query_ids, neighbor_ids, vertex_face_ids, incident_face_ids
    except ImportError:

        vertex_faces: list[list[int]] = [[] for _ in range(int(vertex_count))]
        for face_id, tri in enumerate(face_array):
            for vertex_id in tri:
                vertex_faces[int(vertex_id)].append(face_id)
        pair_query: list[int] = []
        pair_neighbor: list[int] = []
        for face_id, tri in enumerate(face_array):
            candidates: set[int] = set()
            for vertex_id in tri:
                candidates.update(vertex_faces[int(vertex_id)])
            ordered = sorted(candidates)
            pair_query.extend([face_id] * len(ordered))
            pair_neighbor.extend(ordered)
        return (
            np.asarray(pair_query, dtype=np.int64),
            np.asarray(pair_neighbor, dtype=np.int64),
            vertex_face_ids,
            incident_face_ids,
        )

def _bilateral_filter_face_normals(
    normals: np.ndarray,
    centers: np.ndarray,
    areas: np.ndarray,
    query_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    sigma_dist: float,
    theta_radians: float,
) -> np.ndarray:
    """Apply Eq. (2)-(3) in Li & Nan (2021) using flattened adjacency pairs.

    The paper writes A_fi (query-face area) in Eq. (3). This factor is
    constant across the query face's neighborhood and is retained explicitly;
    it cancels when the weighted vector is normalized.
    """
    cos_theta = float(np.cos(theta_radians))
    beta_denominator = max((1.0 - cos_theta) ** 2, 1e-24)
    sigma_squared_twice = max(2.0 * sigma_dist * sigma_dist, 1e-24)

    dots = np.einsum("ij,ij->i", normals[query_ids], normals[neighbor_ids])
    dots = np.clip(dots, -1.0, 1.0)
    keep = dots > cos_theta
    q = query_ids[keep]
    n = neighbor_ids[keep]
    d = dots[keep]

    centroid_delta = centers[n] - centers[q]
    distance_squared = np.einsum("ij,ij->i", centroid_delta, centroid_delta)
    alpha = np.exp(-distance_squared / sigma_squared_twice)
    beta = np.exp(-((1.0 - d) ** 2) / beta_denominator)
    weights = areas[q] * alpha * beta
    weighted_normals = normals[n] * weights[:, None]

    weighted_sum = np.column_stack((
        np.bincount(q, weights=weighted_normals[:, 0], minlength=len(normals)),
        np.bincount(q, weights=weighted_normals[:, 1], minlength=len(normals)),
        np.bincount(q, weights=weighted_normals[:, 2], minlength=len(normals)),
    ))
    lengths = np.linalg.norm(weighted_sum, axis=1)
    filtered = normals.copy()
    valid = np.isfinite(lengths) & (lengths > 1e-15)
    filtered[valid] = weighted_sum[valid] / lengths[valid, None]
    return filtered

def _update_vertices(
    vertices: np.ndarray,
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    filtered_normals: np.ndarray,
    vertex_face_ids: np.ndarray,
    incident_face_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Eq. (5) in Li & Nan (2021) without changing mesh connectivity.

    The equation uses the filtered normal for the signed projection and the
    current face normal as the displacement direction, exactly as printed.
    The displacement is the unweighted mean over incident faces.
    """
    offsets = face_centers[incident_face_ids] - vertices[vertex_face_ids]
    projections = np.einsum(
        "ij,ij->i", offsets, filtered_normals[incident_face_ids]
    )
    contributions = projections[:, None] * face_normals[incident_face_ids]
    counts = np.bincount(vertex_face_ids, minlength=len(vertices)).astype(np.float64)
    displacement = np.zeros_like(vertices)
    used = counts > 0.0
    for axis in range(3):
        sums = np.bincount(
            vertex_face_ids,
            weights=contributions[:, axis],
            minlength=len(vertices),
        )
        displacement[used, axis] = sums[used] / counts[used]
    updated = vertices + displacement
    return updated, displacement

def attenuate_filter_displacement(
    original_vertices: np.ndarray,
    filtered_vertices: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, float]:
    """Optionally scale the paper filter's vertex displacement as a safety layer.

    A scale of 1.0 returns the unmodified Li--Nan filter result; 0.0 restores
    the input positions; intermediate values linearly attenuate the displacement.
    This control is an application-side extension, not part of Li & Nan (2021).
    It is not a general self-intersection or Z-Fighting guarantee.
    """
    original = np.asarray(original_vertices, dtype=np.float64)
    filtered = np.asarray(filtered_vertices, dtype=np.float64)
    amount = float(scale)
    if original.ndim != 2 or original.shape[1] != 3 or filtered.shape != original.shape:
        raise FeaturePreservingFilterError(
            "original_vertices and filtered_vertices must be matching Nx3 arrays."
        )
    if not np.isfinite(original).all() or not np.isfinite(filtered).all():
        raise FeaturePreservingFilterError("filter displacement arrays must contain only finite coordinates.")
    if not np.isfinite(amount) or amount < 0.0 or amount > 1.0:
        raise FeaturePreservingFilterError("filter displacement scale must be within [0, 1].")
    output = original + amount * (filtered - original)
    if not np.isfinite(output).all():
        raise FeaturePreservingFilterError("attenuated filter output contains non-finite coordinates.")
    max_displacement = float(np.max(np.linalg.norm(output - original, axis=1))) if len(output) else 0.0
    return output, max_displacement

def _csr_offsets_from_sorted_ids(sorted_ids: np.ndarray, item_count: int) -> np.ndarray:
    counts = np.bincount(np.asarray(sorted_ids, dtype=np.int64), minlength=int(item_count))
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))

def filter_mesh_li_nan(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    normal_angle_degrees: float = 20.0,
    iterations: int = 10,
    convergence_threshold: Optional[float] = None,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
    preview_cb: Optional[Callable[[int, int, np.ndarray, np.ndarray, np.ndarray, float], None]] = None,
    locked_vertices: Optional[np.ndarray] = None,
    max_step_fraction: float = 0.50,
    compute_manager=None,
) -> tuple[np.ndarray, np.ndarray, MeshFilterStats]:
    """Filter a triangulated mesh using the first phase of Li & Nan (2021).

    Parameters follow Section 3.1 of the paper:
      * normal_angle_degrees: theta in Eqs. (2)-(3), paper default 20 degrees;
      * sigma_dist: estimated internally as the average unique edge length;
      * iterations: paper reports 10-20 iterations as an efficient practical
        range; the default here is 10 (the lower end of that reported range);
      * convergence_threshold: optional maximum vertex displacement in mesh
        units. The paper describes this stopping condition but does not give a
        universal numeric threshold, so no numeric default is invented.

    The method changes vertex positions only. It preserves the face index
    array and does not weld, split, delete, or retriangulate faces.
    """
    filtered_vertices, filtered_faces, diagonal = _validate_mesh_arrays(vertices, faces)

    theta = float(normal_angle_degrees)
    if not np.isfinite(theta) or theta <= 0.0 or theta >= 90.0:
        raise FeaturePreservingFilterError("normal_angle_degrees must be between 0 and 90 degrees.")
    if isinstance(iterations, bool) or int(iterations) != iterations or not 1 <= int(iterations) <= 100:
        raise FeaturePreservingFilterError("iterations must be an integer between 1 and 100.")
    iterations = int(iterations)
    if not np.isfinite(float(max_step_fraction)) or not 0.0 < float(max_step_fraction) <= 1.0:
        raise FeaturePreservingFilterError("max_step_fraction must be in (0, 1].")
    if locked_vertices is None:
        locked_mask = np.zeros(len(filtered_vertices), dtype=bool)
    else:
        locked_raw = np.asarray(locked_vertices)
        if locked_raw.dtype == bool and locked_raw.shape == (len(filtered_vertices),):
            locked_mask = locked_raw.copy()
        else:
            locked_ids = np.asarray(locked_vertices, dtype=np.int64).reshape(-1)
            if len(locked_ids) and (int(locked_ids.min()) < 0 or int(locked_ids.max()) >= len(filtered_vertices)):
                raise FeaturePreservingFilterError("locked_vertices contains out-of-range vertex indices.")
            locked_mask = np.zeros(len(filtered_vertices), dtype=bool)
            locked_mask[locked_ids] = True
    locked_reference = filtered_vertices.copy()
    if convergence_threshold is not None:
        convergence_threshold = float(convergence_threshold)
        if not np.isfinite(convergence_threshold) or convergence_threshold < 0.0:
            raise FeaturePreservingFilterError("convergence_threshold must be finite and non-negative.")

    topology_started = time.perf_counter()
    sigma_dist = _average_unique_edge_length(filtered_vertices, filtered_faces)
    (
        face_query_ids,
        face_neighbor_ids,
        vertex_face_ids,
        incident_face_ids,
    ) = _topological_face_neighborhoods(len(filtered_vertices), filtered_faces)

    face_query_offsets = _csr_offsets_from_sorted_ids(face_query_ids, len(filtered_faces))
    vertex_order = np.argsort(vertex_face_ids, kind="stable")
    sorted_vertex_ids = vertex_face_ids[vertex_order]
    vertex_offsets = _csr_offsets_from_sorted_ids(sorted_vertex_ids, len(filtered_vertices))
    vertex_incident_faces = np.asarray(incident_face_ids[vertex_order], dtype=np.uint64)
    topology_build_seconds = max(time.perf_counter() - topology_started, 0.0)
    filtering_started = time.perf_counter()
    filter_backend_name = "CPU"
    filter_gpu_batches = 0
    filter_gpu_seconds = 0.0
    filter_cpu_reference_seconds = 0.0
    filter_fallback_reason = ""
    gpu_filter_verified = False
    filter_pair_count = int(len(face_neighbor_ids) + len(vertex_incident_faces))
    gpu_filter_threshold = 12000
    if compute_manager is not None:
        if getattr(compute_manager, "backend", None) is None:
            filter_fallback_reason = str(
                getattr(compute_manager, "fallback_reason", "No verified GPU backend is available for reconstruction filtering.")
            )
        elif filter_pair_count < gpu_filter_threshold:
            filter_fallback_reason = (
                f"Li-Nan filter workload has {filter_pair_count:,} adjacency entries, below the "
                f"{gpu_filter_threshold:,} entry GPU threshold; CPU avoids GPU transfer overhead."
            )
    theta_radians = float(np.deg2rad(theta))
    final_max_displacement = 0.0
    completed = 0

    for iteration in range(iterations):
        face_normals, face_centers, face_areas = _face_geometry(filtered_vertices, filtered_faces)
        accelerated = None
        reference_result = None
        if compute_manager is not None and filter_pair_count >= gpu_filter_threshold:
            accelerated = compute_manager.filter_mesh_step(
                filtered_vertices, face_normals, face_centers, face_areas,
                face_query_offsets, face_neighbor_ids, vertex_offsets,
                vertex_incident_faces, sigma_dist, theta_radians,
            )
        if accelerated is not None:
            gpu_normals, gpu_vertices, gpu_displacement = accelerated
            if not gpu_filter_verified:
                reference_started = time.perf_counter()
                reference_normals = _bilateral_filter_face_normals(
                    face_normals, face_centers, face_areas,
                    face_query_ids, face_neighbor_ids, sigma_dist, theta_radians,
                )
                reference_vertices, reference_displacement = _update_vertices(
                    filtered_vertices, face_centers, face_normals, reference_normals,
                    vertex_face_ids, incident_face_ids,
                )
                reference_elapsed = max(time.perf_counter() - reference_started, 0.0)
                reference_result = (reference_normals, reference_vertices, reference_displacement)
                filter_cpu_reference_seconds += reference_elapsed
                if compute_manager is not None and hasattr(compute_manager, "record_filter_cpu_reference"):
                    compute_manager.record_filter_cpu_reference(reference_elapsed)
                parity_ok = (
                    np.asarray(gpu_normals).shape == reference_normals.shape
                    and np.asarray(gpu_vertices).shape == reference_vertices.shape
                    and np.isfinite(gpu_normals).all()
                    and np.isfinite(gpu_vertices).all()
                    and np.isfinite(gpu_displacement).all()
                    and np.asarray(gpu_displacement).shape == reference_displacement.shape
                    and np.allclose(gpu_normals, reference_normals, rtol=8e-4, atol=8e-6)
                    and np.allclose(gpu_vertices, reference_vertices, rtol=8e-4, atol=8e-6)
                    and np.allclose(gpu_displacement, reference_displacement, rtol=8e-4, atol=8e-6)
                )
                if not parity_ok:
                    filter_fallback_reason = (
                        f"{getattr(compute_manager, 'backend_name', 'GPU')} Li-Nan filter result failed "
                        "the first-iteration CPU parity check; all Stage-1 filter iterations use CPU."
                    )
                    if hasattr(compute_manager, "disable_filter_kernel"):
                        compute_manager.disable_filter_kernel(filter_fallback_reason)
                    accelerated = None
                else:
                    gpu_filter_verified = True
            if accelerated is not None:
                filtered_normals = np.asarray(gpu_normals, dtype=np.float64)
                proposed_vertices = np.asarray(gpu_vertices, dtype=np.float64)
                displacement = np.asarray(gpu_displacement, dtype=np.float64)
                filter_backend_name = str(getattr(compute_manager, "backend_name", "GPU"))
                filter_gpu_batches += 1
        if accelerated is None:
            if reference_result is not None:
                filtered_normals, proposed_vertices, displacement = reference_result
            else:
                filtered_normals = _bilateral_filter_face_normals(
                    face_normals,
                    face_centers,
                    face_areas,
                    face_query_ids,
                    face_neighbor_ids,
                    sigma_dist,
                    theta_radians,
                )
                proposed_vertices, displacement = _update_vertices(
                    filtered_vertices,
                    face_centers,
                    face_normals,
                    filtered_normals,
                    vertex_face_ids,
                    incident_face_ids,
                )
        if not np.isfinite(proposed_vertices).all():
            raise FeaturePreservingFilterError(
                f"vertex update produced non-finite coordinates at iteration {iteration + 1}; "
                "the input mesh may be numerically unstable."
            )

        if np.any(locked_mask):
            proposed_vertices[locked_mask] = locked_reference[locked_mask]
        displacement = proposed_vertices - filtered_vertices

        current_triangles = filtered_vertices[filtered_faces]
        current_cross = np.cross(
            current_triangles[:, 1] - current_triangles[:, 0],
            current_triangles[:, 2] - current_triangles[:, 0],
        )
        current_area = np.linalg.norm(current_cross, axis=1)
        area_epsilon = max(diagonal * diagonal, 1.0) * 1e-14
        accepted_vertices = None
        accepted_step = 0.0
        trial_step = float(max_step_fraction)
        for _line_search in range(9):
            trial_vertices = filtered_vertices + displacement * trial_step
            if np.any(locked_mask):
                trial_vertices[locked_mask] = locked_reference[locked_mask]
            trial_triangles = trial_vertices[filtered_faces]
            trial_cross = np.cross(
                trial_triangles[:, 1] - trial_triangles[:, 0],
                trial_triangles[:, 2] - trial_triangles[:, 0],
            )
            trial_area = np.linalg.norm(trial_cross, axis=1)
            alignment = np.einsum("ij,ij->i", current_cross, trial_cross)
            safe = (
                np.isfinite(trial_vertices).all()
                and np.isfinite(trial_area).all()
                and np.all(trial_area > area_epsilon)
                and np.all(alignment > 0.0)
            )
            if safe:
                accepted_vertices = trial_vertices
                accepted_step = trial_step
                break
            trial_step *= 0.5
        if accepted_vertices is None:

            accepted_vertices = filtered_vertices.copy()
            accepted_step = 0.0

        actual_displacement = accepted_vertices - filtered_vertices
        final_max_displacement = float(
            np.max(np.linalg.norm(actual_displacement, axis=1))
        ) if len(actual_displacement) else 0.0
        filtered_vertices = accepted_vertices
        completed = iteration + 1
        if progress_cb is not None:
            progress_cb(completed, iterations, final_max_displacement)
        if preview_cb is not None:
            preview_cb(
                completed, iterations, filtered_vertices.copy(), filtered_faces.copy(),
                filtered_normals.copy(), final_max_displacement,
            )
        if convergence_threshold is not None and final_max_displacement <= convergence_threshold:
            break

    _face_geometry(filtered_vertices, filtered_faces)
    filtering_seconds = max(time.perf_counter() - filtering_started, 0.0)
    if compute_manager is not None:
        filter_gpu_seconds = float(getattr(compute_manager, "filter_gpu_seconds_total", 0.0))
        filter_cpu_reference_seconds = max(
            filter_cpu_reference_seconds,
            float(getattr(compute_manager, "filter_cpu_reference_seconds", 0.0)),
        )
        if not filter_fallback_reason:
            filter_fallback_reason = str(getattr(compute_manager, "filter_fallback_reason", ""))
    stats = MeshFilterStats(
        input_vertices=int(len(filtered_vertices)),
        input_faces=int(len(filtered_faces)),
        average_edge_length=float(sigma_dist),
        normal_angle_degrees=theta,
        requested_iterations=iterations,
        completed_iterations=completed,
        final_max_vertex_displacement=final_max_displacement,
        topology_build_seconds=float(topology_build_seconds),
        filtering_seconds=float(filtering_seconds),
        filter_compute_backend=filter_backend_name,
        filter_gpu_batches=int(filter_gpu_batches),
        filter_gpu_seconds=float(filter_gpu_seconds),
        filter_cpu_reference_seconds=float(filter_cpu_reference_seconds),
        filter_fallback_reason=filter_fallback_reason,
    )
    return filtered_vertices, filtered_faces, stats
