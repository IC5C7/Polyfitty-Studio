"""Hierarchical feature-preserving mesh decimation from Li & Nan (2021), §3.3.

This module is Stage 3 of the Li & Nan pipeline. It consumes the triangulated
mesh and per-face planar-region result produced by Stages 1 and 2. The same
safety-checked QEM core also serves the shared output face-budget step when no
planar-region labels are available.

Paper-derived operations:
* A vertex uses one fitted region-plane quadric when all its incident faces
  have the same assigned planar-region ID. The paper names this `Q_Pv` but does
  not expand it separately; this implementation instantiates it with Eq. (7)'s
  plane quadric `p p^T` for the Stage-2 least-squares support plane.
* Otherwise, its initial quadric is the unweighted sum of its incident
  triangle-plane quadrics.
* An edge quadric is the sum of the endpoint quadrics.
* The new vertex minimizes the resulting homogeneous quadratic. A full-rank
  3x3 system is solved; for a singular system, the two endpoints and midpoint
  are evaluated and the lowest-cost placement is selected.
* Collapses are selected greedily from a min-cost priority queue.

Application safety layer (not specified as such by the paper):
* Boundary collapses are optional and can be disabled to preserve open boundaries.
* The simplicial link condition and duplicate-face checks are enforced.
* Candidate collapses that create degenerate or flipped triangles are rejected.
* An optional close-surface guard rejects newly introduced intersections.
These safeguards are not attributed to Li & Nan (2021).
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from typing import Callable, Optional

import numpy as np

try:
    from .feature_preserving import _validate_mesh_arrays
    from .planar_regions import PlanarExtractionResult
except ImportError:
    from feature_preserving import _validate_mesh_arrays
    from planar_regions import PlanarExtractionResult

class FeaturePreservingDecimationError(ValueError):
    """Raised when hierarchical feature-preserving decimation cannot proceed."""

@dataclass(frozen=True)
class MeshDecimationStats:
    input_vertex_count: int
    input_used_vertex_count: int
    input_unused_vertex_count: int
    input_face_count: int
    output_vertex_count: int
    output_face_count: int
    target_vertex_count: int
    successful_collapses: int
    target_reached: bool
    initial_component_count: int
    final_component_count: int
    initial_boundary_edge_count: int
    final_boundary_edge_count: int
    initial_non_manifold_edge_count: int
    final_non_manifold_edge_count: int
    rejected_boundary_or_non_manifold: int
    rejected_link_condition: int
    rejected_duplicate_face: int
    rejected_face_flip_or_degenerate: int
    collapse_intersection_guard_enabled: bool
    rejected_collapse_new_crossing: int
    rejected_collapse_new_near_coplanar: int
    rejected_collapse_gap_reduction: int
    collapse_guard_candidate_pairs_measured: int
    collapse_guard_index_rebuilds: int
    topology_build_seconds: float = 0.0
    component_check_seconds: float = 0.0
    quadric_setup_seconds: float = 0.0
    initial_queue_seconds: float = 0.0
    collapse_seconds: float = 0.0
    final_validation_seconds: float = 0.0
    queue_compaction_count: int = 0
    queue_stale_entries_removed: int = 0
    queue_peak_entries: int = 0
    queue_placement_reuses: int = 0
    queue_cached_cost_requeues: int = 0
    topology_compact_map_entries: int = 0
    topology_compact_edge_entries: int = 0
    topology_materialized_edge_entries: int = 0
    adaptive_qem_tier: str = "small"
    adaptive_queue_compact_min_entries: int = 0
    adaptive_queue_stale_ratio: float = 0.0
    adaptive_queue_stale_floor: int = 0
    adaptive_queue_stale_divisor: int = 0
    adaptive_queue_memory_budget_bytes: int = 0
    adaptive_progress_callback_stride: int = 1
    progress_callback_count: int = 0
    estimated_topology_bytes: int = 0
    estimated_queue_entry_bytes: int = 0
    estimated_queue_peak_bytes: int = 0
    estimated_queue_stale_peak_bytes: int = 0
    queue_memory_compaction_count: int = 0
    topology_storage_mode: str = "compact-edge-incidence/lazy-materialization"
    candidate_evaluation_backend: str = "CPU-serial"
    candidate_evaluation_workers: int = 1
    candidate_evaluation_count: int = 0
    candidate_geometry_checks: int = 0
    candidate_evaluation_batches: int = 0
    candidate_evaluation_seconds: float = 0.0
    candidate_geometry_seconds: float = 0.0

@dataclass(frozen=True)
class MeshDecimationResult:
    vertices: np.ndarray
    faces: np.ndarray
    face_region_ids: np.ndarray
    original_vertex_indices: np.ndarray
    original_face_indices: np.ndarray
    stats: MeshDecimationStats
    collapse_guard_report: dict | None = None

_QEM_HEAP_COMPACT_MIN_SIZE = 4096
_QEM_HEAP_COMPACT_STALE_FACTOR = 2.0
_QEM_REUSE_UNCHANGED_EDGE_COSTS = True
_QEM_ADAPTIVE_POLICY_ENABLED = True
_QEM_PARALLEL_MIN_CANDIDATES = 2048
_QEM_PARALLEL_MAX_WORKERS = 8
_QEM_PARALLEL_CHUNK_MIN = 256

@dataclass(frozen=True)
class AdaptiveQEMPolicy:
    """Mesh-size-aware queue and callback policy for deterministic QEM."""

    tier: str
    compact_min_entries: int
    stale_ratio: float
    stale_divisor: int
    stale_floor: int
    stale_memory_budget_bytes: int
    max_progress_callbacks: int

def _adaptive_qem_policy(
    vertex_count: int,
    face_count: int,
    requested_collapses: int,
) -> AdaptiveQEMPolicy:
    """Choose conservative queue/memory limits from input mesh complexity.

    The topology representation itself remains compact-edge-incidence with
    lazy set materialization at every size. Larger meshes receive earlier
    stale-heap compaction and fewer Python progress callback invocations.
    Heap compaction only removes stale generations and never changes live
    `(cost, serial, edge, generation)` ordering.
    """
    face_count = max(0, int(face_count))
    vertex_count = max(0, int(vertex_count))
    requested_collapses = max(0, int(requested_collapses))

    overridden = (
        int(_QEM_HEAP_COMPACT_MIN_SIZE) != 4096
        or float(_QEM_HEAP_COMPACT_STALE_FACTOR) != 2.0
    )
    if overridden or not _QEM_ADAPTIVE_POLICY_ENABLED:
        max_callbacks = min(512, max(1, requested_collapses))
        return AdaptiveQEMPolicy(
            tier="manual" if overridden else "fixed",
            compact_min_entries=max(1, int(_QEM_HEAP_COMPACT_MIN_SIZE)),
            stale_ratio=max(0.05, float(_QEM_HEAP_COMPACT_STALE_FACTOR)),
            stale_divisor=2,
            stale_floor=1024,
            stale_memory_budget_bytes=0,
            max_progress_callbacks=max_callbacks,
        )

    if face_count < 10_000 and vertex_count < 8_000:
        tier = "small"
        min_entries, ratio, divisor, floor, budget, max_callbacks = (
            4096, 2.0, 2, 1024, 2 * 1024 * 1024, 512
        )
    elif face_count < 50_000 and vertex_count < 40_000:
        tier = "medium"
        min_entries, ratio, divisor, floor, budget, max_callbacks = (
            6144, 2.0, 2, 1024, 4 * 1024 * 1024, 512
        )
    elif face_count < 250_000 and vertex_count < 180_000:
        tier = "large"
        min_entries, ratio, divisor, floor, budget, max_callbacks = (
            12_288, 1.50, 4, 3072, 6 * 1024 * 1024, 768
        )
    else:
        tier = "xlarge"
        min_entries, ratio, divisor, floor, budget, max_callbacks = (
            24_576, 1.35, 5, 6144, 12 * 1024 * 1024, 1024
        )

    max_callbacks = min(max(1, int(max_callbacks)), max(1, requested_collapses))
    return AdaptiveQEMPolicy(
        tier=tier,
        compact_min_entries=int(min_entries),
        stale_ratio=float(ratio),
        stale_divisor=int(divisor),
        stale_floor=int(floor),
        stale_memory_budget_bytes=int(budget),
        max_progress_callbacks=max(1, int(max_callbacks)),
    )

def _estimated_heap_entry_bytes() -> int:
    """Conservative CPython heap-entry memory estimate used for pressure limits."""
    sample = (0.0, 0, (0, 1), 0)
    return int(
        sys.getsizeof(sample)
        + sys.getsizeof(sample[2])
        + sys.getsizeof(sample[0])
        + 2 * sys.getsizeof(sample[1])
    )

def _estimate_topology_memory_bytes(
    vertex_faces: list[set[int]],
    vertex_neighbors: list[set[int]],
    edge_faces: dict,
) -> int:
    """Estimate Python-managed topology memory without recursively double counting IDs."""
    total = (
        sys.getsizeof(vertex_faces)
        + sys.getsizeof(vertex_neighbors)
        + sys.getsizeof(edge_faces)
    )
    total += sum(sys.getsizeof(items) for items in vertex_faces)
    total += sum(sys.getsizeof(items) for items in vertex_neighbors)
    total += 8 * (
        sum(len(items) for items in vertex_faces)
        + sum(len(items) for items in vertex_neighbors)
    )
    total += len(edge_faces) * (sys.getsizeof((0, 1)) + 2 * sys.getsizeof(0))
    for incidence in edge_faces.values():
        total += sys.getsizeof(incidence)
        if isinstance(incidence, tuple):
            total += len(incidence) * sys.getsizeof(0)
        elif isinstance(incidence, set):
            total += len(incidence) * sys.getsizeof(0)
    return int(total)

def _should_compact_adaptive_heap(
    heap_size: int,
    live_entries: int,
    stale_entries: int,
    policy: AdaptiveQEMPolicy,
    entry_bytes: int,
) -> tuple[bool, bool]:
    """Return (should_compact, memory_pressure_triggered)."""
    heap_size = max(0, int(heap_size))
    live_entries = max(0, int(live_entries))
    stale_entries = max(0, int(stale_entries))
    if heap_size < policy.compact_min_entries or stale_entries < policy.stale_floor:
        return False, False
    ratio_limit = max(
        policy.compact_min_entries,
        int(max(1, live_entries) * policy.stale_ratio),
    )
    stale_ratio_limit = max(policy.stale_floor, live_entries // max(1, policy.stale_divisor))
    ratio_pressure = heap_size > ratio_limit and stale_entries > stale_ratio_limit
    memory_pressure = bool(
        policy.stale_memory_budget_bytes > 0
        and stale_entries * max(1, int(entry_bytes)) >= policy.stale_memory_budget_bytes
    )
    return bool(ratio_pressure or memory_pressure), memory_pressure

def _compact_qem_heap(
    heap: list[tuple],
    queued_generation: dict[tuple[int, int], int],
    edge_faces: dict[tuple[int, int], int | tuple[int, ...] | set[int]],
    active_vertices: np.ndarray,
) -> int:
    """Drop stale lazy-deletion entries without changing live queue ordering.

    Queue entries retain their original serial number and scalar edge cost.
    The optimal point remains in ``edge_placement_cache`` and is deliberately
    not duplicated in stale heap records. Filtering by generation and active
    topology, followed by ``heapify``, preserves the exact live queue order.
    No edge cost or placement is recomputed during compaction.
    """
    previous_size = len(heap)
    if previous_size == 0:
        return 0
    write_index = 0
    for read_index in range(previous_size):
        item = heap[read_index]
        _cost, _serial, edge, generation = item
        first, second = edge
        if (
            queued_generation.get(edge) == generation
            and edge in edge_faces
            and active_vertices[first]
            and active_vertices[second]
        ):
            heap[write_index] = item
            write_index += 1
        elif queued_generation.get(edge) == generation:
            queued_generation.pop(edge, None)
    removed = previous_size - write_index
    if removed:
        del heap[write_index:]
        heapq.heapify(heap)
    return int(removed)

def _edge_key(a: int, b: int) -> tuple[int, int]:
    """Canonical tuple edge used by the QEM priority queue and caches."""
    return (a, b) if a < b else (b, a)

def _plane_quadric(plane_coefficients: np.ndarray) -> np.ndarray:
    p = np.asarray(plane_coefficients, dtype=np.float64)
    return np.outer(p, p)

def _face_plane_quadrics(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(cross, axis=1)
    normals = cross / lengths[:, None]
    offsets = -np.einsum("ij,ij->i", normals, triangles[:, 0])
    planes = np.column_stack((normals, offsets))
    return np.einsum("ni,nj->nij", planes, planes)

def _region_plane_quadrics(
    planar_regions: PlanarExtractionResult,
) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for region in planar_regions.regions:
        coeff = np.concatenate((
            np.asarray(region.plane.normal, dtype=np.float64),
            np.asarray([region.plane.offset], dtype=np.float64),
        ))
        if coeff.shape != (4,) or not np.isfinite(coeff).all():
            raise FeaturePreservingDecimationError(
                f"planar region {region.region_id} has an invalid fitted plane."
            )
        output[int(region.region_id)] = _plane_quadric(coeff)
    return output

def _initial_vertex_quadrics(
    vertex_count: int,
    faces: np.ndarray,
    face_region_ids: np.ndarray,
    face_quadrics: np.ndarray,
    region_quadrics: dict[int, np.ndarray],
) -> np.ndarray:
    """Build per-vertex quadrics with vectorized face-to-vertex accumulation.

    Each triangle contributes its plane quadric once to each of its three
    vertices. Three indexed scatter-adds replace nested Python loops over
    per-vertex incident-face sets. Min/max scatter reductions preserve the
    Li--Nan rule that a vertex wholly inside one valid planar region uses that
    region's fitted-plane quadric instead of the sum of its face quadrics.
    """
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    labels = np.asarray(face_region_ids, dtype=np.int64).reshape(-1)
    quadrics = np.zeros((int(vertex_count), 4, 4), dtype=np.float64)
    if len(triangles) == 0:
        return quadrics
    if labels.shape != (len(triangles),) or np.asarray(face_quadrics).shape != (len(triangles), 4, 4):
        raise FeaturePreservingDecimationError("face quadric/region arrays do not match the input topology")

    face_quadrics = np.asarray(face_quadrics, dtype=np.float64)
    for corner in range(3):
        np.add.at(quadrics, triangles[:, corner], face_quadrics)

    minimum_label = np.full(int(vertex_count), np.iinfo(np.int64).max, dtype=np.int64)
    maximum_label = np.full(int(vertex_count), np.iinfo(np.int64).min, dtype=np.int64)
    for corner in range(3):
        vertex_ids = triangles[:, corner]
        np.minimum.at(minimum_label, vertex_ids, labels)
        np.maximum.at(maximum_label, vertex_ids, labels)

    uniform_region = (minimum_label == maximum_label) & (minimum_label >= 0)
    for region_id, region_quadric in region_quadrics.items():
        mask = uniform_region & (minimum_label == int(region_id))
        if np.any(mask):
            quadrics[mask] = np.asarray(region_quadric, dtype=np.float64)
    return quadrics

def _incidence_count(value) -> int:
    """Return the number of incident faces without allocating a wrapper."""
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return 1
    return len(value)

def _mutable_incidence_set(mapping: dict, key) -> set[int]:
    """Materialize one compact incidence value as a native set on first edit."""
    current = mapping.get(key)
    if isinstance(current, set):
        return current
    if current is None:
        materialized: set[int] = set()
    elif isinstance(current, (int, np.integer)):
        materialized = {int(current)}
    else:
        materialized = {int(value) for value in current}
    mapping[key] = materialized
    return materialized

def _build_topology(
    faces: np.ndarray,
    vertex_count: int,
) -> tuple[
    list[set[int]],
    list[set[int]],
    dict[tuple[int, int], int | tuple[int, ...] | set[int]],
]:
    """Build mutable vertex one-rings and compact edge-incidence values.

    Vertex one-rings remain native sets because QEM reads and edits them on
    every collapse. Edge incidence stores a scalar face ID for boundary edges
    and a short tuple for the usual two-face manifold edge. An edge value is
    converted to a native set only when a collapse first edits that edge.

    A separate global face-key map is unnecessary: every proposed surviving
    triangle contains the retained endpoint, so any pre-existing face with the
    same unordered vertices is already in that endpoint's affected one-ring.
    The link-condition check therefore rules out external duplicates, while
    ``generated_keys`` detects duplicates produced within the affected patch.
    """
    vertex_faces = [set() for _ in range(int(vertex_count))]
    vertex_neighbors = [set() for _ in range(int(vertex_count))]
    edge_faces: dict[tuple[int, int], int | tuple[int, ...] | set[int]] = {}

    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    for face_id, tri in enumerate(triangles):
        a, b, c = (int(value) for value in tri)
        vertex_faces[a].add(face_id)
        vertex_faces[b].add(face_id)
        vertex_faces[c].add(face_id)
        for first, second in ((a, b), (b, c), (c, a)):
            edge = (first, second) if first < second else (second, first)
            incident = edge_faces.get(edge)
            if incident is None:
                edge_faces[edge] = face_id
            elif isinstance(incident, (int, np.integer)):
                if int(incident) != face_id:
                    edge_faces[edge] = (int(incident), face_id)
            elif isinstance(incident, set):
                incident.add(face_id)
            elif face_id not in incident:
                edge_faces[edge] = incident + (face_id,)
            vertex_neighbors[first].add(second)
            vertex_neighbors[second].add(first)
    return vertex_faces, vertex_neighbors, edge_faces

def _count_components(faces: np.ndarray) -> int:
    """Count connected mesh islands by shared vertices, including non-manifold fans.

    A mesh component is a connected component of the mesh's vertex graph. Faces
    that share a vertex are therefore connected even when that vertex is a
    non-manifold junction, or when a shared edge has three or more incident
    faces. Counting only edges with exactly two incident faces incorrectly
    splits such valid connected islands and can reject an otherwise safe QEM
    result as having introduced new components.
    """
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    face_count = int(len(triangles))
    if face_count == 0:
        return 0

    parent = list(range(face_count))
    rank = [0] * face_count

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: int, second: int) -> None:
        root_a = find(first)
        root_b = find(second)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1

    first_face_by_vertex: dict[int, int] = {}
    for face_id, triangle in enumerate(triangles):
        for raw_vertex in triangle:
            vertex_id = int(raw_vertex)
            owner = first_face_by_vertex.get(vertex_id)
            if owner is None:
                first_face_by_vertex[vertex_id] = face_id
            else:
                union(face_id, owner)

    return len({find(face_id) for face_id in range(face_count)})

def _edge_counts(edge_faces: dict[tuple[int, int], int | tuple[int, ...] | set[int]]) -> tuple[int, int]:
    boundary = 0
    non_manifold = 0
    for incident in edge_faces.values():
        count_faces = 1 if isinstance(incident, (int, np.integer)) else len(incident)
        boundary += count_faces == 1
        non_manifold += count_faces > 2
    return int(boundary), int(non_manifold)

def _quadric_cost(quadric: np.ndarray, position: np.ndarray) -> float:
    """Evaluate x_h.T @ Q @ x_h without allocating a homogeneous vector."""
    x, y, z = (float(position[0]), float(position[1]), float(position[2]))
    q = quadric
    value = (
        q[0, 0] * x * x + q[1, 1] * y * y + q[2, 2] * z * z
        + 2.0 * q[0, 1] * x * y
        + 2.0 * q[0, 2] * x * z
        + 2.0 * q[1, 2] * y * z
        + 2.0 * q[0, 3] * x
        + 2.0 * q[1, 3] * y
        + 2.0 * q[2, 3] * z
        + q[3, 3]
    )
    if not np.isfinite(value):
        return float("inf")
    return max(float(value), 0.0)

def _optimal_edge_placement(
    vertices: np.ndarray,
    quadrics: np.ndarray,
    first: int,
    second: int,
) -> tuple[float, np.ndarray]:
    """Return the minimum-QEM point constrained to the collapsing edge.

    QEM collapse selection is intentionally sequential: accepting one edge
    changes nearby topology and all adjacent edge costs, so speculative GPU
    batches would alter the greedy priority queue or require rollback. This
    scalar hot path is therefore allocation-light and keeps one authoritative
    cost equation for CPU/GPU-independent decimation.
    """
    quadric = quadrics[first] + quadrics[second]
    p0 = vertices[first]
    direction = vertices[second] - p0
    dx, dy, dz = (float(direction[0]), float(direction[1]), float(direction[2]))
    x, y, z = (float(p0[0]), float(p0[1]), float(p0[2]))
    length_sq = dx * dx + dy * dy + dz * dz
    if not np.isfinite(length_sq) or length_sq <= 1e-24:
        midpoint = 0.5 * (p0 + vertices[second])
        return _quadric_cost(quadric, midpoint), midpoint

    q = quadric

    qh0 = (
        q[0, 0] * x + q[0, 1] * y + q[0, 2] * z + q[0, 3],
        q[1, 0] * x + q[1, 1] * y + q[1, 2] * z + q[1, 3],
        q[2, 0] * x + q[2, 1] * y + q[2, 2] * z + q[2, 3],
    )
    numerator = dx * qh0[0] + dy * qh0[1] + dz * qh0[2]
    qd = (
        q[0, 0] * dx + q[0, 1] * dy + q[0, 2] * dz,
        q[1, 0] * dx + q[1, 1] * dy + q[1, 2] * dz,
        q[2, 0] * dx + q[2, 1] * dy + q[2, 2] * dz,
    )
    denominator = dx * qd[0] + dy * qd[1] + dz * qd[2]

    base_cost = _quadric_cost(quadric, p0)
    best_cost = base_cost
    best_t = 0.0

    for candidate_t in (0.5, 1.0):
        candidate_cost = base_cost + 2.0 * numerator * candidate_t + denominator * candidate_t * candidate_t
        if np.isfinite(candidate_cost):
            candidate_cost = max(float(candidate_cost), 0.0)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_t = candidate_t
    if np.isfinite(denominator) and denominator > 1e-18:
        optimum_t = min(1.0, max(0.0, -numerator / denominator))
        candidate_cost = base_cost + 2.0 * numerator * optimum_t + denominator * optimum_t * optimum_t
        if np.isfinite(candidate_cost):
            candidate_cost = max(float(candidate_cost), 0.0)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_t = optimum_t
    return float(best_cost), np.asarray(p0 + best_t * direction, dtype=np.float64)

def _qem_candidate_costs_vectorized(
    vertices: np.ndarray,
    quadrics: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate independent edge placements with the scalar QEM operation order.

    The expression order intentionally mirrors :func:`_optimal_edge_placement`.
    Keeping every candidate independent makes the vectorized result stable and
    lets NumPy release the GIL while operating on large contiguous batches.
    """
    candidate_edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    count_edges = len(candidate_edges)
    costs = np.full(count_edges, np.inf, dtype=np.float64)
    positions = np.zeros((count_edges, 3), dtype=np.float64)
    if count_edges == 0:
        return costs, positions

    first = candidate_edges[:, 0]
    second = candidate_edges[:, 1]
    p0 = np.asarray(vertices[first], dtype=np.float64)
    p1 = np.asarray(vertices[second], dtype=np.float64)
    direction = p1 - p0
    dx, dy, dz = direction[:, 0], direction[:, 1], direction[:, 2]
    x, y, z = p0[:, 0], p0[:, 1], p0[:, 2]
    length_sq = dx * dx + dy * dy + dz * dz
    quadric = np.asarray(quadrics[first] + quadrics[second], dtype=np.float64)

    qh0_x = quadric[:, 0, 0] * x + quadric[:, 0, 1] * y + quadric[:, 0, 2] * z + quadric[:, 0, 3]
    qh0_y = quadric[:, 1, 0] * x + quadric[:, 1, 1] * y + quadric[:, 1, 2] * z + quadric[:, 1, 3]
    qh0_z = quadric[:, 2, 0] * x + quadric[:, 2, 1] * y + quadric[:, 2, 2] * z + quadric[:, 2, 3]
    numerator = dx * qh0_x + dy * qh0_y + dz * qh0_z
    qd_x = quadric[:, 0, 0] * dx + quadric[:, 0, 1] * dy + quadric[:, 0, 2] * dz
    qd_y = quadric[:, 1, 0] * dx + quadric[:, 1, 1] * dy + quadric[:, 1, 2] * dz
    qd_z = quadric[:, 2, 0] * dx + quadric[:, 2, 1] * dy + quadric[:, 2, 2] * dz
    denominator = dx * qd_x + dy * qd_y + dz * qd_z

    base_cost = (
        quadric[:, 0, 0] * x * x + quadric[:, 1, 1] * y * y + quadric[:, 2, 2] * z * z
        + 2.0 * quadric[:, 0, 1] * x * y
        + 2.0 * quadric[:, 0, 2] * x * z
        + 2.0 * quadric[:, 1, 2] * y * z
        + 2.0 * quadric[:, 0, 3] * x
        + 2.0 * quadric[:, 1, 3] * y
        + 2.0 * quadric[:, 2, 3] * z
        + quadric[:, 3, 3]
    )
    base_cost = np.where(np.isfinite(base_cost), np.maximum(base_cost, 0.0), np.inf)
    best_cost = base_cost.copy()
    best_t = np.zeros(count_edges, dtype=np.float64)

    for candidate_t in (0.5, 1.0):
        candidate_cost = (
            base_cost + 2.0 * numerator * candidate_t
            + denominator * candidate_t * candidate_t
        )
        finite = np.isfinite(candidate_cost)
        candidate_cost = np.where(finite, np.maximum(candidate_cost, 0.0), np.inf)
        better = candidate_cost < best_cost
        best_cost = np.where(better, candidate_cost, best_cost)
        best_t = np.where(better, candidate_t, best_t)

    valid_optimum = np.isfinite(denominator) & (denominator > 1e-18)
    optimum_t = np.clip(-numerator / np.where(valid_optimum, denominator, 1.0), 0.0, 1.0)
    optimum_cost = (
        base_cost + 2.0 * numerator * optimum_t
        + denominator * optimum_t * optimum_t
    )
    finite_optimum = valid_optimum & np.isfinite(optimum_cost)
    optimum_cost = np.where(finite_optimum, np.maximum(optimum_cost, 0.0), np.inf)
    better_optimum = optimum_cost < best_cost
    best_cost = np.where(better_optimum, optimum_cost, best_cost)
    best_t = np.where(better_optimum, optimum_t, best_t)

    degenerate = ~np.isfinite(length_sq) | (length_sq <= 1e-24)
    computed_positions = p0 + best_t[:, None] * direction
    midpoint_positions = 0.5 * (p0 + p1)
    computed_positions[degenerate] = midpoint_positions[degenerate]

    if np.any(degenerate):
        midpoint = midpoint_positions[degenerate]
        q = quadric[degenerate]
        mx, my, mz = midpoint[:, 0], midpoint[:, 1], midpoint[:, 2]
        midpoint_cost = (
            q[:, 0, 0] * mx * mx + q[:, 1, 1] * my * my + q[:, 2, 2] * mz * mz
            + 2.0 * q[:, 0, 1] * mx * my
            + 2.0 * q[:, 0, 2] * mx * mz
            + 2.0 * q[:, 1, 2] * my * mz
            + 2.0 * q[:, 0, 3] * mx
            + 2.0 * q[:, 1, 3] * my
            + 2.0 * q[:, 2, 3] * mz
            + q[:, 3, 3]
        )
        best_cost[degenerate] = np.where(
            np.isfinite(midpoint_cost), np.maximum(midpoint_cost, 0.0), np.inf
        )

    finite_positions = np.isfinite(computed_positions).all(axis=1)
    best_cost = np.where(finite_positions, best_cost, np.inf)
    costs[:] = best_cost
    positions[:] = computed_positions
    return costs, positions

def _qem_candidate_geometry_safe_vectorized(
    vertices: np.ndarray,
    faces: np.ndarray,
    active_faces: np.ndarray,
    vertex_faces: list[set[int]],
    edges: np.ndarray,
    positions: np.ndarray,
    area_epsilon: float,
) -> np.ndarray:
    """Check face degeneracy/flips for independent candidates in one read-only pass."""
    candidate_edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    candidate_positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    safe = np.ones(len(candidate_edges), dtype=bool)
    if not len(candidate_edges):
        return safe

    candidate_ids: list[int] = []
    face_ids: list[int] = []
    for candidate_id, (first, second) in enumerate(candidate_edges):
        affected = vertex_faces[int(first)] | vertex_faces[int(second)]
        for face_id in affected:
            if active_faces[int(face_id)]:
                candidate_ids.append(candidate_id)
                face_ids.append(int(face_id))
    if not face_ids:
        return safe

    candidate_index = np.asarray(candidate_ids, dtype=np.int64)
    selected_face_ids = np.asarray(face_ids, dtype=np.int64)
    triangles = np.asarray(faces[selected_face_ids], dtype=np.int64)
    selected_edges = candidate_edges[candidate_index]
    first = selected_edges[:, 0]
    second = selected_edges[:, 1]
    contains_first = np.any(triangles == first[:, None], axis=1)
    contains_second = np.any(triangles == second[:, None], axis=1)
    survives = ~(contains_first & contains_second)
    if not np.any(survives):
        return safe

    candidate_index = candidate_index[survives]
    triangles = triangles[survives]
    first = first[survives]
    second = second[survives]
    new_position = candidate_positions[candidate_index]

    old_triangles = np.asarray(vertices[triangles], dtype=np.float64)
    old_cross = np.cross(
        old_triangles[:, 1] - old_triangles[:, 0],
        old_triangles[:, 2] - old_triangles[:, 0],
    )
    old_length_sq = np.einsum("ij,ij->i", old_cross, old_cross)

    remapped = np.where(triangles == second[:, None], first[:, None], triangles)
    new_triangles = np.asarray(vertices[remapped], dtype=np.float64)
    new_triangles = np.where(
        (remapped == first[:, None])[:, :, None],
        new_position[:, None, :],
        new_triangles,
    )
    new_cross = np.cross(
        new_triangles[:, 1] - new_triangles[:, 0],
        new_triangles[:, 2] - new_triangles[:, 0],
    )
    new_length_sq = np.einsum("ij,ij->i", new_cross, new_cross)
    alignment = np.einsum("ij,ij->i", old_cross, new_cross)
    invalid = (
        ~np.isfinite(old_length_sq)
        | (old_length_sq <= area_epsilon ** 2)
        | ~np.isfinite(new_length_sq)
        | (new_length_sq <= area_epsilon ** 2)
        | ~np.isfinite(alignment)
        | (alignment <= 0.20 * np.sqrt(old_length_sq * new_length_sq))
    )
    candidate_invalid = np.zeros(len(candidate_edges), dtype=bool)
    np.logical_or.at(candidate_invalid, candidate_index, invalid)
    safe &= ~candidate_invalid
    return safe

def _parallel_qem_candidate_evaluation(
    vertices: np.ndarray,
    quadrics: np.ndarray,
    faces: np.ndarray,
    active_faces: np.ndarray,
    vertex_faces: list[set[int]],
    edges: list[tuple[int, int]] | np.ndarray,
    area_epsilon: float,
    compute_manager=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str, float, float]:
    """Evaluate a deterministic Edge batch on CPU workers or a GPU cost kernel.

    The cost/placement phase and the read-only geometry phase are separate
    barriers. Chunks are submitted concurrently but gathered in input order.
    Neither workers nor the optional device kernel modify topology, quadrics,
    queue state, or candidate ordering.
    """
    candidate_edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    count_edges = len(candidate_edges)
    if count_edges < _QEM_PARALLEL_MIN_CANDIDATES:
        raise ValueError("parallel QEM evaluation requires a sufficiently large candidate batch")
    cpu_count = max(1, int(os.cpu_count() or 1))
    workers = max(1, min(cpu_count, int(_QEM_PARALLEL_MAX_WORKERS), count_edges // _QEM_PARALLEL_CHUNK_MIN))
    chunk_count = max(workers, min(workers * 4, (count_edges + _QEM_PARALLEL_CHUNK_MIN - 1) // _QEM_PARALLEL_CHUNK_MIN))
    chunks = [chunk for chunk in np.array_split(candidate_edges, chunk_count) if len(chunk)]

    gpu_result = None
    gpu_backend_name = ""
    gpu_operation = getattr(compute_manager, "qem_candidate_batch", None)
    cost_started = time.perf_counter()
    if callable(gpu_operation):
        gpu_result = gpu_operation(vertices, quadrics, candidate_edges)
        if gpu_result is not None:
            gpu_backend_name = str(getattr(compute_manager, "backend_name", "GPU")).upper()

    if gpu_result is not None:
        all_costs, all_positions = gpu_result
        all_costs = np.asarray(all_costs, dtype=np.float64).reshape(-1)
        all_positions = np.asarray(all_positions, dtype=np.float64).reshape(-1, 3)
        if len(all_costs) != count_edges or len(all_positions) != count_edges:
            raise FeaturePreservingDecimationError(
                f"{gpu_backend_name} QEM candidate backend returned mismatched batch dimensions."
            )
        candidate_cost_seconds = max(time.perf_counter() - cost_started, 0.0)
    else:
        def evaluate_cost(chunk: np.ndarray):
            return _qem_candidate_costs_vectorized(vertices, quadrics, chunk)

        if workers <= 1 or len(chunks) <= 1:
            cost_parts = [evaluate_cost(chunk) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="PolyfittyQEMCost") as executor:

                cost_parts = list(executor.map(evaluate_cost, chunks))
        all_costs = np.concatenate([item[0] for item in cost_parts])
        all_positions = np.concatenate([item[1] for item in cost_parts], axis=0)
        candidate_cost_seconds = max(time.perf_counter() - cost_started, 0.0)

    ranges = []
    cursor = 0
    for chunk in chunks:
        ranges.append((cursor, cursor + len(chunk)))
        cursor += len(chunk)

    def evaluate_geometry(chunk_range):
        start, stop = chunk_range
        return _qem_candidate_geometry_safe_vectorized(
            vertices,
            faces,
            active_faces,
            vertex_faces,
            candidate_edges[start:stop],
            all_positions[start:stop],
            area_epsilon,
        )

    geometry_started = time.perf_counter()
    if workers <= 1 or len(ranges) <= 1:
        geometry_parts = [evaluate_geometry(item) for item in ranges]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="PolyfittyQEMGuard") as executor:
            geometry_parts = list(executor.map(evaluate_geometry, ranges))
    all_geometry = np.concatenate(geometry_parts) if geometry_parts else np.empty(0, dtype=bool)
    candidate_geometry_seconds = max(time.perf_counter() - geometry_started, 0.0)

    if gpu_result is not None:
        backend_name = f"{gpu_backend_name}+CPU-Multicore"
    else:
        backend_name = "CPU-Multicore" if workers > 1 else "CPU-Vectorized"
    return (
        all_costs,
        all_positions,
        all_geometry,
        workers,
        backend_name,
        candidate_cost_seconds,
        candidate_geometry_seconds,
    )

def _cross_components(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
    ax = float(p1[0] - p0[0])
    ay = float(p1[1] - p0[1])
    az = float(p1[2] - p0[2])
    bx = float(p2[0] - p0[0])
    by = float(p2[1] - p0[1])
    bz = float(p2[2] - p0[2])
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

def _collapse_geometry_is_safe(
    vertices: np.ndarray,
    faces: np.ndarray,
    active_faces: np.ndarray,
    vertex_faces: list[set[int]],
    first: int,
    second: int,
    new_position: np.ndarray,
    area_epsilon: float,
    *,
    affected_face_ids: set[int] | None = None,
) -> bool:
    affected_faces = affected_face_ids
    if affected_faces is None:
        affected_faces = set(vertex_faces[first]).union(vertex_faces[second])
    for face_id in affected_faces:
        if not active_faces[face_id]:
            continue
        tri = faces[face_id]
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if (a == first or b == first or c == first) and (a == second or b == second or c == second):
            continue

        old_cross = _cross_components(vertices[a], vertices[b], vertices[c])
        old_length_sq = old_cross[0] ** 2 + old_cross[1] ** 2 + old_cross[2] ** 2
        if not np.isfinite(old_length_sq) or old_length_sq <= area_epsilon ** 2:
            return False

        na = first if a == second else a
        nb = first if b == second else b
        nc = first if c == second else c
        pa = new_position if na == first else vertices[na]
        pb = new_position if nb == first else vertices[nb]
        pc = new_position if nc == first else vertices[nc]
        new_cross = _cross_components(pa, pb, pc)
        new_length_sq = new_cross[0] ** 2 + new_cross[1] ** 2 + new_cross[2] ** 2
        if not np.isfinite(new_length_sq) or new_length_sq <= area_epsilon ** 2:
            return False
        alignment = (
            old_cross[0] * new_cross[0]
            + old_cross[1] * new_cross[1]
            + old_cross[2] * new_cross[2]
        )

        if not np.isfinite(alignment) or alignment <= 0.20 * np.sqrt(old_length_sq * new_length_sq):
            return False
    return True

def _topology_rejection_reason(
    faces: np.ndarray,
    active_faces: np.ndarray,
    vertex_faces: list[set[int]],
    vertex_neighbors: list[set[int]],
    edge_faces: dict[tuple[int, int], int | tuple[int, ...] | set[int]],
    first: int,
    second: int,
    *,
    allow_boundary: bool = False,
    affected_face_ids: set[int] | None = None,
    generated_face_keys: set[tuple[int, int, int]] | None = None,
) -> Optional[str]:
    incident_edge_faces = edge_faces.get(_edge_key(first, second))
    incident_count = _incidence_count(incident_edge_faces)
    if incident_count == 1 and not allow_boundary:
        return "boundary"
    if incident_count not in ({1, 2} if allow_boundary else {2}):
        return "non_manifold"

    for vertex_id in (first, second):
        for neighbor_id in vertex_neighbors[vertex_id]:
            adjacent = edge_faces.get(_edge_key(vertex_id, neighbor_id))
            count_faces = _incidence_count(adjacent)
            if count_faces == 1 and not allow_boundary:
                return "boundary"
            if count_faces > 2:
                return "non_manifold"

    opposite_vertices: set[int] = set()
    incident_ids = (
        (int(incident_edge_faces),)
        if isinstance(incident_edge_faces, (int, np.integer))
        else incident_edge_faces
    )
    for face_id in incident_ids:
        if not active_faces[face_id]:
            return "non_manifold"
        tri = faces[face_id]
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        if a not in (first, second):
            opposite = a
        elif b not in (first, second):
            opposite = b
        elif c not in (first, second):
            opposite = c
        else:
            return "non_manifold"
        opposite_vertices.add(opposite)
    expected_opposites = 1 if incident_count == 1 else 2
    if len(opposite_vertices) != expected_opposites:
        return "link"

    neighbors_first = vertex_neighbors[first]
    neighbors_second = vertex_neighbors[second]
    if any(value not in neighbors_first or value not in neighbors_second for value in opposite_vertices):
        return "link"
    smaller, larger = (neighbors_first, neighbors_second) if len(neighbors_first) <= len(neighbors_second) else (neighbors_second, neighbors_first)
    common_count = sum(value in larger for value in smaller)
    if common_count != expected_opposites:
        return "link"

    affected_faces = affected_face_ids
    if affected_faces is None:
        affected_faces = set(vertex_faces[first]).union(vertex_faces[second])
    generated_keys = generated_face_keys if generated_face_keys is not None else set()
    generated_keys.clear()
    for face_id in affected_faces:
        if not active_faces[face_id]:
            continue
        tri = faces[face_id]
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        if (a == first or b == first or c == first) and (a == second or b == second or c == second):
            continue
        replaced_values = (
            first if a == second else a,
            first if b == second else b,
            first if c == second else c,
        )
        replaced = tuple(sorted(replaced_values))
        if replaced in generated_keys:
            return "duplicate"
        generated_keys.add(replaced)
    return None

def _remove_face_topology(
    face_id: int,
    faces: np.ndarray,
    vertex_faces: list[set[int]],
    vertex_neighbors: list[set[int]],
    edge_faces: dict[tuple[int, int], int | tuple[int, ...] | set[int]],
) -> None:
    tri = tuple(int(v) for v in faces[face_id])
    for vertex_id in tri:
        vertex_faces[vertex_id].discard(face_id)
    a, b, c = tri
    for u, v in ((a, b), (b, c), (c, a)):
        edge = _edge_key(u, v)
        if edge not in edge_faces:
            continue
        edge_set = _mutable_incidence_set(edge_faces, edge)
        edge_set.discard(int(face_id))
        if not edge_set:
            edge_faces.pop(edge, None)
            vertex_neighbors[u].discard(v)
            vertex_neighbors[v].discard(u)

def _add_face_topology(
    face_id: int,
    faces: np.ndarray,
    vertex_faces: list[set[int]],
    vertex_neighbors: list[set[int]],
    edge_faces: dict[tuple[int, int], int | tuple[int, ...] | set[int]],
) -> None:
    a, b, c = (int(v) for v in faces[face_id])
    for vertex_id in (a, b, c):
        vertex_faces[vertex_id].add(face_id)
    for u, v in ((a, b), (b, c), (c, a)):
        edge = _edge_key(u, v)
        _mutable_incidence_set(edge_faces, edge).add(int(face_id))
        vertex_neighbors[u].add(v)
        vertex_neighbors[v].add(u)

def decimate_mesh_li_nan(
    vertices: np.ndarray,
    faces: np.ndarray,
    planar_result: PlanarExtractionResult | None,
    *,
    target_vertex_count: int,
    progress_cb: Optional[Callable[[int, int, int], None]] = None,
    visualization_cb: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int], None]] = None,
    visualization_stride: int = 32,
    allow_boundary_collapses: bool = False,
    collapse_intersection_guard: bool = False,
    collapse_guard_center_distance_fraction: float = 0.10,
    collapse_guard_normal_angle_degrees: float = 20.0,
    collapse_guard_min_overlap_ratio: float = 0.10,
    collapse_guard_near_gap_fraction: float = 0.01,
    collapse_guard_min_gap_fraction: float = 0.50,
    compute_manager=None,
) -> MeshDecimationResult:
    """Simplify a triangulated mesh using Li & Nan's hierarchical QEM.

    Args:
        vertices: Input mesh vertices. In the Li & Nan pipeline these are the
            Stage-1 filtered coordinates used for planar extraction.
        faces: Triangulated faces for the input vertex array.
        planar_result: Optional output of `extract_planar_regions_li_nan` for
            this exact face ordering and vertex geometry. When omitted, plain
            triangle-plane quadrics are used for shared mesh-budget reduction.
        target_vertex_count: Explicit vertex target. The paper reports about
            twice the number of extracted planar regions as a useful empirical
            starting point.
        allow_boundary_collapses: Whether open boundary edges may be collapsed.
        progress_cb: Optional callback `(completed_collapses, requested_collapses,
            current_vertex_count)`.
        visualization_cb: Optional callback receiving the current mesh, active
            face-region IDs, and the most recently collapsed edge segments as
            `(vertices, faces, region_ids, edge_segments, done, total, vertices_left)`.
        visualization_stride: Number of accepted collapses between viewport frames.
        compute_manager: Optional verified GPU compute manager. Candidate
            placement may use a backend-provided deterministic float64 batch
            kernel; CPU multi-core evaluation remains the reference fallback.

    Returns compacted mesh arrays plus retained region IDs and original-index
    mappings. The input arrays and `planar_result` are not modified. Vertices
    not referenced by any triangle are excluded from the decimation count and
    omitted during output compaction; their number is exposed in the stats.

    Note: boundary/manifold/link/flip guards and the optional local
    intersection guard are application safeguards beyond the paper. If they
    prevent the requested reduction, stats report `target_reached=False`
    rather than silently relaxing those safeguards.
    """
    mesh_vertices, mesh_faces, diagonal = _validate_mesh_arrays(vertices, faces)
    if planar_result is not None and not isinstance(planar_result, PlanarExtractionResult):
        raise FeaturePreservingDecimationError(
            "planar_result must be None or the output of extract_planar_regions_li_nan()."
        )
    if planar_result is not None and planar_result.face_region_ids.shape != (len(mesh_faces),):
        raise FeaturePreservingDecimationError(
            "planar_result face labels do not match the input face count/order."
        )
    raw_target = target_vertex_count
    if isinstance(raw_target, bool) or int(raw_target) != raw_target or int(raw_target) < 3:
        raise FeaturePreservingDecimationError(
            "target_vertex_count must be an integer of at least 3."
        )
    target = int(raw_target)
    if collapse_intersection_guard:
        try:
            from .mesh_collapse_guard import LocalCollapseIntersectionGuard
        except ImportError:
            from mesh_collapse_guard import LocalCollapseIntersectionGuard
    else:
        LocalCollapseIntersectionGuard = None

    if planar_result is None:
        face_region_ids = np.full(len(mesh_faces), -1, dtype=np.int64)
        region_quadrics = {}
    else:
        face_region_ids = np.asarray(planar_result.face_region_ids, dtype=np.int64).copy()
        present_region_ids = {int(x) for x in np.unique(face_region_ids) if x >= 0}
        region_quadrics = _region_plane_quadrics(planar_result)
        missing_regions = present_region_ids - set(region_quadrics)
        if missing_regions:
            raise FeaturePreservingDecimationError(
                "planar_result is missing fitted planes for region IDs: "
                + ", ".join(str(x) for x in sorted(missing_regions))
            )

    work_vertices = mesh_vertices.copy()
    work_faces = mesh_faces.copy()
    active_faces = np.ones(len(work_faces), dtype=bool)
    active_vertices = np.ones(len(work_vertices), dtype=bool)
    topology_started = time.perf_counter()
    vertex_faces, vertex_neighbors, edge_faces = _build_topology(
        work_faces, len(work_vertices)
    )
    estimated_topology_bytes = _estimate_topology_memory_bytes(
        vertex_faces, vertex_neighbors, edge_faces
    )
    topology_build_seconds = max(time.perf_counter() - topology_started, 0.0)
    component_started = time.perf_counter()
    used_vertices = np.asarray([bool(items) for items in vertex_faces], dtype=bool)
    active_vertices &= used_vertices
    input_vertex_count = int(np.count_nonzero(active_vertices))
    initial_boundary, initial_non_manifold = _edge_counts(edge_faces)
    initial_components = _count_components(mesh_faces)
    component_check_seconds = max(time.perf_counter() - component_started, 0.0)
    target = min(target, input_vertex_count)
    requested_work = max(0, input_vertex_count - target)
    adaptive_policy = _adaptive_qem_policy(
        input_vertex_count, len(work_faces), requested_work
    )
    progress_callback_stride = max(
        1,
        int((requested_work + adaptive_policy.max_progress_callbacks - 1)
            // adaptive_policy.max_progress_callbacks)
        if requested_work > adaptive_policy.max_progress_callbacks else 1,
    )
    estimated_queue_entry_bytes = _estimated_heap_entry_bytes()

    collapse_guard = None
    if collapse_intersection_guard:
        try:
            collapse_guard = LocalCollapseIntersectionGuard(
                work_vertices,
                work_faces,
                center_distance_fraction=collapse_guard_center_distance_fraction,
                normal_angle_degrees=collapse_guard_normal_angle_degrees,
                min_overlap_ratio=collapse_guard_min_overlap_ratio,
                near_gap_fraction=collapse_guard_near_gap_fraction,
                min_gap_fraction=collapse_guard_min_gap_fraction,
            )
        except Exception as exc:
            raise FeaturePreservingDecimationError(
                f"could not initialize collapse intersection guard: {exc}"
            ) from exc

    quadric_started = time.perf_counter()
    face_quadrics = _face_plane_quadrics(work_vertices, work_faces)
    vertex_quadrics = _initial_vertex_quadrics(
        len(work_vertices),
        work_faces,
        face_region_ids,
        face_quadrics,
        region_quadrics,
    )
    quadric_setup_seconds = max(time.perf_counter() - quadric_started, 0.0)
    area_epsilon = max(diagonal * diagonal, 1.0) * 1e-14

    heap: list[tuple[float, int, tuple[int, int], int]] = []
    serial = count()
    edge_generation: dict[tuple[int, int], int] = {}
    queued_generation: dict[tuple[int, int], int] = {}

    edge_placement_cache: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    edge_geometry_safe_cache: dict[tuple[int, int], bool] = {}
    queue_live_entries = 0
    queue_cached_cost_requeues = 0
    candidate_evaluation_backend = "CPU-serial"
    candidate_evaluation_workers = 1
    candidate_evaluation_count = 0
    candidate_geometry_checks = 0
    candidate_evaluation_batches = 0
    candidate_evaluation_seconds = 0.0
    candidate_geometry_seconds = 0.0

    def push_edge(
        edge: tuple[int, int],
        *,
        recompute: bool = True,
        candidate_data: tuple[float, np.ndarray, bool] | None = None,
    ) -> None:
        nonlocal queue_live_entries, queue_cached_cost_requeues, candidate_evaluation_count
        first, second = edge
        if first == second or not active_vertices[first] or not active_vertices[second]:
            return
        if edge not in edge_faces:
            return
        cached = edge_placement_cache.get(edge)
        if candidate_data is not None:
            cost, position, geometry_safe = candidate_data
            edge_geometry_safe_cache[edge] = bool(geometry_safe)
            position = np.asarray(position, dtype=np.float64)
            edge_placement_cache[edge] = (
                float(cost), float(position[0]), float(position[1]), float(position[2])
            )
        elif recompute or cached is None:
            cost, position = _optimal_edge_placement(
                work_vertices, vertex_quadrics, first, second
            )
            candidate_evaluation_count += 1

            edge_geometry_safe_cache.pop(edge, None)
            if not np.isfinite(cost):
                edge_placement_cache.pop(edge, None)
                return
            position = np.asarray(position, dtype=np.float64)
            edge_placement_cache[edge] = (
                float(cost), float(position[0]), float(position[1]), float(position[2])
            )
        else:
            cost = cached[0]
            queue_cached_cost_requeues += 1

            edge_geometry_safe_cache.pop(edge, None)
        previous_queued_generation = queued_generation.get(edge)
        if previous_queued_generation is not None:
            queue_live_entries -= 1
        generation = edge_generation.get(edge, 0) + 1
        edge_generation[edge] = generation
        queued_generation[edge] = generation
        queue_live_entries += 1

        heapq.heappush(heap, (cost, next(serial), edge, generation))

    initial_queue_started = time.perf_counter()
    initial_edges = sorted(edge_faces)
    if len(initial_edges) >= _QEM_PARALLEL_MIN_CANDIDATES:
        (
            batch_costs,
            batch_positions,
            batch_geometry_safe,
            candidate_evaluation_workers,
            batch_backend,
            batch_cost_seconds,
            batch_geometry_seconds,
        ) = _parallel_qem_candidate_evaluation(
            work_vertices,
            vertex_quadrics,
            work_faces,
            active_faces,
            vertex_faces,
            initial_edges,
            area_epsilon,
            compute_manager=compute_manager,
        )
        candidate_evaluation_seconds += batch_cost_seconds
        candidate_geometry_seconds += batch_geometry_seconds
        candidate_evaluation_batches += 1
        candidate_geometry_checks += len(initial_edges)
        candidate_evaluation_count += len(initial_edges)
        candidate_evaluation_backend = str(batch_backend)
        for edge, cost, position, geometry_safe in zip(
            initial_edges, batch_costs, batch_positions, batch_geometry_safe
        ):
            if not np.isfinite(cost):
                continue
            push_edge(
                edge,
                candidate_data=(float(cost), position, bool(geometry_safe)),
            )
    else:
        for edge in initial_edges:
            push_edge(edge)
    initial_queue_seconds = max(time.perf_counter() - initial_queue_started, 0.0)
    queue_peak_entries = len(heap)
    estimated_queue_peak_bytes = len(heap) * estimated_queue_entry_bytes
    estimated_queue_stale_peak_bytes = 0
    queue_compaction_count = 0
    queue_memory_compaction_count = 0
    queue_stale_entries_removed = 0
    queue_placement_reuses = 0
    progress_callback_count = 0
    last_progress_done = -1

    successful_collapses = 0
    visualization_stride = max(1, int(visualization_stride))
    recent_collapse_segments: list[np.ndarray] = []
    recent_active_faces: set[int] = set()
    rejected_boundary_or_non_manifold = 0
    rejected_link_condition = 0
    rejected_duplicate_face = 0
    rejected_face_flip_or_degenerate = 0
    rejected_collapse_new_crossing = 0
    rejected_collapse_new_near_coplanar = 0
    rejected_collapse_gap_reduction = 0
    current_vertex_count = input_vertex_count
    active_vertex_count = current_vertex_count

    def target_not_reached() -> bool:
        return active_vertex_count > target

    collapse_started = time.perf_counter()

    affected_face_scratch: set[int] = set()
    generated_face_key_scratch: set[tuple[int, int, int]] = set()
    touched_edge_scratch: set[tuple[int, int]] = set()
    while target_not_reached() and heap:
        _queued_cost, _order, edge, generation = heapq.heappop(heap)
        first, second = edge
        if queued_generation.get(edge) != generation:
            continue
        queued_generation.pop(edge, None)
        queue_live_entries -= 1
        if edge_generation.get(edge) != generation:
            continue
        if (
            not active_vertices[first]
            or not active_vertices[second]
            or edge not in edge_faces
        ):
            continue
        affected_face_scratch.clear()
        affected_face_scratch.update(vertex_faces[first])
        affected_face_scratch.update(vertex_faces[second])
        reason = _topology_rejection_reason(
            work_faces,
            active_faces,
            vertex_faces,
            vertex_neighbors,
            edge_faces,
            first,
            second,
            allow_boundary=allow_boundary_collapses,
            affected_face_ids=affected_face_scratch,
            generated_face_keys=generated_face_key_scratch,
        )
        if reason is not None:
            if reason in ("boundary", "non_manifold"):
                rejected_boundary_or_non_manifold += 1
            elif reason == "link":
                rejected_link_condition += 1
            elif reason == "duplicate":
                rejected_duplicate_face += 1
            continue

        cached_placement = edge_placement_cache.get(edge)
        if cached_placement is None:

            continue
        position = np.asarray(
            (cached_placement[1], cached_placement[2], cached_placement[3]),
            dtype=np.float64,
        )
        queue_placement_reuses += 1
        cached_geometry_safe = edge_geometry_safe_cache.pop(edge, None)
        if cached_geometry_safe is None:
            geometry_started = time.perf_counter()
            cached_geometry_safe = _collapse_geometry_is_safe(
                work_vertices,
                work_faces,
                active_faces,
                vertex_faces,
                first,
                second,
                position,
                area_epsilon,
                affected_face_ids=affected_face_scratch,
            )
            candidate_geometry_checks += 1
            candidate_geometry_seconds += max(time.perf_counter() - geometry_started, 0.0)
        if not cached_geometry_safe:
            rejected_face_flip_or_degenerate += 1
            continue

        collapse_segment = np.empty((2, 3), dtype=np.float64)
        collapse_segment[0] = work_vertices[first]
        collapse_segment[1] = work_vertices[second]
        affected_face_set = affected_face_scratch
        if collapse_guard is not None:
            decision = collapse_guard.check_candidate(
                work_vertices,
                work_faces,
                active_faces,
                affected_face_set,
                first,
                second,
                position,
            )
            if decision.reject:
                if decision.reason == "new_crossing":
                    rejected_collapse_new_crossing += 1
                elif decision.reason == "new_near_coplanar":
                    rejected_collapse_new_near_coplanar += 1
                else:
                    rejected_collapse_gap_reduction += 1
                continue

        affected_faces = sorted(affected_face_set)
        touched_edges = touched_edge_scratch
        touched_edges.clear()
        for face_id in affected_faces:
            old_tri = work_faces[face_id]
            old_a, old_b, old_c = int(old_tri[0]), int(old_tri[1]), int(old_tri[2])
            touched_edges.add(_edge_key(old_a, old_b))
            touched_edges.add(_edge_key(old_b, old_c))
            touched_edges.add(_edge_key(old_c, old_a))
            _remove_face_topology(
                face_id,
                work_faces,
                vertex_faces,
                vertex_neighbors,
                edge_faces,
            )
            tri = work_faces[face_id]
            tri_a, tri_b, tri_c = int(tri[0]), int(tri[1]), int(tri[2])
            contains_first = tri_a == first or tri_b == first or tri_c == first
            contains_second = tri_a == second or tri_b == second or tri_c == second
            if contains_first and contains_second:
                active_faces[face_id] = False
                continue
            if contains_second:
                work_faces[face_id, work_faces[face_id] == second] = first
            _add_face_topology(
                face_id,
                work_faces,
                vertex_faces,
                vertex_neighbors,
                edge_faces,
            )
            new_tri = work_faces[face_id]
            new_a, new_b, new_c = int(new_tri[0]), int(new_tri[1]), int(new_tri[2])
            touched_edges.add(_edge_key(new_a, new_b))
            touched_edges.add(_edge_key(new_b, new_c))
            touched_edges.add(_edge_key(new_c, new_a))

        work_vertices[first] = position
        vertex_quadrics[first] += vertex_quadrics[second]
        active_vertices[second] = False
        active_vertex_count -= 1
        if allow_boundary_collapses and not vertex_faces[first] and active_vertices[first]:
            active_vertices[first] = False
            active_vertex_count -= 1
        successful_collapses += 1
        recent_collapse_segments.append(collapse_segment)
        recent_active_faces.update(face_id for face_id in affected_faces if active_faces[face_id])

        if collapse_guard is not None:
            collapse_guard.record_accepted_collapse(
                work_vertices, work_faces, active_faces, affected_face_set
            )

        for touched_edge in sorted(touched_edges):
            if touched_edge in edge_faces:

                push_edge(
                    touched_edge,
                    recompute=(first in touched_edge or not _QEM_REUSE_UNCHANGED_EDGE_COSTS),
                )
            else:

                if queued_generation.pop(touched_edge, None) is not None:
                    queue_live_entries -= 1
                edge_placement_cache.pop(touched_edge, None)
                edge_geometry_safe_cache.pop(touched_edge, None)

        queue_peak_entries = max(queue_peak_entries, len(heap))
        stale_entries = max(0, len(heap) - queue_live_entries)
        estimated_queue_peak_bytes = max(
            estimated_queue_peak_bytes,
            len(heap) * estimated_queue_entry_bytes,
        )
        estimated_queue_stale_peak_bytes = max(
            estimated_queue_stale_peak_bytes,
            stale_entries * estimated_queue_entry_bytes,
        )
        should_compact, memory_pressure_triggered = _should_compact_adaptive_heap(
            len(heap), queue_live_entries, stale_entries,
            adaptive_policy, estimated_queue_entry_bytes,
        )
        if should_compact:
            removed = _compact_qem_heap(
                heap, queued_generation, edge_faces, active_vertices
            )
            if removed:
                queue_compaction_count += 1
                queue_stale_entries_removed += removed
                if memory_pressure_triggered:
                    queue_memory_compaction_count += 1
                queue_live_entries = len(heap)

        if progress_cb is not None and (
            successful_collapses % progress_callback_stride == 0
            or not target_not_reached()
            or not heap
        ):
            progress_cb(successful_collapses, requested_work, active_vertex_count)
            progress_callback_count += 1
            last_progress_done = successful_collapses
        if visualization_cb is not None and (
            successful_collapses % visualization_stride == 0 or not target_not_reached() or not heap
        ):
            active_face_ids = np.flatnonzero(active_faces).astype(np.int64)
            active_region_ids = face_region_ids[active_face_ids].copy()
            if recent_active_faces:
                active_mask = np.isin(active_face_ids, np.fromiter(recent_active_faces, dtype=np.int64))
                active_region_ids[active_mask] = -2
            visualization_cb(
                work_vertices.copy(),
                work_faces[active_face_ids].copy(),
                active_region_ids,
                np.asarray(recent_collapse_segments, dtype=np.float64).reshape(-1, 2, 3),
                int(successful_collapses),
                int(requested_work),
                int(active_vertex_count),
            )
            recent_collapse_segments.clear()
            recent_active_faces.clear()

    if visualization_cb is not None and recent_collapse_segments:
        active_face_ids = np.flatnonzero(active_faces).astype(np.int64)
        active_region_ids = face_region_ids[active_face_ids].copy()
        if recent_active_faces:
            active_mask = np.isin(active_face_ids, np.fromiter(recent_active_faces, dtype=np.int64))
            active_region_ids[active_mask] = -2
        visualization_cb(
            work_vertices.copy(),
            work_faces[active_face_ids].copy(),
            active_region_ids,
            np.asarray(recent_collapse_segments, dtype=np.float64).reshape(-1, 2, 3),
            int(successful_collapses),
            int(requested_work),
            int(active_vertex_count),
        )

    if progress_cb is not None and last_progress_done != successful_collapses:
        progress_cb(successful_collapses, requested_work, active_vertex_count)
        progress_callback_count += 1
        last_progress_done = successful_collapses

    collapse_seconds = max(time.perf_counter() - collapse_started, 0.0)
    final_validation_started = time.perf_counter()
    surviving_face_ids = np.flatnonzero(active_faces).astype(np.int64)
    surviving_faces_old_indices = work_faces[surviving_face_ids]
    used_old_vertex_ids = np.unique(surviving_faces_old_indices.reshape(-1)).astype(np.int64)
    old_to_new = np.full(len(work_vertices), -1, dtype=np.int64)
    old_to_new[used_old_vertex_ids] = np.arange(len(used_old_vertex_ids), dtype=np.int64)
    output_vertices = work_vertices[used_old_vertex_ids].copy()
    output_faces = old_to_new[surviving_faces_old_indices].astype(np.int64, copy=True)
    output_region_ids = face_region_ids[surviving_face_ids].copy()

    final_topology = _build_topology(output_faces, len(output_vertices))
    final_vertex_faces, final_vertex_neighbors, final_edge_faces = final_topology
    final_boundary, final_non_manifold = _edge_counts(final_edge_faces)
    final_components = _count_components(output_faces)

    if final_components != initial_components:
        raise FeaturePreservingDecimationError(
            "application safety invariant failed: decimation changed the connected-component count "
            f"({initial_components} → {final_components}); output rejected."
        )
    if not allow_boundary_collapses and final_boundary != initial_boundary:
        raise FeaturePreservingDecimationError(
            "application safety invariant failed: protected boundary-edge count changed; output rejected."
        )
    if allow_boundary_collapses and final_boundary > initial_boundary:
        raise FeaturePreservingDecimationError(
            "application safety invariant failed: decimation created new boundary edges; output rejected."
        )
    if final_non_manifold > initial_non_manifold:
        raise FeaturePreservingDecimationError(
            "application safety invariant failed: decimation created new non-manifold edges; output rejected."
        )

    final_validation_seconds = max(time.perf_counter() - final_validation_started, 0.0)
    stats = MeshDecimationStats(
        input_vertex_count=int(len(mesh_vertices)),
        input_used_vertex_count=input_vertex_count,
        input_unused_vertex_count=int(len(mesh_vertices) - input_vertex_count),
        input_face_count=int(len(mesh_faces)),
        output_vertex_count=int(len(output_vertices)),
        output_face_count=int(len(output_faces)),
        target_vertex_count=target,
        successful_collapses=int(successful_collapses),
        target_reached=bool(len(output_vertices) <= target),
        initial_component_count=int(initial_components),
        final_component_count=int(final_components),
        initial_boundary_edge_count=int(initial_boundary),
        final_boundary_edge_count=int(final_boundary),
        initial_non_manifold_edge_count=int(initial_non_manifold),
        final_non_manifold_edge_count=int(final_non_manifold),
        rejected_boundary_or_non_manifold=int(rejected_boundary_or_non_manifold),
        rejected_link_condition=int(rejected_link_condition),
        rejected_duplicate_face=int(rejected_duplicate_face),
        rejected_face_flip_or_degenerate=int(rejected_face_flip_or_degenerate),
        collapse_intersection_guard_enabled=bool(collapse_guard is not None),
        rejected_collapse_new_crossing=int(rejected_collapse_new_crossing),
        rejected_collapse_new_near_coplanar=int(rejected_collapse_new_near_coplanar),
        rejected_collapse_gap_reduction=int(rejected_collapse_gap_reduction),
        collapse_guard_candidate_pairs_measured=int(collapse_guard.pairs_tested if collapse_guard is not None else 0),
        collapse_guard_index_rebuilds=int(collapse_guard.index_rebuild_count if collapse_guard is not None else 0),
        topology_build_seconds=float(topology_build_seconds),
        component_check_seconds=float(component_check_seconds),
        quadric_setup_seconds=float(quadric_setup_seconds),
        initial_queue_seconds=float(initial_queue_seconds),
        collapse_seconds=float(collapse_seconds),
        final_validation_seconds=float(final_validation_seconds),
        queue_compaction_count=int(queue_compaction_count),
        queue_stale_entries_removed=int(queue_stale_entries_removed),
        queue_peak_entries=int(queue_peak_entries),
        queue_placement_reuses=int(queue_placement_reuses),
        queue_cached_cost_requeues=int(queue_cached_cost_requeues),
        topology_compact_map_entries=int(len(edge_faces)),
        topology_compact_edge_entries=int(sum(not isinstance(value, set) for value in edge_faces.values())),
        topology_materialized_edge_entries=int(sum(isinstance(value, set) for value in edge_faces.values())),
        adaptive_qem_tier=str(adaptive_policy.tier),
        adaptive_queue_compact_min_entries=int(adaptive_policy.compact_min_entries),
        adaptive_queue_stale_ratio=float(adaptive_policy.stale_ratio),
        adaptive_queue_stale_floor=int(adaptive_policy.stale_floor),
        adaptive_queue_stale_divisor=int(adaptive_policy.stale_divisor),
        adaptive_queue_memory_budget_bytes=int(adaptive_policy.stale_memory_budget_bytes),
        adaptive_progress_callback_stride=int(progress_callback_stride),
        progress_callback_count=int(progress_callback_count),
        estimated_topology_bytes=int(estimated_topology_bytes),
        estimated_queue_entry_bytes=int(estimated_queue_entry_bytes),
        estimated_queue_peak_bytes=int(estimated_queue_peak_bytes),
        estimated_queue_stale_peak_bytes=int(estimated_queue_stale_peak_bytes),
        queue_memory_compaction_count=int(queue_memory_compaction_count),
        topology_storage_mode="compact-edge-incidence/lazy-materialization",
        candidate_evaluation_backend=str(candidate_evaluation_backend),
        candidate_evaluation_workers=int(candidate_evaluation_workers),
        candidate_evaluation_count=int(candidate_evaluation_count),
        candidate_geometry_checks=int(candidate_geometry_checks),
        candidate_evaluation_batches=int(candidate_evaluation_batches),
        candidate_evaluation_seconds=float(candidate_evaluation_seconds),
        candidate_geometry_seconds=float(candidate_geometry_seconds),
    )
    return MeshDecimationResult(
        vertices=output_vertices,
        faces=output_faces,
        face_region_ids=output_region_ids,
        original_vertex_indices=used_old_vertex_ids,
        original_face_indices=surviving_face_ids,
        stats=stats,
        collapse_guard_report=collapse_guard.report() if collapse_guard is not None else None,
    )
