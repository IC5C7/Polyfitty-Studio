from __future__ import annotations

"""Planar-region and intra-contour extraction from Li & Nan (2021), Section 3.2.

This is Stage 2 only. It consumes an already triangulated mesh (normally the
Stage-1 filtered mesh) and returns per-face planar-region labels, fitted region
planes, region adjacency, and intra-contour edges. It does not decimate or
change the input vertex/face arrays.

Source-reading note:
The prose in Section 3.2 says neighbouring faces are grouped when their
filtered normals are sufficiently close, and the experiments mention a 15°
normal-proximity tolerance. The printed comparison in Algorithm 1 appears as
"sin_0(theta)" in the available paper copy and is not defined in the prose.
To make the threshold explicit and interpretable, this implementation exposes
an angular tolerance in degrees and uses dot(n_i, n_j) > cos(tolerance).
This is an explicit implementation interpretation of the prose/experiment,
not a claim that the ambiguous printed symbol has been unambiguously resolved.

The paper specifies an area threshold and a fitting threshold but does not
provide universal numerical values. They are required arguments; no numeric
values are invented here. The near-coplanar adjacency refinement is described
in prose but not given a separate equation/threshold. This implementation
uses the same explicit fitting threshold for the least-squares residual of the
combined adjacent regions, and the same angular tolerance for their fitted
normal proximity. This operationalization is documented as an implementation
choice rather than a separately published paper default.
"""

from dataclasses import dataclass
import heapq
import time
from typing import Optional

import numpy as np

try:
    from .feature_preserving import _face_geometry, _validate_mesh_arrays
except ImportError:
    from feature_preserving import _face_geometry, _validate_mesh_arrays

class PlanarRegionExtractionError(ValueError):
    """Raised when planar-region extraction cannot safely process a mesh."""

@dataclass(frozen=True)
class FittedPlane:
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    rms_distance: float
    max_distance: float
    point_count: int

@dataclass(frozen=True)
class PlanarRegion:
    region_id: int
    face_indices: np.ndarray
    vertex_indices: np.ndarray
    area: float
    plane: FittedPlane

@dataclass(frozen=True)
class ContourFeature:
    """One mesh edge separating at least two assigned planar regions.

    For a manifold edge with exactly two adjacent planar regions, `vertices`
    contains the two endpoints projected onto the intersection line of the
    fitted region planes, as described in Section 3.2. If the planes are
    parallel/coincident within numerical tolerance, or the edge is
    non-manifold, original edge endpoints are retained and `projected` is
    False. This avoids inventing an intersection line when none is uniquely
    defined.
    """

    vertex_indices: tuple[int, int]
    adjacent_face_indices: tuple[int, ...]
    adjacent_region_ids: tuple[int, ...]
    vertices: np.ndarray
    projected: bool
    line_point: Optional[np.ndarray] = None
    line_direction: Optional[np.ndarray] = None

@dataclass(frozen=True)
class PlanarExtractionStats:
    input_vertices: int
    input_faces: int
    boundary_edge_count: int
    non_manifold_edge_count: int
    coarse_region_count: int
    refined_region_count: int
    ungrouped_face_count: int
    ungrouped_region_count: int
    region_adjacency_count: int
    contour_edge_count: int
    projected_contour_count: int
    edge_incidence_seconds: float = 0.0
    coarse_region_seconds: float = 0.0
    refinement_seconds: float = 0.0
    plane_fit_seconds: float = 0.0
    contour_seconds: float = 0.0

@dataclass(frozen=True)
class PlanarExtractionResult:
    face_region_ids: np.ndarray
    regions: tuple[PlanarRegion, ...]
    region_adjacency: tuple[tuple[int, int], ...]
    contours: tuple[ContourFeature, ...]
    stats: PlanarExtractionStats

def _build_edge_incidence(
    faces: np.ndarray,
) -> tuple[dict[tuple[int, int], list[int]], list[list[int]]]:
    """Return edge -> incident faces and face -> edge-neighbour faces.

    The paper's region growing text explicitly says to traverse triangles via
    common edges. A face pair therefore becomes a growth-neighbour pair only
    when the two faces share an entire indexed edge (two vertex indices).
    """
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, tri in enumerate(faces):
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(edge, []).append(face_id)

    face_neighbors: list[set[int]] = [set() for _ in range(len(faces))]
    for incident in edge_to_faces.values():
        if len(incident) < 2:
            continue

        unique_incident = sorted(set(incident))
        for i, first in enumerate(unique_incident):
            for second in unique_incident[i + 1:]:
                face_neighbors[first].add(second)
                face_neighbors[second].add(first)
    return edge_to_faces, [sorted(items) for items in face_neighbors]

def _grow_coarse_regions(
    filtered_normals: np.ndarray,
    face_neighbors: list[list[int]],
    normal_angle_degrees: float,
    visualization_cb=None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Algorithm 1's seed-and-grow grouping using common-edge adjacency."""
    dot_threshold = float(np.cos(np.deg2rad(normal_angle_degrees)))
    labels = np.full(len(filtered_normals), -1, dtype=np.int64)
    groups: list[np.ndarray] = []
    total_faces = int(len(filtered_normals))
    visualization_stride = max(1, int(np.ceil(max(1, total_faces) / 24.0)))
    processed_faces = 0
    next_visualization = visualization_stride
    last_visualization = 0

    for seed in range(len(filtered_normals)):
        if labels[seed] != -1:
            continue
        region_id = len(groups)
        labels[seed] = region_id
        stack = [seed]
        members: list[int] = []
        seed_normal = filtered_normals[seed]

        while stack:
            current = stack.pop()
            members.append(current)
            processed_faces += 1

            for neighbor in face_neighbors[current]:
                if labels[neighbor] != -1:
                    continue
                similarity = float(np.dot(seed_normal, filtered_normals[neighbor]))
                if similarity > dot_threshold:
                    labels[neighbor] = region_id
                    stack.append(neighbor)
            if visualization_cb is not None and processed_faces >= next_visualization:
                visualization_cb(
                    labels.copy(), np.asarray(members, dtype=np.int64), processed_faces, total_faces,
                    "COARSE REGION GROWTH",
                )
                last_visualization = processed_faces
                next_visualization += visualization_stride

        groups.append(np.asarray(members, dtype=np.int64))

    if visualization_cb is not None and total_faces and last_visualization != total_faces:
        visualization_cb(
            labels.copy(), np.asarray(members, dtype=np.int64), total_faces, total_faces,
            "COARSE REGION GROWTH COMPLETE",
        )
    return labels, groups

def _unique_vertices_for_faces(faces: np.ndarray, face_ids: np.ndarray) -> np.ndarray:
    if len(face_ids) == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(faces[face_ids].reshape(-1))

_COVARIANCE_PLANE_FIT_MIN_POINTS = 256
_COVARIANCE_PLANE_FIT_SAFE_EIGEN_RATIO = 1e-14

def _fit_plane_from_points(points: np.ndarray, reference_normal: np.ndarray) -> FittedPlane:
    """Fit a least-squares plane, using a 3x3 covariance eigensolve for large regions.

    The smallest right-singular vector of the centered N×3 point matrix is
    exactly the eigenvector of its 3×3 covariance matrix with the smallest
    eigenvalue. For large regions, forming that covariance and solving the
    fixed-size symmetric system avoids an O(N) SVD workspace. Ill-conditioned
    or small point sets retain the original SVD path for numerical robustness.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        raise PlanarRegionExtractionError("a planar region has fewer than three distinct vertices.")
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    normal = None

    if len(points) >= _COVARIANCE_PLANE_FIT_MIN_POINTS:
        covariance = centered.T @ centered
        if np.isfinite(covariance).all():
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            largest = max(float(eigenvalues[-1]), 0.0)
            second = max(float(eigenvalues[1]), 0.0)

            if (
                np.isfinite(eigenvalues).all()
                and largest > 0.0
                and second > largest * _COVARIANCE_PLANE_FIT_SAFE_EIGEN_RATIO
            ):
                normal = np.asarray(eigenvectors[:, 0], dtype=np.float64)

    if normal is None:
        _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if len(singular_values) < 3 or singular_values[1] <= max(singular_values[0], 1.0) * 1e-12:
            raise PlanarRegionExtractionError("a region is too close to collinear for a stable plane fit.")
        normal = np.asarray(vh[-1], dtype=np.float64)

    normal_length = float(np.linalg.norm(normal))
    if not np.isfinite(normal_length) or normal_length <= 1e-15:
        raise PlanarRegionExtractionError("least-squares plane fitting produced an invalid normal.")
    normal /= normal_length
    reference = np.asarray(reference_normal, dtype=np.float64).reshape(3)
    if float(np.dot(normal, reference)) < 0.0:
        normal = -normal

    offset = -float(np.dot(normal, centroid))
    distances = np.abs(points @ normal + offset)
    return FittedPlane(
        normal=normal,
        offset=offset,
        centroid=centroid,
        rms_distance=float(np.sqrt(np.mean(distances * distances))),
        max_distance=float(np.max(distances)),
        point_count=int(len(points)),
    )

def _fit_plane(
    vertices: np.ndarray,
    vertex_ids: np.ndarray,
    face_normals: np.ndarray,
    face_ids: np.ndarray,
) -> FittedPlane:
    points = vertices[vertex_ids]
    if len(points) < 3:
        raise PlanarRegionExtractionError("a planar region has fewer than three distinct vertices.")
    reference = np.mean(face_normals[face_ids], axis=0)
    return _fit_plane_from_points(points, reference)

def _region_area(face_areas: np.ndarray, face_ids: np.ndarray) -> float:
    return float(np.sum(face_areas[face_ids]))

def _region_plane_cost(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_face_ids: np.ndarray,
    target_plane: FittedPlane,
) -> float:
    """Eq. (6): sum of absolute distances of source-region vertices to plane."""
    vertex_ids = _unique_vertices_for_faces(faces, source_face_ids)
    points = vertices[vertex_ids]
    distances = np.abs(points @ target_plane.normal + target_plane.offset)
    return float(np.sum(distances))

def _combined_region_fit_cost(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    first_face_ids: np.ndarray,
    second_face_ids: np.ndarray,
) -> tuple[float, FittedPlane]:
    combined_face_ids = np.concatenate((first_face_ids, second_face_ids))
    vertex_ids = _unique_vertices_for_faces(faces, combined_face_ids)
    plane = _fit_plane(vertices, vertex_ids, face_normals, combined_face_ids)
    points = vertices[vertex_ids]
    cost = float(np.sum(np.abs(points @ plane.normal + plane.offset)))
    return cost, plane

def _make_region_adjacency(
    edge_to_faces: dict[tuple[int, int], list[int]],
    face_labels: np.ndarray,
) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {
        int(region_id): set() for region_id in np.unique(face_labels) if region_id >= 0
    }
    for incident in edge_to_faces.values():
        region_ids = sorted({int(face_labels[f]) for f in incident if face_labels[f] >= 0})
        if len(region_ids) < 2:
            continue
        for i, first in enumerate(region_ids):
            for second in region_ids[i + 1:]:
                adjacency[first].add(second)
                adjacency[second].add(first)
    return adjacency

def _merge_region_into(
    source_id: int,
    target_id: int,
    groups: dict[int, np.ndarray],
    areas: dict[int, float],
    adjacency: dict[int, set[int]],
) -> None:
    """Merge one graph node into another and propagate its graph edges."""
    if source_id == target_id or source_id not in groups or target_id not in groups:
        return
    groups[target_id] = np.concatenate((groups[target_id], groups[source_id]))
    areas[target_id] += areas[source_id]
    del groups[source_id]
    del areas[source_id]

    inherited = set(adjacency.get(source_id, set()))
    inherited.discard(target_id)
    for neighbor_id in inherited:
        if neighbor_id not in groups:
            continue
        adjacency.setdefault(neighbor_id, set()).discard(source_id)
        adjacency[neighbor_id].add(target_id)
        adjacency.setdefault(target_id, set()).add(neighbor_id)
    adjacency.setdefault(target_id, set()).discard(source_id)
    adjacency.pop(source_id, None)

def _refine_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    coarse_groups: list[np.ndarray],
    edge_to_faces: dict[tuple[int, int], list[int]],
    small_region_area_threshold: float,
    fitting_threshold: float,
    normal_angle_degrees: float,
) -> tuple[dict[int, np.ndarray], set[int]]:
    """Apply Algorithm 1's small-region redistribution and adjacent fusion.

    Region vertex sets and fitted planes are cached. A region's cache is
    invalidated only when that region absorbs another region; candidate
    comparisons against the same target no longer repeat identical unique-
    vertex extraction and least-squares fitting. Pair fits use the sorted
    union of cached vertex IDs instead of rebuilding unique IDs from all
    concatenated triangle indices for every adjacency candidate.
    """
    groups = {i: ids.copy() for i, ids in enumerate(coarse_groups)}
    areas = {i: _region_area(face_areas, ids) for i, ids in enumerate(coarse_groups)}
    labels = np.full(len(faces), -1, dtype=np.int64)
    for region_id, face_ids in groups.items():
        labels[face_ids] = region_id
    adjacency = _make_region_adjacency(edge_to_faces, labels)
    ungrouped_region_ids: set[int] = set()

    vertex_cache: dict[int, np.ndarray] = {}
    plane_cache: dict[int, FittedPlane] = {}

    def region_vertices(region_id: int) -> np.ndarray:
        cached = vertex_cache.get(region_id)
        if cached is None:
            cached = _unique_vertices_for_faces(faces, groups[region_id])
            vertex_cache[region_id] = cached
        return cached

    def region_plane(region_id: int) -> FittedPlane:
        cached = plane_cache.get(region_id)
        if cached is None:
            face_ids = groups[region_id]
            cached = _fit_plane_from_points(
                vertices[region_vertices(region_id)],
                np.mean(face_normals[face_ids], axis=0),
            )
            plane_cache[region_id] = cached
        return cached

    def remove_region_cache(region_id: int) -> None:
        vertex_cache.pop(region_id, None)
        plane_cache.pop(region_id, None)

    def merge_region_cache(target_id: int, source_id: int, merged_vertex_ids: np.ndarray) -> None:
        vertex_cache[target_id] = merged_vertex_ids

        merged_faces = groups[target_id]
        plane_cache[target_id] = _fit_plane_from_points(
            vertices[merged_vertex_ids],
            np.mean(face_normals[merged_faces], axis=0),
        )
        remove_region_cache(source_id)

    initial_order = sorted(groups, key=lambda rid: (areas[rid], rid))
    for source_id in initial_order:
        if source_id not in groups or areas[source_id] >= small_region_area_threshold:
            continue
        candidate_ids = sorted(adjacency.get(source_id, set()))
        if not candidate_ids:
            ungrouped_region_ids.add(source_id)
            for neighbor_id in adjacency.get(source_id, set()):
                adjacency.get(neighbor_id, set()).discard(source_id)
            adjacency.pop(source_id, None)
            groups.pop(source_id, None)
            areas.pop(source_id, None)
            remove_region_cache(source_id)
            continue

        source_vertices = region_vertices(source_id)
        source_points = vertices[source_vertices]
        best_target: Optional[int] = None
        best_cost = float("inf")
        for target_id in candidate_ids:
            if target_id not in groups:
                continue
            target_plane = region_plane(target_id)
            distances = np.abs(source_points @ target_plane.normal + target_plane.offset)
            cost = float(np.sum(distances))
            if cost < best_cost:
                best_cost = cost
                best_target = target_id

        if best_target is not None and best_cost < fitting_threshold:
            merged_vertex_ids = np.union1d(
                region_vertices(best_target), source_vertices
            ).astype(np.int64, copy=False)
            _merge_region_into(source_id, best_target, groups, areas, adjacency)
            merge_region_cache(best_target, source_id, merged_vertex_ids)
        else:
            ungrouped_region_ids.add(source_id)
            for neighbor_id in adjacency.get(source_id, set()):
                adjacency.get(neighbor_id, set()).discard(source_id)
            adjacency.pop(source_id, None)
            groups.pop(source_id, None)
            areas.pop(source_id, None)
            remove_region_cache(source_id)

    dot_threshold = float(np.cos(np.deg2rad(normal_angle_degrees)))
    if fitting_threshold > 0.0 and groups:
        versions: dict[int, int] = {region_id: 0 for region_id in groups}
        for region_id in groups:
            region_vertices(region_id)
            region_plane(region_id)

        merge_heap: list[tuple[float, int, int, int, int, FittedPlane, np.ndarray]] = []
        enqueued_pairs: set[tuple[int, int, int, int]] = set()

        def enqueue_pair(first_id: int, second_id: int) -> None:
            if first_id == second_id or first_id not in groups or second_id not in groups:
                return
            first_id, second_id = sorted((int(first_id), int(second_id)))
            if second_id not in adjacency.get(first_id, set()):
                return
            first_version = versions[first_id]
            second_version = versions[second_id]
            pair_key = (first_id, second_id, first_version, second_version)
            if pair_key in enqueued_pairs:
                return
            enqueued_pairs.add(pair_key)
            if float(np.dot(plane_cache[first_id].normal, plane_cache[second_id].normal)) <= dot_threshold:
                return
            combined_vertex_ids = np.union1d(
                region_vertices(first_id), region_vertices(second_id)
            ).astype(np.int64, copy=False)
            combined_face_ids = np.concatenate((groups[first_id], groups[second_id]))
            combined_plane = _fit_plane_from_points(
                vertices[combined_vertex_ids],
                np.mean(face_normals[combined_face_ids], axis=0),
            )
            cost = float(np.sum(np.abs(
                vertices[combined_vertex_ids] @ combined_plane.normal + combined_plane.offset
            )))
            if np.isfinite(cost):
                heapq.heappush(
                    merge_heap,
                    (cost, first_id, second_id, first_version, second_version,
                     combined_plane, combined_vertex_ids),
                )

        for first_id in sorted(groups):
            for second_id in sorted(adjacency.get(first_id, set())):
                if second_id > first_id:
                    enqueue_pair(first_id, second_id)

        while merge_heap:
            cost, first_id, second_id, first_version, second_version, combined_plane, combined_vertex_ids = heapq.heappop(merge_heap)
            if first_id not in groups or second_id not in groups:
                continue
            if versions[first_id] != first_version or versions[second_id] != second_version:
                continue
            if second_id not in adjacency.get(first_id, set()):
                continue
            if cost >= fitting_threshold:
                break

            target_id, source_id = first_id, second_id
            if (areas[source_id], -source_id) > (areas[target_id], -target_id):
                target_id, source_id = source_id, target_id
            _merge_region_into(source_id, target_id, groups, areas, adjacency)
            versions[target_id] += 1
            versions[source_id] += 1
            vertex_cache[target_id] = combined_vertex_ids
            plane_cache[target_id] = combined_plane
            remove_region_cache(source_id)

            for neighbor_id in sorted(adjacency.get(target_id, set())):
                enqueue_pair(target_id, neighbor_id)

    return groups, ungrouped_region_ids

def _plane_intersection_line(
    first: FittedPlane,
    second: FittedPlane,
    scale: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    n1, n2 = first.normal, second.normal
    direction = np.cross(n1, n2)
    denom = float(np.dot(direction, direction))
    if not np.isfinite(denom) or denom <= max(scale * scale, 1.0) * 1e-24:
        return None

    b1, b2 = -first.offset, -second.offset
    point = (
        b1 * np.cross(n2, direction) + b2 * np.cross(direction, n1)
    ) / denom
    direction /= np.sqrt(denom)
    if not np.isfinite(point).all() or not np.isfinite(direction).all():
        return None
    return point, direction

def _project_edge_to_line(
    edge_points: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray,
) -> np.ndarray:
    offsets = edge_points - line_point
    parameters = offsets @ line_direction
    return line_point[None, :] + parameters[:, None] * line_direction[None, :]

def extract_planar_regions_li_nan(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    small_region_area_threshold: float,
    fitting_threshold: float,
    normal_angle_degrees: float = 15.0,
    visualization_cb=None,
) -> PlanarExtractionResult:
    """Extract planar regions and intra-contours following Li & Nan (2021).

    Args:
        vertices: Nx3 positions of a triangulated mesh. Normally this is the
            output vertex array from `filter_mesh_li_nan` (Stage 1).
        faces: Mx3 triangle indices. This function never changes this array.
        small_region_area_threshold: Area threshold for Algorithm 1's small
            region refinement. Required because the paper gives no universal
            numeric value. Units are the square of the mesh's coordinate unit.
        fitting_threshold: Eq. (6) distance-sum threshold. Required because
            the paper gives no universal numeric value. Units are mesh units.
        normal_angle_degrees: Face-normal angular tolerance used for region
            growing. Default 15° matches the example used in the paper's
            experiments; it is not claimed to be universally optimal.

    Returns:
        A result with region id per face (-1 means ungrouped), least-squares
        supporting planes, region adjacency, and intra-contour mesh edges.

    This phase does not weld vertices, alter topology, delete faces, or
    simplify the mesh. Those operations belong to later stages.
    """
    mesh_vertices, mesh_faces, diagonal = _validate_mesh_arrays(vertices, faces)
    area_threshold = float(small_region_area_threshold)
    fit_threshold = float(fitting_threshold)
    angle = float(normal_angle_degrees)
    if not np.isfinite(area_threshold) or area_threshold < 0.0:
        raise PlanarRegionExtractionError("small_region_area_threshold must be finite and non-negative.")
    if not np.isfinite(fit_threshold) or fit_threshold < 0.0:
        raise PlanarRegionExtractionError("fitting_threshold must be finite and non-negative.")
    if not np.isfinite(angle) or angle <= 0.0 or angle >= 90.0:
        raise PlanarRegionExtractionError("normal_angle_degrees must be between 0 and 90 degrees.")

    face_normals, _face_centers, face_areas = _face_geometry(mesh_vertices, mesh_faces)
    edge_started = time.perf_counter()
    edge_to_faces, face_neighbors = _build_edge_incidence(mesh_faces)
    edge_incidence_seconds = max(time.perf_counter() - edge_started, 0.0)
    coarse_started = time.perf_counter()
    _coarse_labels, coarse_groups = _grow_coarse_regions(
        face_normals,
        face_neighbors,
        angle,
        visualization_cb=visualization_cb,
    )
    coarse_region_seconds = max(time.perf_counter() - coarse_started, 0.0)
    refinement_started = time.perf_counter()
    refined_groups, ungrouped_region_ids = _refine_regions(
        mesh_vertices,
        mesh_faces,
        face_normals,
        face_areas,
        coarse_groups,
        edge_to_faces,
        area_threshold,
        fit_threshold,
        angle,
    )
    refinement_seconds = max(time.perf_counter() - refinement_started, 0.0)

    output_labels = np.full(len(mesh_faces), -1, dtype=np.int64)
    regions: list[PlanarRegion] = []
    plane_fit_started = time.perf_counter()
    ordered_refined_ids = sorted(refined_groups)
    region_visualization_stride = max(1, int(np.ceil(max(1, len(ordered_refined_ids)) / 24.0)))
    for new_id, old_id in enumerate(ordered_refined_ids):
        face_ids = np.sort(np.asarray(refined_groups[old_id], dtype=np.int64))
        output_labels[face_ids] = new_id
        vertex_ids = _unique_vertices_for_faces(mesh_faces, face_ids)
        plane = _fit_plane(mesh_vertices, vertex_ids, face_normals, face_ids)
        regions.append(PlanarRegion(
            region_id=new_id,
            face_indices=face_ids,
            vertex_indices=vertex_ids,
            area=_region_area(face_areas, face_ids),
            plane=plane,
        ))
        if visualization_cb is not None and (
            new_id % region_visualization_stride == 0 or new_id == len(ordered_refined_ids) - 1
        ):
            visualization_cb(
                output_labels.copy(), face_ids.copy(), new_id + 1, len(ordered_refined_ids),
                "REFINED REGION PLANE FIT",
            )

    plane_fit_seconds = max(time.perf_counter() - plane_fit_started, 0.0)
    contour_started = time.perf_counter()
    final_adjacency = _make_region_adjacency(edge_to_faces, output_labels)
    graph_edges = tuple(sorted(
        (first, second)
        for first, neighbors in final_adjacency.items()
        for second in neighbors
        if first < second
    ))

    region_by_id = {region.region_id: region for region in regions}
    contours: list[ContourFeature] = []
    boundary_edges = 0
    non_manifold_edges = 0
    ordered_edges = sorted(edge_to_faces.items())
    contour_visualization_stride = max(1, int(np.ceil(max(1, len(ordered_edges)) / 24.0)))
    for edge_index, (edge, incident_faces) in enumerate(ordered_edges):
        unique_faces = tuple(sorted(set(incident_faces)))
        if len(unique_faces) == 1:
            boundary_edges += 1
            continue
        if len(unique_faces) > 2:
            non_manifold_edges += 1

        adjacent_regions = tuple(sorted({
            int(output_labels[face_id])
            for face_id in unique_faces
            if output_labels[face_id] >= 0
        }))
        if len(adjacent_regions) < 2:
            continue

        original_edge_points = mesh_vertices[np.asarray(edge, dtype=np.int64)].copy()
        projected = False
        output_edge_points = original_edge_points
        line_point = None
        line_direction = None
        if len(unique_faces) == 2 and len(adjacent_regions) == 2:
            line = _plane_intersection_line(
                region_by_id[adjacent_regions[0]].plane,
                region_by_id[adjacent_regions[1]].plane,
                diagonal,
            )
            if line is not None:
                line_point, line_direction = line
                output_edge_points = _project_edge_to_line(
                    original_edge_points, line_point, line_direction
                )
                projected = True

        contours.append(ContourFeature(
            vertex_indices=edge,
            adjacent_face_indices=unique_faces,
            adjacent_region_ids=adjacent_regions,
            vertices=output_edge_points,
            projected=projected,
            line_point=line_point,
            line_direction=line_direction,
        ))
        if visualization_cb is not None and (
            edge_index % contour_visualization_stride == 0 or edge_index == len(ordered_edges) - 1
        ):
            visualization_cb(
                output_labels.copy(), np.asarray(unique_faces, dtype=np.int64),
                edge_index + 1, len(ordered_edges), "CONTOUR EDGE EXTRACTION",
            )

    contour_seconds = max(time.perf_counter() - contour_started, 0.0)
    stats = PlanarExtractionStats(
        input_vertices=int(len(mesh_vertices)),
        input_faces=int(len(mesh_faces)),
        boundary_edge_count=int(boundary_edges),
        non_manifold_edge_count=int(non_manifold_edges),
        coarse_region_count=int(len(coarse_groups)),
        refined_region_count=int(len(regions)),
        ungrouped_face_count=int(np.count_nonzero(output_labels < 0)),
        ungrouped_region_count=int(len(ungrouped_region_ids)),
        region_adjacency_count=int(len(graph_edges)),
        contour_edge_count=int(len(contours)),
        projected_contour_count=int(sum(contour.projected for contour in contours)),
        edge_incidence_seconds=float(edge_incidence_seconds),
        coarse_region_seconds=float(coarse_region_seconds),
        refinement_seconds=float(refinement_seconds),
        plane_fit_seconds=float(plane_fit_seconds),
        contour_seconds=float(contour_seconds),
    )
    return PlanarExtractionResult(
        face_region_ids=output_labels,
        regions=tuple(regions),
        region_adjacency=graph_edges,
        contours=tuple(contours),
        stats=stats,
    )
