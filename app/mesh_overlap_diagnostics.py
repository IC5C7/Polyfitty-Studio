from __future__ import annotations

"""Geometric diagnostics for close, nearly parallel, non-edge-adjacent faces.

This module does not repair or delete geometry. It reports projected triangle
intersection area and the separation of the two supporting planes over that
intersection. These measurements are useful Z-fighting/self-overlap candidates,
not a complete triangle-triangle self-intersection proof.
"""

from collections import defaultdict
from typing import Any
import math

import numpy as np
from scipy.spatial import cKDTree

def _signed_area_2d(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    return 0.5 * float(
        np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - poly[:, 1] * np.roll(poly[:, 0], -1))
    )

def _clip_convex_polygon(subject: np.ndarray, clipper: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Sutherland-Hodgman clipping of one convex 2D polygon by another."""
    output = [np.asarray(p, dtype=np.float64) for p in subject]
    clip = np.asarray(clipper, dtype=np.float64)
    if _signed_area_2d(clip) < 0.0:
        clip = clip[::-1]

    for index in range(len(clip)):
        a = clip[index]
        b = clip[(index + 1) % len(clip)]
        direction = b - a

        def side(point: np.ndarray) -> float:
            offset = point - a
            return float(direction[0] * offset[1] - direction[1] * offset[0])

        incoming = output
        output = []
        if not incoming:
            break
        previous = incoming[-1]
        previous_side = side(previous)
        for current in incoming:
            current_side = side(current)
            previous_inside = previous_side >= -eps
            current_inside = current_side >= -eps
            denominator = previous_side - current_side

            if current_inside:
                if not previous_inside and abs(denominator) > 1e-24:
                    output.append(previous + (current - previous) * (previous_side / denominator))
                output.append(current)
            elif previous_inside and abs(denominator) > 1e-24:
                output.append(previous + (current - previous) * (previous_side / denominator))
            previous = current
            previous_side = current_side

    if len(output) < 3:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)

def _barycentric_height(points: np.ndarray, triangle_xy: np.ndarray, triangle_height: np.ndarray) -> np.ndarray:
    """Interpolate a triangle's height along the common projection normal."""
    a, b, c = triangle_xy
    matrix = np.column_stack((b - a, c - a))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= 1e-18:
        return np.full(len(points), np.nan, dtype=np.float64)
    uv = np.linalg.solve(matrix, (points - a).T).T
    weights = np.column_stack((1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]))
    return weights @ triangle_height

def _face_edge_pairs(faces: np.ndarray) -> set[tuple[int, int]]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        a, b, c = map(int, face)
        edge_faces[tuple(sorted((a, b)))].append(face_index)
        edge_faces[tuple(sorted((b, c)))].append(face_index)
        edge_faces[tuple(sorted((c, a)))].append(face_index)
    adjacent: set[tuple[int, int]] = set()
    for incident in edge_faces.values():
        if len(incident) > 1:
            for i in range(len(incident)):
                for j in range(i + 1, len(incident)):
                    adjacent.add(tuple(sorted((incident[i], incident[j]))))
    return adjacent

def analyze_near_coplanar_face_pairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    center_distance: float,
    normal_angle_degrees: float = 2.0,
    min_overlap_ratio: float = 0.10,
    max_plane_separation: float | None = None,
    exclude_edge_adjacent: bool = True,
) -> dict[str, Any]:
    """Report close, nearly parallel face pairs with 2D overlap and 3D gap.

    Parameters are explicit because the paper does not define Z-fighting
    thresholds. ``overlap_ratio_min`` divides intersection area by the smaller
    of the two projected triangle areas. ``max_plane_separation`` is measured
    only over the projected intersection polygon. If omitted, no pair is
    labelled near-coplanar; all measured candidates are still returned.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("vertices must be a finite Nx3 array")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must be an Mx3 triangle index array")
    if len(f) and (int(f.min()) < 0 or int(f.max()) >= len(v)):
        raise ValueError("faces contain out-of-range vertex indices")
    if center_distance <= 0 or not math.isfinite(center_distance):
        raise ValueError("center_distance must be a finite positive distance")
    if not (0.0 <= normal_angle_degrees <= 90.0):
        raise ValueError("normal_angle_degrees must be within [0, 90]")
    if not (0.0 <= min_overlap_ratio <= 1.0):
        raise ValueError("min_overlap_ratio must be within [0, 1]")
    if max_plane_separation is not None and (max_plane_separation < 0 or not math.isfinite(max_plane_separation)):
        raise ValueError("max_plane_separation must be finite and non-negative")

    if len(f) == 0:
        return {
            "face_count": 0,
            "candidate_pair_count": 0,
            "projected_overlap_pair_count": 0,
            "near_coplanar_overlap_pair_count": 0,
            "parameters": {
                "center_distance": float(center_distance),
                "normal_angle_degrees": float(normal_angle_degrees),
                "min_overlap_ratio": float(min_overlap_ratio),
                "max_plane_separation": max_plane_separation,
            },
            "pairs": [],
        }

    triangles = v[f]
    centers = triangles.mean(axis=1)
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw_normals, axis=1)
    normals = np.divide(
        raw_normals,
        lengths[:, None],
        out=np.zeros_like(raw_normals),
        where=lengths[:, None] > 1e-20,
    )
    areas = 0.5 * lengths
    adjacent = _face_edge_pairs(f) if exclude_edge_adjacent else set()
    pairs = np.asarray(list(cKDTree(centers).query_pairs(r=float(center_distance))), dtype=np.int64)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)

    angle_cosine = math.cos(math.radians(float(normal_angle_degrees)))
    measured: list[dict[str, Any]] = []
    for face_i, face_j in pairs:
        i, j = int(face_i), int(face_j)
        if exclude_edge_adjacent and (i, j) in adjacent:
            continue
        dot = float(np.dot(normals[i], normals[j]))
        if abs(dot) < angle_cosine:
            continue
        if areas[i] <= 1e-20 or areas[j] <= 1e-20:
            continue

        n = normals[i].copy()
        if dot < 0.0:
            n = -n
        axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(n)))]
        u = np.cross(n, axis)
        u_length = float(np.linalg.norm(u))
        if u_length <= 1e-20:
            continue
        u /= u_length
        w = np.cross(n, u)
        origin = triangles[i, 0]

        tri_i_rel = triangles[i] - origin
        tri_j_rel = triangles[j] - origin
        tri_i_xy = np.column_stack((tri_i_rel @ u, tri_i_rel @ w))
        tri_j_xy = np.column_stack((tri_j_rel @ u, tri_j_rel @ w))
        tri_i_h = tri_i_rel @ n
        tri_j_h = tri_j_rel @ n

        polygon = _clip_convex_polygon(tri_i_xy, tri_j_xy)
        overlap_area = abs(_signed_area_2d(polygon)) if len(polygon) >= 3 else 0.0
        ratio_i = float(overlap_area / areas[i])
        ratio_j = float(overlap_area / areas[j])
        ratio_min = float(min(ratio_i, ratio_j))

        if len(polygon) >= 3 and overlap_area > 1e-18:
            height_i = _barycentric_height(polygon, tri_i_xy, tri_i_h)
            height_j = _barycentric_height(polygon, tri_j_xy, tri_j_h)
            gap = height_j - height_i
            finite_gap = gap[np.isfinite(gap)]
            if len(finite_gap):
                max_gap = float(np.max(np.abs(finite_gap)))
                mean_gap = float(abs(np.mean(finite_gap)))
                min_gap = float(np.min(finite_gap))
                max_signed_gap = float(np.max(finite_gap))
            else:
                max_gap = mean_gap = min_gap = max_signed_gap = None
        else:
            max_gap = mean_gap = min_gap = max_signed_gap = None

        overlap = bool(ratio_min >= min_overlap_ratio and overlap_area > 1e-18)
        near = bool(
            overlap
            and max_plane_separation is not None
            and max_gap is not None
            and max_gap <= max_plane_separation
        )
        measured.append({
            "face_i": i,
            "face_j": j,
            "centroid_distance": float(np.linalg.norm(centers[i] - centers[j])),
            "normal_angle_degrees": float(math.degrees(math.acos(min(1.0, abs(dot))))),
            "projected_overlap_area": float(overlap_area),
            "overlap_ratio_face_i": ratio_i,
            "overlap_ratio_face_j": ratio_j,
            "overlap_ratio_min": ratio_min,
            "max_plane_separation_over_overlap": max_gap,
            "mean_signed_plane_separation_over_overlap": mean_gap,
            "min_signed_plane_separation_over_overlap": min_gap,
            "max_signed_plane_separation_over_overlap": max_signed_gap,
            "projected_overlap": overlap,
            "near_coplanar_overlap_candidate": near,
        })

    measured.sort(
        key=lambda item: (
            not item["near_coplanar_overlap_candidate"],
            not item["projected_overlap"],
            -(item["overlap_ratio_min"]),
            item["max_plane_separation_over_overlap"] if item["max_plane_separation_over_overlap"] is not None else math.inf,
        )
    )
    return {
        "face_count": int(len(f)),
        "candidate_pair_count": int(len(measured)),
        "projected_overlap_pair_count": int(sum(x["projected_overlap"] for x in measured)),
        "near_coplanar_overlap_pair_count": int(sum(x["near_coplanar_overlap_candidate"] for x in measured)),
        "parameters": {
            "center_distance": float(center_distance),
            "normal_angle_degrees": float(normal_angle_degrees),
            "min_overlap_ratio": float(min_overlap_ratio),
            "max_plane_separation": None if max_plane_separation is None else float(max_plane_separation),
            "exclude_edge_adjacent": bool(exclude_edge_adjacent),
        },
        "pairs": measured,
    }
