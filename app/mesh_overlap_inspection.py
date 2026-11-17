from __future__ import annotations

"""Stage-wise inspection of near-parallel projected triangle overlaps.

This is an application diagnostic, not a Z-fighting repair algorithm and not
part of Li & Nan (2021). Thresholds are explicit screening choices.
"""

from dataclasses import dataclass
from typing import Any, Iterable
import math

import numpy as np
from scipy.spatial import cKDTree

if __package__:
    from .mesh_overlap_diagnostics import (
        _barycentric_height,
        _clip_convex_polygon,
        _face_edge_pairs,
        _signed_area_2d,
        analyze_near_coplanar_face_pairs,
    )
else:
    from mesh_overlap_diagnostics import (
        _barycentric_height,
        _clip_convex_polygon,
        _face_edge_pairs,
        _signed_area_2d,
        analyze_near_coplanar_face_pairs,
    )

def mean_unique_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must be an Mx3 triangle index array")
    if len(f) == 0:
        return 0.0
    edges = np.sort(np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1)
    unique_edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(v[unique_edges[:, 0]] - v[unique_edges[:, 1]], axis=1)
    positive = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    return float(np.mean(positive)) if len(positive) else 0.0

def measure_face_pair(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_i: int,
    face_j: int,
    *,
    min_overlap_ratio: float = 0.10,
    max_plane_separation: float | None = None,
) -> dict[str, Any]:
    """Measure one specified face pair, independent of center-distance screening."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    i, j = int(face_i), int(face_j)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("vertices must be a finite Nx3 array")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must be an Mx3 triangle index array")
    if i == j or min(i, j) < 0 or max(i, j) >= len(f):
        raise ValueError("face pair indices must identify two distinct faces")
    if len(f) and (int(f.min()) < 0 or int(f.max()) >= len(v)):
        raise ValueError("faces contain out-of-range vertex indices")
    if not (0.0 <= min_overlap_ratio <= 1.0):
        raise ValueError("min_overlap_ratio must be within [0, 1]")
    if max_plane_separation is not None and (max_plane_separation < 0 or not math.isfinite(max_plane_separation)):
        raise ValueError("max_plane_separation must be finite and non-negative")

    tri_i = v[f[i]]
    tri_j = v[f[j]]
    cross_i = np.cross(tri_i[1] - tri_i[0], tri_i[2] - tri_i[0])
    cross_j = np.cross(tri_j[1] - tri_j[0], tri_j[2] - tri_j[0])
    len_i = float(np.linalg.norm(cross_i))
    len_j = float(np.linalg.norm(cross_j))
    if len_i <= 1e-20 or len_j <= 1e-20:
        return {
            "face_i": i, "face_j": j, "valid": False,
            "reason": "degenerate_triangle", "projected_overlap": False,
            "near_coplanar_overlap_candidate": False,
        }

    ni = cross_i / len_i
    nj = cross_j / len_j
    dot = float(np.dot(ni, nj))
    angle = math.degrees(math.acos(min(1.0, abs(dot))))
    n = ni.copy() if dot >= 0.0 else -ni.copy()
    axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, axis)
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    origin = tri_i[0]

    rel_i = tri_i - origin
    rel_j = tri_j - origin
    xy_i = np.column_stack((rel_i @ u, rel_i @ w))
    xy_j = np.column_stack((rel_j @ u, rel_j @ w))
    h_i = rel_i @ n
    h_j = rel_j @ n
    poly = _clip_convex_polygon(xy_i, xy_j)
    overlap_area = abs(_signed_area_2d(poly)) if len(poly) >= 3 else 0.0
    area_i_3d = 0.5 * len_i
    area_j_3d = 0.5 * len_j
    ratio_i = float(overlap_area / area_i_3d)
    ratio_j = float(overlap_area / area_j_3d)
    ratio_min = min(ratio_i, ratio_j)

    poly_i_3d: list[list[float]] = []
    poly_j_3d: list[list[float]] = []
    poly_mid_3d: list[list[float]] = []
    if len(poly) >= 3 and overlap_area > 1e-18:
        heights_i = _barycentric_height(poly, xy_i, h_i)
        heights_j = _barycentric_height(poly, xy_j, h_j)
        gaps = heights_j - heights_i
        valid = np.isfinite(gaps)
        if np.any(valid):
            abs_gaps = np.abs(gaps[valid])
            max_gap = float(np.max(abs_gaps))
            mean_abs_gap = float(np.mean(abs_gaps))
            min_signed_gap = float(np.min(gaps[valid]))
            max_signed_gap = float(np.max(gaps[valid]))
            for point, hi, hj in zip(poly, heights_i, heights_j):
                pi = origin + u * point[0] + w * point[1] + n * hi
                pj = origin + u * point[0] + w * point[1] + n * hj
                poly_i_3d.append([float(x) for x in pi])
                poly_j_3d.append([float(x) for x in pj])
                poly_mid_3d.append([float(x) for x in 0.5 * (pi + pj)])
        else:
            max_gap = mean_abs_gap = min_signed_gap = max_signed_gap = None
    else:
        max_gap = mean_abs_gap = min_signed_gap = max_signed_gap = None

    overlap = bool(overlap_area > 1e-18 and ratio_min >= min_overlap_ratio)
    near = bool(
        overlap and max_plane_separation is not None and max_gap is not None
        and max_gap <= max_plane_separation
    )
    sign_epsilon = max(1e-12, 1e-10 * float(np.max(np.ptp(v, axis=0))))
    crosses = bool(
        overlap and min_signed_gap is not None and max_signed_gap is not None
        and min_signed_gap < -sign_epsilon and max_signed_gap > sign_epsilon
    )
    touches = bool(
        overlap and not crosses and min_signed_gap is not None and max_signed_gap is not None
        and min_signed_gap <= sign_epsilon and max_signed_gap >= -sign_epsilon
    )
    if crosses:
        classification = "crossing_surface_candidate"
    elif near:
        classification = "near_coplanar_overlap"
    elif touches:
        classification = "touching_or_near_contact_candidate"
    elif overlap:
        classification = "separated_projected_overlap"
    else:
        classification = "close_parallel_no_significant_overlap"

    return {
        "face_i": i,
        "face_j": j,
        "valid": True,
        "normal_angle_degrees": float(angle),
        "centroid_distance": float(np.linalg.norm(tri_i.mean(axis=0) - tri_j.mean(axis=0))),
        "projected_overlap_area": float(overlap_area),
        "overlap_ratio_face_i": ratio_i,
        "overlap_ratio_face_j": ratio_j,
        "overlap_ratio_min": float(ratio_min),
        "max_plane_separation_over_overlap": max_gap,
        "mean_abs_plane_separation_over_overlap": mean_abs_gap,
        "min_signed_plane_separation_over_overlap": min_signed_gap,
        "max_signed_plane_separation_over_overlap": max_signed_gap,
        "projected_overlap": overlap,
        "near_coplanar_overlap_candidate": near,
        "surface_crossing_candidate": crosses,
        "touching_candidate": touches,
        "classification": classification,
        "overlap_polygon_face_i_3d": poly_i_3d,
        "overlap_polygon_face_j_3d": poly_j_3d,
        "overlap_polygon_mid_3d": poly_mid_3d,
    }

def analyze_mesh_stage(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    label: str,
    center_distance_fraction: float = 0.10,
    normal_angle_degrees: float = 2.0,
    min_overlap_ratio: float = 0.10,
    near_gap_fraction: float = 0.01,
) -> dict[str, Any]:
    """Screen one mesh using thresholds normalized to that mesh's own edge scale."""
    mean_edge = mean_unique_edge_length(vertices, faces)
    if mean_edge <= 0.0:
        raise ValueError("mesh has no positive unique-edge scale")
    center_distance = center_distance_fraction * mean_edge
    max_gap = near_gap_fraction * mean_edge
    screen = analyze_near_coplanar_face_pairs(
        vertices, faces,
        center_distance=center_distance,
        normal_angle_degrees=normal_angle_degrees,
        min_overlap_ratio=min_overlap_ratio,
        max_plane_separation=max_gap,
        exclude_edge_adjacent=True,
    )
    detailed = [
        measure_face_pair(
            vertices, faces, p["face_i"], p["face_j"],
            min_overlap_ratio=min_overlap_ratio,
            max_plane_separation=max_gap,
        )
        for p in screen["pairs"]
    ]
    detailed.sort(key=lambda p: (
        not p.get("near_coplanar_overlap_candidate", False),
        not p.get("projected_overlap", False),
        -(p.get("overlap_ratio_min") or 0.0),
        p.get("max_plane_separation_over_overlap") if p.get("max_plane_separation_over_overlap") is not None else math.inf,
    ))
    return {
        "label": label,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "mean_unique_edge_length": mean_edge,
        "screen_parameters": {
            "center_distance_fraction_of_local_mean_edge": float(center_distance_fraction),
            "center_distance": float(center_distance),
            "normal_angle_degrees": float(normal_angle_degrees),
            "minimum_overlap_ratio": float(min_overlap_ratio),
            "near_gap_fraction_of_local_mean_edge": float(near_gap_fraction),
            "near_gap_limit": float(max_gap),
            "exclude_edge_adjacent": True,
        },
        "candidate_pair_count": len(detailed),
        "projected_overlap_pair_count": sum(bool(p.get("projected_overlap")) for p in detailed),
        "near_coplanar_overlap_pair_count": sum(bool(p.get("near_coplanar_overlap_candidate")) for p in detailed),
        "surface_crossing_pair_count": sum(bool(p.get("surface_crossing_candidate")) for p in detailed),
        "touching_pair_count": sum(bool(p.get("touching_candidate")) for p in detailed),
        "pairs": detailed,
    }

def compare_tracked_face_pairs(
    stages: dict[str, tuple[np.ndarray, np.ndarray]],
    original_face_pairs: Iterable[tuple[int, int]],
    *,
    min_overlap_ratio: float = 0.10,
    center_distance_fraction: float = 0.10,
    normal_angle_degrees: float = 2.0,
    near_gap_fraction: float = 0.01,
) -> list[dict[str, Any]]:
    """Measure the same source-face IDs at each topology-preserving filter stage.

    The mapping is valid only where the face ordering and face count correspond
    to the source. The decimated result should not be passed here unless its
    faces have been mapped back with the decimator's original_face_indices.
    """
    unique = sorted({tuple(sorted((int(a), int(b)))) for a, b in original_face_pairs})
    rows: list[dict[str, Any]] = []

    stage_scales = {
        name: mean_unique_edge_length(vertices, faces)
        for name, (vertices, faces) in stages.items()
    }
    for pair in unique:
        row: dict[str, Any] = {"source_face_i": pair[0], "source_face_j": pair[1], "stages": {}}
        for stage_name, (vertices, faces) in stages.items():
            if max(pair) >= len(faces):
                row["stages"][stage_name] = {"valid": False, "reason": "face_id_not_present"}
                continue
            local_scale = stage_scales[stage_name]
            gap_limit = near_gap_fraction * local_scale
            measurement = measure_face_pair(
                vertices, faces, pair[0], pair[1],
                min_overlap_ratio=min_overlap_ratio,
                max_plane_separation=gap_limit,
            )
            measurement["screened_candidate"] = bool(
                measurement.get("centroid_distance", math.inf) <= center_distance_fraction * local_scale
                and measurement.get("normal_angle_degrees", math.inf) <= normal_angle_degrees
            )
            measurement["max_gap_fraction_of_mean_edge"] = (
                measurement["max_plane_separation_over_overlap"] / local_scale
                if local_scale > 0.0 and measurement.get("max_plane_separation_over_overlap") is not None
                else None
            )
            row["stages"][stage_name] = {
                k: v for k, v in measurement.items()
                if not k.startswith("overlap_polygon_")
            }
        rows.append(row)
    return rows
