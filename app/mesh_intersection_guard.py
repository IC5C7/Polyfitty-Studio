from __future__ import annotations

"""Optional local displacement guard for newly introduced close-surface crossings.

This is an application-side extension. It is not part of Li & Nan (2021), and
its candidate screen is not an exhaustive triangle-triangle intersection test.
"""

from typing import Any
import math

import numpy as np

if __package__:
    from .mesh_overlap_inspection import mean_unique_edge_length, measure_face_pair
    from .mesh_overlap_diagnostics import analyze_near_coplanar_face_pairs
else:
    from mesh_overlap_inspection import mean_unique_edge_length, measure_face_pair
    from mesh_overlap_diagnostics import analyze_near_coplanar_face_pairs

def _screen_pairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    center_distance_fraction: float,
    normal_angle_degrees: float,
    min_overlap_ratio: float,
) -> set[tuple[int, int]]:
    mean_edge = mean_unique_edge_length(vertices, faces)
    if mean_edge <= 0.0:
        return set()
    result = analyze_near_coplanar_face_pairs(
        vertices,
        faces,
        center_distance=center_distance_fraction * mean_edge,
        normal_angle_degrees=normal_angle_degrees,
        min_overlap_ratio=min_overlap_ratio,
        max_plane_separation=None,
        exclude_edge_adjacent=True,
    )
    return {
        tuple(sorted((int(pair["face_i"]), int(pair["face_j"]))))
        for pair in result["pairs"]
    }

def _reference_constraint(
    reference: dict[str, Any],
    min_gap_fraction: float,
    epsilon: float,
) -> dict[str, Any]:
    if not reference.get("valid", False):
        return {"mode": "skip", "reason": "invalid_reference_pair", "minimum_gap": None}
    if reference.get("surface_crossing_candidate", False):
        return {"mode": "skip", "reason": "preexisting_crossing", "minimum_gap": None}
    if not reference.get("projected_overlap", False):
        return {"mode": "no_new_crossing", "reason": "no_reference_overlap", "minimum_gap": None}

    minimum = reference.get("min_signed_plane_separation_over_overlap")
    maximum = reference.get("max_signed_plane_separation_over_overlap")
    if minimum is None or maximum is None:
        return {"mode": "skip", "reason": "missing_reference_gap", "minimum_gap": None}

    minimum = float(minimum)
    maximum = float(maximum)
    if minimum >= -epsilon and maximum > epsilon:
        positive_reference_gap = max(0.0, minimum)
        return {
            "mode": "preserve_positive_gap",
            "reason": "reference_positive_side_contact" if reference.get("touching_candidate") else "reference_surfaces_separated_positive",
            "minimum_gap": max(epsilon, positive_reference_gap * min_gap_fraction),
        }
    if maximum <= epsilon and minimum < -epsilon:
        negative_reference_gap = max(0.0, abs(maximum))
        return {
            "mode": "preserve_negative_gap",
            "reason": "reference_negative_side_contact" if reference.get("touching_candidate") else "reference_surfaces_separated_negative",
            "minimum_gap": max(epsilon, negative_reference_gap * min_gap_fraction),
        }
    if minimum > epsilon:
        return {
            "mode": "preserve_positive_gap",
            "reason": "reference_surfaces_separated_positive",
            "minimum_gap": max(epsilon, minimum * min_gap_fraction),
        }
    if maximum < -epsilon:
        return {
            "mode": "preserve_negative_gap",
            "reason": "reference_surfaces_separated_negative",
            "minimum_gap": max(epsilon, abs(maximum) * min_gap_fraction),
        }
    return {"mode": "skip", "reason": "reference_gap_ambiguous_near_zero", "minimum_gap": None}

def _satisfies_constraint(
    measurement: dict[str, Any],
    constraint: dict[str, Any],
    epsilon: float,
) -> bool:
    mode = constraint["mode"]
    if mode == "skip":
        return True
    if not measurement.get("valid", False):
        return False
    if not measurement.get("projected_overlap", False):
        return True
    if measurement.get("surface_crossing_candidate", False):
        return False
    minimum = measurement.get("min_signed_plane_separation_over_overlap")
    maximum = measurement.get("max_signed_plane_separation_over_overlap")
    if minimum is None or maximum is None:
        return False
    required = float(constraint.get("minimum_gap") or 0.0)
    if mode == "preserve_positive_gap":
        return float(minimum) >= required - 2.0 * epsilon
    if mode == "preserve_negative_gap":
        return float(maximum) <= -required + 2.0 * epsilon
    return True

def guard_filter_displacement(
    original_vertices: np.ndarray,
    proposed_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_gap_fraction: float = 0.5,
    center_distance_fraction: float = 0.10,
    candidate_normal_angle_degrees: float = 20.0,
    min_overlap_ratio: float = 0.10,
    line_search_iterations: int = 20,
    pair_sweeps: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Locally attenuate filter movement when tracked close face pairs become unsafe.

    The guard screens the union of close, nearly parallel projected-overlap
    candidates in the source and proposed states. For a source pair with a
    stable signed gap, the guard prevents a sign-changing crossing and retains
    at least ``min_gap_fraction`` of the reference minimum separation. For a
    pair without a source overlap, it prevents a newly introduced crossing.
    Pairs already crossing or touching in the source are reported but not
    repaired automatically.

    The operation changes vertex positions only. It never splits, welds,
    deletes, or retriangulates faces.
    """
    original = np.asarray(original_vertices, dtype=np.float64)
    proposed = np.asarray(proposed_vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if original.ndim != 2 or original.shape[1] != 3 or proposed.shape != original.shape:
        raise ValueError("original_vertices and proposed_vertices must be matching Nx3 arrays")
    if not np.isfinite(original).all() or not np.isfinite(proposed).all():
        raise ValueError("vertex arrays must contain only finite coordinates")
    if face_array.ndim != 2 or face_array.shape[1] != 3 or len(face_array) == 0:
        raise ValueError("faces must be a non-empty Mx3 triangle index array")
    if int(face_array.min()) < 0 or int(face_array.max()) >= len(original):
        raise ValueError("faces contain out-of-range vertex indices")
    if not (0.0 < float(min_gap_fraction) <= 1.0):
        raise ValueError("min_gap_fraction must be within (0, 1]")
    if not (0.0 < float(center_distance_fraction) <= 1.0):
        raise ValueError("center_distance_fraction must be within (0, 1]")
    if not (0.0 < float(candidate_normal_angle_degrees) < 90.0):
        raise ValueError("candidate_normal_angle_degrees must be within (0, 90)")
    if not (0.0 <= float(min_overlap_ratio) <= 1.0):
        raise ValueError("min_overlap_ratio must be within [0, 1]")
    if isinstance(line_search_iterations, bool) or int(line_search_iterations) != line_search_iterations or not 1 <= int(line_search_iterations) <= 60:
        raise ValueError("line_search_iterations must be an integer within [1, 60]")
    if isinstance(pair_sweeps, bool) or int(pair_sweeps) != pair_sweeps or not 1 <= int(pair_sweeps) <= 10:
        raise ValueError("pair_sweeps must be an integer within [1, 10]")

    displacement = proposed - original
    diagonal = float(np.linalg.norm(np.ptp(original, axis=0)))
    epsilon = max(1e-12, diagonal * 1e-10)
    source_pairs = _screen_pairs(
        original,
        face_array,
        center_distance_fraction=center_distance_fraction,
        normal_angle_degrees=candidate_normal_angle_degrees,
        min_overlap_ratio=min_overlap_ratio,
    )
    proposed_pairs = _screen_pairs(
        proposed,
        face_array,
        center_distance_fraction=center_distance_fraction,
        normal_angle_degrees=candidate_normal_angle_degrees,
        min_overlap_ratio=min_overlap_ratio,
    )
    candidate_pairs = sorted(source_pairs | proposed_pairs)

    if not candidate_pairs:
        return proposed.copy(), {
            "enabled": True,
            "candidate_pair_count": len(candidate_pairs),
            "guarded_pair_count": 0,
            "preexisting_crossing_pair_count": 0,
            "preexisting_touching_pair_count": 0,
            "unresolved_pair_count": 0,
            "attenuated_vertex_count": 0,
            "minimum_vertex_scale": 1.0,
            "mean_attenuated_vertex_scale": 1.0,
            "min_gap_fraction": float(min_gap_fraction),
            "candidate_normal_angle_degrees": float(candidate_normal_angle_degrees),
            "center_distance_fraction": float(center_distance_fraction),
            "minimum_overlap_ratio": float(min_overlap_ratio),
            "pairs": [],
        }

    pair_records: list[dict[str, Any]] = []
    constraints: dict[tuple[int, int], dict[str, Any]] = {}
    for face_i, face_j in candidate_pairs:
        reference = measure_face_pair(
            original, face_array, face_i, face_j,
            min_overlap_ratio=min_overlap_ratio,
        )
        constraint = _reference_constraint(reference, float(min_gap_fraction), epsilon)
        constraints[(face_i, face_j)] = constraint
        pair_records.append({
            "face_i": face_i,
            "face_j": face_j,
            "reference_classification": reference.get("classification"),
            "reference_touching_candidate": bool(reference.get("touching_candidate", False)),
            "constraint_mode": constraint["mode"],
            "constraint_reason": constraint["reason"],
            "required_minimum_gap": constraint.get("minimum_gap"),
            "line_search_scale": 1.0,
            "guarded": False,
            "unresolved": False,
        })

    records_by_pair = {(item["face_i"], item["face_j"]): item for item in pair_records}
    vertex_scales = np.ones(len(original), dtype=np.float64)
    guarded_pairs: set[tuple[int, int]] = set()

    def current_vertices() -> np.ndarray:
        return original + displacement * vertex_scales[:, None]

    ordered_pairs = []
    for face_i, face_j in candidate_pairs:
        constraint = constraints[(face_i, face_j)]
        if constraint["mode"] == "skip":
            continue
        proposed_measurement = measure_face_pair(
            proposed, face_array, face_i, face_j,
            min_overlap_ratio=min_overlap_ratio,
        )
        if not _satisfies_constraint(proposed_measurement, constraint, epsilon):
            ordered_pairs.append((face_i, face_j))
    ordered_pairs.sort(key=lambda pair: (
        not measure_face_pair(
            proposed, face_array, pair[0], pair[1],
            min_overlap_ratio=min_overlap_ratio,
        ).get("surface_crossing_candidate", False),
        pair,
    ))

    for _sweep in range(int(pair_sweeps)):
        changed_this_sweep = False
        for face_i, face_j in ordered_pairs:
            pair = (face_i, face_j)
            constraint = constraints[pair]
            base_current = current_vertices()
            current_measurement = measure_face_pair(
                base_current, face_array, face_i, face_j,
                min_overlap_ratio=min_overlap_ratio,
            )
            if _satisfies_constraint(current_measurement, constraint, epsilon):
                continue

            affected_vertices = np.unique(face_array[[face_i, face_j]].reshape(-1))
            old_scales = vertex_scales[affected_vertices].copy()

            def evaluate(local_factor: float) -> tuple[bool, dict[str, Any]]:
                trial = base_current.copy()
                trial[affected_vertices] = (
                    original[affected_vertices]
                    + displacement[affected_vertices] * (old_scales * local_factor)[:, None]
                )
                measured = measure_face_pair(
                    trial, face_array, face_i, face_j,
                    min_overlap_ratio=min_overlap_ratio,
                )
                return _satisfies_constraint(measured, constraint, epsilon), measured

            zero_safe, _zero_measurement = evaluate(0.0)
            if not zero_safe:
                records_by_pair[pair]["unresolved"] = True
                continue

            low = 0.0
            high = 1.0
            for _iteration in range(int(line_search_iterations)):
                middle = 0.5 * (low + high)
                is_safe, _measurement = evaluate(middle)
                if is_safe:
                    low = middle
                else:
                    high = middle
            chosen_scale = max(0.0, low * (1.0 - 1e-8))
            if chosen_scale < 1.0 - 1e-7:
                vertex_scales[affected_vertices] = old_scales * chosen_scale
                records_by_pair[pair]["line_search_scale"] = float(chosen_scale)
                records_by_pair[pair]["guarded"] = True
                guarded_pairs.add(pair)
                changed_this_sweep = True
        if not changed_this_sweep:
            break

    guarded = current_vertices()
    unresolved = 0
    for face_i, face_j in candidate_pairs:
        pair = (face_i, face_j)
        constraint = constraints[pair]
        if constraint["mode"] == "skip":
            continue
        measured = measure_face_pair(
            guarded, face_array, face_i, face_j,
            min_overlap_ratio=min_overlap_ratio,
        )
        is_safe = _satisfies_constraint(measured, constraint, epsilon)
        records_by_pair[pair]["final_classification"] = measured.get("classification")
        records_by_pair[pair]["final_min_signed_gap"] = measured.get(
            "min_signed_plane_separation_over_overlap"
        )
        records_by_pair[pair]["final_max_signed_gap"] = measured.get(
            "max_signed_plane_separation_over_overlap"
        )
        if not is_safe:
            records_by_pair[pair]["unresolved"] = True
            unresolved += 1

    attenuated = vertex_scales < 1.0 - 1e-7
    changed_scales = vertex_scales[attenuated]
    stats = {
        "enabled": True,
        "candidate_pair_count": int(len(candidate_pairs)),
        "guarded_pair_count": int(len(guarded_pairs)),
        "preexisting_crossing_pair_count": int(sum(
            record["constraint_reason"] == "preexisting_crossing" for record in pair_records
        )),
        "preexisting_touching_pair_count": int(sum(
            bool(record["reference_touching_candidate"]) for record in pair_records
        )),
        "unresolved_pair_count": int(unresolved),
        "attenuated_vertex_count": int(np.count_nonzero(attenuated)),
        "minimum_vertex_scale": float(np.min(changed_scales)) if len(changed_scales) else 1.0,
        "mean_attenuated_vertex_scale": float(np.mean(changed_scales)) if len(changed_scales) else 1.0,
        "min_gap_fraction": float(min_gap_fraction),
        "candidate_normal_angle_degrees": float(candidate_normal_angle_degrees),
        "center_distance_fraction": float(center_distance_fraction),
        "minimum_overlap_ratio": float(min_overlap_ratio),
        "line_search_iterations": int(line_search_iterations),
        "pair_sweeps": int(pair_sweeps),
        "pairs": pair_records,
    }
    return guarded, stats
