from __future__ import annotations

"""Batched local overlap veto for experimental safe edge collapse.

This is an application-side extension, not part of Li & Nan (2021). It screens
close, nearly parallel, projected-overlap face pairs using explicit thresholds.
It is not an exhaustive triangle-triangle intersection algorithm.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any
import math

import numpy as np
from scipy.spatial import cKDTree

if __package__:
    from .mesh_overlap_inspection import mean_unique_edge_length, measure_face_pair
else:
    from mesh_overlap_inspection import mean_unique_edge_length, measure_face_pair

@dataclass(frozen=True)
class CollapseGuardDecision:
    reject: bool
    reason: str | None
    face_pair: tuple[int, int] | None

class LocalCollapseIntersectionGuard:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        center_distance_fraction: float = 0.10,
        normal_angle_degrees: float = 20.0,
        min_overlap_ratio: float = 0.10,
        near_gap_fraction: float = 0.01,
        min_gap_fraction: float = 0.50,
        rebuild_interval: int = 32,
    ) -> None:
        self.faces_count = int(len(faces))
        self.center_distance_fraction = float(center_distance_fraction)
        self.normal_angle_degrees = float(normal_angle_degrees)
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.near_gap_fraction = float(near_gap_fraction)
        self.min_gap_fraction = float(min_gap_fraction)
        self.rebuild_interval = int(rebuild_interval)
        self.mean_edge_length = mean_unique_edge_length(vertices, faces)
        if not math.isfinite(self.mean_edge_length) or self.mean_edge_length <= 0.0:
            raise ValueError("collapse intersection guard requires a positive mean edge length")
        if not (0.0 < self.center_distance_fraction <= 1.0):
            raise ValueError("center_distance_fraction must be within (0, 1]")
        if not (0.0 < self.normal_angle_degrees < 90.0):
            raise ValueError("normal_angle_degrees must be within (0, 90)")
        if not (0.0 <= self.min_overlap_ratio <= 1.0):
            raise ValueError("min_overlap_ratio must be within [0, 1]")
        if not (0.0 < self.near_gap_fraction <= 1.0):
            raise ValueError("near_gap_fraction must be within (0, 1]")
        if not (0.0 < self.min_gap_fraction <= 1.0):
            raise ValueError("min_gap_fraction must be within (0, 1]")
        if self.rebuild_interval < 1:
            raise ValueError("rebuild_interval must be positive")

        self.center_distance = self.center_distance_fraction * self.mean_edge_length
        self.near_gap_limit = self.near_gap_fraction * self.mean_edge_length
        self.numeric_epsilon = max(
            1e-12,
            float(np.linalg.norm(np.ptp(np.asarray(vertices, dtype=np.float64), axis=0))) * 1e-10,
        )
        self.pairs_tested = 0
        self.rejected_new_crossing = 0
        self.rejected_new_near_coplanar = 0
        self.rejected_gap_reduction = 0
        self.preexisting_crossing_pairs: set[tuple[int, int]] = set()
        self.preexisting_near_pairs: set[tuple[int, int]] = set()
        self.sample_rejections: list[dict[str, Any]] = []
        self.index_rebuild_count = 0
        self.accepted_since_rebuild = 0
        self.max_stale_center_drift = 0.0
        self._snapshot_centers_by_face = np.zeros((self.faces_count, 3), dtype=np.float64)
        self._active_face_ids = np.empty(0, dtype=np.int64)
        self._tree: cKDTree | None = None
        self._moved_face_ids: set[int] = set()
        self.spatial_queries = 0
        self.last_spatial_screen_pair_count = 0
        self._pair_faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
        self._refresh(vertices, faces, np.ones(self.faces_count, dtype=bool), initial=True)

    def _refresh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        active_faces: np.ndarray,
        *,
        initial: bool = False,
    ) -> None:
        active_ids = np.flatnonzero(active_faces).astype(np.int64)
        active_triangles = np.asarray(faces, dtype=np.int64)[active_ids]
        centers = np.asarray(vertices, dtype=np.float64)[active_triangles].mean(axis=1)
        self._active_face_ids = active_ids
        self._snapshot_centers_by_face.fill(0.0)
        self._snapshot_centers_by_face[active_ids] = centers
        self._tree = cKDTree(centers) if len(centers) else None
        self._moved_face_ids.clear()
        self.max_stale_center_drift = 0.0
        self.accepted_since_rebuild = 0
        if not initial:
            self.index_rebuild_count += 1

    def _face_coordinates(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_id: int,
        *,
        first: int | None = None,
        second: int | None = None,
        new_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tri = np.asarray(faces[face_id], dtype=np.int64).copy()
        if first is not None and second is not None:
            tri[tri == second] = first
        points = np.asarray(vertices, dtype=np.float64)[tri].copy()
        if first is not None and new_position is not None:
            points[tri == first] = np.asarray(new_position, dtype=np.float64)
        return tri, points

    def _measure(self, tri_a: np.ndarray, tri_b: np.ndarray) -> dict[str, Any]:
        return measure_face_pair(
            np.vstack((tri_a, tri_b)),
            self._pair_faces,
            0,
            1,
            min_overlap_ratio=self.min_overlap_ratio,
            max_plane_separation=self.near_gap_limit,
        )

    def _transition_conflict(self, before: dict[str, Any], after: dict[str, Any]) -> str | None:
        if not after.get("valid", False):
            return None
        if float(after.get("centroid_distance", math.inf)) > self.center_distance:
            return None
        if float(after.get("normal_angle_degrees", math.inf)) > self.normal_angle_degrees:
            return None
        if not after.get("projected_overlap", False):
            return None

        before_crossing = bool(before.get("surface_crossing_candidate", False))
        after_crossing = bool(after.get("surface_crossing_candidate", False))
        before_near = bool(before.get("near_coplanar_overlap_candidate", False))
        after_near = bool(after.get("near_coplanar_overlap_candidate", False))
        if after_crossing and not before_crossing:
            return "new_crossing"
        if after_near and not before_near:
            return "new_near_coplanar"
        if before_crossing:
            self.preexisting_crossing_pairs.add((int(before["face_i"]), int(before["face_j"])))
            return None
        if before_near:
            self.preexisting_near_pairs.add((int(before["face_i"]), int(before["face_j"])))
        if not before.get("projected_overlap", False):
            return None

        before_min = before.get("min_signed_plane_separation_over_overlap")
        before_max = before.get("max_signed_plane_separation_over_overlap")
        after_min = after.get("min_signed_plane_separation_over_overlap")
        after_max = after.get("max_signed_plane_separation_over_overlap")
        if None in (before_min, before_max, after_min, after_max):
            return None
        before_min = float(before_min)
        before_max = float(before_max)
        after_min = float(after_min)
        after_max = float(after_max)
        eps = self.numeric_epsilon
        if before_min >= -eps and before_max > eps:
            required = max(eps, self.min_gap_fraction * max(0.0, before_min))
            if after_min < required - 2.0 * eps:
                return "reduced_positive_gap"
        elif before_max <= eps and before_min < -eps:
            required = max(eps, self.min_gap_fraction * max(0.0, abs(before_max)))
            if after_max > -required + 2.0 * eps:
                return "reduced_negative_gap"
        return None

    def check_candidate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        active_faces: np.ndarray,
        affected_face_ids: set[int],
        first: int,
        second: int,
        new_position: np.ndarray,
    ) -> CollapseGuardDecision:
        predicted: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        deleted: set[int] = set()
        for face_id in affected_face_ids:
            if not active_faces[face_id]:
                continue
            tri, points = self._face_coordinates(
                vertices, faces, face_id,
                first=first, second=second, new_position=new_position,
            )
            if len(set(int(x) for x in tri)) != 3:
                deleted.add(int(face_id))
                continue
            predicted[int(face_id)] = (tri, points)
        if not predicted:
            return CollapseGuardDecision(False, None, None)

        changed_ids = sorted(predicted)
        changed_set = set(changed_ids)
        candidate_pairs: set[tuple[int, int]] = set()
        moved_ids = np.fromiter(
            (face_id for face_id in self._moved_face_ids
             if active_faces[face_id] and face_id not in changed_set and face_id not in deleted),
            dtype=np.int64,
        )
        moved_centers = (
            np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)[moved_ids]].mean(axis=1)
            if len(moved_ids) else np.empty((0, 3), dtype=np.float64)
        )
        for face_id in changed_ids:
            predicted_center = predicted[face_id][1].mean(axis=0)
            if self._tree is not None:
                self.spatial_queries += 1
                static_neighbors = self._tree.query_ball_point(
                    predicted_center, r=self.center_distance + self.numeric_epsilon
                )
                for local_partner in static_neighbors:
                    partner = int(self._active_face_ids[int(local_partner)])
                    if partner == face_id or partner in changed_set or partner in deleted or partner in self._moved_face_ids:
                        continue
                    candidate_pairs.add((min(face_id, partner), max(face_id, partner)))
            if len(moved_ids):
                distances = np.linalg.norm(moved_centers - predicted_center, axis=1)
                for partner in moved_ids[distances <= self.center_distance + self.numeric_epsilon]:
                    candidate_pairs.add((min(face_id, int(partner)), max(face_id, int(partner))))
        for face_a, face_b in combinations(changed_ids, 2):
            candidate_pairs.add((min(face_a, face_b), max(face_a, face_b)))
        self.last_spatial_screen_pair_count = len(candidate_pairs)

        for face_a, face_b in sorted(candidate_pairs):
            if not active_faces[face_a] or not active_faces[face_b] or face_a in deleted or face_b in deleted:
                continue
            if face_a in predicted:
                tri_a_after, points_a_after = predicted[face_a]
            else:
                tri_a_after, points_a_after = self._face_coordinates(vertices, faces, face_a)
            if face_b in predicted:
                tri_b_after, points_b_after = predicted[face_b]
            else:
                tri_b_after, points_b_after = self._face_coordinates(vertices, faces, face_b)
            if len(set(int(x) for x in tri_a_after) & set(int(x) for x in tri_b_after)) >= 2:
                continue
            if float(np.linalg.norm(points_a_after.mean(axis=0) - points_b_after.mean(axis=0))) > self.center_distance:
                continue
            after = self._measure(points_a_after, points_b_after)
            if not after.get("valid", False):
                continue
            if float(after.get("normal_angle_degrees", math.inf)) > self.normal_angle_degrees:
                continue
            if not after.get("projected_overlap", False):
                continue
            _tri_a_before, points_a_before = self._face_coordinates(vertices, faces, face_a)
            _tri_b_before, points_b_before = self._face_coordinates(vertices, faces, face_b)
            before = self._measure(points_a_before, points_b_before)
            before["face_i"] = face_a
            before["face_j"] = face_b
            after["face_i"] = face_a
            after["face_j"] = face_b
            self.pairs_tested += 1
            reason = self._transition_conflict(before, after)
            if reason is not None:
                if reason == "new_crossing":
                    self.rejected_new_crossing += 1
                elif reason == "new_near_coplanar":
                    self.rejected_new_near_coplanar += 1
                else:
                    self.rejected_gap_reduction += 1
                if len(self.sample_rejections) < 100:
                    self.sample_rejections.append({
                        "face_i": face_a,
                        "face_j": face_b,
                        "reason": reason,
                        "before_classification": before.get("classification"),
                        "after_classification": after.get("classification"),
                        "before_min_signed_gap": before.get("min_signed_plane_separation_over_overlap"),
                        "before_max_signed_gap": before.get("max_signed_plane_separation_over_overlap"),
                        "after_min_signed_gap": after.get("min_signed_plane_separation_over_overlap"),
                        "after_max_signed_gap": after.get("max_signed_plane_separation_over_overlap"),
                        "before_overlap_ratio": before.get("overlap_ratio_min"),
                        "after_overlap_ratio": after.get("overlap_ratio_min"),
                    })
                return CollapseGuardDecision(True, reason, (face_a, face_b))
        return CollapseGuardDecision(False, None, None)

    def record_accepted_collapse(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        active_faces: np.ndarray,
        changed_face_ids: set[int],
    ) -> None:
        self.accepted_since_rebuild += 1
        for face_id in changed_face_ids:
            if not active_faces[face_id]:
                self._moved_face_ids.discard(int(face_id))
                continue
            self._moved_face_ids.add(int(face_id))
            current_center = np.asarray(vertices, dtype=np.float64)[faces[face_id]].mean(axis=0)
            drift = float(np.linalg.norm(current_center - self._snapshot_center_for_face(face_id)))
            if math.isfinite(drift):
                self.max_stale_center_drift = max(self.max_stale_center_drift, drift)
        if self.accepted_since_rebuild >= self.rebuild_interval:
            self._refresh(vertices, faces, active_faces)

    def _snapshot_center_for_face(self, face_id: int) -> np.ndarray:
        return self._snapshot_centers_by_face[int(face_id)]

    def report(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "center_distance_fraction_of_input_mean_edge": self.center_distance_fraction,
            "center_distance": self.center_distance,
            "normal_angle_degrees": self.normal_angle_degrees,
            "minimum_overlap_ratio": self.min_overlap_ratio,
            "near_gap_fraction_of_input_mean_edge": self.near_gap_fraction,
            "near_gap_limit": self.near_gap_limit,
            "minimum_preserved_gap_fraction": self.min_gap_fraction,
            "spatial_search_radius_fraction_of_input_mean_edge": self.center_distance_fraction,
            "spatial_index_rebuild_interval_successful_collapses": self.rebuild_interval,
            "spatial_queries_for_changed_faces": self.spatial_queries,
            "moved_faces_since_last_index_rebuild": len(self._moved_face_ids),
            "max_moved_face_center_drift_since_index_rebuild": self.max_stale_center_drift,
            "index_rebuild_count": self.index_rebuild_count,
            "last_candidate_pair_count": self.last_spatial_screen_pair_count,
            "candidate_face_pairs_measured": self.pairs_tested,
            "rejected_new_crossing": self.rejected_new_crossing,
            "rejected_new_near_coplanar": self.rejected_new_near_coplanar,
            "rejected_gap_reduction": self.rejected_gap_reduction,
            "rejected_total": self.rejected_new_crossing + self.rejected_new_near_coplanar + self.rejected_gap_reduction,
            "preexisting_crossing_pairs_encountered": len(self.preexisting_crossing_pairs),
            "preexisting_near_pairs_encountered": len(self.preexisting_near_pairs),
            "sample_rejections": list(self.sample_rejections),
            "screen_is_exhaustive": False,
        }
