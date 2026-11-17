from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional
import time
import traceback
import sys

import numpy as np

try:
    from .coordinate_frame import as_transform_matrix, points_to_source_local
    from .live_preview import planar_group_preview_mesh, merge_preview_meshes
except ImportError:
    from coordinate_frame import as_transform_matrix, points_to_source_local
    from live_preview import planar_group_preview_mesh, merge_preview_meshes

try:
    from .polyfit_runtime import load_polyfit_binding, PolyFitBindingError
except ImportError:
    from polyfit_runtime import load_polyfit_binding, PolyFitBindingError

try:
    from .pointio import load_points, reduce_points, write_vg_ascii, convert_up_axis, merge_mesh_vertices_by_distance, weld_coincident_mesh_vertices
    from .planarize import planarize_mesh_geometry, build_lowpoly_plane_patches
    from .resolution import get_output_face_budget
    from .postprocess import process_obj_e_stage
    from .mesh_io import load_obj_trimesh
except ImportError:
    from pointio import load_points, reduce_points, write_vg_ascii, convert_up_axis, merge_mesh_vertices_by_distance, weld_coincident_mesh_vertices
    from planarize import planarize_mesh_geometry, build_lowpoly_plane_patches
    from resolution import get_output_face_budget
    from postprocess import process_obj_e_stage
    from mesh_io import load_obj_trimesh

class PipelineFailure(RuntimeError):
    pass

def _emit(progress_cb: Callable[[int, str], None], value: int, message: str) -> None:
    progress_cb(max(0, min(100, int(value))), message)

def extract_planes_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.0,
    min_plane_points: int = 250,
    max_planes: int = 24,
    iterations: int = 350,
    progress: Optional[Callable[[int, str], None]] = None,
    seed: int = 17,
    preserve_small_features: bool = False,
    small_feature_min_points: int = 40,
    stage_preview_cb: Optional[Callable[..., None]] = None,
) -> tuple[list[dict], np.ndarray, dict]:
    """Extract dominant planes using deterministic, NumPy-based RANSAC.

    RANSAC hypotheses are scored on a bounded sample for speed, then verified/refit
    against the full remaining cloud. Groups are disjoint and reference original
    point indices, as required by PolyFit's ASCII .vg format.
    """
    n = len(points)
    if n < 3:
        raise PipelineFailure("Fewer than three input points were provided.")

    extent = np.ptp(points, axis=0)
    diagonal = float(np.linalg.norm(extent))
    if diagonal <= 1e-12:
        raise PipelineFailure("The point cloud has near-zero extent. Check the input coordinates.")
    threshold = float(distance_threshold) if distance_threshold > 0 else max(diagonal * 0.0025, 1e-7)
    min_count = max(30, int(min_plane_points))
    small_min_count = max(12, min(int(small_feature_min_points), min_count)) if preserve_small_features else min_count
    rng = np.random.default_rng(seed)
    remaining = np.arange(n, dtype=np.int64)
    configured_max_planes = max(1, int(max_planes))
    groups: list[dict] = []
    normals = np.zeros((n, 3), dtype=np.float32)
    live_plane_meshes: list[tuple[np.ndarray, np.ndarray]] = []
    color_palette = np.array([
        [0.15, 0.72, 0.96], [0.95, 0.48, 0.24], [0.49, 0.82, 0.43],
        [0.76, 0.52, 0.96], [0.98, 0.78, 0.28], [0.28, 0.84, 0.77],
        [0.93, 0.42, 0.58], [0.58, 0.70, 0.95], [0.80, 0.76, 0.55],
    ], dtype=np.float32)

    max_score_points = min(n, 65000)
    for plane_index in range(configured_max_planes):
        loop_minimum = small_min_count if preserve_small_features else min_count
        if len(remaining) < loop_minimum:
            break
        rem_points = points[remaining]
        if len(rem_points) > max_score_points:
            local_sample = rng.choice(len(rem_points), size=max_score_points, replace=False)
            score_points = rem_points[local_sample]
        else:
            local_sample = None
            score_points = rem_points
        if len(score_points) < loop_minimum:
            break

        best_count = 0
        best_plane = None
        best_ratio = 0.0

        cell_size = max(diagonal * 0.03, threshold * 8.0)
        cell_keys = np.floor((rem_points - rem_points.min(axis=0)) / cell_size).astype(np.int64)
        _, cell_inverse, cell_counts = np.unique(
            cell_keys, axis=0, return_inverse=True, return_counts=True
        )
        cell_order = np.argsort(cell_inverse)
        cell_starts = np.concatenate(([0], np.cumsum(cell_counts)[:-1])).astype(np.int64)
        valid_cells = np.flatnonzero(cell_counts >= 3)
        valid_prob = cell_counts[valid_cells].astype(np.float64)
        if len(valid_prob):
            valid_prob /= valid_prob.sum()

        support_floor = small_min_count if preserve_small_features else min_count
        minimum_sample_inliers = max(10, int(support_floor * len(score_points) / max(1, len(rem_points)) * 0.35))
        for iteration in range(iterations):
            if len(valid_cells) and rng.random() < 0.85:
                cell_id = int(rng.choice(valid_cells, p=valid_prob))
                begin = int(cell_starts[cell_id])
                finish = begin + int(cell_counts[cell_id])
                members = cell_order[begin:finish]
                tri_ids = rng.choice(members, size=3, replace=False)
                p0, p1, p2 = rem_points[tri_ids]
            else:
                tri_ids = rng.integers(0, len(score_points), size=3)
                if len(set(int(x) for x in tri_ids)) < 3:
                    continue
                p0, p1, p2 = score_points[tri_ids]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-10:
                continue
            normal /= norm
            d = -float(np.dot(normal, p0))
            inliers = np.abs(score_points @ normal + d) <= threshold
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
                best_ratio = count / len(score_points)
            if progress and iteration % 25 == 0:

                frac = (plane_index + iteration / max(1, iterations)) / max(1, max_planes)
                progress(12 + int(28 * min(1.0, frac)), f"Searching planar candidates {plane_index + 1}/{max_planes} · RANSAC {iteration}/{iterations}")
            if best_ratio > 0.80 and iteration > 80:
                break

        ratio_floor = 0.0001 if preserve_small_features else 0.0005
        minimum_ratio = max(ratio_floor, support_floor / max(1, len(rem_points)) * 0.25)
        if best_plane is None or best_count < minimum_sample_inliers or best_ratio < minimum_ratio:
            break

        full_dist = np.abs(rem_points @ best_plane[:3] + best_plane[3])
        inlier_local = np.flatnonzero(full_dist <= threshold)
        required_full_support = small_min_count if preserve_small_features else min_count
        if len(inlier_local) < required_full_support:
            break

        fit_points = rem_points[inlier_local]
        if len(fit_points) > 30000:
            fit_points = fit_points[rng.choice(len(fit_points), 30000, replace=False)]
        centroid = fit_points.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(fit_points - centroid, full_matrices=False)
            refined_normal = vh[-1]
        except np.linalg.LinAlgError:
            refined_normal = best_plane[:3]
        refined_normal = refined_normal / max(float(np.linalg.norm(refined_normal)), 1e-12)

        dominant = int(np.argmax(np.abs(refined_normal)))
        if refined_normal[dominant] < 0:
            refined_normal = -refined_normal
        refined_d = -float(np.dot(refined_normal, centroid))
        refined_dist = np.abs(rem_points @ refined_normal + refined_d)
        refined_local = np.flatnonzero(refined_dist <= threshold)
        if len(refined_local) >= required_full_support:
            inlier_local = refined_local
        global_indices = remaining[inlier_local]
        plane = [float(refined_normal[0]), float(refined_normal[1]), float(refined_normal[2]), float(refined_d)]
        color = color_palette[plane_index % len(color_palette)].copy()
        group = {"plane": plane, "indices": global_indices, "color": color}
        groups.append(group)
        normals[global_indices] = refined_normal.astype(np.float32)

        if stage_preview_cb is not None:
            patch_vertices, patch_faces = planar_group_preview_mesh(points, global_indices, refined_normal)
            if len(patch_faces):
                live_plane_meshes.append((patch_vertices, patch_faces))
                live_vertices, live_faces = merge_preview_meshes(live_plane_meshes)
                stage_preview_cb(
                    live_vertices, live_faces,
                    f"B · PLANAR SEGMENTS · {len(groups)} patches / {len(live_faces):,} faces",
                )

        keep = np.ones(len(remaining), dtype=bool)
        keep[inlier_local] = False
        remaining = remaining[keep]
        if progress:
            progress(12 + int(28 * min(1.0, (plane_index + 1) / max(1, max_planes))),
                     f"Extracted plane {plane_index + 1} · {len(global_indices):,} points")

    if not groups:
        raise PipelineFailure(
            f"No planes were found. Increase the RANSAC tolerance or lower the minimum points per plane. (Automatic tolerance: {threshold:.6g})"
        )

    unassigned = np.flatnonzero(np.linalg.norm(normals, axis=1) == 0)
    if len(unassigned):
        unassigned_points = points[unassigned]
        planes = np.asarray([g["plane"] for g in groups], dtype=np.float64)
        best_dist = np.full(len(unassigned), np.inf, dtype=np.float64)
        best_idx = np.zeros(len(unassigned), dtype=np.int32)
        for pi, plane in enumerate(planes):
            dist = np.abs(unassigned_points @ plane[:3] + plane[3])
            mask = dist < best_dist
            best_dist[mask] = dist[mask]
            best_idx[mask] = pi
        normals[unassigned] = planes[best_idx, :3].astype(np.float32)

    coverage = sum(len(g["indices"]) for g in groups) / max(1, n)
    stats = {
        "point_count": n,
        "plane_count": len(groups),
        "assigned_points": int(sum(len(g["indices"]) for g in groups)),
        "coverage": float(coverage),
        "distance_threshold": threshold,
        "remaining_points": int(len(remaining)),
        "max_planes": int(configured_max_planes),
        "plane_limit_reached": bool(
            len(groups) >= configured_max_planes and len(remaining) >= small_min_count
        ),
    }
    return groups, normals, stats

def _normal_cluster_count(groups: list[dict], angle_degrees: float = 20.0) -> int:
    """Count distinct unoriented plane-normal directions (parallel = same cluster)."""
    if not groups:
        return 0
    cosine = float(np.cos(np.deg2rad(angle_degrees)))
    representatives: list[np.ndarray] = []
    for group in sorted(groups, key=lambda g: len(g.get("indices", [])), reverse=True):
        normal = np.asarray(group["plane"][:3], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal /= norm
        if not any(abs(float(np.dot(normal, existing))) >= cosine for existing in representatives):
            representatives.append(normal)
    return len(representatives)

def choose_reconstruction_engine(
    groups: list[dict], stats: dict, mesh_input_stats: Optional[dict] = None,
    max_auto_polyfit_planes: int = 32,
) -> tuple[str, str, int]:
    """Choose PolyFit as the AUTO primary engine for every supported input.

    Mesh topology is retained as diagnostic context for the supervisor, which
    may retry Li & Nan 2021 only after the isolated PolyFit attempt fails.
    """
    coverage = float(stats.get("coverage", 0.0))
    plane_count = int(stats.get("plane_count", len(groups)))
    normal_clusters = _normal_cluster_count(groups)
    if mesh_input_stats is not None:
        reason = (
            "triangulated mesh · PolyFit primary with Li & Nan 2021 fallback "
            f"available: {int(mesh_input_stats.get('faces', 0)):,} source triangles, "
            f"{int(mesh_input_stats.get('components', 0)):,} connected components"
        )
        return "polyfit", reason, normal_clusters
    if plane_count > max_auto_polyfit_planes:
        reason = (
            f"AUTO complexity note: {plane_count} planar candidates exceed the preferred UI hint "
            f"({max_auto_polyfit_planes}); continuing with PolyFit native optimization"
        )
        return "polyfit", reason, normal_clusters
    if plane_count >= 3 and coverage >= 0.42 and normal_clusters <= 10:
        reason = (
            f"planar structure detected: {plane_count} planes, "
            f"{coverage * 100:.1f}% coverage, {normal_clusters} normal directions"
        )
        return "polyfit", reason, normal_clusters
    reason = (
        f"point-cloud input routed to PolyFit: {plane_count} planes, "
        f"{coverage * 100:.1f}% coverage, {normal_clusters} normal directions"
    )
    return "polyfit", reason, normal_clusters

def _inspect_mesh_input(path: Path, merge_distance: float = 0.0) -> Optional[dict]:
    """Return topology diagnostics for triangle-mesh inputs, including GLB scenes."""
    try:
        import trimesh
        loaded = trimesh.load(str(path), force="scene", process=False)
        geometries = []
        if isinstance(loaded, trimesh.Scene):
            for node in loaded.graph.nodes_geometry:
                try:
                    transform, geometry_name = loaded.graph[node]
                    geom = loaded.geometry[geometry_name].copy()
                    if isinstance(geom, trimesh.Trimesh) and len(geom.faces):
                        geom.apply_transform(transform)
                        geometries.append(geom)
                except Exception:
                    continue
        elif isinstance(loaded, trimesh.Trimesh) and len(loaded.faces):
            geometries.append(loaded)
        meshes = [g for g in geometries if len(g.faces) > 0]
        if not meshes:
            return None
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        original_vertices = int(len(mesh.vertices))
        original_faces = int(len(mesh.faces))

        mesh, seam_weld_stats = weld_coincident_mesh_vertices(mesh)
        seam_welded_vertices = int(seam_weld_stats.get("welded_vertices", 0))
        seam_removed_faces = int(seam_weld_stats.get("removed_faces", 0))
        distance_merge_stats = {
            "input_vertices": int(len(mesh.vertices)), "output_vertices": int(len(mesh.vertices)),
            "merged_vertices": 0, "removed_faces": 0, "rejected_chain_merges": 0,
        }
        if merge_distance > 0.0:
            merged_vertices, merged_faces, distance_merge_stats = merge_mesh_vertices_by_distance(
                np.asarray(mesh.vertices), np.asarray(mesh.faces), merge_distance
            )
            if len(merged_faces) == 0:
                raise ValueError(
                    "Merge by Distance removed every triangle. Reduce the merge distance "
                    "or disable vertex merging."
                )
            mesh = trimesh.Trimesh(vertices=merged_vertices, faces=merged_faces, process=False)
        edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
        boundary_edges = int(np.count_nonzero(edge_counts == 1))
        nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
        try:
            components = len(mesh.split(only_watertight=False))
        except Exception:
            components = -1
        return {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "original_vertices": original_vertices,
            "original_faces": original_faces,
            "seam_welded_vertices": seam_welded_vertices,
            "merged_vertices": int(seam_welded_vertices + distance_merge_stats.get("merged_vertices", 0)),
            "removed_faces": int(seam_removed_faces + distance_merge_stats.get("removed_faces", 0)),
            "merge_rejected_chain_welds": int(distance_merge_stats.get("rejected_chain_merges", 0)),
            "merge_distance": float(merge_distance if merge_distance > 0.0 else 0.0),
            "components": int(components),
            "boundary_edges": boundary_edges,
            "nonmanifold_edges": nonmanifold_edges,
            "watertight": bool(mesh.is_watertight),
            "surface_area": float(mesh.area),
        }
    except Exception as exc:

        if merge_distance > 0.0:
            raise PipelineFailure(f"Input mesh Merge by Distance failed: {exc}") from exc

        return None

def _mesh_sample_budget(face_count: int, max_points: int) -> int:
    """Avoid massively oversampling a low-poly source mesh into a point cloud."""
    cap = int(max_points) if int(max_points) > 0 else 300000
    if face_count <= 0:
        return cap

    return min(cap, max(6000, int(face_count) * 16))

def _load_source_mesh_geometry(
    path: Path, source_up_axis: str = "y", merge_distance: float = 0.0,
    source_frame_transform=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a mesh in the shared source-object frame and Studio Z-up coordinates."""
    import trimesh

    loaded = trimesh.load(str(path), force="scene", process=False)
    meshes = []
    if isinstance(loaded, trimesh.Scene):
        for node in loaded.graph.nodes_geometry:
            try:
                transform, geometry_name = loaded.graph[node]
                geometry = loaded.geometry[geometry_name]
                if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces):
                    part = geometry.copy()
                    part.apply_transform(transform)
                    meshes.append(part)
            except Exception:
                continue
    elif isinstance(loaded, trimesh.Trimesh) and len(loaded.faces):
        meshes.append(loaded)
    if not meshes:
        raise PipelineFailure("Could not load the original triangles from the input mesh.")
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    mesh, _ = weld_coincident_mesh_vertices(mesh)
    if merge_distance > 0.0:
        vertices, faces, _ = merge_mesh_vertices_by_distance(
            np.asarray(mesh.vertices), np.asarray(mesh.faces), merge_distance
        )
        if len(faces) == 0:
            raise PipelineFailure("No valid triangles remain after merging the input mesh.")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    world_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = points_to_source_local(world_vertices, source_frame_transform)
    vertices = convert_up_axis(vertices, source_up_axis)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return vertices, faces

def _decimate_mesh_arrays(
    vertices: np.ndarray, faces: np.ndarray, target_faces: int, log_cb=None
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Studio's built-in triangle-plane QEM without an Open3D dependency."""
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    target = max(4, int(target_faces))
    if len(tris) <= target:
        return verts, tris

    if __package__:
        from .feature_decimation import decimate_mesh_li_nan
    else:
        from feature_decimation import decimate_mesh_li_nan

    unique_edges = np.unique(
        np.sort(np.vstack((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]])), axis=1),
        axis=0,
    )
    euler_characteristic = int(len(verts) - len(unique_edges) + len(tris))
    estimated_target_vertices = max(4, int(round(target * 0.5 + euler_characteristic)))
    estimated_target_vertices = min(estimated_target_vertices, len(verts) - 1)
    try:
        result = None
        out_vertices = np.empty((0, 3), dtype=np.float64)
        out_faces = np.empty((0, 3), dtype=np.int64)
        for target_vertices in range(estimated_target_vertices, 3, -1):
            candidate = decimate_mesh_li_nan(
                verts,
                tris,
                None,
                target_vertex_count=target_vertices,
                allow_boundary_collapses=False,
                collapse_intersection_guard=False,
            )
            candidate_vertices = np.asarray(candidate.vertices, dtype=np.float64)
            candidate_faces = np.asarray(candidate.faces, dtype=np.int64)
            if len(candidate_vertices) >= 4 and 4 <= len(candidate_faces) < len(tris):
                result = candidate
                out_vertices = candidate_vertices
                out_faces = candidate_faces
                if len(out_faces) <= target:
                    break
        if result is None:
            raise RuntimeError("QEM produced no useful reduction.")
        if len(out_faces) > target:
            raise RuntimeError(
                f"QEM stopped at {len(out_faces):,} faces, above the requested {target:,}-face budget."
            )
        if log_cb:
            log_cb(
                f"Low-poly limit: {len(tris):,} → {len(out_faces):,} faces "
                f"(target {target:,}) · shared triangle-plane QEM core · "
                f"{result.stats.successful_collapses:,} collapses"
            )
        return out_vertices, out_faces
    except Exception as exc:
        raise PipelineFailure(
            f"Could not apply the output polygon limit ({target:,} faces) with the shared QEM core: {exc}"
        ) from exc

def _write_triangle_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    import trimesh
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise PipelineFailure("No valid output triangles remain after planar fitting.")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    if not path.exists() or path.stat().st_size == 0:
        raise PipelineFailure("Failed to save the low-poly OBJ.")

def _is_architectural_plane_set(groups: list[dict], stats: dict) -> bool:
    """Detect a strong architectural plane set independently of selected engine."""
    plane_count = int(stats.get("plane_count", len(groups)))
    coverage = float(stats.get("coverage", 0.0))
    normal_clusters = int(stats.get("normal_clusters", _normal_cluster_count(groups)))
    return plane_count >= 3 and coverage >= 0.35 and normal_clusters <= 12

def _planar_dissolve_tolerance(points: Optional[np.ndarray], plane_stats: Optional[dict]) -> float | None:
    """Choose a sub-millimetre, scale-aware tolerance for source-planar face joining."""
    if points is None:
        return None
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(cloud) == 0:
        return None
    diagonal = float(np.linalg.norm(np.ptp(cloud, axis=0)))
    threshold = float((plane_stats or {}).get("distance_threshold", 0.0) or 0.0)
    return max(
        diagonal * 2.5e-5,
        min(max(threshold, 0.0) * 0.01, 0.001),
        1e-7,
    )

def _apply_e_stage(
    output_path: Path,
    config: dict,
    log_cb,
    progress_cb=None,
    stage_preview_cb=None,
    planar_tolerance: float | None = None,
    max_normal_angle_degrees: float = 0.081,
) -> dict:
    """Apply Studio's separate E1/E2 cleanup after the official reconstruction."""
    if "merge_mesh_output" in config:
        output_cleanup_enabled = bool(config.get("merge_mesh_output", False))
        distance = float(config.get("merge_distance", 0.0001)) if output_cleanup_enabled else 0.0
    else:
        output_cleanup_enabled = float(config.get("e1_merge_distance", 0.001)) > 0.0
        distance = float(config.get("e1_merge_distance", 0.001))
    if progress_cb is not None:
        _emit(progress_cb, 90, "Studio post-process · E1/E2 topology cleanup")
    try:
        stats = process_obj_e_stage(
            output_path,
            merge_distance=distance,
            progress_cb=(lambda value, message: _emit(progress_cb, value, message)) if progress_cb else None,
            log_cb=log_cb,
            preview_cb=stage_preview_cb,
            planar_tolerance=planar_tolerance,
            max_normal_angle_degrees=max_normal_angle_degrees,
        )
    except Exception as exc:
        raise PipelineFailure(f"Studio E1/E2 post-processing failed: {exc}") from exc

    if progress_cb is not None:
        _emit(progress_cb, 97, "E · Topology cleanup complete")
    return stats

def _source_constrained_planar_fit(
    input_path: Path,
    output_path: Path,
    points: np.ndarray,
    groups: list[dict],
    plane_stats: dict,
    config: dict,
    mesh_input_stats: Optional[dict],
    log_cb,
    progress_cb,
    stage_preview_cb=None,
    mesh_preview_cb=None,
    source_colors: Optional[np.ndarray] = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Create a compact planar fit from original mesh topology or point patches."""
    source_faces = int((mesh_input_stats or {}).get("faces", 0))
    budget = get_output_face_budget(
        str(config.get("resolution_preset", "standard")),
        source_faces=source_faces,
    )
    configured_budget = int(config.get("output_face_budget", budget) or budget)
    budget = min(budget, configured_budget) if source_faces <= 0 else min(budget, configured_budget, source_faces)
    _emit(progress_cb, 82, "Shared planar fitting · Generate low-poly mesh aligned to original boundaries")

    if mesh_input_stats:
        merge_input = bool(config.get("merge_mesh_input", config.get("merge_mesh_vertices", False)))
        merge_distance = float(config.get("merge_distance", 0.0001)) if merge_input else 0.0
        vertices, faces = _load_source_mesh_geometry(
            input_path, str(config.get("source_up_axis", "y")), merge_distance,
            source_frame_transform=config.get("source_frame_transform"),
        )
        before_faces = len(faces)
        original_vertices = vertices.copy()
        original_faces = faces.copy()
        topology_budget_fallback = False
        topology_decimation_skipped = False
        source_is_watertight = bool((mesh_input_stats or {}).get("watertight", False))
        source_components = int((mesh_input_stats or {}).get("components", 1))
        source_nonmanifold_edges = int((mesh_input_stats or {}).get("nonmanifold_edges", 0))
        source_boundary_edges = int((mesh_input_stats or {}).get("boundary_edges", 0))
        safe_for_quadric_decimation = (
            source_is_watertight
            and source_components == 1
            and source_nonmanifold_edges == 0
            and source_boundary_edges == 0
        )
        if len(faces) > budget and not safe_for_quadric_decimation:
            topology_decimation_skipped = True
            log_cb(
                "Skipped pre-fit Quadric Decimation: source topology is open, multi-component, "
                f"or non-manifold ({source_components} components / {source_boundary_edges} boundary / "
                f"{source_nonmanifold_edges} non-manifold edges). Preserving the original surface "
                "connectivity to prevent collapsed/intersecting faces."
            )
        elif len(faces) > budget:
            vertices, faces = _decimate_mesh_arrays(vertices, faces, budget, log_cb=log_cb)
        if source_is_watertight and len(faces) < before_faces:

            try:
                import trimesh
                candidate_mesh = trimesh.Trimesh(
                    vertices=np.asarray(vertices, dtype=np.float64),
                    faces=np.asarray(faces, dtype=np.int64), process=False,
                )
                candidate_components = len(candidate_mesh.split(only_watertight=False))
                expected_components = max(1, int((mesh_input_stats or {}).get("components", 1)))
                topology_ok = candidate_mesh.is_watertight and candidate_components == expected_components
            except Exception:
                topology_ok = False
            if not topology_ok:
                vertices, faces = original_vertices, original_faces
                topology_budget_fallback = True
                log_cb(
                    "Cancelled low-poly simplification because it damaged the original watertight topology. "
                    f"Preserving the original connected, hole-free topology ({before_faces:,} faces)."
                )
        diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0))) if len(vertices) else 1.0

        ransac_tolerance = max(float(plane_stats.get("distance_threshold", 0.0)), 0.0)
        snap_tolerance = max(ransac_tolerance * 0.35, diagonal * 0.00025)
        scaffold_vertices = np.asarray(vertices, dtype=np.float64).copy()
        scaffold_faces = np.asarray(faces, dtype=np.int64).copy()

        source_topology_is_known = bool(mesh_input_stats and "watertight" in mesh_input_stats)
        source_topology_is_open_or_fragmented = source_topology_is_known and (
            not source_is_watertight
            or source_components != 1
            or source_nonmanifold_edges > 0
            or source_boundary_edges > 0
        )
        if source_topology_is_open_or_fragmented:
            vertices, faces = scaffold_vertices, scaffold_faces
            fit_stats = {
                "planarized_faces": 0,
                "clipped_faces": 0,
                "degenerate_faces_removed": 0,
                "duplicate_faces_removed": 0,
                "projected_vertices": 0,
                "fallback_vertices": 0,
                "skipped_vertices": int(len(vertices)),
                "max_vertex_displacement": 0.0,
                "topology_snap_fallback": True,
            }
            log_cb(
                "Skipped source-constrained planar vertex snapping: input mesh is open, "
                f"multi-component, or non-manifold ({source_components} components / "
                f"{source_boundary_edges} boundary / {source_nonmanifold_edges} non-manifold edges). "
                "Preserving original surface coordinates; E1/E2 will only merge safe shared vertices and coplanar faces."
            )
        else:
            vertices, faces, fit_stats = planarize_mesh_geometry(
                vertices, faces, points, groups, snap_tolerance,
                max_angle_degrees=12.0,
                support_margin_ratio=0.006,
                clip_outside_support=False,
            )
        if source_is_watertight:
            try:
                import trimesh
                fitted_mesh = trimesh.Trimesh(
                    vertices=np.asarray(vertices, dtype=np.float64),
                    faces=np.asarray(faces, dtype=np.int64), process=False,
                )
                fitted_components = len(fitted_mesh.split(only_watertight=False))
                expected_components = max(1, int((mesh_input_stats or {}).get("components", 1)))
                fitted_topology_ok = fitted_mesh.is_watertight and fitted_components == expected_components
            except Exception:
                fitted_topology_ok = False
            if not fitted_topology_ok:

                vertices, faces = scaffold_vertices, scaffold_faces
                fit_stats["planarized_faces"] = 0
                fit_stats["topology_snap_fallback"] = True
                log_cb(
                    "Cancelled the planar vertex snap because it damaged the original watertight topology. "
                    "Preserving the original triangle connectivity."
                )
            else:
                fit_stats["topology_snap_fallback"] = False
        log_cb(
            f"Original-mesh planar fitting: {before_faces:,} → {len(faces):,} faces · "
            f"planarized {fit_stats['planarized_faces']:,} · "
            f"degenerate {fit_stats.get('degenerate_faces_removed', 0):,} / duplicate {fit_stats.get('duplicate_faces_removed', 0):,} removed · "
            f"max vertex displacement {fit_stats.get('max_vertex_displacement', 0.0):.4g} · "
            f"quality-guard rollback {fit_stats.get('quality_guard_restored_vertices', 0):,} vertices / "
            f"{fit_stats.get('quality_guard_restored_faces', 0):,} face warnings · "
            f"in-plane lengths preserved"
        )
    else:
        vertices, faces, fit_stats = build_lowpoly_plane_patches(points, groups, max_faces=budget)
        if len(faces) == 0:
            raise PipelineFailure("Could not generate low-poly planar patches from the point cloud.")
        log_cb(
            f"Point-cloud low-poly planar patches: {fit_stats['plane_count']} planes · "
            f"{len(faces):,} faces (budget {budget:,})"
        )

    _write_triangle_mesh(output_path, vertices, faces)
    source_fit_tolerance = _planar_dissolve_tolerance(points, plane_stats)
    e_stage_metrics = _apply_e_stage(
        output_path, config, log_cb, progress_cb=progress_cb,
        stage_preview_cb=stage_preview_cb,
        planar_tolerance=source_fit_tolerance,
        max_normal_angle_degrees=1.0,
    )
    metrics, preview_vertices, preview_faces = _mesh_metrics(
        output_path, preview_cb=mesh_preview_cb,
        color_source_points=points, color_source_values=source_colors,
    )
    metrics.update(e_stage_metrics)
    metrics["planarized_faces"] = int(fit_stats.get("planarized_faces", len(faces)))
    metrics["output_face_budget"] = int(budget)
    metrics["reconstruction_method"] = "Unified low-poly planar fit"
    metrics["plane_count"] = int(plane_stats.get("plane_count", len(groups)))
    metrics["plane_coverage"] = float(plane_stats.get("coverage", 0.0))
    metrics["distance_threshold"] = float(plane_stats.get("distance_threshold", 0.0))
    metrics["input_points"] = int(len(points))
    metrics["degenerate_faces_removed"] = int(fit_stats.get("degenerate_faces_removed", 0))
    metrics["duplicate_faces_removed"] = int(fit_stats.get("duplicate_faces_removed", 0))
    metrics["projected_vertices"] = int(fit_stats.get("projected_vertices", 0))
    metrics["fallback_vertices"] = int(fit_stats.get("fallback_vertices", 0))
    metrics["skipped_vertices"] = int(fit_stats.get("skipped_vertices", 0))
    metrics["max_vertex_displacement"] = float(fit_stats.get("max_vertex_displacement", 0.0))
    metrics["topology_budget_fallback"] = bool(locals().get("topology_budget_fallback", False))
    metrics["topology_decimation_skipped"] = bool(locals().get("topology_decimation_skipped", False))
    metrics["planar_snap_skipped_for_topology"] = bool(
        locals().get("source_topology_is_open_or_fragmented", False)
    )
    metrics["vg_path"] = ""
    return metrics, preview_vertices, preview_faces

def _finalize_lowpoly_output(
    output_path: Path,
    source_points: Optional[np.ndarray],
    groups: list[dict],
    plane_stats: dict,
    config: dict,
    source_faces: int,
    log_cb,
    stage_preview_cb=None,
    mesh_preview_cb=None,
    polyfit_facets: Optional[int] = None,
    progress_cb=None,
    source_colors: Optional[np.ndarray] = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Cap polygon count and apply the same plane/footprint fit to every engine."""
    import trimesh
    loaded = load_obj_trimesh(output_path)
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if not parts:
            raise PipelineFailure("No valid mesh was found in the engine output.")
        mesh = trimesh.util.concatenate(parts)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise PipelineFailure("The engine output OBJ contains no valid triangles.")

    budget = get_output_face_budget(str(config.get("resolution_preset", "standard")), source_faces)
    configured_budget = int(config.get("output_face_budget", budget) or budget)
    budget = min(budget, configured_budget) if source_faces <= 0 else min(budget, configured_budget, source_faces)
    before_faces = int(len(mesh.faces))
    original_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    original_faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    vertices, faces = original_vertices, original_faces
    topology_decimation_skipped = False
    topology_budget_fallback = False
    if before_faces > budget:
        try:
            edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
            boundary_edges = int(np.count_nonzero(edge_counts == 1))
            nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
            components = len(mesh.split(only_watertight=False))
            safe_for_decimation = (
                bool(mesh.is_watertight)
                and components == 1
                and boundary_edges == 0
                and nonmanifold_edges == 0
            )
        except Exception:
            boundary_edges = -1
            nonmanifold_edges = -1
            components = -1
            safe_for_decimation = False
        if safe_for_decimation:
            vertices, faces = _decimate_mesh_arrays(
                original_vertices, original_faces, budget, log_cb=log_cb
            )
            try:
                candidate = trimesh.Trimesh(
                    vertices=np.asarray(vertices, dtype=np.float64),
                    faces=np.asarray(faces, dtype=np.int64), process=False,
                )
                valid_topology = bool(candidate.is_watertight) and len(
                    candidate.split(only_watertight=False)
                ) == 1
            except Exception:
                valid_topology = False
            if not valid_topology:
                vertices, faces = original_vertices, original_faces
                topology_budget_fallback = True
                log_cb(
                    "Cancelled pre-fit Quadric Decimation: it opened or fragmented the reconstructed surface. "
                    "Preserving the original face topology."
                )
        else:
            topology_decimation_skipped = True
            log_cb(
                "Skipped pre-fit Quadric Decimation: reconstructed output is open, multi-component, "
                f"or non-manifold ({components} components / {boundary_edges} boundary / "
                f"{nonmanifold_edges} non-manifold edges). Preserving face topology."
            )
    fit_stats = {"planarized_faces": 0, "clipped_faces": 0}
    if source_points is not None and groups:
        diagonal = float(np.linalg.norm(np.ptp(source_points, axis=0)))

        ransac_tolerance = max(float(plane_stats.get("distance_threshold", 0.0)), 0.0)
        snap_tolerance = max(ransac_tolerance * 0.35, diagonal * 0.00025)
        vertices, faces, fit_stats = planarize_mesh_geometry(
            vertices, faces, source_points, groups, snap_tolerance,
            max_angle_degrees=12.0,
            support_margin_ratio=0.006,

            clip_outside_support=False,
        )
    if len(faces) == 0:
        raise PipelineFailure("No output faces remain after shared planar fitting. Adjust the RANSAC tolerance.")
    _write_triangle_mesh(output_path, vertices, faces)
    e_stage_metrics = _apply_e_stage(
        output_path, config, log_cb, progress_cb=progress_cb,
        stage_preview_cb=stage_preview_cb,
        planar_tolerance=_planar_dissolve_tolerance(source_points, plane_stats),
        max_normal_angle_degrees=1.0 if source_points is not None else 0.081,
    )
    e_stage_metrics["topology_decimation_skipped"] = bool(topology_decimation_skipped)
    e_stage_metrics["topology_budget_fallback"] = bool(topology_budget_fallback)
    log_cb(
        f"Shared low-poly post-process: {before_faces:,} → {len(faces):,} faces · "
        f"planarized {fit_stats.get('planarized_faces', 0):,} · "
        f"boundary clipping {fit_stats.get('clipped_faces', 0):,} · "
        f"quality-guard rollback {fit_stats.get('quality_guard_restored_vertices', 0):,} vertices · target {budget:,}"
    )
    metrics, preview_vertices, preview_faces = _mesh_metrics(
        output_path, polyfit_facets=polyfit_facets, preview_cb=mesh_preview_cb,
        color_source_points=source_points, color_source_values=source_colors,
    )
    metrics["planarized_faces"] = int(fit_stats.get("planarized_faces", 0))
    metrics["clipped_faces"] = int(fit_stats.get("clipped_faces", 0))
    metrics["degenerate_faces_removed"] = int(fit_stats.get("degenerate_faces_removed", 0))
    metrics["duplicate_faces_removed"] = int(fit_stats.get("duplicate_faces_removed", 0))
    metrics["projected_vertices"] = int(fit_stats.get("projected_vertices", 0))
    metrics["fallback_vertices"] = int(fit_stats.get("fallback_vertices", 0))
    metrics["skipped_vertices"] = int(fit_stats.get("skipped_vertices", 0))
    metrics["max_vertex_displacement"] = float(fit_stats.get("max_vertex_displacement", 0.0))
    metrics["output_face_budget"] = int(budget)
    metrics.update(e_stage_metrics)
    return metrics, preview_vertices, preview_faces

def _count_obj_polygon_faces(obj_path: Path) -> int:
    """Count original OBJ face records without triangulating or rewriting them."""
    count = 0
    with obj_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.lstrip()
            if stripped.startswith("f ") or stripped.startswith("f\t"):
                count += 1
    return count

def _validate_polyfit_vg_groups(groups: list[dict], point_count: int) -> dict:
    """Validate the exact index-based groups before writing upstream VG input."""
    if not groups:
        raise PipelineFailure("There are no planar groups to write to the PolyFit VG file.")
    if point_count < 3:
        raise PipelineFailure("The PolyFit VG input contains fewer than three points.")

    all_indices = []
    for group_index, group in enumerate(groups, start=1):
        plane = np.asarray(group.get("plane", []), dtype=np.float64)
        indices = np.asarray(group.get("indices", []), dtype=np.int64).reshape(-1)
        if plane.shape != (4,) or not np.isfinite(plane).all():
            raise PipelineFailure(f"The plane equation for VG group {group_index} is invalid.")
        normal_length = float(np.linalg.norm(plane[:3]))
        if normal_length <= 1e-10:
            raise PipelineFailure(f"The normal vector for VG group {group_index} is zero.")
        if len(indices) < 3:
            raise PipelineFailure(f"VG plane group {group_index} contains fewer than three sample points.")
        if int(indices.min()) < 0 or int(indices.max()) >= int(point_count):
            raise PipelineFailure(
                f"VG plane group {group_index} contains out-of-range point indices "
                f"(valid 0..{point_count - 1})."
            )
        all_indices.append(indices)

    joined = np.concatenate(all_indices)
    unique_count = int(np.unique(joined).size)
    total_count = int(len(joined))
    if unique_count != total_count:
        raise PipelineFailure(
            f"There are {total_count - unique_count:,} duplicate point indices across VG plane groups. "
            "Rebuild groups using the original point indices."
        )
    return {
        "groups": int(len(groups)),
        "assigned_indices": total_count,
        "unique_assigned_indices": unique_count,
        "point_count": int(point_count),
    }

def _validate_loaded_polyfit_point_cloud(
    point_cloud, expected_points: Optional[int], expected_groups: Optional[int], log_cb
) -> dict:
    """Confirm PolyFit's VG/BVG parser loaded the complete header and group list."""
    try:
        actual_points = int(point_cloud.num_points())
        loaded_groups = point_cloud.groups()
        actual_groups = int(len(loaded_groups))
    except Exception as exc:
        raise PipelineFailure(f"Could not verify the PolyFit input group / point counts: {exc}") from exc
    if actual_points < 3 or actual_groups == 0:
        raise PipelineFailure(
            f"PolyFit input is empty: points={actual_points:,}, groups={actual_groups:,}."
        )
    if expected_points is not None and actual_points != int(expected_points):
        raise PipelineFailure(
            "VG read validation failed: the point count written to the file differs from the count read by PolyFit "
            f"(written={int(expected_points):,}, loaded={actual_points:,})."
        )
    if expected_groups is not None and actual_groups != int(expected_groups):
        raise PipelineFailure(
            "VG segment read validation failed: not all generated groups were loaded "
            f"(written={int(expected_groups):,}, loaded={actual_groups:,})."
        )
    log_cb(
        "PolyFit VG full-input read verified: "
        f"{actual_points:,} points · {actual_groups:,} planar groups"
        + (" · matches the generated group count" if expected_groups is not None else "")
    )
    return {"point_count": actual_points, "group_count": actual_groups}

def _preview_polyfit_map(polyfit_module, mesh, preview_path: Path, label: str,
                         stage_preview_cb, log_cb) -> None:
    """Persist an upstream Map for live preview and point-set timeout fallback."""
    candidate_required = preview_path.name == "polyfit_candidate_preview.obj"
    import trimesh

    try:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.unlink(missing_ok=True)
        saved = bool(polyfit_module.save_mesh(str(preview_path), mesh))
        if not saved or not preview_path.is_file() or preview_path.stat().st_size == 0:
            raise RuntimeError("PolyFit save_mesh did not produce a non-empty OBJ file.")
    except Exception as exc:
        if candidate_required:
            raise PipelineFailure(f"Could not save the PolyFit timeout-fallback candidate mesh: {exc}") from exc
        log_cb(f"Skipping {label} preview saving. Native reconstruction will continue: {exc}")
        return

    if stage_preview_cb is None:
        return
    try:
        preview = load_obj_trimesh(preview_path)
        if isinstance(preview, trimesh.Scene):
            parts = [g for g in preview.geometry.values() if isinstance(g, trimesh.Trimesh)]
            preview = trimesh.util.concatenate(parts) if parts else None
        if isinstance(preview, trimesh.Trimesh) and len(preview.faces):
            stage_preview_cb(
                np.asarray(preview.vertices, dtype=np.float32),
                np.asarray(preview.faces, dtype=np.uint32),
                label,
            )
    except Exception as exc:
        log_cb(f"Skipped {label} preview display; saved mesh remains available: {exc}")

def _preview_native_reconstructed_model(
    output_path: Path,
    polyfit_facets: int,
    stage_preview_cb,
    log_cb,
) -> dict:
    """Preview and count the official PolyFit stage-(e) model saved by ``polyfit.save_mesh``.

    This is deliberately emitted before Studio's E1/E2 cleanup.  In the upstream
    five-panel pipeline, (e) is the reconstructed model produced after face
    selection, while E1/E2 are additional Studio-side OBJ topology cleanup.
    The OBJ remains polygonal on disk; trimesh triangulation is preview-only.
    """
    try:
        polygon_count = _count_obj_polygon_faces(output_path)
    except Exception as exc:
        log_cb(f"Could not count official PolyFit stage-E OBJ polygons: {exc}")
        return {}

    stats = {
        "polyfit_stage_e_obj_polygon_faces": int(polygon_count),
        "polyfit_stage_e_preview_triangles": 0,
        "polyfit_stage_e_previewed": False,
    }
    if stage_preview_cb is None:
        return stats

    try:
        import trimesh

        loaded = load_obj_trimesh(output_path)
        if isinstance(loaded, trimesh.Scene):
            parts = [
                geometry for geometry in loaded.geometry.values()
                if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces)
            ]
            loaded = trimesh.util.concatenate(parts) if parts else None
        if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
            raise ValueError("the saved PolyFit model contains no previewable faces")

        preview_triangles = int(len(loaded.faces))
        stats["polyfit_stage_e_preview_triangles"] = preview_triangles
        stats["polyfit_stage_e_previewed"] = True
        label = f"E · RECONSTRUCTED MODEL · {polygon_count:,} polygons"
        stage_preview_cb(
            np.asarray(loaded.vertices, dtype=np.float32),
            np.asarray(loaded.faces, dtype=np.uint32),
            label,
        )
        log_cb(
            "PolyFit pipeline stage E · Reconstructed model: "
            f"{polygon_count:,} saved OBJ polygons / {preview_triangles:,} preview triangles · "
            f"{int(polyfit_facets):,} selected native facets · polygon OBJ preserved"
        )
    except Exception as exc:

        log_cb(f"Skipped PolyFit stage-E preview (native OBJ remains valid): {exc}")
    return stats

def _run_official_polyfit_steps(
    polyfit_module,
    point_cloud,
    config: dict,
    work_dir: Path,
    log_cb,
    progress_cb=None,
    stage_preview_cb=None,
):
    """Execute upstream Python Example_2's original step-by-step API sequence."""
    required = ("HypothesisGenerator", "FaceSelection", "SCIP", "save_mesh")
    missing = [name for name in required if not hasattr(polyfit_module, name)]
    if missing:
        raise PipelineFailure(
            "The installed PolyFit binding does not expose the required stepwise API: " + ", ".join(missing)
        )

    try:
        groups = point_cloud.groups()
        if not groups:
            raise PipelineFailure("The PolyFit input point cloud has no planar groups.")

        if progress_cb:
            _emit(progress_cb, 56, "B · Refine planar segments (refine_planes)")
        hypothesis = polyfit_module.HypothesisGenerator(point_cloud)
        hypothesis.refine_planes()
        log_cb("PolyFit upstream step 1/4: HypothesisGenerator.refine_planes() completed")

        if progress_cb:
            _emit(progress_cb, 63, "C · Generate candidate faces (HypothesisGenerator.generate)")
        mesh = hypothesis.generate()
        if not mesh:
            raise PipelineFailure(
                "PolyFit failed to generate candidate faces. Verify that the VG contains all major planes and "
                "that the plane groups form a closed model."
            )
        if hasattr(mesh, "size_of_facets") and int(mesh.size_of_facets()) <= 0:
            raise PipelineFailure("PolyFit candidate-face generation returned an empty result.")
        log_cb("PolyFit upstream step 2/4: HypothesisGenerator.generate() completed")
        _preview_polyfit_map(
            polyfit_module, mesh, work_dir / "polyfit_candidate_preview.obj",
            f"C · CANDIDATE FACES · {int(mesh.size_of_facets()):,} facets",
            stage_preview_cb, log_cb,
        )

        if progress_cb:
            _emit(progress_cb, 70, "C · Compute candidate-face confidences (compute_confidences)")
        hypothesis.compute_confidences(mesh, False)
        log_cb("PolyFit upstream step 3/4: compute_confidences(mesh, False) completed")

        if progress_cb:
            _emit(progress_cb, 77, "D · SCIP optimization / face selection (FaceSelection.optimize)")
        selector = polyfit_module.FaceSelection(point_cloud, mesh)
        weights = (
            float(config.get("fitting", 0.43)),
            float(config.get("coverage", 0.27)),
            float(config.get("complexity", 0.30)),
        )
        selector.optimize(hypothesis, polyfit_module.SCIP, *weights)
        facets = int(mesh.size_of_facets())
        if facets <= 0:
            raise PipelineFailure("SCIP optimization returned no selected faces.")
        log_cb(
            "PolyFit upstream step 4/4: FaceSelection.optimize(SCIP) completed · "
            f"weights fitting/coverage/complexity={weights[0]:.3f}/{weights[1]:.3f}/{weights[2]:.3f} · "
            f"selected facets={facets:,}"
        )
        _preview_polyfit_map(
            polyfit_module, mesh, work_dir / "polyfit_selected_preview.obj",
            f"D · SELECTED FACES · SCIP optimization · {facets:,} facets",
            stage_preview_cb, log_cb,
        )
        return mesh
    except PipelineFailure:
        raise
    except Exception as exc:
        raise PipelineFailure(f"Official PolyFit stepwise execution failed: {exc}") from exc

def _inspect_native_polyfit_output(
    output_path: Path,
    polyfit_facets: int,
    log_cb,
    stage_preview_cb=None,
    mesh_preview_cb=None,
    e_stage_metrics: Optional[dict] = None,
    color_source_points: Optional[np.ndarray] = None,
    color_source_values: Optional[np.ndarray] = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Inspect the final PolyFit OBJ after optional Studio E1/E2 post-processing."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise PipelineFailure("The official PolyFit result OBJ was not created.")

    raw_polygon_faces = _count_obj_polygon_faces(output_path)
    metrics, preview_vertices, preview_faces = _mesh_metrics(
        output_path, polyfit_facets=polyfit_facets, preview_cb=mesh_preview_cb,
        color_source_points=color_source_points, color_source_values=color_source_values,
    )
    metrics["polyfit_native_output"] = True
    metrics["polyfit_native_facets"] = int(polyfit_facets)
    metrics["polyfit_obj_polygon_faces"] = int(raw_polygon_faces)
    metrics["polyfit_preview_triangles"] = int(len(preview_faces))
    metrics["polyfit_output_rewritten"] = bool(e_stage_metrics)
    if e_stage_metrics:
        metrics["polyfitty_postprocess_obj_polygon_faces"] = int(raw_polygon_faces)
    metrics["reconstruction_method"] = (
        "Official PolyFit 1.6.0 · Example_2 + Studio E1/E2 post-process"
        if e_stage_metrics else "Official PolyFit 1.6.0 · native Example_2 stepwise"
    )
    if e_stage_metrics:
        metrics.update(e_stage_metrics)

    if e_stage_metrics:
        log_cb(
            "PolyFit final model after Studio post-process: "
            f"{polyfit_facets:,} upstream facets / {raw_polygon_faces:,} final OBJ polygons / "
            f"{len(preview_faces):,} preview triangles · "
            "Studio E1 distance weld + E2 coplanar face dissolve · No remeshing "
            "or quadric decimation"
        )
    else:
        log_cb(
            "Inspected native PolyFit polygon result: "
            f"{polyfit_facets:,} upstream facets / {raw_polygon_faces:,} OBJ polygons / "
            f"{len(preview_faces):,} preview triangles · No post-process remeshing"
        )
    if raw_polygon_faces and raw_polygon_faces != int(polyfit_facets):
        log_cb(
            "Warning: the PolyFit Map facet count differs from the OBJ face-record count. "
            f"Map={polyfit_facets:,}, post-process OBJ={raw_polygon_faces:,}. Keeping the Studio-cleaned result unchanged."
        )
    if not metrics.get("watertight", False):
        log_cb(
            "The native PolyFit output is an open mesh "
            f"(boundary {metrics.get('boundary_edges', 0):,}, "
            f"non-manifold {metrics.get('nonmanifold_edges', 0):,}). "
            "Studio post-processing preserves open boundaries and does not fill holes. "
            "Check that all required planar segments are present and that the input satisfies the watertight-model assumption."
        )

    if stage_preview_cb is not None:
        final_label = (
            f"STUDIO POSTPROCESS COMPLETE · {raw_polygon_faces:,} final polygons · cleaned output"
            if e_stage_metrics else
            f"E · RECONSTRUCTED MODEL · {raw_polygon_faces:,} polygons · {polyfit_facets:,} native facets"
        )
        stage_preview_cb(preview_vertices, preview_faces, final_label)
    return metrics, preview_vertices, preview_faces

def _transfer_vertex_colors(
    vertices: np.ndarray,
    source_points: Optional[np.ndarray],
    source_colors: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Transfer original sampled surface color to reconstructed vertices."""
    if source_points is None or source_colors is None:
        return None
    points = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(source_colors, dtype=np.float32)
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or len(colors) != len(points) or len(verts) == 0 or colors.ndim != 2 or colors.shape[1] < 3:
        return None
    colors = colors[:, :3]
    if float(np.nanmax(colors)) > 1.5:
        colors = colors / 255.0
    colors = np.clip(np.nan_to_num(colors, nan=0.66, posinf=1.0, neginf=0.0), 0.0, 1.0)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        try:
            _, indices = tree.query(verts, k=1, workers=-1)
        except TypeError:
            _, indices = tree.query(verts, k=1)
        output = colors[np.asarray(indices, dtype=np.int64)]
    except Exception:

        output = np.empty((len(verts), 3), dtype=np.float32)
        chunk = max(1, min(2048, 2_000_000 // max(1, len(points))))
        for start in range(0, len(verts), chunk):
            block = verts[start:start + chunk]
            distances = np.sum((block[:, None, :] - points[None, :, :]) ** 2, axis=2)
            output[start:start + len(block)] = colors[np.argmin(distances, axis=1)]
    alpha = np.ones((len(output), 1), dtype=np.float32)
    return np.ascontiguousarray(np.column_stack((output, alpha)), dtype=np.float32)

def _mesh_metrics(
    obj_path: Path,
    polyfit_facets: Optional[int] = None,
    preview_cb: Optional[Callable[..., None]] = None,
    color_source_points: Optional[np.ndarray] = None,
    color_source_values: Optional[np.ndarray] = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    import trimesh

    loaded = load_obj_trimesh(obj_path)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise PipelineFailure("A mesh was loaded from OBJ, but it contains no valid faces.")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise PipelineFailure("The PolyFit result OBJ contains no valid faces.")

    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mesh_faces = np.asarray(mesh.faces, dtype=np.uint32)
    vertex_colors = _transfer_vertex_colors(
        mesh_vertices, color_source_points, color_source_values
    )
    if preview_cb is not None:
        if vertex_colors is not None:
            try:
                preview_cb(mesh_vertices, mesh_faces, vertex_colors)
            except TypeError:
                preview_cb(mesh_vertices, mesh_faces)
        else:
            preview_cb(mesh_vertices, mesh_faces)

    edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
    try:
        components = len(mesh.split(only_watertight=False))
    except Exception:
        components = -1
    metrics = {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "polyfit_facets": int(polyfit_facets if polyfit_facets is not None else len(mesh.faces)),
        "unique_edges": int(len(mesh.edges_unique)),
        "components": int(components),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "surface_area": float(mesh.area),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "euler_number": int(mesh.euler_number),
        "file_size_mb": float(obj_path.stat().st_size / (1024 * 1024)),
    }
    return metrics, mesh_vertices, mesh_faces

def _run_li_nan_feature_preserving(
    input_path: Path,
    output_path: Path,
    config: dict,
    start_time: float,
    progress_cb,
    log_cb,
    mesh_preview_cb=None,
    stage_preview_cb=None,
) -> dict:
    """Run the three-stage Li & Nan (2021) mesh simplification pipeline.

    Paper stages:
      1. face-normal bilateral filtering and anisotropic vertex updates;
      2. edge-connected planar-region growing/refinement, LS planes, contours;
      3. hierarchical region-constrained QEM edge collapses.

    Boundary locking and the optional intersection guard are application
    safeguards, not claims about the published method.
    """
    supported_mesh_suffixes = {".obj", ".stl", ".off", ".glb", ".gltf", ".3mf", ".ply"}
    if input_path.suffix.lower() not in supported_mesh_suffixes:
        raise PipelineFailure(
            "Li & Nan 2021 feature-preserving simplification requires an existing triangulated mesh "
            "(.obj, .stl, .off, .glb, glTF, 3MF, or mesh-based PLY). Point-cloud inputs are not supported."
        )

    merge_enabled = bool(config.get("merge_mesh_input", config.get("merge_mesh_vertices", False)))
    merge_distance = float(config.get("merge_distance", 0.0001)) if merge_enabled else 0.0
    if not np.isfinite(merge_distance) or merge_distance < 0.0:
        raise PipelineFailure("Merge by Distance must be finite and non-negative.")

    _emit(progress_cb, 3, "Load triangulated source mesh")
    try:
        vertices, faces = _load_source_mesh_geometry(
            input_path,
            str(config.get("source_up_axis", "y")),
            merge_distance,
            source_frame_transform=config.get("source_frame_transform"),
        )
    except Exception as exc:
        raise PipelineFailure(f"Could not load a triangulated source mesh for Li & Nan simplification: {exc}") from exc
    if len(vertices) < 3 or len(faces) < 1:
        raise PipelineFailure("The selected file does not contain a usable triangle mesh.")

    if __package__:
        from .feature_preserving import filter_mesh_li_nan
        from .planar_regions import extract_planar_regions_li_nan
        from .feature_decimation import decimate_mesh_li_nan
        from .mesh_intersection_guard import guard_filter_displacement
        from .preview_visuals import active_face_ids_from_vertex_motion, face_edge_segments
    else:
        from feature_preserving import filter_mesh_li_nan
        from planar_regions import extract_planar_regions_li_nan
        from feature_decimation import decimate_mesh_li_nan
        from mesh_intersection_guard import guard_filter_displacement
        from preview_visuals import active_face_ids_from_vertex_motion, face_edge_segments

    used_vertices = int(len(np.unique(np.asarray(faces, dtype=np.int64).reshape(-1))))
    if used_vertices < 3:
        raise PipelineFailure("The input mesh has fewer than three referenced vertices.")

    filter_iterations = int(config.get("linan_filter_iterations", 10))
    filter_angle = float(config.get("linan_filter_normal_angle", 20.0))
    intersection_guard = True
    region_angle = float(config.get("linan_region_normal_angle", 15.0))
    requested_area_threshold = float(config.get("linan_small_region_area_threshold", 0.0) or 0.0)
    requested_fit_threshold = float(config.get("linan_fitting_threshold", 0.0) or 0.0)
    if not np.isfinite(requested_area_threshold) or requested_area_threshold < 0.0:
        raise PipelineFailure("Small planar-region area threshold must be finite and non-negative (0 = AUTO).")
    if not np.isfinite(requested_fit_threshold) or requested_fit_threshold < 0.0:
        raise PipelineFailure("Planar fitting threshold must be finite and non-negative (0 = AUTO).")

    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 1e-12:
        raise PipelineFailure("The input mesh has near-zero spatial extent.")
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid_areas = face_areas[np.isfinite(face_areas) & (face_areas > 0.0)]
    if not len(valid_areas):
        raise PipelineFailure("The input mesh contains no non-degenerate triangles.")

    area_threshold = requested_area_threshold
    if area_threshold == 0.0:
        area_threshold = float(np.median(valid_areas) * 4.0)
    fit_threshold = requested_fit_threshold
    if fit_threshold == 0.0:
        fit_scale = min(0.01, max(0.001, 0.001 * float(np.sqrt(max(1, len(faces))))))
        fit_threshold = float(diagonal * fit_scale)

    log_cb(
        f"Li & Nan mesh input: {len(vertices):,} vertices / {len(faces):,} triangles · "
        "source frame converted to Studio Z-up"
    )
    if merge_distance > 0.0:
        log_cb(f"Input Mesh Cleanup · Merge by Distance: {merge_distance:g} mesh units")
    log_cb(
        f"Stage 1 · bilateral face-normal filter: {filter_iterations} iterations · "
        f"normal angle {filter_angle:g}° · sigma_dist = average unique edge length"
    )

    _emit(progress_cb, 8, "Li & Nan Stage 1/3 · bilateral normal filtering and vertex updates")
    last_filter_progress = {"percent": 8}

    def report_filter(iteration: int, total: int, max_displacement: float) -> None:
        percent = 8 + int(29 * iteration / max(1, total))
        if percent >= last_filter_progress["percent"] or iteration >= total:
            last_filter_progress["percent"] = percent
            _emit(
                progress_cb,
                percent,
                f"Li & Nan filtering {iteration}/{total} · max displacement {max_displacement:.6g}",
            )

    previous_filter_preview_vertices = np.asarray(vertices, dtype=np.float64).copy()
    reconstruction_compute = None
    reconstruction_filter_compute = None
    compute_mode = str(config.get("reconstruction_compute_mode", "auto") or "auto").strip().lower()
    try:
        if __package__:
            from .gpu_compute import GPUComputeManager
        else:
            from gpu_compute import GPUComputeManager
        reconstruction_compute = GPUComputeManager(
            selection=compute_mode,
            progress_callback=lambda _percent, message: log_cb(str(message)),
        )
        reconstruction_filter_compute = reconstruction_compute
        if reconstruction_compute.backend is not None:
            log_cb(
                f"Reconstruction compute device · {reconstruction_compute.vendor} "
                f"{reconstruction_compute.device_name} · {reconstruction_compute.backend_name}"
            )
            filter_operation = getattr(reconstruction_compute.backend, "filter_mesh_step", None)
            if not callable(filter_operation) and compute_mode in {"auto", "gpu", "gpu-only"}:

                available_filter_backends = {
                    str(getattr(item, "backend", "")).lower()
                    for item in getattr(reconstruction_compute, "devices", [])
                }
                for candidate_backend in ("opencl", "sycl", "cuda"):
                    if candidate_backend not in available_filter_backends:
                        continue
                    try:
                        candidate_manager = GPUComputeManager(
                            selection=candidate_backend,
                            progress_callback=lambda _percent, message: log_cb(str(message)),
                        )
                        candidate_operation = getattr(candidate_manager.backend, "filter_mesh_step", None)
                        if candidate_manager.backend is not None and callable(candidate_operation):
                            reconstruction_filter_compute = candidate_manager
                            log_cb(
                                f"Li-Nan Stage-1 compute uses {candidate_manager.backend_name} "
                                f"on {candidate_manager.vendor} {candidate_manager.device_name}; "
                                f"the primary {reconstruction_compute.backend_name} adapter does not expose this filter kernel."
                            )
                            break
                    except Exception as filter_backend_exc:
                        log_cb(
                            f"Li-Nan {candidate_backend.upper()} filter adapter unavailable · "
                            f"{type(filter_backend_exc).__name__}: {filter_backend_exc}"
                        )
        else:
            log_cb("Reconstruction compute · CPU reference (no verified GPU backend available)")
    except Exception as compute_exc:
        log_cb(
            f"Reconstruction GPU compute initialization unavailable; using CPU reference "
            f"({type(compute_exc).__name__}: {compute_exc})"
        )

    def preview_linan_filter_iteration(iteration, total, current_vertices, current_faces, _filtered_normals, max_displacement):
        nonlocal previous_filter_preview_vertices
        if stage_preview_cb is not None:
            active_face_ids = active_face_ids_from_vertex_motion(
                previous_filter_preview_vertices,
                current_vertices,
                current_faces,
                max_faces=min(256, max(1, int(np.ceil(len(current_faces) * 0.01)))),
            )
            active_edges = face_edge_segments(current_vertices, current_faces, active_face_ids)
            stage_preview_cb(
                np.asarray(current_vertices, dtype=np.float32),
                np.asarray(current_faces, dtype=np.uint32),
                f"LI & NAN · LIVE FACE-NORMAL FILTER {iteration}/{total} · ACTIVE {len(active_face_ids):,} FACES · MAX {max_displacement:.6g}",
                highlight_segments=active_edges,
            )
        previous_filter_preview_vertices = np.asarray(current_vertices, dtype=np.float64).copy()

    try:
        filtered_vertices, filtered_faces, filter_stats = filter_mesh_li_nan(
            vertices,
            faces,
            normal_angle_degrees=filter_angle,
            iterations=filter_iterations,
            progress_cb=report_filter,
            preview_cb=preview_linan_filter_iteration if stage_preview_cb is not None else None,
            compute_manager=reconstruction_filter_compute,
        )
    except Exception as exc:
        raise PipelineFailure(f"Li & Nan bilateral mesh filtering failed; input mesh was not replaced: {exc}") from exc
    if not np.array_equal(filtered_faces, faces):
        raise PipelineFailure("Li & Nan filter unexpectedly changed triangle connectivity; output rejected.")

    try:
        filtered_vertices, filter_guard_stats = guard_filter_displacement(
            vertices,
            filtered_vertices,
            filtered_faces,
        )
    except Exception as exc:
        raise PipelineFailure(
            f"Li & Nan close-surface filtering guard failed; filtered geometry was rejected: {exc}"
        ) from exc
    guarded_triangles = filtered_vertices[filtered_faces]
    guarded_cross = np.cross(
        guarded_triangles[:, 1] - guarded_triangles[:, 0],
        guarded_triangles[:, 2] - guarded_triangles[:, 0],
    )
    guarded_double_areas = np.linalg.norm(guarded_cross, axis=1)
    if not np.isfinite(filtered_vertices).all() or not np.isfinite(guarded_double_areas).all() or np.any(guarded_double_areas <= 2e-15):
        raise PipelineFailure("Li & Nan close-surface filtering guard produced non-finite or degenerate geometry; output rejected.")
    guarded_max_displacement = float(np.max(np.linalg.norm(filtered_vertices - vertices, axis=1))) if len(vertices) else 0.0
    log_cb(
        "Stage 1 close-surface guard ON: "
        f"{filter_guard_stats.get('candidate_pair_count', 0):,} nearby face pairs · "
        f"{filter_guard_stats.get('guarded_pair_count', 0):,} pairs protected · "
        f"{filter_guard_stats.get('attenuated_vertex_count', 0):,} vertices attenuated · "
        f"{filter_guard_stats.get('unresolved_pair_count', 0):,} unresolved pre-existing/ambiguous pairs"
    )
    if stage_preview_cb is not None:
        active_filter_faces = active_face_ids_from_vertex_motion(
            vertices, filtered_vertices, filtered_faces,
            max_faces=min(256, max(1, int(np.ceil(len(filtered_faces) * 0.01)))),
        )
        active_filter_edges = face_edge_segments(filtered_vertices, filtered_faces, active_filter_faces)
        stage_preview_cb(
            filtered_vertices.astype(np.float32),
            filtered_faces.astype(np.uint32),
            f"LI & NAN · STAGE 1 FILTERED · ACTIVE {len(active_filter_faces):,} FACES · {len(filtered_faces):,} TRIANGLES",
            highlight_segments=active_filter_edges,
        )

    log_cb(
        f"Stage 1 complete: {filter_stats.completed_iterations} iterations · "
        f"compute {filter_stats.filter_compute_backend} · GPU batches {filter_stats.filter_gpu_batches} · "
        f"topology {filter_stats.topology_build_seconds:.2f}s · filter {filter_stats.filtering_seconds:.2f}s · "
        f"GPU kernel+transfer {filter_stats.filter_gpu_seconds:.2f}s · "
        f"CPU parity reference {filter_stats.filter_cpu_reference_seconds:.2f}s · "
        f"max raw displacement {filter_stats.final_max_vertex_displacement:.6g} · "
        f"max guarded displacement {guarded_max_displacement:.6g}"
    )
    if filter_stats.filter_fallback_reason:
        log_cb(f"Stage 1 compute note · {filter_stats.filter_fallback_reason}")
    log_cb(
        f"Stage 2 · planar-region extraction: region normal angle {region_angle:g}° · "
        f"small-region area {area_threshold:.6g} · fitting threshold {fit_threshold:.6g}"
    )
    _emit(progress_cb, 40, "Li & Nan Stage 2/3 · region growing, refinement, LS planes and contours")
    def preview_linan_planar_regions(region_ids, active_face_ids, done, total, phase):
        if stage_preview_cb is None:
            return
        active_ids = np.asarray(active_face_ids, dtype=np.int64).reshape(-1)
        active_ids = active_ids[(active_ids >= 0) & (active_ids < len(filtered_faces))]
        active_edges = face_edge_segments(filtered_vertices, filtered_faces, active_ids)
        stage_preview_cb(
            filtered_vertices.astype(np.float32),
            filtered_faces.astype(np.uint32),
            f"LI & NAN · LIVE PLANAR REGIONS · {phase} · {done:,}/{total:,} · ACTIVE {len(active_ids):,} FACES",
            highlight_segments=active_edges,
        )

    try:
        planar_result = extract_planar_regions_li_nan(
            filtered_vertices,
            filtered_faces,
            small_region_area_threshold=area_threshold,
            fitting_threshold=fit_threshold,
            normal_angle_degrees=region_angle,
            visualization_cb=preview_linan_planar_regions if stage_preview_cb is not None else None,
        )
    except Exception as exc:
        raise PipelineFailure(f"Li & Nan planar-region and contour extraction failed: {exc}") from exc

    planar_stats = planar_result.stats
    region_count = len(planar_result.regions)
    plane_coverage = (
        float(np.count_nonzero(planar_result.face_region_ids >= 0) / len(filtered_faces))
        if len(filtered_faces) else 0.0
    )
    log_cb(
        f"Stage 2 timing · edge topology {planar_stats.edge_incidence_seconds:.2f}s · "
        f"coarse region growing {planar_stats.coarse_region_seconds:.2f}s · "
        f"refinement {planar_stats.refinement_seconds:.2f}s · "
        f"least-squares planes {planar_stats.plane_fit_seconds:.2f}s · "
        f"contours {planar_stats.contour_seconds:.2f}s"
    )
    log_cb(
        f"Stage 2 complete: {planar_stats.coarse_region_count:,} coarse → "
        f"{planar_stats.refined_region_count:,} planar regions · "
        f"{planar_stats.ungrouped_face_count:,} ungrouped triangles · "
        f"{planar_stats.contour_edge_count:,} intra-contour edges · "
        f"{planar_stats.projected_contour_count:,} projected to plane-intersection lines · "
        f"plane coverage {plane_coverage:.1%}"
    )

    requested_target = int(config.get("linan_target_vertices", 0) or 0)
    if requested_target < 0:
        raise PipelineFailure("Target vertex count must be 0 (automatic) or a positive integer.")
    if requested_target == 0:
        if region_count > 0:
            target_vertices = min(used_vertices, max(3, 2 * region_count))
            target_source = f"paper-guided AUTO: 2 × {region_count:,} planar regions"
        else:
            target_vertices = min(used_vertices, max(3, int(np.ceil(used_vertices * 0.10))))
            target_source = "AUTO fallback: 10% of input vertices (no planar regions detected)"
    else:
        target_vertices = min(max(3, requested_target), used_vertices)
        target_source = "user-specified target"

    boundary_lock = bool(config.get("linan_boundary_lock", True))
    log_cb(
        f"Stage 3 · hierarchical QEM edge collapse: target {target_vertices:,}/{used_vertices:,} vertices · "
        f"{target_source} · boundary lock {'ON' if boundary_lock else 'OFF'} · "
        f"intersection guard {'ON' if intersection_guard else 'OFF'}"
    )
    _emit(progress_cb, 58, "Li & Nan Stage 3/3 · hierarchical planar-region QEM")
    last_decimation_progress = {"percent": 58}

    def report_decimation(done: int, total: int, current: int) -> None:
        percent = 58 + int(30 * done / max(1, total))
        if percent >= last_decimation_progress["percent"] or done >= total:
            last_decimation_progress["percent"] = percent
            _emit(
                progress_cb,
                percent,
                f"Li & Nan edge collapse {done:,}/{total:,} · {current:,} vertices",
            )

    def preview_linan_qem(current_vertices, current_faces, region_ids, recent_edges, done, total, current_count):
        if stage_preview_cb is None:
            return
        active_faces = np.flatnonzero(np.asarray(region_ids, dtype=np.int64).reshape(-1) == -2)
        active_edges = face_edge_segments(current_vertices, current_faces, active_faces)
        recent = (
            np.asarray(recent_edges, dtype=np.float32).reshape(-1, 2, 3)
            if recent_edges is not None else np.empty((0, 2, 3), dtype=np.float32)
        )
        edge_parts = [part for part in (active_edges, recent) if len(part)]
        highlight_edges = np.concatenate(edge_parts, axis=0) if edge_parts else np.empty((0, 2, 3), dtype=np.float32)
        stage_preview_cb(
            np.asarray(current_vertices, dtype=np.float32),
            np.asarray(current_faces, dtype=np.uint32),
            f"LI & NAN · LIVE HIERARCHICAL QEM {done:,}/{total:,} · {current_count:,} VERTICES · ACTIVE FACES {len(active_faces):,}",
            highlight_segments=highlight_edges,
        )

    try:
        decimation_result = decimate_mesh_li_nan(
            filtered_vertices,
            filtered_faces,
            planar_result,
            target_vertex_count=target_vertices,
            progress_cb=report_decimation,
            visualization_cb=preview_linan_qem if stage_preview_cb is not None else None,
            visualization_stride=max(1, int(np.ceil(max(1, used_vertices - target_vertices) / 32.0))),
            allow_boundary_collapses=not boundary_lock,
            collapse_intersection_guard=intersection_guard,
            compute_manager=reconstruction_compute,
        )
    except Exception as exc:
        raise PipelineFailure(f"Li & Nan hierarchical QEM decimation failed; output was rejected: {exc}") from exc

    dec_stats = decimation_result.stats
    output_vertices = np.asarray(decimation_result.vertices, dtype=np.float64)
    output_faces = np.asarray(decimation_result.faces, dtype=np.int64)
    if not dec_stats.target_reached:
        log_cb(
            "Li & Nan stopped above the requested target because remaining edges were rejected by "
            "manifold/boundary/link/flip or optional intersection safeguards: "
            f"{dec_stats.output_vertex_count:,}/{target_vertices:,} vertices retained."
        )
    log_cb(
        f"Stage 3 timing · topology {dec_stats.topology_build_seconds:.2f}s · "
        f"component checks {dec_stats.component_check_seconds:.2f}s · "
        f"quadric setup {dec_stats.quadric_setup_seconds:.2f}s · "
        f"initial edge queue {dec_stats.initial_queue_seconds:.2f}s · "
        f"sequential collapses {dec_stats.collapse_seconds:.2f}s · "
        f"candidate evaluation {dec_stats.candidate_evaluation_backend} · "
        f"cost {dec_stats.candidate_evaluation_seconds:.3f}s · "
        f"geometry {dec_stats.candidate_geometry_seconds:.3f}s · "
        f"workers {dec_stats.candidate_evaluation_workers} · "
        f"evaluated {dec_stats.candidate_evaluation_count:,} candidates · "
        f"batched geometry checks {dec_stats.candidate_geometry_checks:,} · "
        f"candidate batches {dec_stats.candidate_evaluation_batches:,} · "
        f"queue compactions {dec_stats.queue_compaction_count:,} · "
        f"stale entries removed {dec_stats.queue_stale_entries_removed:,} · "
        f"placement reuses {dec_stats.queue_placement_reuses:,} · "
        f"cached cost requeues {dec_stats.queue_cached_cost_requeues:,} · "
        f"queue peak {dec_stats.queue_peak_entries:,} · "
        f"adaptive QEM {dec_stats.adaptive_qem_tier} · "
        f"queue policy min {dec_stats.adaptive_queue_compact_min_entries:,} · "
        f"stale ratio {dec_stats.adaptive_queue_stale_ratio:.2f} · "
        f"stale budget {dec_stats.adaptive_queue_memory_budget_bytes / (1024 * 1024):.1f} MiB · "
        f"estimated topology {dec_stats.estimated_topology_bytes / (1024 * 1024):.1f} MiB · "
        f"estimated queue peak {dec_stats.estimated_queue_peak_bytes / (1024 * 1024):.1f} MiB · "
        f"memory compactions {dec_stats.queue_memory_compaction_count:,} · "
        f"progress stride {dec_stats.adaptive_progress_callback_stride:,} · "
        f"final validation {dec_stats.final_validation_seconds:.2f}s"
    )
    log_cb(
        f"Stage 3 complete: {dec_stats.successful_collapses:,} collapses · "
        f"vertices {used_vertices:,}→{dec_stats.output_vertex_count:,} · "
        f"triangles {len(filtered_faces):,}→{dec_stats.output_face_count:,} · "
        f"boundary edges {dec_stats.initial_boundary_edge_count:,}→{dec_stats.final_boundary_edge_count:,}"
    )

    output_merge_stats = None
    if bool(config.get("merge_mesh_output", False)):
        output_merge_distance = float(config.get("merge_distance", 0.0001))
        if not np.isfinite(output_merge_distance) or output_merge_distance < 0.0:
            raise PipelineFailure("Output Mesh Cleanup distance must be finite and non-negative.")
        if output_merge_distance > 0.0:
            before_output_vertices = len(output_vertices)
            before_output_faces = len(output_faces)
            output_vertices, output_faces, output_merge_stats = merge_mesh_vertices_by_distance(
                output_vertices, output_faces, output_merge_distance
            )
            if len(output_vertices) < 3 or len(output_faces) < 1:
                raise PipelineFailure(
                    "Output Mesh Cleanup removed every valid triangle. Reduce the merge distance or choose INPUT ONLY."
                )
            log_cb(
                "Output Mesh Cleanup · Merge by Distance: "
                f"{before_output_vertices:,}→{len(output_vertices):,} vertices · "
                f"{before_output_faces:,}→{len(output_faces):,} triangles"
            )

    if stage_preview_cb is not None:
        stage_preview_cb(
            output_vertices.astype(np.float32),
            output_faces.astype(np.uint32),
            f"LI & NAN · FEATURE-PRESERVING OUTPUT · {len(output_faces):,} TRIANGLES",
        )

    _emit(progress_cb, 92, "Save Li & Nan feature-preserving OBJ")
    try:
        _write_triangle_mesh(output_path, output_vertices, output_faces)
        metrics, mesh_vertices, mesh_faces = _mesh_metrics(
            output_path, polyfit_facets=None, preview_cb=mesh_preview_cb
        )
    except Exception as exc:
        raise PipelineFailure(f"Could not save or inspect Li & Nan output mesh: {exc}") from exc

    if output_merge_stats is not None:
        metrics.update({
            "output_merge_distance": float(config.get("merge_distance", 0.0001)),
            "output_merge_vertices_merged": int(output_merge_stats.get("merged_vertices", 0)),
            "output_merge_faces_removed": int(output_merge_stats.get("removed_faces", 0)),
            "output_merge_rejected_chain_welds": int(output_merge_stats.get("rejected_chain_merges", 0)),
        })

    elapsed = float(time.perf_counter() - start_time)
    metrics.update({
        "reconstruction_method": "Li & Nan 2021 · bilateral filtering + planar regions/contours + hierarchical QEM",
        "requested_engine": str(config.get("requested_engine", "LI_NAN")).upper(),
        "resolved_engine": "LI_NAN",
        "texture_bake_source": str(config.get("texture_bake_source", input_path)),
        "feature_preservation_enabled": True,
        "bilateral_filter_enabled": True,
        "planar_region_extraction_enabled": True,
        "contour_projection_enabled": True,
        "region_plane_quadrics_enabled": True,
        "boundary_lock_enabled": bool(boundary_lock),
        "overlap_collapse_guard_enabled": True,
        "input_vertices": int(len(vertices)),
        "input_triangles": int(len(faces)),
        "filtered_vertices": int(len(filtered_vertices)),
        "filtered_triangles": int(len(filtered_faces)),
        "plane_count": int(region_count),
        "plane_coverage": plane_coverage,
        "coarse_region_count": int(planar_stats.coarse_region_count),
        "refined_planar_region_count": int(planar_stats.refined_region_count),
        "intra_contour_edge_count": int(planar_stats.contour_edge_count),
        "projected_contour_edge_count": int(planar_stats.projected_contour_count),
        "ungrouped_face_count": int(planar_stats.ungrouped_face_count),
        "filter_iterations": int(filter_stats.completed_iterations),
        "filter_max_vertex_displacement": float(filter_stats.final_max_vertex_displacement),
        "filter_guarded_max_vertex_displacement": float(guarded_max_displacement),
        "filter_overlap_guard_enabled": True,
        "filter_guard_candidate_pair_count": int(filter_guard_stats.get("candidate_pair_count", 0)),
        "filter_guard_guarded_pair_count": int(filter_guard_stats.get("guarded_pair_count", 0)),
        "filter_guard_unresolved_pair_count": int(filter_guard_stats.get("unresolved_pair_count", 0)),
        "filter_guard_attenuated_vertex_count": int(filter_guard_stats.get("attenuated_vertex_count", 0)),
        "filter_guard_minimum_vertex_scale": float(filter_guard_stats.get("minimum_vertex_scale", 1.0)),
        "filter_normal_angle_degrees": float(filter_angle),
        "region_normal_angle_degrees": float(region_angle),
        "small_region_area_threshold": float(area_threshold),
        "fitting_threshold": float(fit_threshold),
        "thresholds_auto": bool(requested_area_threshold == 0.0 or requested_fit_threshold == 0.0),
        "automatic_fallback_from": str(config.get("automatic_fallback_from", "")),
        "automatic_fallback_reason": str(config.get("automatic_fallback_reason", "")),
        "automatic_fallback_input_source": str(config.get("automatic_fallback_input_source", "")),
        "target_vertices_requested": int(target_vertices),
        "target_vertices_reached": bool(dec_stats.target_reached),
        "successful_edge_collapses": int(dec_stats.successful_collapses),
        "initial_components": int(dec_stats.initial_component_count),
        "final_components": int(dec_stats.final_component_count),
        "initial_boundary_edges": int(dec_stats.initial_boundary_edge_count),
        "final_boundary_edges": int(dec_stats.final_boundary_edge_count),
        "initial_nonmanifold_edges": int(dec_stats.initial_non_manifold_edge_count),
        "final_nonmanifold_edges": int(dec_stats.final_non_manifold_edge_count),
        "rejected_collapse_boundary_nonmanifold": int(dec_stats.rejected_boundary_or_non_manifold),
        "rejected_collapse_link_condition": int(dec_stats.rejected_link_condition),
        "rejected_collapse_duplicate_face": int(dec_stats.rejected_duplicate_face),
        "rejected_collapse_flip_degenerate": int(dec_stats.rejected_face_flip_or_degenerate),
        "collapse_intersection_guard_enabled": bool(dec_stats.collapse_intersection_guard_enabled),
        "rejected_collapse_new_crossing": int(dec_stats.rejected_collapse_new_crossing),
        "rejected_collapse_new_near_coplanar": int(dec_stats.rejected_collapse_new_near_coplanar),
        "rejected_collapse_gap_reduction": int(dec_stats.rejected_collapse_gap_reduction),
        "collapse_guard_candidate_pairs_measured": int(dec_stats.collapse_guard_candidate_pairs_measured),
        "collapse_guard_index_rebuilds": int(dec_stats.collapse_guard_index_rebuilds),
        "qem_topology_build_seconds": float(dec_stats.topology_build_seconds),
        "qem_component_check_seconds": float(dec_stats.component_check_seconds),
        "qem_quadric_setup_seconds": float(dec_stats.quadric_setup_seconds),
        "qem_initial_queue_seconds": float(dec_stats.initial_queue_seconds),
        "qem_collapse_seconds": float(dec_stats.collapse_seconds),
        "qem_queue_compaction_count": int(dec_stats.queue_compaction_count),
        "qem_queue_stale_entries_removed": int(dec_stats.queue_stale_entries_removed),
        "qem_queue_peak_entries": int(dec_stats.queue_peak_entries),
        "qem_queue_placement_reuses": int(dec_stats.queue_placement_reuses),
        "qem_queue_cached_cost_requeues": int(dec_stats.queue_cached_cost_requeues),
        "qem_adaptive_tier": str(dec_stats.adaptive_qem_tier),
        "qem_topology_storage_mode": str(dec_stats.topology_storage_mode),
        "qem_adaptive_queue_compact_min_entries": int(dec_stats.adaptive_queue_compact_min_entries),
        "qem_adaptive_queue_stale_ratio": float(dec_stats.adaptive_queue_stale_ratio),
        "qem_adaptive_queue_stale_floor": int(dec_stats.adaptive_queue_stale_floor),
        "qem_adaptive_queue_stale_divisor": int(dec_stats.adaptive_queue_stale_divisor),
        "qem_adaptive_queue_memory_budget_bytes": int(dec_stats.adaptive_queue_memory_budget_bytes),
        "qem_queue_memory_compaction_count": int(dec_stats.queue_memory_compaction_count),
        "qem_adaptive_progress_callback_stride": int(dec_stats.adaptive_progress_callback_stride),
        "qem_progress_callback_count": int(dec_stats.progress_callback_count),
        "qem_estimated_topology_bytes": int(dec_stats.estimated_topology_bytes),
        "qem_estimated_queue_entry_bytes": int(dec_stats.estimated_queue_entry_bytes),
        "qem_estimated_queue_peak_bytes": int(dec_stats.estimated_queue_peak_bytes),
        "qem_estimated_queue_stale_peak_bytes": int(dec_stats.estimated_queue_stale_peak_bytes),
        "qem_final_validation_seconds": float(dec_stats.final_validation_seconds),
        "elapsed_seconds": elapsed,
    })
    _emit(progress_cb, 100, "Li & Nan feature-preserving simplification complete")
    log_cb(f"Li & Nan output written: {output_path}")
    return {
        "output_path": str(output_path),
        "metrics": metrics,
        "mesh_preview": {"vertices": mesh_vertices, "faces": mesh_faces},
        "plane_stats": {
            "plane_count": int(region_count),
            "coverage": plane_coverage,
            "point_count": int(len(vertices)),
            "distance_threshold": 0.0,
        },
        "elapsed_seconds": elapsed,
    }

def run_reconstruction(config: dict, progress_cb, log_cb, preview_cb, mesh_preview_cb=None, stage_preview_cb=None) -> dict:
    start = time.perf_counter()
    input_path = Path(config["input_path"])
    output_path = Path(config["output_path"])
    if __package__:
        from .workspace import mesh_work_dir
    else:
        from workspace import mesh_work_dir
    work_dir = Path(config.get("work_dir") or mesh_work_dir(input_path))
    work_dir.mkdir(parents=True, exist_ok=True)
    vg_path = work_dir / (input_path.stem + "_autoplanes.vg")

    suffix = input_path.suffix.lower()
    requested = str(config.get("engine_mode", "auto")).lower()
    if requested not in {"auto", "polyfit", "li_nan", "linan", "feature_preserving"}:
        raise PipelineFailure(
            f"Unsupported reconstruction mode '{requested}'. Supported modes are AUTO, PolyFit, and Li & Nan 2021."
        )
    plane_stats = {"plane_count": 0, "coverage": 0.0, "point_count": 0, "distance_threshold": 0.0}
    points = colors = None
    groups: list[dict] = []
    normals = None
    mesh_input_stats: Optional[dict] = None
    polyfit = None

    if requested in {"li_nan", "linan", "feature_preserving"}:
        linan_config = dict(config)
        linan_config.setdefault("requested_engine", requested.upper())
        return _run_li_nan_feature_preserving(
            input_path=input_path,
            output_path=output_path,
            config=linan_config,
            start_time=start,
            progress_cb=progress_cb,
            log_cb=log_cb,
            mesh_preview_cb=mesh_preview_cb,
            stage_preview_cb=stage_preview_cb,
        )
    if suffix in {".vg", ".bvg"}:
        try:
            polyfit = load_polyfit_binding(allow_repair=not getattr(sys, "frozen", False))
        except PolyFitBindingError as exc:
            raise PipelineFailure(str(exc)) from exc
        if str(config.get("source_up_axis", "y")).lower() != "z":
            raise PipelineFailure("VG/BVG inputs require Z-up because plane-group normals and equations are stored in the file.")
        _emit(progress_cb, 4, "A · Input point set · validate existing planar segments")
        if suffix == ".vg":
            try:
                points, colors = load_points(input_path)
                preview_cb(points, colors)
                plane_stats["point_count"] = len(points)
            except Exception as exc:
                log_cb(f".vg preview skipped: {exc}")
        vg_path = input_path
        engine = "polyfit"
        log_cb(f"Existing PolyFit segmented input: {input_path.name}")
    else:
        _emit(progress_cb, 3, "A · Input point cloud / sample source mesh surface")
        merge_enabled = bool(config.get("merge_mesh_input", config.get("merge_mesh_vertices", False)))
        requested_merge_distance = float(config.get("merge_distance", 0.0001)) if merge_enabled else 0.0
        if requested_merge_distance < 0.0:
            raise PipelineFailure("Merge by Distance must be greater than zero.")
        mesh_input_stats = _inspect_mesh_input(input_path, merge_distance=requested_merge_distance)

        if requested == "auto" and mesh_input_stats:
            log_cb(
                "AUTO primary engine: PolyFit "
                f"({mesh_input_stats['vertices']:,} source vertices / {mesh_input_stats['faces']:,} triangles). "
                "If the isolated PolyFit worker fails, times out, or exits unexpectedly, the supervisor retries "
                "with Li & Nan 2021 on the original input mesh."
            )

        try:
            polyfit = load_polyfit_binding(allow_repair=not getattr(sys, "frozen", False))
        except PolyFitBindingError as exc:
            raise PipelineFailure(str(exc)) from exc
        effective_merge_distance = requested_merge_distance if mesh_input_stats else 0.0
        mesh_sample_count = int(config.get("max_points", 180000)) or 300000
        if mesh_input_stats:
            mesh_sample_count = _mesh_sample_budget(
                mesh_input_stats["faces"], int(config.get("max_points", 180000))
            )
            log_cb(
                "Original mesh topology: "
                f"{mesh_input_stats['vertices']:,} vertices / {mesh_input_stats['faces']:,} faces / "
                f"{mesh_input_stats['components']} components / "
                f"{mesh_input_stats['boundary_edges']:,} boundary edges"
            )
            if mesh_input_stats.get("seam_welded_vertices", 0) > 0:
                log_cb(
                    "GLB/mesh duplicate-position vertex indices restored: "
                    f"{mesh_input_stats['seam_welded_vertices']:,} vertices welded "
                    "(12-decimal position match; geometry positions unchanged)"
                )
            if effective_merge_distance > 0.0:
                log_cb(
                    "Merge by Distance applied: "
                    f"{mesh_input_stats['original_vertices']:,} → {mesh_input_stats['vertices']:,} vertices "
                    f"({mesh_input_stats['merged_vertices']:,} vertices merged) · "
                    f"distance {effective_merge_distance:g} · "
                    f"removed triangles {mesh_input_stats['removed_faces']:,} · "
                    f"rejected transitive chain welds {mesh_input_stats.get('merge_rejected_chain_welds', 0):,}"
                )
            if not mesh_input_stats["watertight"]:
                log_cb(
                    "The source mesh is already open. Li & Nan processes the input topology directly; pre-existing boundaries and disconnected shells are not artificially closed."
                )
            if bool(config.get("preserve_small_features", True)):
                log_cb(
                    f"Small-component preservation sampling: target at least 128 points per connected component · "
                    f"{mesh_input_stats['components']} components total"
                )
            log_cb(f"Mesh-complexity surface sample count: {mesh_sample_count:,} (source faces × 16, user maximum applied)")
        points, colors = load_points(
            input_path, mesh_sample_count=mesh_sample_count,
            merge_distance=effective_merge_distance,
        )
        points = points_to_source_local(
            points, config.get("source_frame_transform")
        )
        if mesh_input_stats:
            config["mesh_input_stats"] = mesh_input_stats
        points = convert_up_axis(points, config.get("source_up_axis", "y"))
        raw_count = len(points)
        requested_voxel_size = float(config.get("voxel_size", 0.0))
        effective_voxel_size = requested_voxel_size
        if mesh_input_stats and bool(config.get("preserve_small_features", True)) and requested_voxel_size > 0.0:

            effective_voxel_size = 0.0
            log_cb("Small mesh-component preservation: skipping global voxel downsampling.")
        points, colors = reduce_points(
            points, colors,
            int(config.get("max_points", 180000)),
            effective_voxel_size,
        )
        log_cb(f"Input loaded: {raw_count:,} → {len(points):,} points")
        if suffix in {".obj", ".stl", ".off", ".glb", ".gltf", ".3mf"} and raw_count > 3000:
            log_cb("Mesh surface sampled uniformly by triangle area.")
        preview_cb(points, colors)

        _emit(progress_cb, 10, "B · Planar segments · detect shared planar structure")
        try:
            groups, normals, plane_stats = extract_planes_ransac(
                points,
                distance_threshold=float(config.get("distance_threshold", 0.0)),
                min_plane_points=int(config.get("min_plane_points", 250)),
                max_planes=int(config.get("max_planes", 48 if config.get("preserve_small_features", True) else 24)),
                iterations=int(config.get("ransac_iterations", 350)),
                progress=lambda value, message: _emit(progress_cb, value, message),
                preserve_small_features=bool(config.get("preserve_small_features", True)),
                small_feature_min_points=int(config.get("small_feature_min_points", 40)),
                stage_preview_cb=stage_preview_cb,
            )
            detected_engine, reason, normal_clusters = choose_reconstruction_engine(
                groups, plane_stats, mesh_input_stats=mesh_input_stats
            )
            plane_stats["normal_clusters"] = normal_clusters
            if bool(config.get("preserve_small_features", True)):
                log_cb(
                    f"Small-plane preservation: minimum {int(config.get('small_feature_min_points', 40))} points · "
                    f"Detected planes: {len(groups)} / limit: {int(config.get('max_planes', 48))}"
                )
            if plane_stats.get("plane_limit_reached"):
                log_cb(
                    "Warning: segmentation stopped early after reaching the maximum plane count. "
                    f"{plane_stats.get('remaining_points', 0):,} unclassified points remain. "
                    "Increase the maximum plane count and analyze again."
                )
            log_cb(f"Shared planar analysis: {reason}")
            if requested == "auto":
                engine = "polyfit"
            else:
                engine = requested
                log_cb(f"Keeping requested engine: {engine.upper()} · shared planar fitting enabled")
        except PipelineFailure as exc:
            if requested == "polyfit":
                raise
            engine = "polyfit"
            log_cb(f"Plane analysis warning: continuing with PolyFit; {exc}")

    if suffix not in {".vg", ".bvg"}:
        if not groups:
            raise PipelineFailure("PolyFit was selected, but there are not enough planar candidates. Try Li & Nan 2021 for a triangle mesh, or provide more planar samples for PolyFit.")
        log_cb(
            f"PolyFit geometry suitability: {plane_stats['plane_count']} planes · "
            f"coverage {plane_stats['coverage'] * 100:.1f}% · "
            f"normal clusters {plane_stats.get('normal_clusters', _normal_cluster_count(groups))}"
        )
        vg_validation = _validate_polyfit_vg_groups(groups, len(points))
        log_cb(
            "Pre-write PolyFit VG validation: "
            f"{vg_validation['groups']:,} groups · "
            f"{vg_validation['assigned_indices']:,} unique sample indices · "
            f"{vg_validation['point_count']:,} points"
        )
        _emit(progress_cb, 42, "Create PolyFit VG input file")
        write_vg_ascii(
            vg_path, points, colors, groups, normals,
            progress=lambda v: _emit(progress_cb, 42 + int(8 * v / 40), "Writing PolyFit VG file"),
        )
        log_cb(f"Saved VG input: {vg_path}")

    _emit(progress_cb, 51, "Initialize PolyFit engine")
    if polyfit is None:
        try:
            polyfit = load_polyfit_binding(allow_repair=not getattr(sys, "frozen", False))
        except PolyFitBindingError as exc:
            raise PipelineFailure(str(exc)) from exc
    polyfit.initialize()

    _emit(progress_cb, 53, "Read VG point cloud / segments")
    point_cloud = polyfit.read_point_set(str(vg_path))
    if not point_cloud:
        raise PipelineFailure("PolyFit could not read the input VG/BVG file. Check the segment format.")
    expected_vg_points = len(points) if suffix not in {".vg", ".bvg"} else None
    expected_vg_groups = len(groups) if suffix not in {".vg", ".bvg"} else None
    vg_loaded = _validate_loaded_polyfit_point_cloud(
        point_cloud, expected_vg_points, expected_vg_groups, log_cb
    )
    plane_stats["polyfit_groups"] = vg_loaded["group_count"]
    plane_stats["polyfit_loaded_points"] = vg_loaded["point_count"]
    if suffix in {".vg", ".bvg"}:
        plane_stats["plane_count"] = int(vg_loaded["group_count"])

    log_cb(
        "Running the official PolyFit Python Example_2 stepwise pipeline: "
        "HypothesisGenerator → refine_planes → generate → compute_confidences "
        "→ FaceSelection.optimize(SCIP)"
    )

    mesh = _run_official_polyfit_steps(
        polyfit, point_cloud, config, work_dir, log_cb,
        progress_cb=progress_cb, stage_preview_cb=stage_preview_cb,
    )
    facets = int(mesh.size_of_facets())
    log_cb(f"Official PolyFit FaceSelection result: {facets:,} native facets")

    _emit(progress_cb, 82, "Save official PolyFit reconstructed model")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not polyfit.save_mesh(str(output_path), mesh):
        raise PipelineFailure("Official PolyFit failed to save the OBJ file.")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise PipelineFailure("The official PolyFit output OBJ was not created.")

    if progress_cb is not None:
        _emit(progress_cb, 86, "E · Reconstructed model (official PolyFit output)")
    stage_e_metrics = _preview_native_reconstructed_model(
        output_path, facets, stage_preview_cb, log_cb,
    )

    e_stage_metrics = _apply_e_stage(
        output_path, config, log_cb, progress_cb=progress_cb,
        stage_preview_cb=stage_preview_cb,
        planar_tolerance=_planar_dissolve_tolerance(points, plane_stats),
        max_normal_angle_degrees=1.0 if points is not None else 0.081,
    )
    metrics, mesh_vertices, mesh_faces = _inspect_native_polyfit_output(
        output_path, facets, log_cb,
        stage_preview_cb=stage_preview_cb, mesh_preview_cb=mesh_preview_cb,
        e_stage_metrics=e_stage_metrics,
        color_source_points=points, color_source_values=colors,
    )
    metrics.update(stage_e_metrics)
    metrics["polyfit_plane_groups"] = int(plane_stats.get("polyfit_groups", plane_stats.get("plane_count", 0)))
    metrics["polyfit_source_faces"] = int((mesh_input_stats or {}).get("faces", 0))
    metrics["polyfit_source_watertight"] = bool((mesh_input_stats or {}).get("watertight", False))
    if mesh_input_stats and float(mesh_input_stats.get("surface_area", 0.0) or 0.0) > 1e-12:
        metrics["polyfit_source_area_ratio"] = float(
            metrics.get("surface_area", 0.0) / mesh_input_stats["surface_area"]
        )
    metrics["requested_engine"] = requested.upper()
    metrics["elapsed_seconds"] = float(time.perf_counter() - start)
    metrics["input_points"] = int(plane_stats.get("point_count", len(points) if points is not None else 0))
    metrics["plane_count"] = int(plane_stats.get("polyfit_groups", plane_stats.get("plane_count", 0)))
    metrics["plane_coverage"] = None if suffix in {".vg", ".bvg"} else float(plane_stats.get("coverage", 0.0))
    metrics["distance_threshold"] = float(plane_stats.get("distance_threshold", 0.0))
    metrics["vg_path"] = str(vg_path)
    _emit(progress_cb, 100, "Complete")
    log_cb(f"Output written: {output_path}")
    return {
        "output_path": str(output_path),
        "metrics": metrics,
        "mesh_preview": {"vertices": mesh_vertices, "faces": mesh_faces},
        "plane_stats": plane_stats,
        "elapsed_seconds": metrics["elapsed_seconds"],
    }

def format_exception(exc: BaseException) -> str:
    return f"{exc}\n\n--- Full traceback ---\n{traceback.format_exc()}"
