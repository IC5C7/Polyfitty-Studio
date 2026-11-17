from __future__ import annotations

"""Source-constrained low-poly planar fitting helpers.

The goal is to fit noisy architectural mesh faces to planes detected from the
source, without growing the source footprint or creating a dense remesh.
"""

import numpy as np

def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    reference = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, reference)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(n, u)
    return u, v

def planarize_mesh_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_points: np.ndarray,
    groups: list[dict],
    snap_tolerance: float,
    *,
    max_angle_degrees: float = 58.0,
    support_margin_ratio: float = 0.012,
    clip_outside_support: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Snap mesh faces to detected source planes while keeping in-plane extents.

    Each source plane carries a 2D footprint derived from its actual inlier
    samples. A face is snapped only when it is near and roughly aligned with a
    supported plane. Coordinates move only along the plane normal, so eave and
    roof lengths in the plane are not expanded. Optional clipping removes
    generated faces that overshoot the source plane footprint.
    """
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    cloud = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    if len(verts) == 0 or len(tris) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32), {
            "input_faces": int(len(tris)), "output_faces": 0, "planarized_faces": 0,
            "clipped_faces": 0, "plane_count": 0,
        }
    if int(tris.min()) < 0 or int(tris.max()) >= len(verts):
        raise ValueError("Face indices are outside the vertex array.")

    tri_xyz = verts[tris].copy()
    centroids = tri_xyz.mean(axis=1)
    cross = np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0])
    areas2 = np.linalg.norm(cross, axis=1)
    face_normals = cross / np.maximum(areas2[:, None], 1e-12)
    valid_face = np.isfinite(tri_xyz).all(axis=(1, 2)) & (areas2 > 1e-12)

    tolerance = max(float(snap_tolerance), 1e-10)
    cosine_limit = float(np.cos(np.deg2rad(float(max_angle_degrees))))
    best_score = np.full(len(tris), np.inf, dtype=np.float64)
    best_outside_score = np.full(len(tris), np.inf, dtype=np.float64)
    best_plane = np.full(len(tris), -1, dtype=np.int32)
    support_infos: list[dict] = []

    for group_index, group in enumerate(groups or []):
        plane = np.asarray(group.get("plane", []), dtype=np.float64).reshape(-1)
        ids = np.asarray(group.get("indices", []), dtype=np.int64).reshape(-1)
        ids = ids[(ids >= 0) & (ids < len(cloud))]
        if len(plane) < 4 or len(ids) < 3:
            continue
        normal = plane[:3]
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal = normal / norm
        d = float(plane[3] / norm)
        support = cloud[ids]
        u, v = _plane_basis(normal)
        projected_support = np.column_stack((support @ u, support @ v))
        lo = projected_support.min(axis=0)
        hi = projected_support.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        margin = max(float(np.max(span)) * float(support_margin_ratio), tolerance * 0.2)

        signed_centroid = centroids @ normal + d
        distance = np.abs(signed_centroid)
        alignment = np.abs(face_normals @ normal)
        centroid_uv = np.column_stack((centroids @ u, centroids @ v))
        inside_centroid = (
            (centroid_uv[:, 0] >= lo[0] - margin) & (centroid_uv[:, 0] <= hi[0] + margin) &
            (centroid_uv[:, 1] >= lo[1] - margin) & (centroid_uv[:, 1] <= hi[1] + margin)
        )

        corner_distance = np.abs(tri_xyz @ normal + d)
        corners_near_plane = np.max(corner_distance, axis=1) <= tolerance * 2.0
        near_aligned = (
            valid_face & (distance <= tolerance * 1.5) &
            (alignment >= cosine_limit) & corners_near_plane
        )
        eligible = near_aligned & inside_centroid & (distance <= tolerance)
        if clip_outside_support:
            outside_delta = np.maximum(
                np.maximum(lo[None, :] - centroid_uv, centroid_uv - hi[None, :]), 0.0
            )
            outside_distance = np.linalg.norm(outside_delta, axis=1)
            outside_candidate = near_aligned & ~inside_centroid
            outside_score = distance + outside_distance + (1.0 - alignment) * tolerance * 0.25
            update_outside = outside_candidate & (outside_score < best_outside_score)
            best_outside_score[update_outside] = outside_score[update_outside]
        if not np.any(eligible):
            support_infos.append({
                "normal": normal, "d": d, "u": u, "v": v,
                "lo": lo, "hi": hi, "margin": margin,
                "eligible": np.zeros(len(tris), dtype=bool),
            })
            continue

        score = distance + (1.0 - alignment) * tolerance * 0.25
        update = eligible & (score < best_score)
        best_score[update] = score[update]
        best_plane[update] = len(support_infos)
        support_infos.append({
            "normal": normal, "d": d, "u": u, "v": v,
            "lo": lo, "hi": hi, "margin": margin,
            "eligible": eligible,
        })

    assigned = best_plane >= 0
    clipped = np.zeros(len(tris), dtype=bool)
    if clip_outside_support:
        clipped = (best_plane < 0) & np.isfinite(best_outside_score)

    if clip_outside_support and np.any(assigned):
        for support_index, info in enumerate(support_infos):
            selected = np.flatnonzero(best_plane == support_index)
            if len(selected) == 0:
                continue
            local_tri = tri_xyz[selected]
            uv_tri = np.stack((local_tri @ info["u"], local_tri @ info["v"]), axis=2)
            lo, hi, margin = info["lo"], info["hi"], info["margin"]
            vertices_inside = (
                (uv_tri[:, :, 0] >= lo[0] - margin) & (uv_tri[:, :, 0] <= hi[0] + margin) &
                (uv_tri[:, :, 1] >= lo[1] - margin) & (uv_tri[:, :, 1] <= hi[1] + margin)
            ).all(axis=1)
            clipped[selected[~vertices_inside]] = True

    keep = ~clipped
    if not np.any(keep):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32), {
            "input_faces": int(len(tris)), "output_faces": 0,
            "planarized_faces": int(np.count_nonzero(assigned)),
            "clipped_faces": int(np.count_nonzero(clipped)),
            "degenerate_faces_removed": 0, "duplicate_faces_removed": 0,
            "plane_count": int(len(support_infos)),
        }

    projected_vertices = verts.copy()
    snap_faces = np.flatnonzero(assigned & keep)
    fallback_vertices = 0
    skipped_vertices = 0
    max_vertex_snap = max(tolerance * 1.5, float(np.linalg.norm(np.ptp(verts, axis=0))) * 0.00075)
    if len(snap_faces):
        pair_vertices = tris[snap_faces].reshape(-1)
        pair_planes = np.repeat(best_plane[snap_faces], 3)
        pairs = np.unique(np.column_stack((pair_vertices, pair_planes)), axis=0)
        pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
        split_at = np.flatnonzero(np.diff(pairs[:, 0])) + 1
        ranges = np.concatenate(([0], split_at, [len(pairs)]))
        parallel_cosine = float(np.cos(np.deg2rad(2.0)))
        for begin, finish in zip(ranges[:-1], ranges[1:]):
            vertex_index = int(pairs[begin, 0])
            original = verts[vertex_index]
            constraints: list[tuple[np.ndarray, float, float]] = []
            for plane_index in pairs[begin:finish, 1]:
                info = support_infos[int(plane_index)]
                normal = np.asarray(info["normal"], dtype=np.float64).copy()
                d = float(info["d"])
                dominant = int(np.argmax(np.abs(normal)))
                if normal[dominant] < 0.0:
                    normal = -normal
                    d = -d
                residual = abs(float(np.dot(normal, original) + d))
                parallel_match = next(
                    (i for i, (other, _other_d, _other_residual) in enumerate(constraints)
                     if abs(float(np.dot(normal, other))) >= parallel_cosine),
                    None,
                )
                if parallel_match is None:
                    constraints.append((normal, d, residual))
                elif residual < constraints[parallel_match][2]:

                    constraints[parallel_match] = (normal, d, residual)
            if not constraints:
                continue
            normals = np.asarray([item[0] for item in constraints], dtype=np.float64)
            offsets = np.asarray([item[1] for item in constraints], dtype=np.float64)
            rhs = -(normals @ original + offsets)
            try:

                left, singular, right_t = np.linalg.svd(normals, full_matrices=False)
                if len(singular) == 0:
                    continue
                cutoff = max(float(singular[0]) * 0.06, 1e-8)
                inverse = np.zeros_like(singular)
                np.divide(1.0, singular, out=inverse, where=singular > cutoff)
                displacement = right_t.T @ (inverse * (left.T @ rhs))
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(displacement).all():
                continue

            joint_residual = normals @ (original + displacement) + offsets
            if (
                float(np.linalg.norm(displacement)) > max_vertex_snap or
                float(np.max(np.abs(joint_residual))) > tolerance
            ):
                fallback_vertices += 1
                signed_residuals = normals @ original + offsets
                best_constraint = int(np.argmin(np.abs(signed_residuals)))
                displacement = -float(signed_residuals[best_constraint]) * normals[best_constraint]
            displacement_length = float(np.linalg.norm(displacement))
            if displacement_length <= max_vertex_snap:
                projected_vertices[vertex_index] = original + displacement
            else:
                skipped_vertices += 1

    output_faces = tris[keep].copy()

    quality_guard_restored_vertices = 0
    quality_guard_restored_faces = 0
    if len(output_faces):
        original_tri = verts[output_faces]
        original_cross = np.cross(
            original_tri[:, 1] - original_tri[:, 0],
            original_tri[:, 2] - original_tri[:, 0],
        )
        original_area2 = np.linalg.norm(original_cross, axis=1)
        original_valid = np.isfinite(original_area2) & (original_area2 > 1e-12)
        original_normal = original_cross / np.maximum(original_area2[:, None], 1e-15)

        for _quality_pass in range(8):
            candidate_tri = projected_vertices[output_faces]
            candidate_cross = np.cross(
                candidate_tri[:, 1] - candidate_tri[:, 0],
                candidate_tri[:, 2] - candidate_tri[:, 0],
            )
            candidate_area2 = np.linalg.norm(candidate_cross, axis=1)
            candidate_normal = candidate_cross / np.maximum(candidate_area2[:, None], 1e-15)
            normal_alignment = np.einsum("ij,ij->i", original_normal, candidate_normal)
            area_ratio = candidate_area2 / np.maximum(original_area2, 1e-15)
            unsafe = original_valid & (
                ~np.isfinite(candidate_area2)
                | (candidate_area2 <= 1e-12)
                | (normal_alignment < 0.5)
                | (area_ratio < 0.20)
                | (area_ratio > 5.0)
            )
            if not np.any(unsafe):
                break
            quality_guard_restored_faces += int(np.count_nonzero(unsafe))
            involved = np.unique(output_faces[unsafe].reshape(-1))
            moved = involved[
                np.linalg.norm(projected_vertices[involved] - verts[involved], axis=1) > 1e-12
            ]
            if len(moved) == 0:
                break
            projected_vertices[moved] = verts[moved]
            quality_guard_restored_vertices += int(len(moved))
        else:

            candidate_tri = projected_vertices[output_faces]
            candidate_cross = np.cross(
                candidate_tri[:, 1] - candidate_tri[:, 0],
                candidate_tri[:, 2] - candidate_tri[:, 0],
            )
            candidate_area2 = np.linalg.norm(candidate_cross, axis=1)
            candidate_normal = candidate_cross / np.maximum(candidate_area2[:, None], 1e-15)
            normal_alignment = np.einsum("ij,ij->i", original_normal, candidate_normal)
            area_ratio = candidate_area2 / np.maximum(original_area2, 1e-15)
            still_unsafe = original_valid & (
                ~np.isfinite(candidate_area2)
                | (candidate_area2 <= 1e-12)
                | (normal_alignment < 0.5)
                | (area_ratio < 0.20)
                | (area_ratio > 5.0)
            )
            if np.any(still_unsafe):
                moved = np.linalg.norm(projected_vertices - verts, axis=1) > 1e-12
                quality_guard_restored_vertices += int(np.count_nonzero(moved))
                projected_vertices[moved] = verts[moved]
                quality_guard_restored_faces += int(np.count_nonzero(still_unsafe))

    tri_out = projected_vertices[output_faces]
    area2_out = np.linalg.norm(
        np.cross(tri_out[:, 1] - tri_out[:, 0], tri_out[:, 2] - tri_out[:, 0]), axis=1
    )
    nondegenerate = np.isfinite(area2_out) & (area2_out > 1e-12)

    for _ in range(2):
        collapsed = output_faces[~nondegenerate]
        if len(collapsed) == 0:
            break
        restore = np.unique(collapsed.reshape(-1))
        projected_vertices[restore] = verts[restore]
        tri_out = projected_vertices[output_faces]
        area2_out = np.linalg.norm(
            np.cross(tri_out[:, 1] - tri_out[:, 0], tri_out[:, 2] - tri_out[:, 0]), axis=1
        )
        nondegenerate = np.isfinite(area2_out) & (area2_out > 1e-12)
    degenerate_removed = int(np.count_nonzero(~nondegenerate))
    output_faces = output_faces[nondegenerate]

    duplicate_removed = 0
    if len(output_faces):
        _, first = np.unique(np.sort(output_faces, axis=1), axis=0, return_index=True)
        first.sort()
        duplicate_removed = int(len(output_faces) - len(first))
        output_faces = output_faces[first]

    if len(output_faces):
        used_vertices, inverse = np.unique(output_faces.reshape(-1), return_inverse=True)
        output_vertices = projected_vertices[used_vertices]
        output_faces = inverse.reshape(-1, 3)
    else:
        output_vertices = np.empty((0, 3), dtype=np.float64)
        output_faces = np.empty((0, 3), dtype=np.int64)

    return (
        np.ascontiguousarray(output_vertices, dtype=np.float32),
        np.ascontiguousarray(output_faces, dtype=np.uint32),
        {
            "input_faces": int(len(tris)),
            "output_faces": int(len(output_faces)),
            "planarized_faces": int(np.count_nonzero(assigned & keep)),
            "clipped_faces": int(np.count_nonzero(clipped)),
            "degenerate_faces_removed": degenerate_removed,
            "duplicate_faces_removed": duplicate_removed,
            "projected_vertices": int(np.count_nonzero(np.linalg.norm(projected_vertices - verts, axis=1) > 1e-12)),
            "fallback_vertices": int(fallback_vertices),
            "skipped_vertices": int(skipped_vertices),
            "quality_guard_restored_vertices": int(quality_guard_restored_vertices),
            "quality_guard_restored_faces": int(quality_guard_restored_faces),
            "max_vertex_displacement": float(np.max(np.linalg.norm(projected_vertices - verts, axis=1))) if len(verts) else 0.0,
            "plane_count": int(len(support_infos)),
        },
    )

def build_lowpoly_plane_patches(
    points: np.ndarray,
    groups: list[dict],
    max_faces: int = 12000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build bounded low-poly planar patches directly from detected source planes."""
    try:
        from .live_preview import planar_group_preview_mesh, merge_preview_meshes
    except ImportError:
        from live_preview import planar_group_preview_mesh, merge_preview_meshes

    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(cloud) == 0 or not groups:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32), {
            "plane_count": 0, "output_faces": 0,
        }

    per_group_points = max(48, min(420, int(max(1, max_faces) / max(1, len(groups)) * 0.42)))
    patches = []
    used_groups = 0
    for group in groups:
        vertices, faces = planar_group_preview_mesh(
            cloud, np.asarray(group.get("indices", []), dtype=np.int64),
            np.asarray(group.get("plane", []), dtype=np.float64),
            max_points=per_group_points,
        )
        if len(faces):
            patches.append((vertices, faces))
            used_groups += 1
    vertices, faces = merge_preview_meshes(patches, max_faces=max_faces)
    return vertices, faces, {
        "plane_count": int(used_groups), "output_faces": int(len(faces)),
    }
