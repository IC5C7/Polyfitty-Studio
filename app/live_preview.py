from __future__ import annotations

"""Small, dependency-free geometry helpers for live planar preview patches."""

import numpy as np

def planar_group_preview_mesh(
    points: np.ndarray,
    indices: np.ndarray,
    plane: np.ndarray,
    max_points: int = 1600,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate a detected plane group with a sparse grid-triangulated patch.

    This is a visualization-only surface: every vertex is an average of actual
    inlier samples, and triangles are created only between adjacent occupied
    grid cells. It avoids a convex-hull fill across large gaps in roof/eave data.
    """
    cloud = np.asarray(points, dtype=np.float64)
    ids = np.asarray(indices, dtype=np.int64).reshape(-1)
    ids = ids[(ids >= 0) & (ids < len(cloud))]
    if len(ids) < 12:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)

    samples = cloud[ids]
    if len(samples) > max(12, int(max_points)):
        pick = np.linspace(0, len(samples) - 1, int(max_points), dtype=np.int64)
        samples = samples[pick]

    normal = np.asarray(plane, dtype=np.float64).reshape(-1)[:3]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)
    normal /= norm

    reference = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    axis_u = np.cross(normal, reference)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = np.cross(normal, axis_u)
    center = samples.mean(axis=0)
    relative = samples - center
    uv = np.column_stack((relative @ axis_u, relative @ axis_v))
    lo = uv.min(axis=0)
    hi = uv.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)

    base = int(np.clip(np.sqrt(len(samples)) * 0.9, 8, 48))
    aspect = float(span[0] / span[1])
    nx = int(np.clip(round(base * np.sqrt(aspect)), 6, 64))
    ny = int(np.clip(round(base / np.sqrt(aspect)), 6, 64))
    cell = np.floor((uv - lo) / span * np.array([nx - 1e-9, ny - 1e-9])).astype(np.int64)
    cell[:, 0] = np.clip(cell[:, 0], 0, nx - 1)
    cell[:, 1] = np.clip(cell[:, 1], 0, ny - 1)

    flat = cell[:, 1] * nx + cell[:, 0]
    unique_cells, inverse = np.unique(flat, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.zeros((len(unique_cells), 2), dtype=np.float64)
    np.add.at(sums, inverse, uv)
    uv_vertices = sums / counts[:, None]

    plane_values = np.asarray(plane, dtype=np.float64).reshape(-1)
    plane_d = float(plane_values[3]) if len(plane_values) >= 4 else -float(np.dot(normal, center))
    plane_point = center - (float(np.dot(normal, center)) + plane_d) * normal
    vertices = plane_point + uv_vertices[:, 0:1] * axis_u + uv_vertices[:, 1:2] * axis_v
    lookup = np.full(nx * ny, -1, dtype=np.int64)
    lookup[unique_cells] = np.arange(len(unique_cells), dtype=np.int64)

    faces: list[tuple[int, int, int]] = []
    for y in range(ny - 1):
        row = y * nx
        for x in range(nx - 1):
            ids4 = (
                int(lookup[row + x]),
                int(lookup[row + x + 1]),
                int(lookup[row + nx + x]),
                int(lookup[row + nx + x + 1]),
            )
            present = [value for value in ids4 if value >= 0]
            if len(present) == 4:
                a, b, c, d = ids4
                faces.append((a, b, d))
                faces.append((a, d, c))
            elif len(present) == 3:

                if ids4[0] < 0:
                    faces.append((ids4[1], ids4[3], ids4[2]))
                elif ids4[1] < 0:
                    faces.append((ids4[0], ids4[3], ids4[2]))
                elif ids4[2] < 0:
                    faces.append((ids4[0], ids4[1], ids4[3]))
                else:
                    faces.append((ids4[0], ids4[1], ids4[2]))

    if not faces:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)
    face_array = np.asarray(faces, dtype=np.int64)
    tri = vertices[face_array]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = np.isfinite(area2) & (area2 > 1e-12)
    face_array = face_array[keep]
    if len(face_array) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)
    return np.ascontiguousarray(vertices, dtype=np.float32), np.ascontiguousarray(face_array, dtype=np.uint32)

def merge_preview_meshes(
    meshes: list[tuple[np.ndarray, np.ndarray]],
    max_faces: int = 180_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate a bounded list of independent preview patches."""
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    offset = 0
    for vertices, faces in meshes:
        v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        f = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
        if len(v) == 0 or len(f) == 0:
            continue
        vertices_parts.append(v)
        faces_parts.append(f.astype(np.uint64) + offset)
        offset += len(v)
    if not faces_parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)
    vertices = np.vstack(vertices_parts)
    faces = np.vstack(faces_parts)
    limit = max(1, int(max_faces))
    if len(faces) > limit:
        keep = np.linspace(0, len(faces) - 1, limit, dtype=np.int64)
        faces = faces[keep]
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces, dtype=np.uint32)

def compact_preview_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_faces: int = 60_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a bounded, compact mesh suitable for transient viewport overlays."""
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(verts) < 3 or len(tri) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)
    valid = (tri >= 0).all(axis=1) & (tri < len(verts)).all(axis=1)
    valid &= np.isfinite(verts[np.clip(tri, 0, max(0, len(verts) - 1))]).all(axis=(1, 2))
    tri = tri[valid]
    if len(tri) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)
    limit = max(1, int(max_faces))
    if len(tri) > limit:
        selection = np.linspace(0, len(tri) - 1, limit, dtype=np.int64)
        tri = tri[selection]
    used, inverse = np.unique(tri.reshape(-1), return_inverse=True)
    return (
        np.ascontiguousarray(verts[used], dtype=np.float32),
        np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.uint32),
    )
