"""Small geometry helpers for live reconstruction edge previews."""

from __future__ import annotations

import numpy as np

def active_face_ids_from_vertex_motion(
    previous_vertices: np.ndarray,
    current_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_faces: int = 256,
) -> np.ndarray:
    """Pick faces containing the vertices that moved most in this iteration."""
    previous = np.asarray(previous_vertices, dtype=np.float64).reshape(-1, 3)
    current = np.asarray(current_vertices, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if previous.shape != current.shape or not len(triangles) or len(current) == 0:
        return np.empty(0, dtype=np.int64)
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(current):
        return np.empty(0, dtype=np.int64)
    vertex_motion = np.linalg.norm(current - previous, axis=1)
    face_motion = np.max(vertex_motion[triangles], axis=1)
    candidates = np.flatnonzero(np.isfinite(face_motion) & (face_motion > 0.0))
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    limit = max(1, int(max_faces))
    if len(candidates) > limit:
        candidate_motion = face_motion[candidates]
        selected = np.argpartition(candidate_motion, -limit)[-limit:]
        candidates = candidates[selected]
    order = np.lexsort((candidates, -face_motion[candidates]))
    return candidates[order].astype(np.int64, copy=False)

def face_edge_segments(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    *,
    max_edges: int = 12_000,
) -> np.ndarray:
    """Return unique perimeter edges for selected faces as (N, 2, 3) segments."""
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    selected = np.asarray(face_ids, dtype=np.int64).reshape(-1)
    if not len(verts) or not len(triangles) or not len(selected):
        return np.empty((0, 2, 3), dtype=np.float32)
    selected = selected[(selected >= 0) & (selected < len(triangles))]
    if not len(selected):
        return np.empty((0, 2, 3), dtype=np.float32)
    chosen = triangles[selected]
    chosen = chosen[(chosen >= 0).all(axis=1) & (chosen < len(verts)).all(axis=1)]
    if not len(chosen):
        return np.empty((0, 2, 3), dtype=np.float32)
    edges = np.concatenate(
        (chosen[:, [0, 1]], chosen[:, [1, 2]], chosen[:, [2, 0]]), axis=0
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    limit = max(1, int(max_edges))
    if len(edges) > limit:
        edges = edges[np.linspace(0, len(edges) - 1, limit, dtype=np.int64)]
    return np.ascontiguousarray(verts[edges].reshape(-1, 2, 3), dtype=np.float32)
