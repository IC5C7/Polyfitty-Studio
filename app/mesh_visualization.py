from __future__ import annotations

"""Mesh-normal visualization helpers used by the viewport only."""

import numpy as np

FRONT_FACE_COLOR = np.array([0.24, 0.68, 1.00, 1.0], dtype=np.float32)
BACK_FACE_COLOR = np.array([0.95, 0.12, 0.12, 1.0], dtype=np.float32)

POLYFIT_RECONSTRUCTED_COLOR = np.array([1.00, 0.86, 0.10, 0.96], dtype=np.float32)
STUDIO_POSTPROCESS_COLOR = np.array([0.16, 0.78, 0.48, 0.96], dtype=np.float32)
STUDIO_POST_E1_COLOR = np.array([1.00, 0.60, 0.16, 0.92], dtype=np.float32)
STUDIO_POST_E2_COLOR = np.array([0.46, 0.62, 1.00, 0.94], dtype=np.float32)

def processing_stage_color(label: str) -> np.ndarray | None:
    """Return the solid RGBA color for a named live processing stage.

    A ``None`` result means the caller should use its per-face intermediate
    palette (for example, planar candidates and SCIP-selected faces).
    """
    normalized = str(label or "").strip().upper()
    if "E · RECONSTRUCTED MODEL" in normalized or normalized.startswith("E RECONSTRUCTED MODEL"):
        return POLYFIT_RECONSTRUCTED_COLOR.copy()
    if "STUDIO POSTPROCESS COMPLETE" in normalized:
        return STUDIO_POSTPROCESS_COLOR.copy()
    if "STUDIO POST E1" in normalized or normalized.startswith("STUDIO POST E1"):
        return STUDIO_POST_E1_COLOR.copy()
    if "STUDIO POST E2" in normalized or normalized.startswith("STUDIO POST E2"):
        return STUDIO_POST_E2_COLOR.copy()
    return None

def apply_layer_alpha(colors: np.ndarray, surface_mix: float, layer: str) -> np.ndarray:
    """Copy an RGBA buffer with the A↔B comparison weight applied to alpha."""
    values = np.asarray(colors, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError("Layer colors must have shape (N, 4).")
    mix = float(np.clip(surface_mix, 0.0, 1.0))
    weight = (1.0 - mix) if str(layer).lower() == "source" else mix
    values[:, 3] *= weight
    return np.ascontiguousarray(values, dtype=np.float32)

def valid_triangles(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    valid = (tri >= 0).all(axis=1) & (tri < len(verts)).all(axis=1)
    tri = tri[valid]
    if len(tri):
        xyz = verts[tri]
        valid_finite = np.isfinite(xyz).all(axis=(1, 2))
        tri = tri[valid_finite]
    return verts, tri

def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    verts, tri = valid_triangles(vertices, faces)
    if len(tri) == 0:
        return np.empty((0, 3), dtype=np.float32)
    xyz = verts[tri]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    good = lengths > 1e-15
    normals = np.zeros_like(cross)
    normals[good] = cross[good] / lengths[good, None]
    return normals.astype(np.float32)

def normal_face_colors(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = face_normals(vertices, faces)
    if len(normals) == 0:
        return np.empty((0, 4), dtype=np.float32)
    rgb = np.clip(normals * 0.5 + 0.5, 0.0, 1.0)
    return np.column_stack((rgb, np.ones(len(rgb), dtype=np.float32))).astype(np.float32)

def face_orientation_colors(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Color correctly oriented faces blue and flipped/back-facing faces red.

    Adjacent triangles are propagated using shared-edge winding. Closed components
    are additionally checked with signed volume so a globally inward component is
    also shown red. Open meshes are still checked for local winding conflicts.
    """
    verts, tri = valid_triangles(vertices, faces)
    count = len(tri)
    if count == 0:
        return np.empty((0, 4), dtype=np.float32)

    edge_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for fi, (a, b, c) in enumerate(tri):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(min(u, v)), int(max(u, v)))
            direction = 1 if int(u) < int(v) else -1
            edge_owners.setdefault(key, []).append((fi, direction))

    adjacency: list[list[tuple[int, bool]]] = [[] for _ in range(count)]
    open_face = np.zeros(count, dtype=bool)
    for owners in edge_owners.values():
        if len(owners) == 2:
            (fa, da), (fb, db) = owners
            same_direction = da == db
            adjacency[fa].append((fb, same_direction))
            adjacency[fb].append((fa, same_direction))
        else:

            for fi, _ in owners:
                open_face[fi] = True

    parity = np.full(count, -1, dtype=np.int8)
    components: list[list[int]] = []
    for seed in range(count):
        if parity[seed] >= 0:
            continue
        parity[seed] = 0
        stack = [seed]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour, needs_flip in adjacency[current]:
                expected = int(parity[current]) ^ int(needs_flip)
                if parity[neighbour] < 0:
                    parity[neighbour] = expected
                    stack.append(neighbour)
        components.append(component)

    is_back = parity.astype(bool)
    for component in components:
        ids = np.asarray(component, dtype=np.int64)
        if np.any(open_face[ids]):
            continue
        faces_oriented = tri[ids].copy()
        flip = parity[ids].astype(bool)
        faces_oriented[flip] = faces_oriented[flip][:, [0, 2, 1]]
        xyz = verts[faces_oriented]
        signed_volume6 = np.einsum(
            "ij,ij->i",
            xyz[:, 0],
            np.cross(xyz[:, 1], xyz[:, 2]),
        ).sum()
        if signed_volume6 < 0.0:

            is_back[ids] = ~is_back[ids]

    colors = np.tile(FRONT_FACE_COLOR, (count, 1))
    colors[is_back] = BACK_FACE_COLOR
    return colors

def _hsv_to_rgb(hues: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Vectorized HSV-to-RGB conversion for stable, vivid component colors."""
    h6 = (np.asarray(hues, dtype=np.float64) % 1.0) * 6.0
    sector = np.floor(h6).astype(np.int64) % 6
    fraction = h6 - np.floor(h6)
    s = np.asarray(saturation, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    p = v * (1.0 - s)
    q = v * (1.0 - fraction * s)
    t = v * (1.0 - (1.0 - fraction) * s)
    rgb = np.empty((len(h6), 3), dtype=np.float64)
    choices = (
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    )
    for idx, channels in enumerate(choices):
        mask = sector == idx
        if np.any(mask):
            rgb[mask] = np.column_stack([channel[mask] for channel in channels])
    return rgb.astype(np.float32)

def face_component_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Return one vivid, repeatable random color per edge-connected face island.

    Triangles belong to the same component when they share an undirected edge.
    Merely touching at a single vertex does not connect two face islands.
    Non-manifold edges connect every incident face, matching mesh connectivity.
    Invalid and non-finite triangles are filtered by :func:`valid_triangles`.
    """
    verts, tri = valid_triangles(vertices, faces)
    face_count = len(tri)
    if face_count == 0:
        return np.empty((0, 4), dtype=np.float32)

    edge_pairs = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]), axis=0)
    edge_lo = np.minimum(edge_pairs[:, 0], edge_pairs[:, 1]).astype(np.int64, copy=False)
    edge_hi = np.maximum(edge_pairs[:, 0], edge_pairs[:, 1]).astype(np.int64, copy=False)
    vertex_span = max(int(len(verts)), 1)
    edge_keys = edge_lo * vertex_span + edge_hi
    edge_owners = np.tile(np.arange(face_count, dtype=np.int64), 3)
    order = np.argsort(edge_keys, kind="stable")
    sorted_keys = edge_keys[order]
    sorted_owners = edge_owners[order]
    shared = np.flatnonzero(sorted_keys[1:] == sorted_keys[:-1])

    parent = np.arange(face_count, dtype=np.int64)
    rank = np.zeros(face_count, dtype=np.uint8)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    for edge_idx in shared:
        left = find(int(sorted_owners[edge_idx]))
        right = find(int(sorted_owners[edge_idx + 1]))
        if left == right:
            continue
        if rank[left] < rank[right]:
            left, right = right, left
        parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1

    roots = np.fromiter((find(i) for i in range(face_count)), dtype=np.int64, count=face_count)
    _, component_ids = np.unique(roots, return_inverse=True)
    component_count = int(component_ids.max()) + 1

    rng = np.random.default_rng(int(seed))
    offset = float(rng.uniform(0.0, 1.0))
    hues = (offset + np.arange(component_count, dtype=np.float64) * 0.6180339887498949) % 1.0
    saturation = rng.uniform(0.70, 0.90, component_count)
    value = rng.uniform(0.78, 0.98, component_count)
    component_rgb = _hsv_to_rgb(hues, saturation, value)
    component_colors = np.column_stack((component_rgb, np.ones(component_count, dtype=np.float32)))
    return np.ascontiguousarray(component_colors[component_ids], dtype=np.float32)

def connected_component_mesh_data(
    vertices: np.ndarray,
    faces: np.ndarray,
    seed: int = 0,
    max_faces: int = 250_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a mesh into flat-shaded per-face colors for viewport rendering."""
    verts, tri = valid_triangles(vertices, faces)
    if len(tri) == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint32),
                np.empty((0, 4), dtype=np.float32))
    colors = face_component_colors(verts, tri, seed=seed)
    if len(tri) > int(max_faces):
        take = np.linspace(0, len(tri) - 1, int(max_faces), dtype=np.int64)
        tri = tri[take]
        colors = colors[take]
    expanded_vertices = np.ascontiguousarray(verts[tri].reshape(-1, 3), dtype=np.float32)
    expanded_faces = np.arange(len(expanded_vertices), dtype=np.uint32).reshape(-1, 3)
    vertex_colors = np.repeat(colors, 3, axis=0)
    return expanded_vertices, expanded_faces, np.ascontiguousarray(vertex_colors, dtype=np.float32)

def diagnostic_mesh_data(
    vertices: np.ndarray,
    faces: np.ndarray,
    mode: str,
    max_faces: int = 250_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a preview-only mesh expanded per triangle with per-vertex colors."""
    verts, tri = valid_triangles(vertices, faces)
    if len(tri) == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint32),
                np.empty((0, 4), dtype=np.float32))
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "normals":
        if len(tri) > max_faces:
            take = np.linspace(0, len(tri) - 1, int(max_faces), dtype=np.int64)
            tri = tri[take]
        colors = normal_face_colors(verts, tri)
    else:

        all_colors = face_orientation_colors(verts, tri)
        if len(tri) > max_faces:
            take = np.linspace(0, len(tri) - 1, int(max_faces), dtype=np.int64)
            tri = tri[take]
            colors = all_colors[take]
        else:
            colors = all_colors
    expanded = np.ascontiguousarray(verts[tri].reshape(-1, 3), dtype=np.float32)
    expanded_faces = np.arange(len(expanded), dtype=np.uint32).reshape(-1, 3)
    vertex_colors = np.repeat(colors, 3, axis=0)
    return expanded, expanded_faces, np.ascontiguousarray(vertex_colors, dtype=np.float32)

def normal_line_positions(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_normals: int = 5000,
    length_ratio: float = 0.025,
) -> np.ndarray:
    verts, tri = valid_triangles(vertices, faces)
    if len(tri) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(tri) > max_normals:
        take = np.linspace(0, len(tri) - 1, int(max_normals), dtype=np.int64)
        tri = tri[take]
    xyz = verts[tri]
    centers = xyz.mean(axis=1)
    normals = face_normals(verts, tri).astype(np.float64)
    span = np.ptp(verts, axis=0)
    length = max(float(np.linalg.norm(span)) * float(length_ratio), 1e-6)
    tips = centers + normals * length
    return np.ascontiguousarray(np.stack((centers, tips), axis=1).reshape(-1, 3), dtype=np.float32)
