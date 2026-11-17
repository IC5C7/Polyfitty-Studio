from __future__ import annotations

import numpy as np

DEFAULT_MAX_WIREFRAME_EDGES = 600_000

def blend_surface_opacity(value: float, minimum: float = 0.0, maximum: float = 1.0) -> tuple[float, float]:
    """Return complementary source/result surface alpha values for a 0..100 slider."""
    mix = float(np.clip(value, 0.0, 100.0)) / 100.0
    low = float(np.clip(minimum, 0.0, 1.0))
    high = float(np.clip(maximum, low, 1.0))
    return high - (high - low) * mix, low + (high - low) * mix

def blend_wireframe_alpha(value: float) -> tuple[float, float]:
    """Return complementary source/result wireframe alpha values for a 0..100 slider."""
    mix = float(np.clip(value, 0.0, 100.0)) / 100.0
    return 1.0 - mix, mix

def build_wireframe_positions(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_edges: int = DEFAULT_MAX_WIREFRAME_EDGES,
) -> np.ndarray | None:
    """Build de-duplicated triangle edges as GL_LINES endpoint positions.

    The returned array has shape (2 * edge_count, 3), where every consecutive
    pair of points represents one independent line segment.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(verts) == 0 or len(tri) == 0:
        return None

    valid = (tri >= 0).all(axis=1) & (tri < len(verts)).all(axis=1)
    tri = tri[valid]
    if len(tri) == 0:
        return None

    edges = np.concatenate(
        (tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]),
        axis=0,
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)

    limit = max(1, int(max_edges))
    if len(edges) > limit:
        sample = np.linspace(0, len(edges) - 1, limit, dtype=np.int64)
        edges = edges[sample]

    return np.ascontiguousarray(verts[edges].reshape(-1, 3), dtype=np.float32)

def build_polygon_wireframe_positions(
    vertices: np.ndarray,
    triangles: np.ndarray,
    quads: np.ndarray,
    max_edges: int = DEFAULT_MAX_WIREFRAME_EDGES,
) -> np.ndarray | None:
    """Build unique polygon perimeter edges from mixed triangles and quads.

    Unlike triangulating a quad first and passing it to the triangle wireframe
    builder, this function never creates a visible diagonal across a quad.
    """
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tri = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    quad = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    if len(verts) == 0 or (len(tri) == 0 and len(quad) == 0):
        return None
    parts = []
    if len(tri):
        tri = tri[(tri >= 0).all(axis=1) & (tri < len(verts)).all(axis=1)]
        if len(tri):
            parts.extend((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    if len(quad):
        quad = quad[(quad >= 0).all(axis=1) & (quad < len(verts)).all(axis=1)]
        if len(quad):
            parts.extend((quad[:, [0, 1]], quad[:, [1, 2]], quad[:, [2, 3]], quad[:, [3, 0]]))
    if not parts:
        return None
    edges = np.vstack(parts)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    limit = max(1, int(max_edges))
    if len(edges) > limit:
        sample = np.linspace(0, len(edges) - 1, limit, dtype=np.int64)
        edges = edges[sample]
    return np.ascontiguousarray(verts[edges].reshape(-1, 3), dtype=np.float32)

def build_sharp_edge_positions(
    vertices: np.ndarray,
    faces: np.ndarray,
    threshold_degrees: float = 30.0,
    max_edges: int = 250_000,
) -> np.ndarray | None:
    """Return boundary and crease edges as GL_LINES endpoint positions.

    A crease is an edge shared by exactly two valid triangles whose face-normal
    angle is greater than ``threshold_degrees``. Boundary and non-manifold edges
    are included because they are important visible topology boundaries.
    """
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(verts) == 0 or len(tri) == 0:
        return None

    valid = (tri >= 0).all(axis=1) & (tri < len(verts)).all(axis=1)
    tri = tri[valid]
    if len(tri) == 0:
        return None

    triangle_points = verts[tri]
    finite = np.isfinite(triangle_points).all(axis=(1, 2))
    tri = tri[finite]
    triangle_points = triangle_points[finite]
    if len(tri) == 0:
        return None

    cross = np.cross(
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 0],
    )
    lengths = np.linalg.norm(cross, axis=1)
    nondegenerate = lengths > 1e-12
    tri = tri[nondegenerate]
    cross = cross[nondegenerate]
    lengths = lengths[nondegenerate]
    if len(tri) == 0:
        return None
    normals = cross / lengths[:, None]

    edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]), axis=0)
    owners = np.tile(np.arange(len(tri), dtype=np.int64), 3)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_owners = owners[order]

    starts = np.r_[True, np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)]
    start_indices = np.flatnonzero(starts)
    counts = np.diff(np.r_[start_indices, len(sorted_edges)])
    threshold = float(np.clip(threshold_degrees, 0.0, 180.0))

    selected_groups = np.flatnonzero(counts != 2)
    manifold_groups = np.flatnonzero(counts == 2)
    if len(manifold_groups):
        starts_for_manifold = start_indices[manifold_groups]
        face_a = sorted_owners[starts_for_manifold]
        face_b = sorted_owners[starts_for_manifold + 1]
        cosine = np.clip(np.einsum("ij,ij->i", normals[face_a], normals[face_b]), -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        crease_groups = manifold_groups[angles > threshold]
        if len(crease_groups):
            selected_groups = np.concatenate((selected_groups, crease_groups))

    if len(selected_groups) == 0:
        return None
    selected = sorted_edges[start_indices[np.sort(selected_groups)]]
    limit = max(1, int(max_edges))
    if len(selected) > limit:
        sample = np.linspace(0, len(selected) - 1, limit, dtype=np.int64)
        selected = selected[sample]
    return np.ascontiguousarray(verts[selected].reshape(-1, 3), dtype=np.float32)

def viewport_render_visibility(
    render_mode: str,
    wireframe_enabled: bool,
    sharp_edges_enabled: bool,
    connected_face_islands_enabled: bool,
) -> dict[str, bool]:
    """Resolve base render layers and independent viewport diagnostic overlays.

    PBR/Albedo and Solid Surface select the base fill implementation. The three
    viewport controls are separate diagnostic layers and never inherit a
    visibility restriction from the selected render mode.
    """
    mode = str(render_mode or "diffuse_albedo").strip().lower().replace(" ", "_")
    if mode == "original_color":
        mode = "diffuse_albedo"
    if mode not in {"pbr", "diffuse_albedo", "surface", "wireframe"}:
        mode = "diffuse_albedo"
    islands = bool(connected_face_islands_enabled)
    return {
        "base_surface": mode == "surface" and not islands,
        "material_surface": mode in {"pbr", "diffuse_albedo"} and not islands,
        "wireframe": bool(wireframe_enabled),
        "sharp_edges": bool(sharp_edges_enabled),
        "connected_face_islands": islands,
    }

def viewport_overlay_visibility(
    render_mode: str,
    wireframe_enabled: bool,
    sharp_edges_enabled: bool,
    connected_face_islands_enabled: bool,
) -> dict[str, bool]:
    """Backward-compatible overlay-only view of :func:`viewport_render_visibility`."""
    state = viewport_render_visibility(
        render_mode, wireframe_enabled, sharp_edges_enabled,
        connected_face_islands_enabled,
    )
    return {
        "wireframe": state["wireframe"],
        "sharp_edges": state["sharp_edges"],
        "connected_face_islands": state["connected_face_islands"],
    }

def infront_gl_options() -> dict:
    """Legacy-compatible description of an alpha-blended In Front layer.

    The VTK viewport backend translates the disabled-depth-test flag into a
    dedicated overlay renderer with a clean depth buffer and preserved color
    buffer. The OpenGL enum keys remain here only as a tiny compatibility token.
    """

    return {
        0x0BE2: True,
        0x0B71: False,
        0x0B44: False,
        "glBlendFunc": (0x0302, 0x0303),
    }

def infront_applies(mode: str, item_kind: str) -> bool:
    """Whether In Front depth bypass should apply to a scene item kind."""
    normalized_mode = str(mode).strip().lower()
    normalized_kind = str(item_kind).strip().lower()
    if normalized_mode == "both":
        return normalized_kind in {"surface", "wireframe"}
    return normalized_mode == normalized_kind and normalized_kind in {"surface", "wireframe"}

def face_overlay_applies(enabled: bool, infront_mode: str = "both") -> bool:
    """Whether face layers should bypass depth testing for an overlay preview."""
    return bool(enabled) and infront_applies(infront_mode, "surface")

def build_invert_overlay_mask(
    width: int,
    height: int,
    layers,
    *,
    supersample: int = 2,
) -> np.ndarray:
    """Rasterize screen-space line overlays into an alpha mask.

    Each layer is ``(projected_points, opacity)``. Projected Y is
    measured from the lower-left corner. The returned mask uses the same origin
    as VTK's framebuffer pixel API. Rasterization is vectorized across all line
    segments; only the maximum subpixel step count loops in Python.
    """
    width = int(width)
    height = int(height)
    if width < 1 or height < 1:
        return np.zeros((0, 0), dtype=np.uint8)
    scale = int(np.clip(supersample, 1, 4))
    mask = np.zeros((height, width), dtype=np.uint8)
    grouped: dict[tuple[float, float], list[np.ndarray]] = {}
    for entry in layers or ():
        try:
            projected, opacity = entry
            points = np.asarray(projected, dtype=np.float64).reshape(-1, 3)
            alpha = float(np.clip(opacity, 0.0, 1.0))
            stroke = 1.0
        except Exception:
            continue
        if len(points) < 2 or len(points) % 2 or alpha <= 1e-4:
            continue
        grouped.setdefault((round(alpha, 4), round(stroke, 2)), []).append(points)

    if not grouped:
        return mask

    high_width = width * scale
    high_height = height * scale
    for (alpha, stroke), point_groups in grouped.items():
        points = np.concatenate(point_groups, axis=0).reshape(-1, 2, 3)
        finite = np.isfinite(points).all(axis=(1, 2))
        in_depth = (points[:, :, 2] >= -1.0).all(axis=1) & (points[:, :, 2] <= 1.0).all(axis=1)
        points = points[finite & in_depth]
        if len(points) == 0:
            continue

        xy0 = points[:, 0, :2].copy()
        xy1 = points[:, 1, :2].copy()
        delta = xy1 - xy0
        steps = np.maximum(1, np.ceil(np.max(np.abs(delta), axis=1) * scale).astype(np.int32))
        max_steps = int(steps.max(initial=0))
        high_mask = np.zeros((high_height, high_width), dtype=np.uint8)
        segment_ids = np.arange(len(points), dtype=np.int64)
        for step in range(max_steps):
            active = segment_ids[steps > step]
            if len(active) == 0:
                continue
            t = (step + 0.5) / steps[active].astype(np.float64)
            sample = xy0[active] + delta[active] * t[:, None]
            x = np.floor(sample[:, 0] * scale + 0.5).astype(np.int64)
            y = np.floor((height - 1.0 - sample[:, 1]) * scale + 0.5).astype(np.int64)
            inside = (x >= 0) & (x < high_width) & (y >= 0) & (y < high_height)
            high_mask[y[inside], x[inside]] = 255

        try:
            from PIL import Image, ImageFilter
            stroke_px = max(1, int(round(stroke * scale)))
            if stroke_px > 1:
                filter_size = stroke_px if stroke_px % 2 else stroke_px + 1
                high_image = Image.fromarray(high_mask, mode="L").filter(
                    ImageFilter.MaxFilter(filter_size)
                )
                high_mask = np.asarray(high_image, dtype=np.uint8)
        except Exception:
            pass

        if scale > 1:
            high_mask = np.rint(
                high_mask.reshape(height, scale, width, scale).mean(axis=(1, 3))
            ).astype(np.uint8)
        layer_mask = np.flipud(high_mask)
        weighted = np.rint(layer_mask.astype(np.float32) * alpha).astype(np.uint8)
        np.maximum(mask, weighted, out=mask)
    return mask

def apply_invert_overlay_mask(framebuffer_rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the invert-color operation to covered framebuffer pixels."""
    frame = np.asarray(framebuffer_rgba, dtype=np.uint8)
    coverage = np.asarray(mask, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] < 3 or coverage.shape != frame.shape[:2]:
        raise ValueError("Framebuffer and invert mask dimensions do not match.")
    result = frame.copy()
    amount = coverage.astype(np.uint16)[..., None]
    base = result[..., :3].astype(np.uint16)
    inverted = 255 - base
    result[..., :3] = ((base * (255 - amount) + inverted * amount + 127) // 255).astype(np.uint8)
    return result
