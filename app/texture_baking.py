from __future__ import annotations

"""Cage-projected UV-atlas baking for textured GLB export.

The production GLB path unwraps connected mesh regions with xatlas and only
duplicates vertices at actual UV-chart seams. Atlas texels are rasterized from
the resulting UV triangles. When a source mesh is available, an automatically
extruded two-sided cage casts normal-directed rays into the source surface.
Source UV images, materials, and vertex colors are evaluated at ray hits.
Cage misses are counted and use a nearest-surface fallback.

``bake_triangle_atlas`` remains as a compatibility helper for internal callers
that explicitly need one chart per triangle. GLB export uses
``bake_unwrapped_mesh_atlas`` instead so connected faces are not needlessly
split into independent topology islands.
"""

from math import ceil, sqrt
from pathlib import Path
from typing import Callable, Optional
import time

import numpy as np

try:
    from .coordinate_frame import as_transform_matrix, points_to_source_local
    from .gpu_compute import GPUComputeManager, GPUComputeRequiredError
    from .live_preview import compact_preview_mesh
    from .mesh_io import load_obj_trimesh
except ImportError:
    from coordinate_frame import as_transform_matrix, points_to_source_local
    from gpu_compute import GPUComputeManager, GPUComputeRequiredError
    from live_preview import compact_preview_mesh
    from mesh_io import load_obj_trimesh

ColorSampler = Callable[[np.ndarray, np.ndarray], np.ndarray]
_PROJECTION_INFO = {
    "cage": {"projection_method": "cage_raycast_both_sides"},
    "direct": {"projection_method": "direct_nearest_surface"},
}

PBR_MAP_LABELS = {
    "base_color": "Base Color (Diffuse / Albedo)",
    "normal": "Normal",
    "roughness": "Roughness",
    "metallic": "Metallic",
    "ao": "Ambient Occlusion",
    "height": "Height / Displacement",
    "opacity": "Opacity / Alpha",
    "emissive": "Emissive",
    "orm": "Channel Packing (AO / Roughness / Metallic)",
}
PBR_MAP_DEFAULTS = {
    "base_color": (1.0, 1.0, 1.0, 1.0),
    "normal": (0.5, 0.5, 1.0, 1.0),
    "roughness": (1.0, 1.0, 1.0, 1.0),
    "metallic": (0.0, 0.0, 0.0, 1.0),
    "ao": (1.0, 1.0, 1.0, 1.0),
    "height": (0.5, 0.5, 0.5, 1.0),
    "opacity": (1.0, 1.0, 1.0, 1.0),
    "emissive": (0.0, 0.0, 0.0, 1.0),
    "orm": (1.0, 1.0, 0.0, 1.0),
}

def normalize_pbr_map_selection(values) -> tuple[str, ...]:
    """Return a stable, de-duplicated PBR map selection in UI order."""
    selected = {str(value).strip().lower() for value in (values or ())}
    return tuple(key for key in PBR_MAP_LABELS if key in selected)

def _build_metallic_roughness_atlas(
    resolution: int,
    pbr_atlas_result: dict[str, np.ndarray],
    selected_maps,
) -> np.ndarray:
    """Pack selected scalar maps into glTF G=roughness / B=metallic channels.

    Unselected channels stay white so their material scalar factors remain
    effective instead of being multiplied by an accidental zero texture.
    """
    selected = set(normalize_pbr_map_selection(selected_maps))
    packed = np.zeros((int(resolution), int(resolution), 4), dtype=np.uint8)
    packed[..., 0] = 255
    packed[..., 1] = 255
    packed[..., 2] = 255
    packed[..., 3] = 255
    if "roughness" in selected and "roughness" in pbr_atlas_result:
        packed[..., 1] = np.asarray(pbr_atlas_result["roughness"], dtype=np.uint8)[..., 0]
    if "metallic" in selected and "metallic" in pbr_atlas_result:
        packed[..., 2] = np.asarray(pbr_atlas_result["metallic"], dtype=np.uint8)[..., 0]
    return packed

def _pbr_sample_blocks(sampled: np.ndarray, map_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Split a packed sampler result into normalized RGBA blocks by map key."""
    values = np.asarray(sampled, dtype=np.float32)
    result = {}
    for index, key in enumerate(map_keys):
        block = values[:, index * 4:(index + 1) * 4]
        if block.shape[1] == 3:
            block = np.column_stack((block, np.ones(len(block), dtype=np.float32)))
        if block.shape[1] != 4:
            default = np.asarray(PBR_MAP_DEFAULTS.get(key, (1, 1, 1, 1)), dtype=np.float32)
            block = np.tile(default, (len(values), 1))
        result[key] = np.clip(
            np.nan_to_num(block, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        )
    return result

_KDTREE_QUERY_BATCH_SIZE = 4096
_KDTREE_PARALLEL_QUERY_THRESHOLD = 2048

def _query_nearest_face_candidates(tree, points: np.ndarray, candidate_count: int):
    """Query the compiled SciPy cKDTree in size-aware batches.

    Small blocks stay single-threaded to avoid scheduler overhead; larger
    blocks use all available cKDTree workers. The returned indices are only
    candidate triangle IDs—the exact point/triangle distances are evaluated by
    the selected GPU backend when it is available.
    """
    query_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    workers = -1 if len(query_points) >= _KDTREE_PARALLEL_QUERY_THRESHOLD else 1
    try:
        return tree.query(query_points, k=int(candidate_count), workers=workers)
    except TypeError:

        return tree.query(query_points, k=int(candidate_count))

def _recent_face_edge_segments(
    vertices: np.ndarray,
    faces: np.ndarray,
    completed_faces: int,
    *,
    trail_faces: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique edge segments for the most recently baked faces.

    The first array is the complete rolling trail; the second contains only
    the current face so the viewport can draw a slightly stronger onion-skin
    head. Coordinates, rather than vertex indices, are used for de-duplication
    because UV seams may duplicate coincident target vertices.
    """
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    stop = max(0, min(int(completed_faces), len(triangles)))
    start = max(0, stop - max(1, int(trail_faces)))
    selected = triangles[start:stop]
    if not len(selected) or not len(verts):
        empty = np.empty((0, 2, 3), dtype=np.float32)
        return empty, empty.copy()
    valid = (selected >= 0).all(axis=1) & (selected < len(verts)).all(axis=1)
    selected = selected[valid]
    if not len(selected):
        empty = np.empty((0, 2, 3), dtype=np.float32)
        return empty, empty.copy()

    def _segments_for(face_indices: np.ndarray) -> np.ndarray:
        if not len(face_indices):
            return np.empty((0, 2, 3), dtype=np.float32)
        points = verts[face_indices]
        segments = np.concatenate((
            points[:, [0, 1]], points[:, [1, 2]], points[:, [2, 0]],
        ), axis=0).astype(np.float32, copy=False)
        bounds = np.ptp(points.reshape(-1, 3), axis=0)
        tolerance = max(float(np.linalg.norm(bounds)) * 1e-8, 1e-8)
        quantized = np.rint(segments.astype(np.float64) / tolerance).astype(np.int64)
        left = quantized[:, 0, :]
        right = quantized[:, 1, :]
        swap = (
            (left[:, 0] > right[:, 0])
            | ((left[:, 0] == right[:, 0]) & (left[:, 1] > right[:, 1]))
            | ((left[:, 0] == right[:, 0]) & (left[:, 1] == right[:, 1]) & (left[:, 2] > right[:, 2]))
        )
        keys = quantized.reshape(-1, 6).copy()
        reversed_keys = quantized[:, ::-1, :].reshape(-1, 6)
        keys[swap] = reversed_keys[swap]
        _unique, indices = np.unique(keys, axis=0, return_index=True)
        return np.ascontiguousarray(segments[np.sort(indices)], dtype=np.float32)

    trail = _segments_for(selected)
    current = _segments_for(selected[-1:])
    return trail, current

class BakeCancelledError(RuntimeError):
    """Raised when an in-progress bake is cooperatively cancelled."""

def _check_bake_cancelled(cancel_check: Optional[Callable[[], bool]]) -> None:
    if cancel_check is not None and cancel_check():
        raise BakeCancelledError("Automatic GLB baking was cancelled by the user.")

def _normalize_vertex_rgba(colors: Optional[np.ndarray], vertex_count: int) -> np.ndarray:
    """Normalize explicit vertex colors; unpainted meshes default to white."""
    if colors is None:
        rgba = np.ones((vertex_count, 4), dtype=np.float32)
    else:
        arr = np.asarray(colors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) != vertex_count:
            rgba = np.ones((vertex_count, 4), dtype=np.float32)
        else:
            if arr.size and float(np.nanmax(arr)) > 1.5:
                arr = arr / 255.0
            if arr.shape[1] == 3:
                arr = np.column_stack((arr, np.ones(len(arr), dtype=np.float32)))
            rgba = arr[:, :4]
            rgba = np.nan_to_num(rgba, nan=1.0, posinf=1.0, neginf=0.0)
            rgba = np.clip(rgba, 0.0, 1.0)
    return np.ascontiguousarray(rgba, dtype=np.float32)

def _prepare_bake_target_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    """Conservatively repair invalid target triangles before UV generation.

    Only numerically coincident positions are welded here. The configurable
    E1 distance weld has already run during reconstruction; applying another
    broad tolerance at bake time could collapse nearby but intentionally
    separate surfaces. Degenerate/duplicate triangles are removed and face
    winding is made consistent across two-face manifold edges. Open and
    non-manifold boundaries are preserved rather than guessed/reconnected.
    """
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(verts) < 3 or len(tri) == 0:
        raise ValueError("There is no target triangle surface to bake.")
    if not np.isfinite(verts).all():
        raise ValueError("Target mesh vertices contain non-finite coordinates.")
    if int(tri.min()) < 0 or int(tri.max()) >= len(verts):
        raise ValueError("Target mesh faces contain invalid vertex indices.")

    color_values = None
    if colors is not None:
        color_values = _normalize_vertex_rgba(colors, len(verts)).astype(np.float64)

    diagonal = float(np.linalg.norm(np.ptp(verts, axis=0)))
    weld_tolerance = max(diagonal * 1e-10, 1e-12)

    quantized = np.rint(verts / weld_tolerance).astype(np.int64)
    unique_keys, inverse = np.unique(quantized, axis=0, return_inverse=True)
    unique_count = int(len(unique_keys))
    welded_vertices = np.zeros((unique_count, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=unique_count).astype(np.float64)
    np.add.at(welded_vertices, inverse, verts)
    welded_vertices /= counts[:, None]
    if color_values is not None:
        welded_colors = np.zeros((unique_count, 4), dtype=np.float64)
        np.add.at(welded_colors, inverse, color_values)
        welded_colors /= counts[:, None]
        color_values = welded_colors.astype(np.float32)
    tri = inverse[tri]
    welded_count = int(len(verts) - unique_count)

    original_face_count = int(len(tri))
    valid = (
        (tri[:, 0] != tri[:, 1])
        & (tri[:, 1] != tri[:, 2])
        & (tri[:, 2] != tri[:, 0])
    )
    edges_a = welded_vertices[tri[:, 1]] - welded_vertices[tri[:, 0]]
    edges_b = welded_vertices[tri[:, 2]] - welded_vertices[tri[:, 0]]
    twice_area = np.linalg.norm(np.cross(edges_a, edges_b), axis=1)
    area_epsilon = max(diagonal * diagonal * 1e-14, 1e-24)
    valid &= np.isfinite(twice_area) & (twice_area > area_epsilon)
    removed_degenerate = int(np.count_nonzero(~valid))
    tri = tri[valid]
    if len(tri) == 0:
        raise ValueError(
            "Bake target cleanup removed every face as degenerate. "
            "Review the E1 merge distance and reconstructed mesh."
        )

    canonical = np.sort(tri, axis=1)
    _unique_rows, first_indices = np.unique(canonical, axis=0, return_index=True)
    first_indices.sort()
    removed_duplicates = int(len(tri) - len(first_indices))
    tri = tri[first_indices]

    edge_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, face in enumerate(tri):
        for corner in range(3):
            left = int(face[corner])
            right = int(face[(corner + 1) % 3])
            edge = (min(left, right), max(left, right))
            direction = 1 if left < right else -1
            edge_owners.setdefault(edge, []).append((face_index, direction))

    adjacency: list[list[tuple[int, bool]]] = [[] for _ in range(len(tri))]
    for owners in edge_owners.values():
        if len(owners) != 2:
            continue
        (left_face, left_direction), (right_face, right_direction) = owners
        same_direction = left_direction == right_direction
        adjacency[left_face].append((right_face, same_direction))
        adjacency[right_face].append((left_face, same_direction))

    flip_state = np.full(len(tri), -1, dtype=np.int8)
    components = 0
    orientation_conflicts = 0
    for start in range(len(tri)):
        if flip_state[start] >= 0:
            continue
        components += 1
        flip_state[start] = 0
        stack = [start]
        while stack:
            face_index = stack.pop()
            for neighbour, must_flip in adjacency[face_index]:
                expected = int(flip_state[face_index]) ^ int(must_flip)
                if flip_state[neighbour] < 0:
                    flip_state[neighbour] = expected
                    stack.append(neighbour)
                elif int(flip_state[neighbour]) != expected:
                    orientation_conflicts += 1

    flip_mask = flip_state.astype(bool)
    flipped_count = int(np.count_nonzero(flip_mask))
    if flipped_count:
        tri[flip_mask] = tri[flip_mask][:, [0, 2, 1]]

    boundary_edges = sum(1 for owners in edge_owners.values() if len(owners) == 1)
    nonmanifold_edges = sum(1 for owners in edge_owners.values() if len(owners) > 2)
    stats = {
        "bake_input_vertices": int(len(verts)),
        "bake_output_vertices": int(len(welded_vertices)),
        "bake_welded_duplicate_vertices": welded_count,
        "bake_input_triangles": original_face_count,
        "bake_output_triangles": int(len(tri)),
        "bake_removed_degenerate_triangles": removed_degenerate,
        "bake_removed_duplicate_triangles": removed_duplicates,
        "bake_winding_flipped_triangles": flipped_count,
        "bake_orientation_conflicts": int(orientation_conflicts),
        "bake_face_components": int(components),
        "bake_boundary_edges": int(boundary_edges),
        "bake_nonmanifold_edges": int(nonmanifold_edges),
        "bake_weld_tolerance": float(weld_tolerance),
    }
    return (
        np.ascontiguousarray(welded_vertices, dtype=np.float64),
        np.ascontiguousarray(tri, dtype=np.int64),
        None if color_values is None else np.ascontiguousarray(color_values, dtype=np.float32),
        stats,
    )

def _generate_uv_atlas_layout(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Generate connected UV charts and the seam-split indexed mesh via xatlas."""
    try:
        import xatlas
    except ImportError:

        face_count = int(len(faces))

        tile_size = _auto_tile_size(int(resolution), face_count)
        cells = int(resolution) // int(tile_size)
        capacity = cells * cells
        if face_count > capacity:
            raise ValueError(
                f"The {resolution // 1024}K fallback atlas supports {capacity:,} triangle charts, "
                f"but the reconstructed mesh has {face_count:,} triangles."
            )
        inner = int(tile_size) - 3
        expanded_vertices = np.ascontiguousarray(
            np.asarray(vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int64).reshape(-1)],
            dtype=np.float32,
        )
        expanded_faces = np.arange(face_count * 3, dtype=np.int64).reshape(-1, 3)
        uv = np.empty((face_count * 3, 2), dtype=np.float32)
        for face_index in range(face_count):
            row, col = divmod(face_index, cells)
            x0, y0 = col * tile_size, row * tile_size
            x1, y1 = x0 + 1, y0 + 1
            low_x = x1 + 0.5
            high_x = x1 + inner + 0.5
            high_y = y1 + 0.5
            low_y = y1 + inner + 0.5
            uv[face_index * 3:face_index * 3 + 3] = np.array([
                [low_x / resolution, 1.0 - high_y / resolution],
                [high_x / resolution, 1.0 - high_y / resolution],
                [low_x / resolution, 1.0 - low_y / resolution],
            ], dtype=np.float32)
        utilization = float(face_count * 0.5 * (inner + 1) ** 2 / (resolution * resolution))
        return (
            expanded_vertices,
            np.ascontiguousarray(expanded_faces, dtype=np.int64),
            np.ascontiguousarray(uv, dtype=np.float32),
            face_count,
            utilization,
        )

    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.uint32),
    )
    pack_options = xatlas.PackOptions()
    pack_options.resolution = int(resolution)

    pack_options.padding = max(4, min(16, int(resolution) // 256))
    atlas.generate(pack_options=pack_options)
    vertex_mapping, output_faces, output_uv = atlas[0]
    vertex_mapping = np.asarray(vertex_mapping, dtype=np.int64).reshape(-1)
    output_faces = np.asarray(output_faces, dtype=np.int64).reshape(-1, 3)
    output_uv = np.asarray(output_uv, dtype=np.float32).reshape(-1, 2)
    if (
        len(vertex_mapping) < 3
        or len(output_faces) != len(faces)
        or len(output_uv) != len(vertex_mapping)
        or int(output_faces.min()) < 0
        or int(output_faces.max()) >= len(vertex_mapping)
        or int(vertex_mapping.min()) < 0
        or int(vertex_mapping.max()) >= len(vertices)
        or not np.isfinite(output_uv).all()
        or np.any(output_uv < -1e-5)
        or np.any(output_uv > 1.00001)
    ):
        raise ValueError("xatlas returned an invalid or incomplete UV layout.")

    source_face_by_key = {
        tuple(sorted(map(int, face))): index for index, face in enumerate(faces)
    }
    ordered_output_faces = np.empty_like(output_faces)
    assigned = np.zeros(len(faces), dtype=bool)
    for output_face in output_faces:
        source_ids = vertex_mapping[output_face]
        source_index = source_face_by_key.get(tuple(sorted(map(int, source_ids))))
        if source_index is None or assigned[source_index]:
            raise ValueError("xatlas changed or duplicated target triangles during UV unwrapping.")
        ordered_output_faces[source_index] = output_face
        assigned[source_index] = True
    if not bool(np.all(assigned)):
        raise ValueError("xatlas did not return a UV mapping for every target triangle.")
    output_faces = ordered_output_faces
    output_uv = np.clip(output_uv, 0.0, 1.0)
    try:
        chart_count = int(atlas.get_mesh_chart_count(0))
    except Exception:
        chart_count = -1
    try:
        utilization = float(atlas.get_utilization(0))
    except Exception:
        utilization = 0.0
    return (
        np.ascontiguousarray(vertices[vertex_mapping], dtype=np.float32),
        np.ascontiguousarray(output_faces, dtype=np.int64),
        np.ascontiguousarray(output_uv, dtype=np.float32),
        chart_count,
        utilization,
    )

def _auto_tile_size(resolution: int, face_count: int) -> int:
    """Choose the largest divisor-sized triangle tile that fits every face.

    Divisor-sized tiles let the atlas be reshaped into an exact square grid
    without leaving an unaddressable remainder strip. Lower-poly outputs get
    larger per-triangle charts; dense outputs reduce tile size to fit.
    """
    resolution = int(resolution)
    face_count = max(1, int(face_count))
    grid = max(1, int(ceil(sqrt(face_count))))
    desired = min(512, resolution // grid)
    tile_size = max(
        (size for size in range(8, desired + 1) if resolution % size == 0),
        default=0,
    )
    if tile_size < 8:
        max_faces = (resolution // 8) ** 2
        raise ValueError(
            f"The {resolution // 1024}K atlas supports at most {max_faces:,} triangle charts "
            "at the minimum 8px chart size. Choose a higher texture resolution or reduce the output face count."
        )
    return int(tile_size)

def bake_triangle_atlas(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray],
    resolution: int = 2048,
    tile_size: Optional[int] = None,
    color_sampler: Optional[ColorSampler] = None,
    progress_callback: Optional[Callable[[Optional[float], str], None]] = None,
    preview_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    cage_extrusion_distance: float = 0.0,
    chart_inner_pixels: Optional[int] = None,
    edge_preview_callback: Optional[Callable[[dict], None]] = None,
    diffuse_atlas_out: Optional[dict] = None,
    pbr_map_keys: Optional[tuple[str, ...] | list[str]] = None,
    pbr_atlases_out: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RGBA atlas, per-corner UVs, and expanded triangle vertices.

    ``color_sampler`` optionally evaluates the original source appearance at
    positions throughout each output triangle. If omitted, source appearance
    is interpolated from the supplied vertex RGBA values.
    """
    _check_bake_cancelled(cancel_check)
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if resolution not in (1024, 2048, 4096):
        raise ValueError("Texture resolution must be 1024, 2048, or 4096.")
    if len(verts) < 3 or len(tri) == 0:
        raise ValueError("There is no triangle mesh to bake.")
    if not np.isfinite(verts).all():
        raise ValueError("Mesh vertices contain non-finite values.")
    if int(tri.min()) < 0 or int(tri.max()) >= len(verts):
        raise ValueError("Mesh faces contain invalid vertex indices.")

    if tile_size is None:
        tile_size = _auto_tile_size(resolution, len(tri))
    if tile_size < 8:
        raise ValueError("Atlas tile size must be at least 8 pixels.")
    cells = resolution // int(tile_size)
    capacity = cells * cells
    if len(tri) > capacity:
        raise ValueError(
            f"The {resolution // 1024}K atlas supports {capacity:,} triangle charts "
            f"at {tile_size}px per chart, but the mesh has {len(tri):,} triangles. "
            "Choose a higher texture resolution or reduce the output face count."
        )

    map_keys = tuple(pbr_map_keys or ())
    color_values = None if colors is None else np.asarray(colors, dtype=np.float32)
    vertex_map_rgba: dict[str, np.ndarray] = {}
    if map_keys:
        for map_index, map_key in enumerate(map_keys):
            start = map_index * 4
            block = (
                color_values[:, start:start + 4]
                if color_values is not None
                and color_values.ndim == 2
                and len(color_values) == len(verts)
                and color_values.shape[1] >= start + 3
                else None
            )
            if block is None:
                default = np.asarray(PBR_MAP_DEFAULTS.get(map_key, (1, 1, 1, 1)), dtype=np.float32)
                block = np.tile(default, (len(verts), 1))
            vertex_map_rgba[map_key] = _normalize_vertex_rgba(block, len(verts))
        rgba = vertex_map_rgba.get("base_color")
        if rgba is None:
            rgba = _normalize_vertex_rgba(colors, len(verts))
    else:
        rgba = _normalize_vertex_rgba(colors, len(verts))
    atlas = np.full((resolution, resolution, 4), 255, dtype=np.uint8)
    pbr_atlases = {"base_color": atlas} if map_keys else {}
    for map_key in map_keys:
        if map_key == "base_color":
            continue
        default = np.asarray(PBR_MAP_DEFAULTS.get(map_key, (1, 1, 1, 1)), dtype=np.float32)
        pbr_atlases[map_key] = np.empty((resolution, resolution, 4), dtype=np.uint8)
        pbr_atlases[map_key][...] = np.rint(default * 255.0).astype(np.uint8)
    diffuse_atlas = atlas
    expanded_vertices = np.ascontiguousarray(verts[tri].reshape(-1, 3), dtype=np.float32)

    corner_uvs = np.zeros((len(tri), 3, 2), dtype=np.float32)

    inner = int(tile_size) - 3 if chart_inner_pixels is None else int(chart_inner_pixels)
    if inner < 1 or inner > int(tile_size) - 3:
        raise ValueError("Triangle chart sample size must leave at least two pixels of tile gutter.")
    axis = np.linspace(0.0, 1.0, inner + 1, dtype=np.float32)
    uu, vv = np.meshgrid(axis, axis)
    denom = np.maximum(uu + vv, 1.0)
    bu = uu / denom
    bv = vv / denom
    bw = 1.0 - bu - bv
    bary = np.stack((bw, bu, bv), axis=-1)

    def _write_chart(face_index: int, patch: np.ndarray, target_atlas=None) -> None:
        destination = atlas if target_atlas is None else target_atlas
        row, col = divmod(face_index, cells)
        x0, y0 = col * tile_size, row * tile_size
        patch = np.clip(np.nan_to_num(patch, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        patch8 = np.rint(patch * 255.0).astype(np.uint8)

        x1, y1 = x0 + 1, y0 + 1
        destination[y1:y1 + inner + 1, x1:x1 + inner + 1] = patch8
        destination[y0:y1, x1:x1 + inner + 1] = patch8[:1]
        destination[y1 + inner + 1:y0 + tile_size, x1:x1 + inner + 1] = patch8[-1:]
        destination[y1:y1 + inner + 1, x0:x1] = patch8[:, :1]
        destination[y1:y1 + inner + 1, x1 + inner + 1:x0 + tile_size] = patch8[:, -1:]

        low_y = resolution - (y1 + inner + 0.5)
        high_y = resolution - (y1 + 0.5)
        low_x = x1 + 0.5
        high_x = x1 + inner + 0.5
        corner_uvs[face_index] = np.array([
            [low_x / resolution, high_y / resolution],
            [high_x / resolution, high_y / resolution],
            [low_x / resolution, low_y / resolution],
        ], dtype=np.float32)

    bary_flat = bary.reshape(-1, 3)
    pixels_per_chart = len(bary_flat)
    preview_face_limit = 100_000
    preview_interval = max(1, int(np.ceil(len(tri) / 24.0)))
    next_preview_at = preview_interval
    preview_geometry_sent = False
    last_preview_completed = -1
    live_cage_outer_vertices = None
    live_cage_inner_vertices = None
    live_cage_faces = None
    last_edge_preview_at = 0.0

    def _emit_active_face_edges(completed: int, *, force: bool = False) -> None:
        nonlocal last_edge_preview_at
        if edge_preview_callback is None:
            return
        now = time.perf_counter()
        if not force and now - last_edge_preview_at < (1.0 / 30.0):
            return
        recent, current = _recent_face_edge_segments(verts, tri, completed, trail_faces=10)
        try:
            edge_preview_callback({
                "recent_edge_segments": recent,
                "current_edge_segments": current,
                "completed_faces": int(max(0, min(completed, len(tri)))),
                "total_faces": int(len(tri)),
            })
            last_edge_preview_at = now
        except Exception:
            pass

    def _report(fraction: Optional[float], message: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(fraction, message)
        except Exception:

            pass

    def _emit_live_preview(
        completed_charts: int, force: bool = False,
        active_face_range: tuple[int, int] | None = None,
    ) -> None:
        """Publish bounded atlas, cage-shell and active-chart geometry snapshots.

        The texture is deliberately downsampled and emitted at most ~24 times
        per bake. The target/cage geometry is included only once; each later
        event carries the current chart batch so the GUI can highlight its exact
        surface region. Failures are visualization-only and never abort export.
        """
        nonlocal next_preview_at, preview_geometry_sent, last_preview_completed
        if preview_callback is None:
            return
        completed = max(0, min(int(completed_charts), len(tri)))
        if not force and completed < next_preview_at:
            return
        if completed == last_preview_completed:
            return
        while next_preview_at <= completed:
            next_preview_at += preview_interval
        try:
            from PIL import Image

            max_side = 768
            if resolution > max_side:
                source_image = Image.fromarray(atlas, mode="RGBA")
                resampling = getattr(Image, "Resampling", Image)
                preview_image = np.asarray(
                    source_image.resize((max_side, max_side), resampling.BILINEAR),
                    dtype=np.uint8,
                ).copy()
            else:
                preview_image = atlas.copy()

            payload = {
                "texture_image": preview_image,
                "completed_charts": completed,
                "total_charts": int(len(tri)),
                "fraction": float(completed / max(1, len(tri))),
                "label": f"LIVE BAKE · {completed:,}/{len(tri):,} charts",
            }
            if diffuse_atlas is not None:
                diffuse_source = Image.fromarray(diffuse_atlas, mode="RGBA")
                if resolution > max_side:
                    diffuse_preview = np.asarray(
                        diffuse_source.resize((max_side, max_side), resampling.BILINEAR),
                        dtype=np.uint8,
                    ).copy()
                else:
                    diffuse_preview = np.asarray(diffuse_source, dtype=np.uint8).copy()
                payload["diffuse_texture_image"] = diffuse_preview
            include_geometry = not preview_geometry_sent
            if include_geometry:
                face_count = len(tri)
                if face_count > preview_face_limit:
                    selected_faces = np.linspace(
                        0, face_count - 1, preview_face_limit, dtype=np.int64
                    )
                else:
                    selected_faces = np.arange(face_count, dtype=np.int64)
                payload.update({
                    "vertices": np.ascontiguousarray(
                        expanded_vertices.reshape(-1, 3, 3)[selected_faces].reshape(-1, 3),
                        dtype=np.float32,
                    ),
                    "faces": np.arange(
                        len(selected_faces) * 3, dtype=np.uint32
                    ).reshape(-1, 3),
                    "uvs": np.ascontiguousarray(
                        corner_uvs[selected_faces].reshape(-1, 2), dtype=np.float32
                    ),
                    "preview_face_count": int(len(selected_faces)),
                })
                if live_cage_outer_vertices is not None and live_cage_faces is not None:
                    cage_outer, cage_faces = compact_preview_mesh(
                        live_cage_outer_vertices, live_cage_faces, max_faces=60_000
                    )
                    cage_inner, _ = compact_preview_mesh(
                        live_cage_inner_vertices, live_cage_faces, max_faces=60_000
                    )
                    payload.update({
                        "cage_outer_vertices": cage_outer,
                        "cage_inner_vertices": cage_inner,
                        "cage_faces": cage_faces,
                        "cage_distance": float(max(0.0, cage_extrusion_distance)),
                    })
            if active_face_range is not None:
                active_start = max(0, min(int(active_face_range[0]), len(tri)))
                active_stop = max(active_start, min(int(active_face_range[1]), len(tri)))
                active_ids = np.arange(active_start, active_stop, dtype=np.int64)
                if len(active_ids) > 5_000:
                    active_ids = active_ids[np.linspace(0, len(active_ids) - 1, 5_000, dtype=np.int64)]
                if len(active_ids):
                    active_vertices = np.ascontiguousarray(
                        expanded_vertices.reshape(-1, 3, 3)[active_ids].reshape(-1, 3),
                        dtype=np.float32,
                    )
                    payload.update({
                        "active_vertices": active_vertices,
                        "active_faces": np.arange(
                            len(active_ids) * 3, dtype=np.uint32
                        ).reshape(-1, 3),
                        "active_chart_start": int(active_start),
                        "active_chart_stop": int(active_stop),
                    })
            preview_callback(payload)
            if include_geometry:
                preview_geometry_sent = True
            last_preview_completed = completed
        except Exception:

            pass

    _report(0.0, f"Preparing UV atlas · {len(tri):,} triangle charts")
    _emit_active_face_edges(0, force=True)
    if color_sampler is None:
        _emit_live_preview(0, force=True)
        _check_bake_cancelled(cancel_check)

        face_colors = rgba[tri]
        patches = np.einsum("pi,bic->bpc", bary_flat, face_colors, optimize=True)
        patches8 = np.rint(
            np.clip(np.nan_to_num(patches, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            * 255.0
        ).astype(np.uint8).reshape(len(tri), inner + 1, inner + 1, 4)
        del patches, face_colors
        map_patches8: dict[str, np.ndarray] = {}
        if map_keys:
            for map_key in map_keys:
                if map_key == "base_color":
                    continue
                map_face_colors = vertex_map_rgba[map_key][tri]
                map_patches = np.einsum("pi,bic->bpc", bary_flat, map_face_colors, optimize=True)
                map_patches8[map_key] = np.rint(
                    np.clip(np.nan_to_num(map_patches, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                    * 255.0
                ).astype(np.uint8).reshape(len(tri), inner + 1, inner + 1, 4)
                del map_patches, map_face_colors
        tile_view = atlas.reshape(cells, tile_size, cells, tile_size, 4).transpose(0, 2, 1, 3, 4)
        chart_ids = np.arange(len(tri), dtype=np.int64)
        rows = chart_ids // cells
        cols = chart_ids % cells
        x1 = cols.astype(np.float64) * float(tile_size) + 1.0
        y1 = rows.astype(np.float64) * float(tile_size) + 1.0
        low_x = x1 + 0.5
        high_x = x1 + float(inner) + 0.5
        low_y = float(resolution) - (y1 + float(inner) + 0.5)
        high_y = float(resolution) - (y1 + 0.5)
        corner_uvs[:, 0, 0] = low_x / float(resolution)
        corner_uvs[:, 0, 1] = high_y / float(resolution)
        corner_uvs[:, 1, 0] = high_x / float(resolution)
        corner_uvs[:, 1, 1] = high_y / float(resolution)
        corner_uvs[:, 2, 0] = low_x / float(resolution)
        corner_uvs[:, 2, 1] = low_y / float(resolution)
        tile_view[rows, cols, 1:inner + 2, 1:inner + 2] = patches8
        tile_view[rows, cols, 0, 1:inner + 2] = patches8[:, 0]
        tile_view[rows, cols, 1:inner + 2, 0] = patches8[:, :, 0]

        bottom_start = inner + 2
        if bottom_start < tile_size:
            bottom_patch = np.broadcast_to(
                patches8[:, -1:, :, :],
                (len(tri), tile_size - bottom_start, inner + 1, 4),
            )
            tile_view[rows, cols, bottom_start:tile_size, 1:inner + 2] = bottom_patch
            right_patch = np.broadcast_to(
                patches8[:, :, -1:, :],
                (len(tri), inner + 1, tile_size - bottom_start, 4),
            )
            tile_view[rows, cols, 1:inner + 2, bottom_start:tile_size] = right_patch

        del patches8, tile_view
        for map_key, map_patch_values in map_patches8.items():
            map_tile_view = pbr_atlases[map_key].reshape(
                cells, tile_size, cells, tile_size, 4
            ).transpose(0, 2, 1, 3, 4)
            map_tile_view[rows, cols, 1:inner + 2, 1:inner + 2] = map_patch_values
            map_tile_view[rows, cols, 0, 1:inner + 2] = map_patch_values[:, 0]
            map_tile_view[rows, cols, 1:inner + 2, 0] = map_patch_values[:, :, 0]
            if bottom_start < tile_size:
                map_bottom = np.broadcast_to(
                    map_patch_values[:, -1:, :, :],
                    (len(tri), tile_size - bottom_start, inner + 1, 4),
                )
                map_tile_view[rows, cols, bottom_start:tile_size, 1:inner + 2] = map_bottom
                map_right = np.broadcast_to(
                    map_patch_values[:, :, -1:, :],
                    (len(tri), inner + 1, tile_size - bottom_start, 4),
                )
                map_tile_view[rows, cols, 1:inner + 2, bottom_start:tile_size] = map_right
            del map_tile_view, map_patch_values
        _report(1.0, f"UV atlas · color transfer {len(tri):,}/{len(tri):,} triangles")
        _emit_live_preview(len(tri), force=True, active_face_range=(0, min(len(tri), 1)))
    else:

        target_edges_a = verts[tri[:, 1]] - verts[tri[:, 0]]
        target_edges_b = verts[tri[:, 2]] - verts[tri[:, 0]]
        face_normals = np.cross(target_edges_a, target_edges_b)
        normal_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = np.divide(
            face_normals, normal_lengths,
            out=np.zeros_like(face_normals), where=normal_lengths > 1e-12,
        )

        vertex_normals = np.zeros_like(verts, dtype=np.float64)
        weighted_face_normals = np.cross(target_edges_a, target_edges_b)
        for corner in range(3):
            np.add.at(vertex_normals, tri[:, corner], weighted_face_normals)
        vertex_lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = np.divide(
            vertex_normals, vertex_lengths,
            out=np.zeros_like(vertex_normals), where=vertex_lengths > 1e-12,
        )
        invalid_vertices = vertex_lengths[:, 0] <= 1e-12
        if np.any(invalid_vertices):
            vertex_normals[invalid_vertices] = np.array([0.0, 0.0, 1.0])

        cage_distance_for_preview = max(0.0, float(cage_extrusion_distance))
        if cage_distance_for_preview > 0.0:
            live_cage_outer_vertices = np.ascontiguousarray(
                verts + vertex_normals * cage_distance_for_preview, dtype=np.float32
            )
            live_cage_inner_vertices = np.ascontiguousarray(
                verts - vertex_normals * cage_distance_for_preview, dtype=np.float32
            )
            live_cage_faces = np.ascontiguousarray(tri, dtype=np.uint32)

            _emit_live_preview(0, force=True)

        faces_per_batch = max(1, min(256, 200_000 // max(1, pixels_per_chart)))
        batch_count = (len(tri) + faces_per_batch - 1) // faces_per_batch
        for batch_index, batch_start in enumerate(range(0, len(tri), faces_per_batch)):
            _check_bake_cancelled(cancel_check)
            batch_stop = min(len(tri), batch_start + faces_per_batch)
            _report(
                batch_start / len(tri),
                f"Cage projection · chart batch {batch_index + 1}/{batch_count} "
                f"({batch_start:,}/{len(tri):,} triangles)",
            )
            batch_faces = tri[batch_start:batch_stop]
            batch_triangles = verts[batch_faces]
            sample_positions = np.einsum("pi,bij->bpj", bary_flat, batch_triangles)
            sample_normals = np.einsum(
                "pi,bij->bpj", bary_flat, vertex_normals[batch_faces]
            )
            normal_magnitudes = np.linalg.norm(sample_normals, axis=2, keepdims=True)
            sample_normals = np.divide(
                sample_normals, normal_magnitudes,
                out=np.zeros_like(sample_normals), where=normal_magnitudes > 1e-12,
            )
            flat_bad_normals = normal_magnitudes[:, :, 0] <= 1e-12
            if np.any(flat_bad_normals):
                sample_normals[flat_bad_normals] = np.repeat(
                    face_normals[batch_start:batch_stop],
                    pixels_per_chart, axis=0,
                ).reshape(-1, 3)[flat_bad_normals.reshape(-1)]
            _report(
                None,
                f"Cage ray projection · sampling {batch_start + 1:,}–{batch_stop:,} "
                f"of {len(tri):,} charts",
            )
            _check_bake_cancelled(cancel_check)
            sampled = np.asarray(
                color_sampler(
                    sample_positions.reshape(-1, 3),
                    sample_normals.reshape(-1, 3),
                ), dtype=np.float32
            )
            if sampled.ndim != 2 or sampled.shape[0] != len(sample_positions) * pixels_per_chart or sampled.shape[1] < 3:
                raise ValueError("Source color sampler returned an invalid RGBA array.")
            if sampled.shape[1] == 3:
                sampled = np.column_stack((sampled, np.ones(len(sampled), dtype=np.float32)))
            sampled_map_blocks = _pbr_sample_blocks(sampled, map_keys) if map_keys else None
            sampled = sampled[:, :4].reshape(batch_stop - batch_start, inner + 1, inner + 1, 4)
            _check_bake_cancelled(cancel_check)
            for local_index, patch in enumerate(sampled):
                if local_index % 16 == 0:
                    _check_bake_cancelled(cancel_check)
                face_index = batch_start + local_index
                _write_chart(face_index, patch)
                if sampled_map_blocks is not None:
                    for map_key, map_values in sampled_map_blocks.items():
                        if map_key == "base_color":
                            continue
                        _write_chart(face_index, map_values.reshape(
                            batch_stop - batch_start, inner + 1, inner + 1, 4
                        )[local_index], pbr_atlases[map_key])
                _emit_active_face_edges(face_index + 1)
            _emit_live_preview(
                batch_stop, active_face_range=(batch_start, batch_stop)
            )
            _report(
                batch_stop / len(tri),
                f"Cage projection · completed {batch_stop:,}/{len(tri):,} triangle charts",
            )

    _check_bake_cancelled(cancel_check)
    if diffuse_atlas_out is not None and diffuse_atlas is not None:
        diffuse_atlas_out["image"] = diffuse_atlas
    if pbr_atlases_out is not None and pbr_atlases:
        pbr_atlases_out.update(pbr_atlases)
    _emit_active_face_edges(len(tri), force=True)
    _emit_live_preview(len(tri), force=True)
    _report(1.0, f"UV atlas complete · {len(tri):,} triangle charts")
    return atlas, corner_uvs.reshape(-1, 2), expanded_vertices

def _extend_uv_chart_gutters(atlas: np.ndarray, covered: np.ndarray, iterations: int = 3) -> None:
    """Extend chart-edge texels into xatlas padding to reduce mip bleed."""
    height, width = covered.shape
    filled = covered.copy()
    neighbours = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    for _ in range(max(0, int(iterations))):
        source_mask = filled.copy()
        added = np.zeros_like(filled)
        for dy, dx in neighbours:
            src_y0, src_y1 = max(0, -dy), min(height, height - dy)
            src_x0, src_x1 = max(0, -dx), min(width, width - dx)
            dst_y0, dst_y1 = max(0, dy), min(height, height + dy)
            dst_x0, dst_x1 = max(0, dx), min(width, width + dx)
            src = source_mask[src_y0:src_y1, src_x0:src_x1]
            dst = added[dst_y0:dst_y1, dst_x0:dst_x1]
            take = src & ~dst
            if not np.any(take):
                continue
            atlas_dst = atlas[dst_y0:dst_y1, dst_x0:dst_x1]
            atlas_src = atlas[src_y0:src_y1, src_x0:src_x1]
            atlas_dst[take] = atlas_src[take]
            dst[take] = True
        if not np.any(added):
            break
        filled |= added

def bake_unwrapped_mesh_atlas(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_vertices: np.ndarray,
    uv_faces: np.ndarray,
    colors: Optional[np.ndarray],
    uv_mesh_vertices: Optional[np.ndarray] = None,
    resolution: int = 2048,
    color_sampler: Optional[ColorSampler] = None,
    progress_callback: Optional[Callable[[Optional[float], str], None]] = None,
    preview_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    cage_extrusion_distance: float = 0.0,
    chart_padding: int = 8,
    edge_preview_callback: Optional[Callable[[dict], None]] = None,
    diffuse_atlas_out: Optional[dict] = None,
    pbr_map_keys: Optional[tuple[str, ...] | list[str]] = None,
    pbr_atlases_out: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bake texels into a packed multi-triangle UV atlas.

    Returns the RGBA atlas, UVs flattened per face corner (for live preview),
    and expanded target triangle positions (for compatibility). The live
    viewport geometry itself uses the seam-split indexed arrays so that faces
    within one chart remain connected while the bake is running.
    """
    _check_bake_cancelled(cancel_check)
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    uv = np.asarray(uv_vertices, dtype=np.float32).reshape(-1, 2)
    uv_tri = np.asarray(uv_faces, dtype=np.int64).reshape(-1, 3)
    display_vertices = (
        verts.astype(np.float32, copy=False)
        if uv_mesh_vertices is None
        else np.asarray(uv_mesh_vertices, dtype=np.float32).reshape(-1, 3)
    )
    if resolution not in (1024, 2048, 4096):
        raise ValueError("Texture resolution must be 1024, 2048, or 4096.")
    if len(tri) == 0 or len(tri) != len(uv_tri):
        raise ValueError("The UV atlas and target mesh must contain the same number of triangles.")
    if len(verts) < 3 or int(tri.min()) < 0 or int(tri.max()) >= len(verts):
        raise ValueError("Target mesh has invalid triangle indices.")
    if len(uv) < 3 or int(uv_tri.min()) < 0 or int(uv_tri.max()) >= len(uv):
        raise ValueError("The UV layout contains invalid indices.")
    if len(display_vertices) != len(uv):
        raise ValueError("UV-mapped preview vertices must match the UV coordinate count.")
    if not np.isfinite(verts).all() or not np.isfinite(uv).all():
        raise ValueError("Target geometry or UV layout contains non-finite values.")

    map_keys = tuple(pbr_map_keys or ())
    color_values = None if colors is None else np.asarray(colors, dtype=np.float32)
    vertex_map_rgba: dict[str, np.ndarray] = {}
    if map_keys:
        for map_index, map_key in enumerate(map_keys):
            start = map_index * 4
            block = (
                color_values[:, start:start + 4]
                if color_values is not None
                and color_values.ndim == 2
                and len(color_values) == len(verts)
                and color_values.shape[1] >= start + 3
                else None
            )
            if block is None:
                default = np.asarray(PBR_MAP_DEFAULTS.get(map_key, (1, 1, 1, 1)), dtype=np.float32)
                block = np.tile(default, (len(verts), 1))
            vertex_map_rgba[map_key] = _normalize_vertex_rgba(block, len(verts))
        rgba = vertex_map_rgba.get("base_color", _normalize_vertex_rgba(colors, len(verts)))
    else:
        rgba = _normalize_vertex_rgba(colors, len(verts))
    atlas = np.full((resolution, resolution, 4), 255, dtype=np.uint8)
    pbr_atlases = {"base_color": atlas} if map_keys else {}
    for map_key in map_keys:
        if map_key == "base_color":
            continue
        default = np.asarray(PBR_MAP_DEFAULTS.get(map_key, (1, 1, 1, 1)), dtype=np.float32)
        pbr_atlases[map_key] = np.empty((resolution, resolution, 4), dtype=np.uint8)
        pbr_atlases[map_key][...] = np.rint(default * 255.0).astype(np.uint8)
    diffuse_atlas = atlas
    covered = np.zeros((resolution, resolution), dtype=bool)
    expanded_vertices = np.ascontiguousarray(verts[tri].reshape(-1, 3), dtype=np.float32)
    corner_uvs = np.ascontiguousarray(uv[uv_tri].reshape(-1, 2), dtype=np.float32)

    pixel_uv = np.empty_like(uv, dtype=np.float64)
    pixel_uv[:, 0] = uv[:, 0] * resolution - 0.5
    pixel_uv[:, 1] = (1.0 - uv[:, 1]) * resolution - 0.5

    target_edges_a = verts[tri[:, 1]] - verts[tri[:, 0]]
    target_edges_b = verts[tri[:, 2]] - verts[tri[:, 0]]
    weighted_face_normals = np.cross(target_edges_a, target_edges_b)
    normal_lengths = np.linalg.norm(weighted_face_normals, axis=1, keepdims=True)
    face_normals = np.divide(
        weighted_face_normals, normal_lengths,
        out=np.zeros_like(weighted_face_normals), where=normal_lengths > 1e-12,
    )
    vertex_normals = np.zeros_like(verts, dtype=np.float64)
    if color_sampler is not None or cage_extrusion_distance > 0.0:
        for corner in range(3):
            np.add.at(vertex_normals, tri[:, corner], weighted_face_normals)
        vertex_lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = np.divide(
            vertex_normals, vertex_lengths,
            out=np.zeros_like(vertex_normals), where=vertex_lengths > 1e-12,
        )
        invalid_vertices = vertex_lengths[:, 0] <= 1e-12
        if np.any(invalid_vertices):
            vertex_normals[invalid_vertices] = np.array([0.0, 0.0, 1.0])

    def _report(fraction: Optional[float], message: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(fraction, message)
        except Exception:
            pass

    preview_interval = max(1, int(np.ceil(len(tri) / 24.0)))
    next_preview_at = preview_interval
    preview_geometry_sent = False
    last_preview_completed = -1
    last_edge_preview_at = 0.0
    live_cage_outer = None
    live_cage_inner = None

    if cage_extrusion_distance > 0.0:
        live_cage_outer = np.ascontiguousarray(
            verts + vertex_normals * float(cage_extrusion_distance), dtype=np.float32
        )

    def _emit_active_face_edges(completed: int, *, force: bool = False) -> None:
        nonlocal last_edge_preview_at
        if edge_preview_callback is None:
            return
        now = time.perf_counter()
        if not force and now - last_edge_preview_at < (1.0 / 30.0):
            return
        recent, current = _recent_face_edge_segments(
            display_vertices, uv_tri, completed, trail_faces=10
        )
        try:
            edge_preview_callback({
                "recent_edge_segments": recent,
                "current_edge_segments": current,
                "completed_faces": int(max(0, min(completed, len(tri)))),
                "total_faces": int(len(tri)),
            })
            last_edge_preview_at = now
        except Exception:
            pass

    def _emit_preview(completed: int, force: bool = False, active_range=None) -> None:
        nonlocal next_preview_at, preview_geometry_sent, last_preview_completed
        if preview_callback is None:
            return
        completed = max(0, min(int(completed), len(tri)))
        if not force and completed < next_preview_at:
            return
        if completed == last_preview_completed and not force:
            return
        while next_preview_at <= completed:
            next_preview_at += preview_interval
        try:
            from PIL import Image

            max_side = 768
            if resolution > max_side:
                source_image = Image.fromarray(atlas, mode="RGBA")
                resampling = getattr(Image, "Resampling", Image)
                preview_image = np.asarray(
                    source_image.resize((max_side, max_side), resampling.BILINEAR),
                    dtype=np.uint8,
                ).copy()
            else:
                preview_image = atlas.copy()
            payload = {
                "texture_image": preview_image,
                "completed_charts": completed,
                "total_charts": int(len(tri)),
                "fraction": float(completed / max(1, len(tri))),
                "label": f"LIVE BAKE · {completed:,}/{len(tri):,} triangles · connected UV charts",
            }
            if diffuse_atlas is not None:
                diffuse_source = Image.fromarray(diffuse_atlas, mode="RGBA")
                if resolution > max_side:
                    diffuse_preview = np.asarray(
                        diffuse_source.resize((max_side, max_side), resampling.BILINEAR),
                        dtype=np.uint8,
                    ).copy()
                else:
                    diffuse_preview = np.asarray(diffuse_source, dtype=np.uint8).copy()
                payload["diffuse_texture_image"] = diffuse_preview
            if not preview_geometry_sent:
                face_count = len(tri)
                if face_count > 100_000:
                    selected = np.linspace(0, face_count - 1, 100_000, dtype=np.int64)
                else:
                    selected = np.arange(face_count, dtype=np.int64)
                selected_faces = uv_tri[selected]
                used_vertices = np.unique(selected_faces)
                remap = np.full(len(uv), -1, dtype=np.int64)
                remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
                payload.update({
                    "vertices": np.ascontiguousarray(display_vertices[used_vertices], dtype=np.float32),
                    "faces": np.ascontiguousarray(remap[selected_faces], dtype=np.uint32),
                    "uvs": np.ascontiguousarray(uv[used_vertices], dtype=np.float32),
                    "preview_face_count": int(len(selected)),
                })
                if live_cage_outer is not None:
                    cage_outer, cage_faces = compact_preview_mesh(
                        live_cage_outer, tri, max_faces=60_000
                    )
                    cage_inner, _ = compact_preview_mesh(
                        live_cage_inner, tri, max_faces=60_000
                    )
                    payload.update({
                        "cage_outer_vertices": cage_outer,
                        "cage_inner_vertices": cage_inner,
                        "cage_faces": cage_faces,
                        "cage_distance": float(cage_extrusion_distance),
                    })
            if active_range is not None:
                start, stop = active_range
                ids = np.arange(max(0, start), min(len(tri), stop), dtype=np.int64)
                if len(ids) > 5_000:
                    ids = ids[np.linspace(0, len(ids) - 1, 5_000, dtype=np.int64)]
                if len(ids):
                    active_faces = uv_tri[ids]
                    active_used = np.unique(active_faces)
                    active_remap = np.full(len(uv), -1, dtype=np.int64)
                    active_remap[active_used] = np.arange(len(active_used), dtype=np.int64)
                    payload.update({
                        "active_vertices": np.ascontiguousarray(display_vertices[active_used]),
                        "active_faces": np.ascontiguousarray(active_remap[active_faces], dtype=np.uint32),
                        "active_chart_start": int(start),
                        "active_chart_stop": int(stop),
                    })
            preview_callback(payload)
            if "vertices" in payload:
                preview_geometry_sent = True
            last_preview_completed = completed
        except Exception:
            pass

    _report(0.0, f"Preparing packed UV atlas · {len(tri):,} triangles")
    _emit_active_face_edges(0, force=True)
    _emit_preview(0, force=True)

    pending_points: list[np.ndarray] = []
    pending_normals: list[np.ndarray] = []
    pending_pixels: list[tuple[np.ndarray, np.ndarray]] = []
    pending_count = 0
    max_sample_batch = 10_000

    def _flush_sample_batch() -> None:
        nonlocal pending_points, pending_normals, pending_pixels, pending_count
        if not pending_count:
            return
        sample_points = np.concatenate(pending_points, axis=0)
        sample_normals = np.concatenate(pending_normals, axis=0)
        sampled = np.asarray(color_sampler(sample_points, sample_normals), dtype=np.float32)
        if sampled.ndim != 2 or sampled.shape[0] != pending_count or sampled.shape[1] < 3:
            raise ValueError("Source color sampler returned an invalid RGBA array.")
        if sampled.shape[1] == 3:
            sampled = np.column_stack((sampled, np.ones(len(sampled), dtype=np.float32)))
        sampled = np.clip(np.nan_to_num(sampled, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        sampled_map_blocks = _pbr_sample_blocks(sampled, map_keys) if map_keys else None
        sampled_base = sampled[:, :4]
        cursor = 0
        for xs, ys in pending_pixels:
            count = len(xs)
            patch = np.rint(sampled_base[cursor:cursor + count] * 255.0).astype(np.uint8)
            atlas[ys, xs] = patch
            if sampled_map_blocks is not None:
                for map_key, block in sampled_map_blocks.items():
                    if map_key == "base_color":
                        continue
                    map_patch = np.rint(block[cursor:cursor + count] * 255.0).astype(np.uint8)
                    pbr_atlases[map_key][ys, xs] = map_patch
            covered[ys, xs] = True
            cursor += count
        pending_points = []
        pending_normals = []
        pending_pixels = []
        pending_count = 0

    face_report_interval = max(1, len(tri) // 100)
    for face_index, face in enumerate(tri):
        if face_index % 32 == 0:
            _check_bake_cancelled(cancel_check)
        uv_face = uv_tri[face_index]
        p = pixel_uv[uv_face]
        edge0 = p[1] - p[0]
        edge1 = p[2] - p[0]
        denominator = float(edge0[0] * edge1[1] - edge0[1] * edge1[0])
        if abs(denominator) <= 1e-8:
            continue

        xmin = max(0, int(np.floor(float(np.min(p[:, 0])))))
        xmax = min(resolution - 1, int(np.ceil(float(np.max(p[:, 0])))))
        ymin = max(0, int(np.floor(float(np.min(p[:, 1])))))
        ymax = min(resolution - 1, int(np.ceil(float(np.max(p[:, 1])))))
        if xmax < xmin or ymax < ymin:
            continue

        x_values = np.arange(xmin, xmax + 1, dtype=np.float64)
        stripe_height = max(1, min(128, max_sample_batch // max(1, len(x_values))))
        target_triangle = verts[face]
        vertex_normal_triangle = vertex_normals[face]
        face_normal = face_normals[face_index]
        source_colors = rgba[face]

        for stripe_start in range(ymin, ymax + 1, stripe_height):
            stripe_stop = min(ymax + 1, stripe_start + stripe_height)
            y_values = np.arange(stripe_start, stripe_stop, dtype=np.float64)
            grid_x, grid_y = np.meshgrid(x_values, y_values)
            qx = grid_x - p[0, 0]
            qy = grid_y - p[0, 1]
            b1 = (qx * edge1[1] - qy * edge1[0]) / denominator
            b2 = (edge0[0] * qy - edge0[1] * qx) / denominator
            b0 = 1.0 - b1 - b2
            inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6)
            if not np.any(inside):
                continue
            bary = np.column_stack((b0[inside], b1[inside], b2[inside]))
            bary = np.clip(bary, 0.0, 1.0)
            bary_sum = bary.sum(axis=1, keepdims=True)
            bary = np.divide(bary, bary_sum, out=np.zeros_like(bary), where=bary_sum > 1e-12)
            px = grid_x[inside].astype(np.int32)
            py = grid_y[inside].astype(np.int32)

            if color_sampler is None:
                patch = np.clip(bary @ source_colors, 0.0, 1.0)
                atlas[py, px] = np.rint(patch * 255.0).astype(np.uint8)
                for map_key in map_keys:
                    if map_key == "base_color":
                        continue
                    map_patch = np.clip(bary @ vertex_map_rgba[map_key][face], 0.0, 1.0)
                    pbr_atlases[map_key][py, px] = np.rint(map_patch * 255.0).astype(np.uint8)
                covered[py, px] = True
            else:
                points = bary @ target_triangle
                normals = bary @ vertex_normal_triangle
                lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = np.divide(
                    normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12
                )
                invalid = lengths[:, 0] <= 1e-12
                if np.any(invalid):
                    fallback = face_normal if np.linalg.norm(face_normal) > 1e-12 else np.array([0.0, 0.0, 1.0])
                    normals[invalid] = fallback
                pending_points.append(points)
                pending_normals.append(normals)
                pending_pixels.append((px, py))
                pending_count += len(px)
                if pending_count >= max_sample_batch:
                    _flush_sample_batch()

        completed = face_index + 1
        _emit_active_face_edges(completed)
        if completed % face_report_interval == 0 or completed == len(tri):
            if color_sampler is not None:
                _flush_sample_batch()
            _emit_preview(completed, active_range=(face_index, completed))
            _report(
                completed / len(tri),
                f"UV atlas · rasterized {completed:,}/{len(tri):,} triangles",
            )

    if color_sampler is not None:
        _flush_sample_batch()
    if not np.any(covered):
        raise ValueError("UV rasterization produced no covered atlas texels.")
    _extend_uv_chart_gutters(atlas, covered, iterations=max(1, min(4, int(chart_padding) // 2)))
    for map_key, map_atlas in pbr_atlases.items():
        if map_key != "base_color":
            _extend_uv_chart_gutters(
                map_atlas, covered, iterations=max(1, min(4, int(chart_padding) // 2))
            )
    if diffuse_atlas_out is not None and diffuse_atlas is not None:
        diffuse_atlas_out["image"] = diffuse_atlas
    if pbr_atlases_out is not None and pbr_atlases:
        pbr_atlases_out.update(pbr_atlases)
    _check_bake_cancelled(cancel_check)
    _emit_active_face_edges(len(tri), force=True)
    _emit_preview(len(tri), force=True)
    _report(1.0, f"Packed UV atlas complete · {len(tri):,} triangles")
    return atlas, corner_uvs, expanded_vertices

class _SourceSurfaceAppearanceSampler:
    """Nearest-surface and cage-ray sampler for source appearance.

    The reconstruction mesh is the bake target. Its face normals define an
    automatically offset two-sided cage shell; rays travel inward from that
    shell and intersect the original source triangles. A nearest-surface
    fallback is retained for cage misses; ``sample`` also exposes direct
    nearest-surface sampling when the user does not want a cage.
    """

    def __init__(
        self, source_path: str | Path, source_up_axis: str = "y", candidates: int = 48,
        progress_callback: Optional[Callable[[Optional[float], str], None]] = None,
        compute_device: str = "auto", cancel_check: Optional[Callable[[], bool]] = None,
        pbr_map_keys: Optional[tuple[str, ...] | list[str]] = None,
        source_frame_transform=None,
    ):
        import trimesh
        from scipy.spatial import cKDTree

        try:
            from .pointio import (
                _sample_texture_rgba, _material_base_color,
                _apply_linear_base_color_factor, mesh_vertex_display_colors,
                _material_occlusion_strength,
            )
        except ImportError:
            from pointio import (
                _sample_texture_rgba, _material_base_color,
                _apply_linear_base_color_factor, mesh_vertex_display_colors,
                _material_occlusion_strength,
            )

        self._trimesh = trimesh
        self._sample_texture_rgba = _sample_texture_rgba
        self._material_base_color = _material_base_color
        self._apply_linear_base_color_factor = _apply_linear_base_color_factor
        self._mesh_vertex_display_colors = mesh_vertex_display_colors
        self.records: list[dict] = []
        requested = normalize_pbr_map_selection(pbr_map_keys or ("base_color",))
        self.selected_pbr_maps = requested or ("base_color",)

        self.pbr_map_keys = tuple(dict.fromkeys(
            ("base_color",) + self.selected_pbr_maps + ("orm",)
        ))
        self.candidates = max(1, int(candidates))
        self.has_image_texture = False
        self.available_pbr_maps: set[str] = set()
        self.cage_sample_count = 0
        self.cage_hit_count = 0
        self.cage_fallback_count = 0
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.source_frame_transform = as_transform_matrix(source_frame_transform)
        _check_bake_cancelled(self.cancel_check)
        self.compute_manager = GPUComputeManager(
            selection=compute_device, progress_callback=self.progress_callback
        )

        if self.progress_callback is not None:
            self.progress_callback(None, f"Step 2/7 · Reading source mesh: {Path(source_path).name}")
        loaded = trimesh.load(str(source_path), process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = []
            for node in loaded.graph.nodes_geometry:
                try:
                    transform, geometry_name = loaded.graph[node]
                    geom = loaded.geometry[geometry_name].copy()
                    if hasattr(geom, "apply_transform"):
                        geom.apply_transform(transform)
                    geometries.append(geom)
                except Exception:
                    continue
        else:
            geometries = [loaded]

        axis = str(source_up_axis or "y").lower()
        valid_geometries = [
            geom for geom in geometries
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
        ]
        for geometry_index, geom in enumerate(valid_geometries, start=1):
            _check_bake_cancelled(self.cancel_check)
            if self.progress_callback is not None:
                self.progress_callback(
                    None,
                    f"Step 3/7 · Building source spatial index "
                    f"({geometry_index}/{len(valid_geometries)} mesh parts)",
                )
            mesh_vertices = np.asarray(geom.vertices, dtype=np.float64).copy()
            if axis == "y":
                mesh_vertices = np.column_stack((mesh_vertices[:, 0], -mesh_vertices[:, 2], mesh_vertices[:, 1]))
            elif axis == "x":
                mesh_vertices = np.column_stack((-mesh_vertices[:, 2], mesh_vertices[:, 1], mesh_vertices[:, 0]))
            elif axis != "z":
                raise ValueError(f"Unsupported source up-axis: {source_up_axis!r}.")
            mesh_vertices = points_to_source_local(
                mesh_vertices, self.source_frame_transform
            )

            faces = np.asarray(geom.faces, dtype=np.int64)
            triangles = np.ascontiguousarray(mesh_vertices[faces], dtype=np.float64)
            if len(triangles) == 0:
                continue
            centroids = triangles.mean(axis=1)
            tree = cKDTree(centroids)
            visual = getattr(geom, "visual", None)
            material = getattr(visual, "material", None)
            uv = getattr(visual, "uv", None)
            image = getattr(material, "baseColorTexture", None)
            if image is None:
                image = getattr(material, "image", None)
            if image is not None and not hasattr(image, "convert"):
                try:
                    from PIL import Image
                    image = Image.fromarray(np.asarray(image))
                except Exception:
                    image = None
            if uv is not None:
                uv = np.asarray(uv, dtype=np.float64)
                if uv.ndim != 2 or uv.shape[1] < 2 or len(uv) != len(mesh_vertices):
                    uv = None
            if image is not None and uv is not None:
                self.has_image_texture = True
            source_vertex_color_space = (
                "linear" if Path(source_path).suffix.lower() in {".glb", ".gltf"} else "srgb"
            )
            vertex_colors = mesh_vertex_display_colors(
                geom, vertex_color_space=source_vertex_color_space
            )
            if vertex_colors is None or len(vertex_colors) != len(mesh_vertices):
                vertex_colors = np.ones((len(mesh_vertices), 4), dtype=np.float32)
            else:
                vertex_colors = np.asarray(vertex_colors, dtype=np.float32)
                if vertex_colors.ndim != 2 or vertex_colors.shape[1] < 3:
                    vertex_colors = np.ones((len(mesh_vertices), 4), dtype=np.float32)
                elif vertex_colors.shape[1] == 3:
                    vertex_colors = np.column_stack((vertex_colors, np.ones(len(vertex_colors), dtype=np.float32)))
                vertex_colors = np.clip(np.nan_to_num(vertex_colors[:, :4], nan=1.0), 0.0, 1.0)

            def _image_attribute(*names):
                for name in names:
                    candidate = getattr(material, name, None)
                    if candidate is None:
                        continue
                    if not hasattr(candidate, "convert"):
                        try:
                            from PIL import Image
                            candidate = Image.fromarray(np.asarray(candidate))
                        except Exception:
                            continue
                    return candidate
                return None

            pbr_images = {
                "base_color": image,
                "normal": _image_attribute("normalTexture"),
                "metallic_roughness": _image_attribute("metallicRoughnessTexture"),
                "roughness": _image_attribute("roughnessTexture"),
                "metallic": _image_attribute("metallicTexture"),
                "ao": _image_attribute("occlusionTexture", "aoTexture"),
                "height": _image_attribute("heightTexture", "displacementTexture"),
                "opacity": _image_attribute("opacityTexture", "alphaTexture"),
                "emissive": _image_attribute("emissiveTexture"),
            }
            if pbr_images["base_color"] is not None:
                self.available_pbr_maps.add("base_color")
            for map_key in ("normal", "ao", "height", "opacity", "emissive"):
                if pbr_images[map_key] is not None:
                    self.available_pbr_maps.add(map_key)
            if pbr_images["metallic_roughness"] is not None or pbr_images["roughness"] is not None:
                self.available_pbr_maps.add("roughness")
            if pbr_images["metallic_roughness"] is not None or pbr_images["metallic"] is not None:
                self.available_pbr_maps.add("metallic")
            if pbr_images["ao"] is not None:
                self.available_pbr_maps.add("ao")
            if any(pbr_images.get(key) is not None for key in ("ao", "metallic_roughness", "roughness", "metallic")):
                self.available_pbr_maps.add("orm")

            emissive_factor = np.asarray(
                getattr(material, "emissiveFactor", (0.0, 0.0, 0.0)),
                dtype=np.float32,
            ).reshape(-1)
            if emissive_factor.size < 3 or not np.isfinite(emissive_factor[:3]).all():
                emissive_factor = np.zeros(3, dtype=np.float32)
            else:
                emissive_factor = np.clip(emissive_factor[:3], 0.0, 1.0)

            self.records.append({
                "triangles": triangles,
                "faces": faces,
                "tree": tree,
                "vertex_colors": vertex_colors,
                "uv": uv,
                "image": image,
                "pbr_images": pbr_images,
                "base_color_factor": _material_base_color(material),
                "roughness_factor": float(
                    1.0 if getattr(material, "roughnessFactor", None) is None
                    else np.clip(getattr(material, "roughnessFactor"), 0.0, 1.0)
                ),
                "metallic_factor": float(
                    0.0 if getattr(material, "metallicFactor", None) is None
                    else np.clip(getattr(material, "metallicFactor"), 0.0, 1.0)
                ),
                "occlusion_strength": _material_occlusion_strength(material),
                "emissive_factor": emissive_factor,
            })

        if not self.records:
            raise ValueError("The source file has no triangle surface to sample.")
        if self.progress_callback is not None:
            self.progress_callback(0.20, f"Step 3/7 · Source spatial index ready ({len(self.records)} mesh parts)")

    def _sample_record_faces(
        self, record: dict, face_ids: np.ndarray, barycentric: np.ndarray
    ) -> np.ndarray:
        """Sample selected PBR maps without applying layer/material multiplication."""
        face_ids = np.asarray(face_ids, dtype=np.int64).reshape(-1)
        bary = np.asarray(barycentric, dtype=np.float64).reshape(-1, 3)
        face_vertices = record["faces"][face_ids]
        uv = record["uv"]
        source_uv = (
            np.einsum("ij,ijk->ik", bary, uv[face_vertices])
            if uv is not None else None
        )
        tri_colors = np.clip(
            np.einsum("ij,ijk->ik", bary, record["vertex_colors"][face_vertices]),
            0.0, 1.0,
        )

        image_samples: dict[str, np.ndarray | None] = {}
        for key, image in record["pbr_images"].items():
            image_samples[key] = (
                self._sample_texture_rgba(image, source_uv)
                if image is not None and source_uv is not None else None
            )

        base_image = image_samples.get("base_color")
        if base_image is not None:

            base_color = self._apply_linear_base_color_factor(
                np.asarray(base_image, dtype=np.float32),
                np.asarray(record.get("base_color_factor", (1, 1, 1, 1)), dtype=np.float32),
            )
        else:

            base_color = tri_colors

        def _channel(values, channel: int, fallback: float) -> np.ndarray:
            if values is None:
                return np.full(len(bary), float(fallback), dtype=np.float32)
            return np.clip(np.asarray(values, dtype=np.float32)[:, channel], 0.0, 1.0)

        rough_image = image_samples.get("roughness")
        metallic_image = image_samples.get("metallic")
        mr_image = image_samples.get("metallic_roughness")

        roughness_texel = _channel(
            rough_image if rough_image is not None else mr_image,
            0 if rough_image is not None else 1,
            1.0,
        )
        roughness = np.clip(
            roughness_texel * float(record["roughness_factor"]), 0.0, 1.0
        )
        metallic_texel = _channel(
            metallic_image if metallic_image is not None else mr_image,
            0 if metallic_image is not None else 2,
            1.0,
        )
        metallic = np.clip(
            metallic_texel * float(record["metallic_factor"]), 0.0, 1.0
        )
        ao_texel = _channel(image_samples.get("ao"), 0, 1.0)

        ao_strength = float(np.clip(record.get("occlusion_strength", 1.0), 0.0, 1.0))
        ao = np.clip(1.0 - (1.0 - ao_texel) * ao_strength, 0.0, 1.0)
        height = _channel(image_samples.get("height"), 0, 0.5)
        opacity_image = image_samples.get("opacity")
        if opacity_image is not None:
            opacity = _channel(opacity_image, 0, 1.0)
        elif base_image is not None:
            opacity = _channel(base_image, 3, 1.0)
        else:
            opacity = np.clip(tri_colors[:, 3], 0.0, 1.0)
        emissive_image = image_samples.get("emissive")
        if emissive_image is not None:
            emissive_rgb = np.asarray(emissive_image[:, :3], dtype=np.float32)
        else:
            emissive_rgb = np.tile(
                np.asarray(record["emissive_factor"], dtype=np.float32).reshape(1, 3),
                (len(bary), 1),
            )
        normal_image = image_samples.get("normal")

        channel_values: dict[str, np.ndarray] = {
            "base_color": base_color,
            "normal": (
                np.asarray(normal_image, dtype=np.float32)
                if normal_image is not None else
                np.tile(np.asarray(PBR_MAP_DEFAULTS["normal"], dtype=np.float32), (len(bary), 1))
            ),
            "roughness": np.column_stack((roughness, roughness, roughness, np.ones(len(bary), dtype=np.float32))),
            "metallic": np.column_stack((metallic, metallic, metallic, np.ones(len(bary), dtype=np.float32))),
            "ao": np.column_stack((ao, ao, ao, np.ones(len(bary), dtype=np.float32))),
            "height": np.column_stack((height, height, height, np.ones(len(bary), dtype=np.float32))),
            "opacity": np.column_stack((opacity, opacity, opacity, np.ones(len(bary), dtype=np.float32))),
            "emissive": np.column_stack((emissive_rgb, np.ones(len(bary), dtype=np.float32))),
            "orm": np.column_stack((ao, roughness, metallic, np.ones(len(bary), dtype=np.float32))),
        }
        blocks = [
            np.clip(np.nan_to_num(channel_values[key], nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            for key in self.pbr_map_keys
        ]
        return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

    def sample_cage(
        self, points: np.ndarray, normals: np.ndarray, cage_distance: float
    ) -> np.ndarray:
        """Bake from a generated cage using normal-directed source ray hits.

        ``cage_distance`` is the outward extrusion on each side of the target
        surface. For each target texel, rays start at +/- cage_distance and
        travel inward through the target point for a maximum distance of twice
        the extrusion. If no source triangle is hit, nearest-surface sampling
        supplies a fallback color and is tracked separately.
        """
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        normal = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        if len(normal) != len(query):
            raise ValueError("Cage projection requires one target normal per sample point.")
        if not np.isfinite(query).all() or not np.isfinite(normal).all():
            raise ValueError("Cage projection received non-finite positions or normals.")
        distance = float(cage_distance)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("Cage extrusion distance must be greater than zero.")
        lengths = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = np.divide(normal, lengths, out=np.zeros_like(normal), where=lengths > 1e-12)
        invalid_normals = lengths[:, 0] <= 1e-12
        normal[invalid_normals] = np.array([0.0, 0.0, 1.0])

        count = len(query)
        output = np.concatenate([
            np.tile(np.asarray(PBR_MAP_DEFAULTS[key], dtype=np.float32), (count, 1))
            for key in self.pbr_map_keys
        ], axis=1)
        best_t = np.full(count, np.inf, dtype=np.float64)
        best_record = np.full(count, -1, dtype=np.int32)
        best_face = np.full(count, -1, dtype=np.int64)
        best_bary = np.zeros((count, 3), dtype=np.float64)
        ray_limit = 2.0 * distance
        epsilon = max(1e-10, distance * 1e-8)

        for record_index, record in enumerate(self.records):
            triangles = record["triangles"]
            face_count = len(triangles)
            k = min(self.candidates, face_count)
            for start in range(0, count, _KDTREE_QUERY_BATCH_SIZE):
                _check_bake_cancelled(self.cancel_check)
                stop = min(count, start + _KDTREE_QUERY_BATCH_SIZE)
                block = query[start:stop]
                block_normals = normal[start:stop]

                _d0, ids_center = _query_nearest_face_candidates(record["tree"], block, k)
                _d1, ids_plus = _query_nearest_face_candidates(
                    record["tree"], block + block_normals * distance, k
                )
                _d2, ids_minus = _query_nearest_face_candidates(
                    record["tree"], block - block_normals * distance, k
                )
                ids_center = np.asarray(ids_center, dtype=np.int64).reshape(len(block), k)
                ids_plus = np.asarray(ids_plus, dtype=np.int64).reshape(len(block), k)
                ids_minus = np.asarray(ids_minus, dtype=np.int64).reshape(len(block), k)
                candidate_ids = np.sort(
                    np.concatenate((ids_center, ids_plus, ids_minus), axis=1), axis=1
                )

                unique_mask = np.ones(candidate_ids.shape, dtype=bool)
                unique_mask[:, 1:] = candidate_ids[:, 1:] != candidate_ids[:, :-1]
                unique_counts = unique_mask.sum(axis=1)
                unique_order = np.argsort(~unique_mask, axis=1, kind="stable")
                candidate_ids = np.take_along_axis(
                    candidate_ids, unique_order, axis=1
                )[:, :int(unique_counts.max())]
                candidate_triangles = np.ascontiguousarray(triangles[candidate_ids])

                ray_origins = np.stack((
                    block + block_normals * distance,
                    block - block_normals * distance,
                ), axis=1)
                ray_directions = np.stack((-block_normals, block_normals), axis=1)

                distances, u, v = self.compute_manager.intersect_rays(
                    ray_origins, ray_directions, candidate_triangles,
                    epsilon=epsilon, ray_limit=ray_limit,
                )
                flat_distances = distances.reshape(len(block), -1)
                selected_flat = np.argmin(flat_distances, axis=1)
                selected_t = flat_distances[np.arange(len(block)), selected_flat]
                selected_ray = selected_flat // candidate_ids.shape[1]
                selected_candidate = selected_flat % candidate_ids.shape[1]
                rows = np.arange(len(block), dtype=np.int64)
                selected_u = u[rows, selected_ray, selected_candidate]
                selected_v = v[rows, selected_ray, selected_candidate]
                selected_bary = np.column_stack((
                    1.0 - selected_u - selected_v, selected_u, selected_v
                ))
                selected_bary = np.clip(selected_bary, 0.0, 1.0)
                selected_bary /= np.maximum(selected_bary.sum(axis=1, keepdims=True), 1e-12)
                selected_faces = candidate_ids[rows, selected_candidate]

                current_best = best_t[start:stop]
                improve = selected_t < current_best
                if np.any(improve):
                    current_best[improve] = selected_t[improve]
                    record_slice = best_record[start:stop]
                    face_slice = best_face[start:stop]
                    bary_slice = best_bary[start:stop]
                    record_slice[improve] = record_index
                    face_slice[improve] = selected_faces[improve]
                    bary_slice[improve] = selected_bary[improve]

        hit_mask = best_record >= 0
        for record_index, record in enumerate(self.records):
            mask = hit_mask & (best_record == record_index)
            if np.any(mask):
                output[mask] = self._sample_record_faces(
                    record, best_face[mask], best_bary[mask]
                )

        miss_mask = ~hit_mask
        if np.any(miss_mask):

            output[miss_mask] = self.sample(query[miss_mask])

        self.cage_sample_count += count
        self.cage_hit_count += int(np.count_nonzero(hit_mask))
        self.cage_fallback_count += int(np.count_nonzero(miss_mask))
        return output

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Sample the nearest source surface appearance as a cage-miss fallback."""
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        output = np.concatenate([
            np.tile(np.asarray(PBR_MAP_DEFAULTS[key], dtype=np.float32), (len(query), 1))
            for key in self.pbr_map_keys
        ], axis=1)
        best_distance = np.full(len(query), np.inf, dtype=np.float64)
        best_record = np.full(len(query), -1, dtype=np.int32)
        best_face = np.full(len(query), -1, dtype=np.int64)
        best_bary = np.zeros((len(query), 3), dtype=np.float64)

        for record_index, record in enumerate(self.records):
            triangles = record["triangles"]
            face_count = len(triangles)
            k = min(self.candidates, face_count)
            for start in range(0, len(query), _KDTREE_QUERY_BATCH_SIZE):
                _check_bake_cancelled(self.cancel_check)
                stop = min(len(query), start + _KDTREE_QUERY_BATCH_SIZE)
                block = query[start:stop]
                _centroid_distance, candidate_ids = _query_nearest_face_candidates(
                    record["tree"], block, k
                )
                candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
                if candidate_ids.ndim == 1:
                    candidate_ids = candidate_ids[:, None]
                count = len(block)
                candidate_triangles = np.ascontiguousarray(triangles[candidate_ids])
                d2, _candidate_u, _candidate_v = self.compute_manager.closest_points(
                    block, candidate_triangles
                )
                d2 = np.asarray(d2, dtype=np.float64).reshape(count, k)
                local_choice = np.argmin(d2, axis=1)
                rows = np.arange(count, dtype=np.int64)
                chosen_face_ids = candidate_ids[rows, local_choice]
                selected_u = np.asarray(_candidate_u, dtype=np.float64)[rows, local_choice]
                selected_v = np.asarray(_candidate_v, dtype=np.float64)[rows, local_choice]
                bary = np.column_stack((1.0 - selected_u - selected_v, selected_u, selected_v))
                bary = np.clip(np.nan_to_num(bary, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=0.0), 0.0, 1.0)
                bary_sum = bary.sum(axis=1, keepdims=True)
                bary = np.divide(bary, bary_sum, out=np.full_like(bary, 1.0 / 3.0), where=bary_sum > 1e-12)

                dist = d2[rows, local_choice]
                target_slice = slice(start, stop)
                improve = dist < best_distance[target_slice]
                if np.any(improve):
                    best_distance[target_slice][improve] = dist[improve]
                    best_record[start:stop][improve] = record_index
                    best_face[start:stop][improve] = chosen_face_ids[improve]
                    best_bary[start:stop][improve] = bary[improve]

        valid = best_record >= 0
        for record_index, record in enumerate(self.records):
            mask = valid & (best_record == record_index)
            if np.any(mask):
                output[mask] = self._sample_record_faces(
                    record, best_face[mask], best_bary[mask]
                )

        return output

def _load_source_sampler(
    source_mesh_path: str | Path | None,
    source_up_axis: str = "y",
    progress_callback: Optional[Callable[[Optional[float], str], None]] = None,
    compute_device: str = "auto",
    cancel_check: Optional[Callable[[], bool]] = None,
    candidates: int = 48,
    pbr_map_keys: Optional[tuple[str, ...] | list[str]] = None,
    source_frame_transform=None,
):
    if not source_mesh_path:
        return None, "no source mesh supplied"
    path = Path(source_mesh_path)
    if not path.is_file():
        return None, f"source mesh not found: {path}"
    try:
        if progress_callback is not None:
            progress_callback(None, f"Step 2/7 · Loading original source mesh and materials: {path.name}")
        sampler = _SourceSurfaceAppearanceSampler(
            path, source_up_axis=source_up_axis, progress_callback=progress_callback,
            compute_device=compute_device, cancel_check=cancel_check,
            candidates=candidates, pbr_map_keys=pbr_map_keys,
            source_frame_transform=source_frame_transform,
        )
        return sampler, ""
    except (BakeCancelledError, GPUComputeRequiredError):
        raise
    except ImportError as exc:
        return None, f"source-surface sampling unavailable ({exc}); using transferred vertex colors"
    except Exception as exc:
        return None, f"source-surface sampling skipped ({exc}); using transferred vertex colors"

def load_generated_obj_for_bake(
    obj_path: str | Path,
    color_preview_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    """Load the final generated OBJ as the authoritative GLB bake target.

    The preview NPZ is permitted only as a vertex-color sidecar. Its geometry
    is never used for export, and its colors are accepted only when its vertex
    positions match the generated OBJ in the same order.
    """
    import trimesh

    path = Path(obj_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Generated OBJ is missing or empty: {path}")

    loaded = load_obj_trimesh(path)
    if isinstance(loaded, trimesh.Scene):
        parts = []
        for node in loaded.graph.nodes_geometry:
            try:
                transform, geometry_name = loaded.graph[node]
                geometry = loaded.geometry[geometry_name].copy()
                if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces):
                    geometry.apply_transform(transform)
                    parts.append(geometry)
            except Exception:
                continue
        if not parts:
            raise ValueError(f"Generated OBJ contains no triangle geometry: {path}")
        loaded = trimesh.util.concatenate(parts)

    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) < 3 or len(loaded.faces) == 0:
        raise ValueError(f"Generated OBJ contains no valid triangle mesh: {path}")

    vertices = np.asarray(loaded.vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(loaded.faces, dtype=np.int64).reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise ValueError("Generated OBJ contains non-finite vertex coordinates.")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("Generated OBJ contains invalid triangle indices.")

    colors = None
    colors_reused = False
    preview_path = Path(color_preview_path) if color_preview_path else None
    if preview_path is not None and preview_path.is_file():
        try:
            with np.load(preview_path) as preview:
                preview_vertices = np.asarray(preview["vertices"], dtype=np.float32).reshape(-1, 3)
                preview_colors = (
                    np.asarray(preview["colors"], dtype=np.float32)
                    if "colors" in preview.files else None
                )

            same_geometry = (
                preview_vertices.shape == vertices.shape
                and np.allclose(preview_vertices, vertices, rtol=1e-6, atol=1e-7)
            )
            if (
                same_geometry
                and preview_colors is not None
                and preview_colors.ndim == 2
                and preview_colors.shape[0] == len(vertices)
                and preview_colors.shape[1] >= 3
            ):
                colors = preview_colors
                colors_reused = True
        except (OSError, KeyError, ValueError):

            colors = None

    info = {
        "target_mesh_source": "generated_obj",
        "target_mesh_path": str(path),
        "target_vertex_count": int(len(vertices)),
        "target_triangle_count": int(len(faces)),
        "preview_colors_reused": bool(colors_reused),
    }
    return vertices, faces, colors, info

def export_textured_glb(
    output_path: str | Path,
    vertices: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    colors: Optional[np.ndarray] = None,
    resolution: int = 2048,
    source_mesh_path: str | Path | None = None,
    source_up_axis: str = "y",
    source_frame_transform=None,
    projection_mode: str = "cage",
    cage_extrusion_percent: float = 5.0,
    target_mesh_path: str | Path | None = None,
    vertex_color_preview_path: str | Path | None = None,
    progress_callback: Optional[Callable[[Optional[float], str], None]] = None,
    compute_device: str = "auto",
    preview_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    source_candidates: int = 48,
    edge_preview_callback: Optional[Callable[[dict], None]] = None,
    pbr_maps: Optional[tuple[str, ...] | list[str]] = None,
) -> dict:
    """Bake an embedded-texture GLB using the generated OBJ as authoritative target geometry.

    When ``target_mesh_path`` is supplied, target vertices and triangles are loaded
    from that OBJ; the preview archive contributes vertex colors only when its
    vertex order and coordinates match. Target topology is conservatively
    cleaned before xatlas creates connected UV charts. The exported geometry
    shares vertices inside a chart and only splits them at UV seams. The
    original input mesh is read-only and remains the high-poly appearance
    source; the generated reconstruction OBJ remains the low-poly bake target.
    ``projection_mode='cage'`` projects texels through an automatically
    extruded two-sided cage. ``projection_mode='direct'`` samples the nearest
    source surface without creating a cage. Source UV images, materials and
    vertex colors are evaluated in either mode. Point-cloud inputs fall back
    to transferred output-vertex colors; unpainted inputs are white.
    """
    import trimesh
    from PIL import Image
    from trimesh.visual.texture import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    def _report(fraction: Optional[float], message: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(fraction, message)
        except Exception:
            pass

    _check_bake_cancelled(cancel_check)
    projection_mode = str(projection_mode or "cage").strip().lower()
    if projection_mode not in {"cage", "direct"}:
        raise ValueError("Bake projection mode must be 'cage' or 'direct'.")
    selected_pbr_maps = normalize_pbr_map_selection(
        ("base_color",) if pbr_maps is None else pbr_maps
    )

    baked_pbr_maps = tuple(dict.fromkeys(selected_pbr_maps + ("orm",)))
    sampler_map_keys = tuple(dict.fromkeys(("base_color",) + baked_pbr_maps))
    target_info = {}
    _report(None, "Step 1/7 · Reading generated reconstruction OBJ")
    if target_mesh_path is not None:
        vertices, faces, colors, target_info = load_generated_obj_for_bake(
            target_mesh_path, color_preview_path=vertex_color_preview_path
        )
    _report(0.08, "Step 1/7 · Generated OBJ loaded; validating target geometry")
    if vertices is None or faces is None:
        raise ValueError("A generated OBJ target or explicit target mesh arrays are required for GLB baking.")

    _report(0.09, "Step 1/7 · Cleaning invalid/duplicate target faces and checking winding")
    target_vertices, target_faces, target_colors, topology_info = _prepare_bake_target_geometry(
        vertices, faces, colors
    )
    target_info.update(topology_info)

    _report(0.10, "Step 2/7 · Preparing original source appearance")
    sampler, sampler_warning = _load_source_sampler(
        source_mesh_path, source_up_axis, progress_callback=_report,
        compute_device=compute_device,
        cancel_check=cancel_check,
        candidates=max(1, int(source_candidates)),
        pbr_map_keys=sampler_map_keys,
        source_frame_transform=source_frame_transform,
    )
    _check_bake_cancelled(cancel_check)
    if sampler is None:
        _report(0.20, "Step 3/7 · No source surface index; using transferred vertex colors")
    else:
        _report(0.22, "Step 3/7 · Source surface index ready")
    _report(0.23, "Step 3/7 · Unwrapping connected target surfaces and packing UV charts")
    uv_mesh_vertices, uv_mesh_faces, uv_vertices, uv_chart_count, uv_utilization = (
        _generate_uv_atlas_layout(target_vertices, target_faces, resolution)
    )
    chart_padding = max(4, min(16, int(resolution) // 256))
    try:
        import xatlas as _xatlas_available
        uv_layout_method = "xatlas_connected_charts"
    except ImportError:
        uv_layout_method = "per_triangle_fallback_charts"
    target_info.update({
        "uv_layout_method": uv_layout_method,
        "uv_chart_count": int(uv_chart_count),
        "uv_atlas_utilization": float(uv_utilization),
        "uv_chart_padding_pixels": int(chart_padding),
        "uv_seam_vertices_added": int(max(0, len(uv_mesh_vertices) - len(target_vertices))),
    })
    bounds_diagonal = float(np.linalg.norm(np.ptp(target_vertices, axis=0))) if len(target_vertices) else 0.0
    extrusion_percent = float(cage_extrusion_percent)
    if projection_mode == "cage":
        if not np.isfinite(extrusion_percent) or not (0.1 <= extrusion_percent <= 50.0):
            raise ValueError("Cage extrusion must be between 0.1% and 50% of the result bounds.")
        cage_distance = max(bounds_diagonal * extrusion_percent / 100.0, 1e-7)
    else:
        cage_distance = 0.0

    projection_label = (
        "Cage ray projection" if projection_mode == "cage"
        else "Direct nearest-surface sampling"
    )

    _report(
        0.25,
        f"Step 4/7 · {projection_label} · Baking {resolution}² UV atlas · "
        f"{len(target_faces):,} triangles / {uv_chart_count:,} UV charts",
    )

    def _atlas_progress(fraction: Optional[float], message: str) -> None:
        if fraction is None:
            _report(None, f"Step 4/7 · {message}")
        else:
            mapped = 0.25 + max(0.0, min(1.0, float(fraction))) * 0.58
            _report(mapped, f"Step 4/7 · {message}")

    color_sampler = (
        (lambda sample_points, sample_normals: sampler.sample_cage(
            sample_points, sample_normals, cage_distance
        )) if projection_mode == "cage" else
        (lambda sample_points, sample_normals: sampler.sample(sample_points))
    ) if sampler is not None else None

    diffuse_atlas_result: dict = {}
    pbr_atlas_result: dict = {}
    if uv_layout_method == "per_triangle_fallback_charts":

        fallback_colors = target_colors
        if sampler is not None:
            if projection_mode == "cage":
                _report(0.30, "Step 4/7 · Cage projection · sampling original appearance at target vertices")
                triangles = target_vertices[target_faces]
                face_normals = np.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                )
                vertex_normals = np.zeros_like(target_vertices, dtype=np.float64)
                for corner in range(3):
                    np.add.at(vertex_normals, target_faces[:, corner], face_normals)
                normal_lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
                vertex_normals = np.divide(
                    vertex_normals, normal_lengths,
                    out=np.zeros_like(vertex_normals), where=normal_lengths > 1e-12,
                )
                invalid_normals = normal_lengths[:, 0] <= 1e-12
                if np.any(invalid_normals):
                    vertex_normals[invalid_normals] = np.array([0.0, 0.0, 1.0])
                fallback_colors = sampler.sample_cage(
                    target_vertices, vertex_normals, cage_distance
                )
            else:
                _report(0.30, "Step 4/7 · Direct bake · sampling original appearance at target vertices")
                fallback_colors = sampler.sample(target_vertices)
            target_info["source_appearance_sampling_mode"] = (
                "cage_projection_at_target_vertices_then_barycentric_interpolation"
                if projection_mode == "cage" else
                "direct_nearest_surface_at_target_vertices_then_barycentric_interpolation"
            )
            target_info["source_appearance_vertex_samples"] = int(len(target_vertices))
        else:
            target_info["source_appearance_sampling_mode"] = "provided_vertex_colors"
            target_info["source_appearance_vertex_samples"] = 0

        atlas, corner_uvs, expanded_vertices = bake_triangle_atlas(
            target_vertices, target_faces, fallback_colors,
            resolution=resolution,
            tile_size=None,
            color_sampler=None,
            progress_callback=_atlas_progress,
            preview_callback=preview_callback,
            cancel_check=cancel_check,
            cage_extrusion_distance=cage_distance if sampler is not None else 0.0,
            chart_inner_pixels=None,
            edge_preview_callback=edge_preview_callback,
            diffuse_atlas_out=diffuse_atlas_result,
            pbr_map_keys=sampler_map_keys,
            pbr_atlases_out=pbr_atlas_result,
        )
        uv_mesh_vertices = expanded_vertices
        uv_mesh_faces = np.arange(len(target_faces) * 3, dtype=np.int64).reshape(-1, 3)
        uv_vertices = corner_uvs.reshape(-1, 2)
        target_info["uv_chart_padding_pixels"] = 1
    else:
        atlas, _corner_uvs, _expanded_vertices = bake_unwrapped_mesh_atlas(
            target_vertices, target_faces, uv_vertices, uv_mesh_faces, target_colors,
            uv_mesh_vertices=uv_mesh_vertices,
            resolution=resolution,
            color_sampler=color_sampler,
            progress_callback=_atlas_progress,
            preview_callback=preview_callback,
            cancel_check=cancel_check,
            cage_extrusion_distance=cage_distance if sampler is not None else 0.0,
            chart_padding=chart_padding,
            edge_preview_callback=edge_preview_callback,
            diffuse_atlas_out=diffuse_atlas_result,
            pbr_map_keys=sampler_map_keys,
            pbr_atlases_out=pbr_atlas_result,
        )
    _check_bake_cancelled(cancel_check)
    _report(0.85, "Step 5/7 · Finalizing texture image and PBR material")
    base_rgba = None
    if "base_color" in selected_pbr_maps:
        base_rgba = np.asarray(atlas, dtype=np.uint8).copy()
    elif "opacity" in selected_pbr_maps:
        base_rgba = np.full((resolution, resolution, 4), 255, dtype=np.uint8)
    if base_rgba is not None and "opacity" in selected_pbr_maps:
        opacity_channel = np.asarray(pbr_atlas_result["opacity"], dtype=np.uint8)[..., 0]
        base_rgba[..., 3] = opacity_channel
    base_color_image = (
        Image.fromarray(base_rgba, mode="RGBA") if base_rgba is not None else None
    )
    has_alpha = (
        "opacity" in selected_pbr_maps
        or ("base_color" in selected_pbr_maps and bool(np.any(atlas[..., 3] < 255)))
    )
    material = PBRMaterial(
        name="Polyfitty_Baked_PBR",
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        baseColorTexture=base_color_image,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
        alphaMode="BLEND" if has_alpha else None,
    )
    visual = TextureVisuals(uv=uv_vertices, material=material)
    mesh = trimesh.Trimesh(
        vertices=uv_mesh_vertices,
        faces=uv_mesh_faces,
        visual=visual,
        process=False,
    )
    target_face_components = int(topology_info.get("bake_face_components", -1))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _report(None, "Step 6/7 · Serializing embedded texture, UVs, and PBR material into GLB")
    studio_source_frame = as_transform_matrix(source_frame_transform)
    if np.allclose(studio_source_frame, np.eye(4), atol=1e-10):
        exported = mesh.export(file_type="glb")
    else:
        scene = trimesh.Scene()
        scene.add_geometry(
            mesh,
            node_name="Polyfitty_Source_Origin",
            geom_name="Polyfitty_Baked_Reconstruction",
            transform=studio_source_frame,
        )
        exported = scene.export(file_type="glb")
    try:
        from .asset_export import patch_glb_specular, patch_glb_diffuse_preview, patch_glb_pbr_maps
    except ImportError:
        from asset_export import patch_glb_specular, patch_glb_diffuse_preview, patch_glb_pbr_maps
    exported = patch_glb_specular(exported if isinstance(exported, bytes) else bytes(exported), 0.25)
    diffuse_preview_atlas = diffuse_atlas_result.get("image")
    if diffuse_preview_atlas is None:
        diffuse_preview_atlas = atlas
    if diffuse_preview_atlas is not None:
        try:
            exported = patch_glb_diffuse_preview(
                exported, Image.fromarray(np.asarray(diffuse_preview_atlas, dtype=np.uint8), mode="RGBA")
            )
            target_info["diffuse_preview_embedded"] = True
        except Exception as exc:
            target_info["diffuse_preview_embedded"] = False
            target_info["diffuse_preview_warning"] = f"Could not embed color-matched Diffuse preview image: {exc}"

    pbr_images = {
        key: Image.fromarray(np.asarray(pbr_atlas_result[key], dtype=np.uint8), mode="RGBA")
        for key in baked_pbr_maps if key in pbr_atlas_result
    }
    if base_color_image is not None:
        pbr_images["base_color"] = base_color_image

    try:
        exported = patch_glb_pbr_maps(
            exported, pbr_images, baked_pbr_maps, baked_maps=baked_pbr_maps
        )
        target_info["pbr_map_metadata_embedded"] = True
    except Exception as exc:
        target_info["pbr_map_metadata_embedded"] = False
        target_info["pbr_map_metadata_warning"] = f"Could not connect selected PBR maps: {exc}"
        raise
    _check_bake_cancelled(cancel_check)
    _report(None, f"Step 7/7 · Writing GLB file ({len(exported):,} bytes)")
    path.write_bytes(exported)
    _check_bake_cancelled(cancel_check)
    _report(1.0, f"GLB export complete · {path.stat().st_size / (1024 * 1024):.2f} MB")
    return {
        "path": str(path),
        **target_info,
        "target_mesh_source": target_info.get("target_mesh_source", "provided_mesh_arrays"),
        "resolution": int(resolution),
        "triangles": int(len(uv_mesh_faces)),
        "mesh_vertices_after_uv_unwrap": int(len(uv_mesh_vertices)),
        "target_face_components_before_uv_unwrap": target_face_components,
        "texture_bytes": int(atlas.nbytes),
        "pbr_maps_requested": list(selected_pbr_maps),
        "pbr_maps_baked": list(baked_pbr_maps),
        "pbr_maps_connected": list(baked_pbr_maps),
        "pbr_material_response_auto_preserved": True,
        "pbr_map_images": {
            key: [int(image.width), int(image.height)]
            for key, image in pbr_images.items()
        },
        "file_bytes": int(path.stat().st_size),
        "chart_padding_pixels": int(chart_padding),
        "source_surface_sampled": bool(sampler is not None),
        "source_has_image_texture": bool(sampler is not None and sampler.has_image_texture),
        "source_sampler_warning": sampler_warning,
        "projection_mode": projection_mode,
        "projection_method": (
            _PROJECTION_INFO[projection_mode]["projection_method"]
            if sampler is not None else "vertex_color_fallback"
        ),
        "source_mesh_path": str(source_mesh_path) if source_mesh_path else "",
        "source_mesh_preserved_for_bake": bool(sampler is not None and source_mesh_path),
        "source_frame_transform": studio_source_frame.tolist(),
        "cage_extrusion_percent": extrusion_percent,
        "cage_extrusion_distance": cage_distance,
        "cage_sample_count": int(sampler.cage_sample_count) if sampler is not None else 0,
        "cage_hit_count": int(sampler.cage_hit_count) if sampler is not None else 0,
        "cage_fallback_count": int(sampler.cage_fallback_count) if sampler is not None else 0,
        **(sampler.compute_manager.info() if sampler is not None else {
            "compute_backend": "CPU", "compute_device": "CPU",
            "compute_precision": "float64", "compute_status": "CPU · vertex-color transfer only",
            "gpu_fallback_reason": "Source-surface projection was not required",
            "gpu_batch_count": 0,
        }),
        "cage_hit_ratio": (
            float(sampler.cage_hit_count / sampler.cage_sample_count)
            if sampler is not None and sampler.cage_sample_count else 0.0
        ),
    }
