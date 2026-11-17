from __future__ import annotations

"""Shared source-object coordinate-frame helpers."""

import numpy as np

def as_transform_matrix(value=None) -> np.ndarray:
    """Return a validated affine 4x4 transform, defaulting to identity."""
    if value is None:
        return np.eye(4, dtype=np.float64)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError("Source frame transform must contain 16 values.")
    matrix = matrix.reshape(4, 4).copy()
    if not np.isfinite(matrix).all():
        raise ValueError("Source frame transform contains non-finite values.")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("Source frame transform must be affine.")
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-12:
        raise ValueError("Source frame transform must be invertible.")
    return matrix

def axis_conversion_matrix(source_up_axis: str = "y") -> np.ndarray:
    """Return the exact proper rotation used by pointio.convert_up_axis."""
    axis = str(source_up_axis or "y").lower()
    matrix = np.eye(4, dtype=np.float64)
    if axis == "z":
        return matrix
    if axis == "y":
        matrix[:3, :3] = np.array(
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            dtype=np.float64,
        )
        return matrix
    if axis == "x":
        matrix[:3, :3] = np.array(
            ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        return matrix
    raise ValueError(f"Unsupported source up-axis: {source_up_axis!r}; choose x, y or z.")

def source_frame_to_studio(source_frame=None, source_up_axis: str = "y") -> np.ndarray:
    """Express a source object transform in the Studio Z-up coordinate frame."""
    frame = as_transform_matrix(source_frame)
    conversion = axis_conversion_matrix(source_up_axis)
    return conversion @ frame @ np.linalg.inv(conversion)

def transform_points(points, matrix) -> np.ndarray:
    """Apply a 4x4 affine transform to an Nx3 point array."""
    values = np.asarray(points)
    shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    transform = as_transform_matrix(matrix)
    homogeneous = np.column_stack((flat, np.ones(len(flat), dtype=np.float64)))
    result = (homogeneous @ transform.T)[:, :3]
    return result.reshape(shape).astype(values.dtype if values.dtype.kind == "f" else np.float64, copy=False)

def points_to_source_local(points, source_frame=None) -> np.ndarray:
    """Convert source-world coordinates into the selected source-object frame."""
    frame = as_transform_matrix(source_frame)
    return transform_points(points, np.linalg.inv(frame))

def extract_scene_source_frame(scene_or_mesh) -> np.ndarray:
    """Choose the first valid mesh node's world transform as the shared source origin."""
    graph = getattr(scene_or_mesh, "graph", None)
    geometry = getattr(scene_or_mesh, "geometry", None)
    nodes = getattr(graph, "nodes_geometry", ()) if graph is not None else ()
    if graph is None or geometry is None:
        return np.eye(4, dtype=np.float64)
    for node in nodes:
        try:
            transform, geometry_name = graph[node]
            item = geometry[geometry_name]
            if getattr(item, "vertices", None) is not None and len(item.vertices) >= 3:
                if getattr(item, "faces", None) is None or len(item.faces) == 0:
                    continue
                return as_transform_matrix(transform)
        except Exception:
            continue
    return np.eye(4, dtype=np.float64)

def source_frame_oriented_bounds(points, source_frame=None) -> dict | None:
    """Build an object-aligned bounding box in the shared source frame.

    ``points`` are expressed in Studio world coordinates. The bounds are
    measured after transforming points back into the source object's local
    frame, then the eight corners are transformed through the exact same
    source frame for viewport placement.
    """
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(values) == 0:
        return None
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    if len(values) == 0:
        return None

    frame = as_transform_matrix(source_frame)
    local = points_to_source_local(values, frame)
    lower = np.min(local, axis=0)
    upper = np.max(local, axis=0)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        return None

    local_corners = np.asarray(
        [
            (lower[0], lower[1], lower[2]),
            (upper[0], lower[1], lower[2]),
            (lower[0], upper[1], lower[2]),
            (upper[0], upper[1], lower[2]),
            (lower[0], lower[1], upper[2]),
            (upper[0], lower[1], upper[2]),
            (lower[0], upper[1], upper[2]),
            (upper[0], upper[1], upper[2]),
        ],
        dtype=np.float64,
    )
    world_corners = transform_points(local_corners, frame)
    axis_scales = np.linalg.norm(frame[:3, :3], axis=0)
    dimensions = np.maximum(upper - lower, 0.0) * axis_scales
    return {
        "local_bounds": (lower, upper),
        "local_corners": local_corners,
        "world_corners": world_corners,
        "dimensions": dimensions,
        "source_frame": frame.copy(),
    }

def source_frame_oriented_axis_edges(local_bounds, source_frame) -> np.ndarray:
    """Return fixed origin-frame X/Y bottom edges and the Z top edge.

    The edge selection is deliberately independent of camera position. X and
    Y always use the lower-Z bottom edges that meet at the source-frame lower
    corner. Z uses the vertical edge rising from that same corner; its upper
    endpoint is the fixed top anchor for the Z dimension label. All segments
    are returned in Studio world coordinates through the exact source frame.
    """
    lower = np.asarray(local_bounds[0], dtype=np.float64).reshape(3)
    upper = np.asarray(local_bounds[1], dtype=np.float64).reshape(3)
    frame = as_transform_matrix(source_frame)

    origin_bottom = (lower[0], lower[1], lower[2])
    local_edges = np.asarray(
        (
            (origin_bottom, (upper[0], lower[1], lower[2])),
            (origin_bottom, (lower[0], upper[1], lower[2])),
            (origin_bottom, (lower[0], lower[1], upper[2])),
        ),
        dtype=np.float64,
    )
    return transform_points(local_edges.reshape(-1, 3), frame).reshape(3, 2, 3)

def source_frame_oriented_axis_edge_candidates(local_bounds, source_frame) -> np.ndarray:
    """Return all four object-aligned bounding-box edges for each XYZ axis.

    The result has shape ``(3, 4, 2, 3)``: axis, candidate edge, endpoint,
    coordinate. Each group contains the four edges parallel to that axis, in
    the exact source-object frame and transformed back into Studio world space.
    """
    lower = np.asarray(local_bounds[0], dtype=np.float64).reshape(3)
    upper = np.asarray(local_bounds[1], dtype=np.float64).reshape(3)
    frame = as_transform_matrix(source_frame)
    candidates = []

    for axis in range(3):
        other_axes = [index for index in range(3) if index != axis]
        axis_edges = []
        for first_side in (0, 1):
            for second_side in (0, 1):
                start = lower.copy()
                end = lower.copy()
                start[other_axes[0]] = lower[other_axes[0]] if first_side == 0 else upper[other_axes[0]]
                end[other_axes[0]] = start[other_axes[0]]
                start[other_axes[1]] = lower[other_axes[1]] if second_side == 0 else upper[other_axes[1]]
                end[other_axes[1]] = start[other_axes[1]]
                start[axis] = lower[axis]
                end[axis] = upper[axis]
                axis_edges.append((start, end))
        world_edges = transform_points(np.asarray(axis_edges, dtype=np.float64).reshape(-1, 3), frame)
        candidates.append(world_edges.reshape(4, 2, 3))
    return np.asarray(candidates, dtype=np.float64)

def source_frame_opposite_camera_z_side_axis_edges(
    edge_candidates,
    camera_position,
    source_frame,
    local_bounds,
    axis_index: int,
) -> np.ndarray:
    """Keep X/Y edges on the bbox height opposite the camera's source-local Z side.

    When the camera is above the source object's local origin (local Z >= 0),
    X/Y labels are anchored to the lower-Z/bottom bbox edges. When it is below
    the source origin (local Z < 0), they are anchored to the upper-Z/top edges.
    The camera and edge coordinates are transformed through the exact source
    frame, so rotated and translated assets use their own origin orientation.
    Only X/Y are accepted; Z keeps its independent second-nearest-visible rule.
    """
    edges = np.asarray(edge_candidates, dtype=np.float64)
    if edges.ndim != 3 or edges.shape[1:] != (2, 3):
        return np.empty((0, 2, 3), dtype=np.float64)
    try:
        axis = int(axis_index)
    except (TypeError, ValueError, OverflowError):
        return np.empty((0, 2, 3), dtype=np.float64)
    if axis not in (0, 1):
        return np.empty((0, 2, 3), dtype=np.float64)

    frame = as_transform_matrix(source_frame)
    camera = np.asarray(camera_position, dtype=np.float64).reshape(1, 3)
    if not np.isfinite(camera).all():
        return np.empty((0, 2, 3), dtype=np.float64)
    camera_local_z = float(points_to_source_local(camera, frame)[0, 2])
    lower = np.asarray(local_bounds[0], dtype=np.float64).reshape(3)
    upper = np.asarray(local_bounds[1], dtype=np.float64).reshape(3)
    target_z = float(lower[2] if camera_local_z >= 0.0 else upper[2])

    local_edges = points_to_source_local(edges.reshape(-1, 3), frame).reshape(-1, 2, 3)
    edge_z = np.mean(local_edges[:, :, 2], axis=1)
    tolerance = max(1e-9, abs(float(upper[2] - lower[2])) * 1e-8)
    keep = np.isfinite(edge_z) & np.isclose(edge_z, target_z, rtol=1e-8, atol=tolerance)
    return edges[keep].copy()

def select_camera_visible_axis_edge(
    edge_candidates,
    camera_position,
    project_point,
    depth_at,
    viewport_size,
    *,
    sample_count: int = 13,
    depth_tolerance: float = 0.0015,
    selection_rank: int = 0,
):
    """Select a ranked camera-nearest axis edge with an unobstructed visible segment.

    ``project_point`` maps a Studio-world point to VTK display coordinates
    (x, y, normalized-depth). ``depth_at`` reads the current scene depth at a
    display pixel. A candidate sample is occluded only when the scene's depth
    is meaningfully closer than the candidate edge. If depth reads are not
    available, valid projected samples remain eligible as a safe fallback.

    The returned mapping contains the selected edge index, visible-segment
    anchor fraction, world anchor, projected anchor, and visible sample count.
    Candidates are ranked by the camera-to-visible-edge distance, then by the
    length of the unobstructed run for deterministic tie-breaking. ``selection_rank``
    is zero-based: 0 selects the nearest visible edge, 1 selects the second-nearest.
    If fewer candidates are visible than requested, the farthest available rank
    is used so a valid label remains visible whenever at least one edge is visible.
    """
    edges = np.asarray(edge_candidates, dtype=np.float64)
    if edges.ndim != 3 or edges.shape[1:] != (2, 3) or len(edges) == 0:
        return None
    camera = np.asarray(camera_position, dtype=np.float64).reshape(3)
    if not np.isfinite(camera).all():
        return None
    width, height = int(viewport_size[0]), int(viewport_size[1])
    if width <= 0 or height <= 0:
        return None

    count = max(3, int(sample_count))
    fractions = np.linspace(0.08, 0.92, count, dtype=np.float64)
    ranked = []
    for edge_index, edge in enumerate(edges):
        if not np.isfinite(edge).all():
            continue
        projected_samples = []
        visible = []
        for fraction in fractions:
            world = edge[0] + (edge[1] - edge[0]) * float(fraction)
            try:
                display = np.asarray(project_point(world), dtype=np.float64).reshape(-1)
            except Exception:
                display = np.empty(0, dtype=np.float64)
            valid = (
                display.size >= 3 and np.isfinite(display[:3]).all()
                and 0.0 <= display[2] <= 1.0
                and 0.0 <= display[0] < width
                and 0.0 <= display[1] < height
            )
            if not valid:
                projected_samples.append(None)
                visible.append(False)
                continue

            is_visible = True
            if depth_at is not None:
                try:
                    scene_depth = float(depth_at(int(round(display[0])), int(round(display[1]))))
                    if np.isfinite(scene_depth) and 0.0 <= scene_depth <= 1.0:
                        is_visible = scene_depth >= float(display[2]) - float(depth_tolerance)
                except Exception:

                    is_visible = True
            projected_samples.append(display[:3].copy())
            visible.append(bool(is_visible))

        runs = []
        run_start = None
        for sample_index, is_visible in enumerate(visible + [False]):
            if is_visible and run_start is None:
                run_start = sample_index
            elif not is_visible and run_start is not None:
                runs.append((run_start, sample_index - 1))
                run_start = None
        if not runs:
            continue
        run_start, run_end = max(
            runs,
            key=lambda bounds: (
                bounds[1] - bounds[0] + 1,
                -abs((bounds[0] + bounds[1]) * 0.5 - (count - 1) * 0.5),
            ),
        )
        anchor_index = (run_start + run_end) // 2
        anchor_fraction = float(fractions[anchor_index])
        anchor_world = edge[0] + (edge[1] - edge[0]) * anchor_fraction
        anchor_display = projected_samples[anchor_index]
        if anchor_display is None:
            continue
        distance = float(np.linalg.norm(anchor_world - camera))
        if not np.isfinite(distance):
            continue
        visible_count = run_end - run_start + 1
        ranked.append((distance, -visible_count, edge_index, {
            "edge_index": int(edge_index),
            "fraction": anchor_fraction,
            "world_point": anchor_world,
            "display_point": anchor_display,
            "visible_samples": int(visible_count),
        }))

    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    try:
        requested_rank = max(0, int(selection_rank))
    except (TypeError, ValueError, OverflowError):
        requested_rank = 0
    selected_rank = min(requested_rank, len(ranked) - 1)
    selected = dict(ranked[selected_rank][3])
    selected["selection_rank"] = int(selected_rank)
    selected["candidate_count"] = int(len(ranked))
    return selected

def source_frame_camera_pose(
    local_bounds,
    source_frame=None,
    viewport_size=(1280, 720),
    horizontal_fov_degrees=67.0,
    *,
    margin=1.28,
    azimuth_degrees=35.0,
    elevation_degrees=22.0,
) -> dict:
    """Compute an origin-relative camera pose that fits an oriented source box.

    The camera's default view direction and up vector are defined in source
    object space, then rotated into Studio world space by the source frame.
    The focal point is the center of the complete local bounds transformed
    from that same origin, so translated/rotated assets receive a stable view.
    """
    lower = np.asarray(local_bounds[0], dtype=np.float64).reshape(3)
    upper = np.asarray(local_bounds[1], dtype=np.float64).reshape(3)
    if (
        not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(upper < lower)
    ):
        raise ValueError("Camera bounds must be finite ordered XYZ limits.")

    frame = as_transform_matrix(source_frame)
    linear = frame[:3, :3]
    left, _singular, right_svd = np.linalg.svd(linear)
    rotation = left @ right_svd
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_svd

    azimuth = np.radians(float(azimuth_degrees))
    elevation = np.radians(float(elevation_degrees))
    local_direction = np.asarray((
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ), dtype=np.float64)
    direction = rotation @ local_direction
    direction /= max(float(np.linalg.norm(direction)), 1e-12)

    view_up = rotation @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    view_up /= max(float(np.linalg.norm(view_up)), 1e-12)
    forward = -direction
    right = np.cross(forward, view_up)
    if float(np.linalg.norm(right)) <= 1e-9:
        view_up = rotation @ np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        right = np.cross(forward, view_up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    camera_up = np.cross(right, forward)
    camera_up /= max(float(np.linalg.norm(camera_up)), 1e-12)

    local_corners = np.asarray(
        [
            (x, y, z)
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )
    world_corners = transform_points(local_corners, frame)
    local_center = (lower + upper) * 0.5
    focal = transform_points(local_center.reshape(1, 3), frame)[0]
    relative = world_corners - focal

    width, height = (float(viewport_size[0]), float(viewport_size[1]))
    aspect = max(width / max(height, 1.0), 1e-3)
    horizontal_fov = np.radians(float(np.clip(horizontal_fov_degrees, 1.0, 170.0)))
    tan_horizontal = max(float(np.tan(horizontal_fov * 0.5)), 1e-6)
    tan_vertical = max(float(tan_horizontal / aspect), 1e-6)

    depth_offset = relative @ direction
    horizontal_extent = np.abs(relative @ right)
    vertical_extent = np.abs(relative @ camera_up)
    required_distance = max(
        float(np.max(depth_offset + horizontal_extent / tan_horizontal)),
        float(np.max(depth_offset + vertical_extent / tan_vertical)),
        0.05,
    )
    distance = max(required_distance * max(float(margin), 1.0), 0.1)
    position = focal + direction * distance

    return {
        "position": position,
        "focal": focal,
        "view_up": camera_up,
        "distance": distance,
        "direction": direction,
        "right": right,
        "camera_up": camera_up,
        "world_corners": world_corners,
    }
