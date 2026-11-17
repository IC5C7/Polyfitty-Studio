from __future__ import annotations

"""OBJ mesh readers that preserve concave polygon geometry.

Trimesh's generic OBJ loader triangulates n-gons with a fan in some versions.
That is only valid for convex polygons. Studio's E2 dissolve may produce
concave n-gons, so fan triangulation creates triangles outside the polygon,
which show up as folded / intersecting fragments in previews and baked GLBs.
This module performs deterministic ear clipping in a stable 2-D projection.
"""

from pathlib import Path
from typing import Iterable

import numpy as np

def _cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])

def _segments_intersect_2d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
    area_epsilon: float, linear_epsilon: float,
) -> bool:
    """Return True for proper crossings, non-adjacent touches, and overlaps."""
    o1 = _cross2(a, b, c)
    o2 = _cross2(a, b, d)
    o3 = _cross2(c, d, a)
    o4 = _cross2(c, d, b)

    proper_ab = (o1 > area_epsilon and o2 < -area_epsilon) or (o1 < -area_epsilon and o2 > area_epsilon)
    proper_cd = (o3 > area_epsilon and o4 < -area_epsilon) or (o3 < -area_epsilon and o4 > area_epsilon)
    if proper_ab and proper_cd:
        return True

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray, orientation: float) -> bool:
        lower = np.minimum(p, r) - linear_epsilon
        upper = np.maximum(p, r) + linear_epsilon
        return bool(
            abs(orientation) <= area_epsilon
            and np.all(q >= lower)
            and np.all(q <= upper)
        )

    return (
        on_segment(a, c, b, o1)
        or on_segment(a, d, b, o2)
        or on_segment(c, a, d, o3)
        or on_segment(c, b, d, o4)
    )

def _newell_normal(points: np.ndarray) -> np.ndarray:
    q = np.roll(points, -1, axis=0)
    normal = np.array((
        np.sum((points[:, 1] - q[:, 1]) * (points[:, 2] + q[:, 2])),
        np.sum((points[:, 2] - q[:, 2]) * (points[:, 0] + q[:, 0])),
        np.sum((points[:, 0] - q[:, 0]) * (points[:, 1] + q[:, 1])),
    ), dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("OBJ polygon has a degenerate Newell normal.")
    return normal / length

def triangulate_obj_polygon(indices: Iterable[int], vertices: np.ndarray) -> list[list[int]]:
    """Triangulate a simple (possibly concave) OBJ polygon using ear clipping.

    Args:
        indices: Zero-based vertex indices in polygon winding order.
        vertices: Nx3 positions.

    Returns:
        N-2 triangle index triplets. Raises ValueError instead of silently
        producing invalid fan triangles if the polygon is self-intersecting
        or otherwise cannot be triangulated reliably.
    """
    ids = [int(index) for index in indices]
    if len(ids) < 3:
        raise ValueError("OBJ face contains fewer than three vertices.")
    if min(ids) < 0 or max(ids) >= len(vertices):
        raise ValueError("OBJ face references a vertex outside the vertex array.")

    if len(ids) > 3 and ids[0] == ids[-1]:
        ids.pop()
    cleaned: list[int] = []
    for index in ids:
        if not cleaned or cleaned[-1] != index:
            cleaned.append(index)
    ids = cleaned
    if len(ids) > 3 and ids[0] == ids[-1]:
        ids.pop()
    if len(ids) < 3:
        raise ValueError("OBJ face collapses after removing repeated vertices.")
    if len(set(ids)) != len(ids):

        raise ValueError("OBJ face contains a repeated non-consecutive vertex index.")
    if len(ids) == 3:
        return [ids]

    points = np.asarray(vertices[np.asarray(ids, dtype=np.int64)], dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("OBJ polygon contains non-finite vertex coordinates.")

    normal = _newell_normal(points)
    reference = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    axis_u = np.cross(normal, reference)
    axis_length = float(np.linalg.norm(axis_u))
    if axis_length <= 1e-12:
        raise ValueError("OBJ polygon has no stable projection axis.")
    axis_u /= axis_length
    axis_v = np.cross(normal, axis_u)
    projected = np.column_stack((points @ axis_u, points @ axis_v))

    span = max(float(np.ptp(projected, axis=0).max()), 1e-12)
    epsilon = max(span * span * 1e-12, 1e-14)
    linear_epsilon = max(span * 1e-10, 1e-12)
    next_points = np.roll(projected, -1, axis=0)

    edge_lengths = np.linalg.norm(next_points - projected, axis=1)
    if np.any(edge_lengths <= linear_epsilon):
        raise ValueError("OBJ polygon contains coincident consecutive corners.")
    count = len(ids)
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if first == second or first_next == second or second_next == first:
                continue
            if _segments_intersect_2d(
                projected[first], projected[first_next],
                projected[second], projected[second_next],
                epsilon, linear_epsilon,
            ):
                raise ValueError("OBJ polygon boundary self-intersects or touches itself.")
    signed_area = 0.5 * float(np.sum(
        projected[:, 0] * next_points[:, 1]
        - next_points[:, 0] * projected[:, 1]
    ))
    if abs(signed_area) <= epsilon:
        raise ValueError("OBJ polygon has near-zero projected area.")
    orientation = 1.0 if signed_area > 0.0 else -1.0

    remaining = list(range(len(ids)))
    triangles: list[list[int]] = []
    guard = 0
    max_guard = len(ids) * 4
    while len(remaining) > 3 and guard < max_guard:
        guard += 1
        ear_found = False
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = projected[previous], projected[current], projected[following]
            if orientation * _cross2(a, b, c) <= epsilon:
                continue

            contains_vertex = False
            for candidate in remaining:
                if candidate in (previous, current, following):
                    continue
                point = projected[candidate]
                edge_1 = orientation * _cross2(a, b, point)
                edge_2 = orientation * _cross2(b, c, point)
                edge_3 = orientation * _cross2(c, a, point)
                if edge_1 >= -epsilon and edge_2 >= -epsilon and edge_3 >= -epsilon:
                    contains_vertex = True
                    break
            if contains_vertex:
                continue

            triangles.append([ids[previous], ids[current], ids[following]])
            remaining.pop(position)
            ear_found = True
            break

        if ear_found:
            continue

        removed_collinear = False
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            if abs(_cross2(projected[previous], projected[current], projected[following])) > epsilon:
                continue
            segment = projected[following] - projected[previous]
            offset = projected[current] - projected[previous]
            if (
                float(np.dot(offset, segment)) >= -epsilon
                and float(np.dot(offset, offset)) <= float(np.dot(segment, segment)) + epsilon
            ):
                remaining.pop(position)
                removed_collinear = True
                break
        if not removed_collinear:
            raise ValueError(
                "OBJ polygon could not be ear-clipped; it may be self-intersecting or non-simple."
            )

    if len(remaining) != 3:
        raise ValueError("OBJ polygon triangulation did not converge.")
    final_cross = abs(_cross2(
        projected[remaining[0]], projected[remaining[1]], projected[remaining[2]]
    ))
    if final_cross <= epsilon:
        raise ValueError("OBJ polygon ended with a degenerate triangle.")
    triangles.append([ids[remaining[0]], ids[remaining[1]], ids[remaining[2]]])
    return triangles

def read_obj_triangles(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read OBJ positions and triangulate all faces without concave fan errors."""
    obj_path = Path(path)
    vertices: list[tuple[float, float, float]] = []
    polygons: list[tuple[int, list[int]]] = []
    max_polygon_vertices = 0

    with obj_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("v ") or stripped.startswith("v\t"):
                fields = stripped.split()
                if len(fields) < 4:
                    raise ValueError(f"Invalid OBJ vertex at line {line_number}.")
                try:
                    xyz = tuple(float(value) for value in fields[1:4])
                except ValueError as exc:
                    raise ValueError(f"Invalid OBJ vertex at line {line_number}.") from exc
                vertices.append(xyz)
            elif stripped.startswith("f ") or stripped.startswith("f\t"):
                fields = stripped.split()[1:]
                if len(fields) < 3:
                    continue
                indices: list[int] = []
                for token in fields:
                    vertex_token = token.split("/", 1)[0]
                    try:
                        raw_index = int(vertex_token)
                    except ValueError as exc:
                        raise ValueError(f"Invalid OBJ face index at line {line_number}.") from exc
                    if raw_index == 0:
                        raise ValueError(f"OBJ indices are 1-based; found zero at line {line_number}.")
                    resolved = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                    if resolved < 0 or resolved >= len(vertices):
                        raise ValueError(f"OBJ face index is out of range at line {line_number}.")
                    indices.append(resolved)
                max_polygon_vertices = max(max_polygon_vertices, len(indices))
                polygons.append((line_number, indices))

    vertex_array = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(vertex_array) < 3 or not polygons:
        raise ValueError(f"OBJ contains no usable triangle geometry: {obj_path}")
    if not np.isfinite(vertex_array).all():
        raise ValueError(f"OBJ contains non-finite vertices: {obj_path}")

    triangles: list[list[int]] = []
    for line_number, indices in polygons:
        try:
            triangles.extend(triangulate_obj_polygon(indices, vertex_array))
        except ValueError as exc:
            raise ValueError(f"Cannot safely triangulate OBJ face at line {line_number}: {exc}") from exc

    face_array = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    stats = {
        "obj_polygon_faces": int(len(polygons)),
        "triangulated_faces": int(len(face_array)),
        "max_polygon_vertices": int(max_polygon_vertices),
        "triangulation_method": "ear_clipping_concave_safe",
    }
    return vertex_array, face_array, stats

def load_obj_trimesh(path: str | Path):
    """Load a generated OBJ as a trimesh using concave-safe triangulation."""
    import trimesh

    vertices, faces, _ = read_obj_triangles(path)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
