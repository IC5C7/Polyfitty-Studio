from __future__ import annotations

"""Topology-preserving Studio post-processing for Wavefront OBJ files.

E1 welds geometric vertices within a world-unit distance while leaving each
face corner's independent UV/normal references intact. Weld clusters are
bounded so transitive chains cannot exceed the requested distance. E2 dissolves only
manifold shared edges between coplanar, same-winding polygons with matching
object/group/material/smoothing state and continuous corner UV/normal data.
The implementation edits polygon loops directly; it never remeshes or
quadric-decimates the reconstructed result.
"""

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Callable
import math

import numpy as np

@dataclass
class _Corner:
    vertex: int
    texcoord: int | None = None
    normal: int | None = None

@dataclass
class _Polygon:
    corners: list[_Corner]
    state: tuple[str, str, str, str]
    order: int
    active: bool = True

def _resolve_obj_index(value: str, count: int, kind: str) -> int:
    try:
        raw = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid OBJ {kind} index: {value!r}") from exc
    if raw == 0:
        raise ValueError(f"OBJ {kind} indices are 1-based; zero is invalid.")
    index = raw - 1 if raw > 0 else count + raw
    if index < 0 or index >= count:
        raise ValueError(f"OBJ {kind} index {raw} is out of range (count={count}).")
    return index

def _parse_obj(path: Path):
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    normals: list[list[float]] = []
    polygons: list[_Polygon] = []
    passthrough: list[str] = []
    state = ["", "", "", ""]

    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            passthrough.append(raw)
            continue
        fields = stripped.split()
        command = fields[0]
        if command == "v":
            if len(fields) < 4:
                raise ValueError(f"OBJ vertex on line {line_number} has fewer than 3 coordinates.")
            values = [float(value) for value in fields[1:]]
            if not np.isfinite(values[:3]).all():
                raise ValueError(f"OBJ vertex on line {line_number} is not finite.")
            vertices.append(values)
        elif command == "vt":
            if len(fields) < 2:
                raise ValueError(f"OBJ texture coordinate on line {line_number} is incomplete.")
            texcoords.append([float(value) for value in fields[1:]])
        elif command == "vn":
            if len(fields) < 4:
                raise ValueError(f"OBJ normal on line {line_number} is incomplete.")
            normals.append([float(value) for value in fields[1:4]])
        elif command in {"o", "g", "usemtl", "s"}:
            index = {"o": 0, "g": 1, "usemtl": 2, "s": 3}[command]
            state[index] = " ".join(fields[1:])
        elif command == "f":
            values = stripped.split("#", 1)[0].split()[1:]
            corners: list[_Corner] = []
            for token in values:
                parts = token.split("/")
                vertex = _resolve_obj_index(parts[0], len(vertices), "vertex")
                texcoord = None
                normal = None
                if len(parts) > 1 and parts[1]:
                    texcoord = _resolve_obj_index(parts[1], len(texcoords), "texture")
                if len(parts) > 2 and parts[2]:
                    normal = _resolve_obj_index(parts[2], len(normals), "normal")
                corners.append(_Corner(vertex, texcoord, normal))
            if len(corners) >= 3:
                polygons.append(_Polygon(corners, tuple(state), len(polygons)))
        else:

            passthrough.append(raw)

    if len(vertices) < 3 or not polygons:
        raise ValueError("The OBJ does not contain enough vertices and polygon faces for Studio post-processing.")
    return vertices, texcoords, normals, polygons, passthrough

def _union_find_vertex_map(vertices: np.ndarray, distance: float) -> tuple[np.ndarray, int, int]:
    count = len(vertices)
    parent = np.arange(count, dtype=np.int64)
    rank = np.zeros(count, dtype=np.uint8)
    cluster_size = np.ones(count, dtype=np.int64)
    cluster_center = vertices[:, :3].copy()
    cluster_radius = np.zeros(count, dtype=np.float64)
    rejected_chain_merges = 0

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = int(parent[root])
        while parent[index] != index:
            nxt = int(parent[index])
            parent[index] = root
            index = nxt
        return root

    if distance > 0.0 and count:
        origin = np.min(vertices[:, :3], axis=0)
        cells = np.floor((vertices[:, :3] - origin) / distance).astype(np.int64)
        buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        distance2 = distance * distance
        offsets = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
        for index, cell in enumerate(cells):
            cx, cy, cz = map(int, cell)
            point = vertices[index, :3]
            for dx, dy, dz in offsets:
                for other in buckets.get((cx + dx, cy + dy, cz + dz), ()):
                    delta = point - vertices[other, :3]
                    if float(np.dot(delta, delta)) <= distance2:
                        a, b = find(index), find(other)
                        if a == b:
                            continue
                        total = int(cluster_size[a] + cluster_size[b])
                        center = (
                            cluster_center[a] * float(cluster_size[a])
                            + cluster_center[b] * float(cluster_size[b])
                        ) / float(total)
                        radius = max(
                            float(cluster_radius[a] + np.linalg.norm(cluster_center[a] - center)),
                            float(cluster_radius[b] + np.linalg.norm(cluster_center[b] - center)),
                        )
                        radius_limit = distance * 0.5 + max(distance * 1e-12, 1e-15)
                        if radius > radius_limit:
                            rejected_chain_merges += 1
                            continue
                        if rank[a] < rank[b]:
                            a, b = b, a
                        parent[b] = a
                        cluster_size[a] = total
                        cluster_center[a] = center
                        cluster_radius[a] = radius
                        if rank[a] == rank[b]:
                            rank[a] += 1
            buckets[(cx, cy, cz)].append(index)

    roots = np.fromiter((find(i) for i in range(count)), dtype=np.int64, count=count)
    _, inverse = np.unique(roots, return_inverse=True)
    unique_count = int(inverse.max()) + 1 if len(inverse) else 0
    merged = np.zeros((unique_count, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=unique_count).astype(np.float64)
    np.add.at(merged, inverse, vertices[:, :3])
    if unique_count:
        merged /= counts[:, None]
    return inverse, count - unique_count, rejected_chain_merges

def _remove_consecutive_duplicate_corners(corners: list[_Corner]) -> list[_Corner]:
    result: list[_Corner] = []
    for corner in corners:
        if not result or result[-1].vertex != corner.vertex:
            result.append(corner)
    if len(result) > 1 and result[0].vertex == result[-1].vertex:
        result.pop()
    return result

def _canonical_polygon_key(
    corners: list[_Corner],
    texcoords: list[list[float]],
    normals: list[list[float]],
) -> tuple:
    """Canonicalize a face including per-corner UV and normal attributes."""
    signature = []
    for corner in corners:
        uv = None if corner.texcoord is None else tuple(
            round(float(value), 12) for value in texcoords[corner.texcoord]
        )
        normal = None if corner.normal is None else tuple(
            round(float(value), 12) for value in normals[corner.normal]
        )
        signature.append((corner.vertex, uv, normal))
    rotations = [tuple(signature[index:] + signature[:index]) for index in range(len(signature))]
    return min(rotations) if rotations else ()

def _newell_normal(vertices: np.ndarray, corners: list[_Corner]) -> tuple[np.ndarray, float]:
    points = vertices[np.asarray([corner.vertex for corner in corners], dtype=np.int64), :3]
    if len(points) < 3:
        return np.zeros(3), 0.0
    nxt = np.roll(points, -1, axis=0)
    normal = np.array([
        np.sum((points[:, 1] - nxt[:, 1]) * (points[:, 2] + nxt[:, 2])),
        np.sum((points[:, 2] - nxt[:, 2]) * (points[:, 0] + nxt[:, 0])),
        np.sum((points[:, 0] - nxt[:, 0]) * (points[:, 1] + nxt[:, 1])),
    ], dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if length <= 1e-15:
        return np.zeros(3), 0.0
    return normal / length, length * 0.5

def _corner_value_match(
    a: _Corner, b: _Corner, attribute: str,
    texcoords: list[list[float]], normals: list[list[float]],
) -> bool:
    ia, ib = getattr(a, attribute), getattr(b, attribute)
    if ia is None or ib is None:
        return ia is None and ib is None
    values = texcoords if attribute == "texcoord" else normals
    va = np.asarray(values[ia], dtype=np.float64)
    vb = np.asarray(values[ib], dtype=np.float64)
    if attribute == "normal":
        la, lb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        if la <= 1e-12 or lb <= 1e-12:
            return np.allclose(va, vb, atol=1e-8, rtol=0.0)
        return float(np.dot(va / la, vb / lb)) >= 0.999999
    width = min(len(va), len(vb), 2)
    return width > 0 and np.allclose(va[:width], vb[:width], atol=1e-8, rtol=0.0)

def _try_merge_polygons(
    vertices: np.ndarray,
    a: _Polygon,
    b: _Polygon,
    edge: tuple[int, int],
    texcoords: list[list[float]],
    normals: list[list[float]],
    planar_tolerance: float,
    normal_cosine: float = 0.999999,
) -> list[_Corner] | None:
    if a.state != b.state:
        return None
    av = [corner.vertex for corner in a.corners]
    bv = [corner.vertex for corner in b.corners]
    u, v = edge
    ai = next((i for i in range(len(av)) if av[i] == u and av[(i + 1) % len(av)] == v), None)
    if ai is None:
        ai = next((i for i in range(len(av)) if av[i] == v and av[(i + 1) % len(av)] == u), None)
    if ai is None:
        return None

    if av[ai] != u:
        u, v = v, u
    bj = next((i for i in range(len(bv)) if bv[i] == v and bv[(i + 1) % len(bv)] == u), None)
    if bj is None:
        return None

    an, area_a = _newell_normal(vertices, a.corners)
    bn, area_b = _newell_normal(vertices, b.corners)
    if area_a <= 1e-14 or area_b <= 1e-14 or float(np.dot(an, bn)) < normal_cosine:
        return None
    a_origin = vertices[av[0], :3]
    b_points = vertices[np.asarray(bv, dtype=np.int64), :3]
    if np.max(np.abs((b_points - a_origin) @ an)) > planar_tolerance:
        return None

    a_corner_by_vertex = {corner.vertex: corner for corner in a.corners}
    b_corner_by_vertex = {corner.vertex: corner for corner in b.corners}
    for shared_vertex in (u, v):
        ca, cb = a_corner_by_vertex[shared_vertex], b_corner_by_vertex[shared_vertex]
        if not _corner_value_match(ca, cb, "texcoord", texcoords, normals):
            return None
        if not _corner_value_match(ca, cb, "normal", texcoords, normals):
            return None

    a_chain = [a.corners[(ai + offset) % len(a.corners)] for offset in range(1, len(a.corners) + 1)]
    b_chain = [b.corners[(bj + offset) % len(b.corners)] for offset in range(1, len(b.corners) + 1)]
    merged = a_chain + b_chain[1:-1]
    merged = _remove_consecutive_duplicate_corners(merged)
    unique_ids = [corner.vertex for corner in merged]
    if len(unique_ids) < 3 or len(set(unique_ids)) != len(unique_ids):
        return None
    merged_normal, merged_area = _newell_normal(vertices, merged)
    if merged_area <= 1e-14 or float(np.dot(merged_normal, an)) < normal_cosine:
        return None
    merged_points = vertices[np.asarray(unique_ids, dtype=np.int64), :3]
    merged_origin = merged_points[0]
    if float(np.max(np.abs((merged_points - merged_origin) @ merged_normal))) > planar_tolerance:
        return None
    return merged

def _greedy_dissolve_component(
    vertices: np.ndarray,
    polygons: list[_Polygon],
    component: list[int],
    texcoords: list[list[float]],
    normals: list[list[float]],
    planar_tolerance: float,
    normal_cosine: float = 0.999999,
) -> tuple[list[_Polygon], int]:
    """Conservatively reduce components that cannot become one simple loop.

    OBJ faces cannot encode inner rings. An annulus must therefore remain
    partitioned around its hole. Safe disjoint face pairs are dissolved in
    batches; the edge map is rebuilt between batches so no stale adjacency is
    used and all per-corner UV/normal and face-state checks remain active.
    """
    active: dict[int, _Polygon] = {index: polygons[index] for index in component}
    dissolved = 0
    while len(active) > 1:
        edge_owners: dict[tuple[int, int], list[int]] = defaultdict(list)
        for face_id, polygon in active.items():
            loop = [corner.vertex for corner in polygon.corners]
            for index, left in enumerate(loop):
                right = loop[(index + 1) % len(loop)]
                if left != right:
                    edge_owners[(min(left, right), max(left, right))].append(face_id)

        reserved_faces: set[int] = set()
        updates: dict[int, _Polygon] = {}
        removed_faces: set[int] = set()
        for edge in sorted(edge_owners):
            owners = edge_owners[edge]
            if len(owners) != 2:
                continue
            left_id, right_id = sorted(owners)
            if left_id in reserved_faces or right_id in reserved_faces:
                continue
            left, right = active[left_id], active[right_id]
            merged_corners = _try_merge_polygons(
                vertices, left, right, edge, texcoords, normals, planar_tolerance, normal_cosine
            )
            if merged_corners is None:
                continue
            updates[left_id] = _Polygon(
                merged_corners, left.state, min(left.order, right.order), True
            )
            removed_faces.add(right_id)
            reserved_faces.update((left_id, right_id))

        if not updates:
            break
        active.update(updates)
        for face_id in removed_faces:
            active.pop(face_id, None)
        dissolved += len(updates)

    result = sorted(active.values(), key=lambda polygon: polygon.order)
    return result, dissolved

def _dissolve_coplanar_faces(
    vertices: np.ndarray,
    polygons: list[_Polygon],
    texcoords: list[list[float]],
    normals: list[list[float]],
    planar_tolerance: float | None = None,
    normal_cosine: float = 0.999999,
) -> tuple[list[_Polygon], int]:
    """Dissolve maximal safe coplanar components into single polygon loops.

    Pairwise loop stitching can leave interior facets behind when a planar grid
    contains many adjacent triangles. Instead, build connectivity only across
    edges that pass the complete material/winding/UV/normal/planarity checks,
    then derive each component's boundary from its directed edge loops. A
    component is merged only when that boundary is exactly one simple cycle;
    components with holes, branching vertices, or ambiguous topology are kept
    unchanged.
    """
    if not polygons:
        return polygons, 0
    span = np.ptp(vertices[:, :3], axis=0)
    scale = max(float(np.linalg.norm(span)), 1.0)
    planar_tolerance = (
        max(scale * 1e-7, 1e-9)
        if planar_tolerance is None
        else max(float(planar_tolerance), scale * 1e-7, 1e-9)
    )

    edge_owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, polygon in enumerate(polygons):
        loop = [corner.vertex for corner in polygon.corners]
        for index, left in enumerate(loop):
            right = loop[(index + 1) % len(loop)]
            if left != right:
                edge_owners[(min(left, right), max(left, right))].append(face_id)

    adjacency: list[list[int]] = [[] for _ in polygons]
    merge_edges: set[tuple[int, int]] = set()
    for edge, owners in edge_owners.items():
        if len(owners) != 2:
            continue
        left_id, right_id = owners
        if _try_merge_polygons(
            vertices, polygons[left_id], polygons[right_id], edge,
            texcoords, normals, planar_tolerance, normal_cosine,
        ) is None:
            continue
        merge_edges.add(edge)
        adjacency[left_id].append(right_id)
        adjacency[right_id].append(left_id)

    visited: set[int] = set()
    output: list[_Polygon] = []
    dissolved = 0
    for seed in range(len(polygons)):
        if seed in visited:
            continue
        stack = [seed]
        component: list[int] = []
        visited.add(seed)
        while stack:
            face_id = stack.pop()
            component.append(face_id)
            for neighbour in adjacency[face_id]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)

        if len(component) < 2:
            output.append(polygons[seed])
            continue

        component_set = set(component)

        def append_safe_fallback() -> None:
            nonlocal dissolved
            fallback_faces, fallback_count = _greedy_dissolve_component(
                vertices, polygons, component, texcoords, normals, planar_tolerance, normal_cosine
            )
            output.extend(fallback_faces)
            dissolved += fallback_count

        outgoing: dict[int, tuple[int, _Corner]] = {}
        incoming_count: dict[int, int] = defaultdict(int)
        boundary_edge_count = 0
        invalid_boundary = False
        for face_id in component:
            polygon = polygons[face_id]
            loop = polygon.corners
            for index, corner in enumerate(loop):
                next_corner = loop[(index + 1) % len(loop)]
                left, right = corner.vertex, next_corner.vertex
                if left == right:
                    invalid_boundary = True
                    break
                edge = (min(left, right), max(left, right))
                owners = edge_owners.get(edge, ())
                if edge in merge_edges and len(owners) == 2 and all(owner in component_set for owner in owners):
                    continue
                if left in outgoing:
                    invalid_boundary = True
                    break
                outgoing[left] = (right, corner)
                incoming_count[right] += 1
                boundary_edge_count += 1
            if invalid_boundary:
                break

        if invalid_boundary or boundary_edge_count < 3 or not outgoing:
            append_safe_fallback()
            continue
        if any(incoming_count.get(vertex, 0) != 1 for vertex in outgoing):
            append_safe_fallback()
            continue
        if any(vertex not in outgoing for vertex in incoming_count):
            append_safe_fallback()
            continue

        start_vertex = min(outgoing, key=lambda vertex: (
            min(polygons[index].order for index in component), vertex
        ))
        boundary_corners: list[_Corner] = []
        current = start_vertex
        seen_vertices: set[int] = set()
        while current not in seen_vertices and current in outgoing:
            seen_vertices.add(current)
            next_vertex, corner = outgoing[current]
            boundary_corners.append(corner)
            current = next_vertex
        if current != start_vertex or len(seen_vertices) != boundary_edge_count:

            append_safe_fallback()
            continue

        unique_ids = [corner.vertex for corner in boundary_corners]
        if len(unique_ids) < 3 or len(set(unique_ids)) != len(unique_ids):
            append_safe_fallback()
            continue
        merged_normal, merged_area = _newell_normal(vertices, boundary_corners)
        reference_normal, _ = _newell_normal(vertices, polygons[component[0]].corners)
        boundary_points = vertices[np.asarray(unique_ids, dtype=np.int64), :3]
        boundary_origin = boundary_points[0]
        boundary_planarity_error = float(
            np.max(np.abs((boundary_points - boundary_origin) @ merged_normal))
        )
        if (
            merged_area <= 1e-14
            or float(np.dot(merged_normal, reference_normal)) < normal_cosine
            or boundary_planarity_error > planar_tolerance
        ):
            append_safe_fallback()
            continue

        state = polygons[component[0]].state
        if any(polygons[index].state != state for index in component):
            append_safe_fallback()
            continue
        output.append(_Polygon(
            boundary_corners, state, min(polygons[index].order for index in component), True
        ))
        dissolved += len(component) - 1

    output.sort(key=lambda polygon: polygon.order)
    return output, dissolved

def _write_obj(
    path: Path,
    vertices: np.ndarray,
    original_vertex_records: list[list[float]],
    texcoord_records: list[list[float]],
    normal_records: list[list[float]],
    vertex_extras: list[list[float]],
    polygons: list[_Polygon],
    passthrough: list[str],
) -> None:
    lines = list(passthrough)
    for index, vertex in enumerate(vertices):
        extras = vertex_extras[index] if index < len(vertex_extras) else []
        values = [*vertex.tolist(), *extras]
        lines.append("v " + " ".join(f"{float(value):.12g}" for value in values))
    for record in texcoord_records:
        lines.append("vt " + " ".join(f"{float(value):.12g}" for value in record))
    for record in normal_records:
        lines.append("vn " + " ".join(f"{float(value):.12g}" for value in record))

    current_state = (None, None, None, None)
    for polygon in polygons:
        if polygon.state != current_state:
            for command, value in zip(("o", "g", "usemtl", "s"), polygon.state):
                if value != current_state[{"o": 0, "g": 1, "usemtl": 2, "s": 3}[command]]:
                    if value:
                        lines.append(f"{command} {value}")
                    elif command in {"o", "g", "usemtl"}:
                        lines.append(command)
                    elif command == "s":
                        lines.append("s off")
            current_state = polygon.state
        tokens = []
        for corner in polygon.corners:
            v = corner.vertex + 1
            vt = "" if corner.texcoord is None else str(corner.texcoord + 1)
            vn = "" if corner.normal is None else str(corner.normal + 1)
            if corner.normal is not None:
                tokens.append(f"{v}/{vt}/{vn}")
            elif corner.texcoord is not None:
                tokens.append(f"{v}/{vt}")
            else:
                tokens.append(str(v))
        lines.append("f " + " ".join(tokens))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def process_obj_e_stage(
    path: str | Path,
    merge_distance: float = 0.001,
    progress_cb: Callable[[int, str], None] | None = None,
    log_cb: Callable[[str], None] | None = None,
    preview_cb: Callable[[np.ndarray, np.ndarray, str], None] | None = None,
    planar_tolerance: float | None = None,
    max_normal_angle_degrees: float = 0.081,
) -> dict:
    """Run Studio's E1 distance weld and E2 coplanar polygon dissolve in-place.

    This cleanup follows PolyFit's official (e) reconstructed-model output; it is
    not itself the upstream PolyFit Figure 1 pipeline stage (e).
    Returns counters that can be added to the reconstruction metrics. OBJ UVs,
    normals, material assignments and polygon face loops remain per-corner.
    """
    obj_path = Path(path)
    original_records, texcoords, normals, polygons, passthrough = _parse_obj(obj_path)
    vertices_raw = np.asarray([record[:3] for record in original_records], dtype=np.float64)
    vertex_map, merged_count, rejected_chain_merges = _union_find_vertex_map(
        vertices_raw, max(0.0, float(merge_distance))
    )
    unique_count = int(vertex_map.max()) + 1 if len(vertex_map) else 0
    vertices = np.zeros((unique_count, 3), dtype=np.float64)
    counts = np.bincount(vertex_map, minlength=unique_count).astype(np.float64)
    np.add.at(vertices, vertex_map, vertices_raw)
    if unique_count:
        vertices /= counts[:, None]

    e1_polygons: list[_Polygon] = []
    removed_degenerate = 0
    seen_faces: set[tuple[tuple[int, ...], tuple[str, str, str, str]]] = set()
    for polygon in polygons:
        remapped = [_Corner(int(vertex_map[corner.vertex]), corner.texcoord, corner.normal)
                    for corner in polygon.corners]
        remapped = _remove_consecutive_duplicate_corners(remapped)
        ids = [corner.vertex for corner in remapped]

        if len(ids) < 3 or len(set(ids)) != len(ids):
            removed_degenerate += 1
            continue
        key = (_canonical_polygon_key(remapped, texcoords, normals), polygon.state)
        if key in seen_faces:
            removed_degenerate += 1
            continue
        seen_faces.add(key)
        e1_polygons.append(_Polygon(remapped, polygon.state, polygon.order))

    used_vertices = sorted({corner.vertex for polygon in e1_polygons for corner in polygon.corners})
    compact = np.full(unique_count, -1, dtype=np.int64)
    compact[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    compact_vertices = vertices[used_vertices]
    representative = np.full(unique_count, -1, dtype=np.int64)
    for old_index, cluster_index in enumerate(vertex_map):
        if representative[int(cluster_index)] < 0:
            representative[int(cluster_index)] = old_index
    compact_extras = [original_records[int(representative[index])][3:] for index in used_vertices]
    for polygon in e1_polygons:
        for corner in polygon.corners:
            corner.vertex = int(compact[corner.vertex])

    def emit_preview(preview_polygons, preview_label):
        if preview_cb is None:
            return
        try:
            triangles = []
            for polygon in preview_polygons:
                loop = [corner.vertex for corner in polygon.corners]
                for index in range(1, len(loop) - 1):
                    triangles.append((loop[0], loop[index], loop[index + 1]))
            if triangles:
                preview_cb(
                    compact_vertices.astype(np.float32),
                    np.asarray(triangles, dtype=np.uint32),
                    preview_label,
                )
        except Exception as exc:
            if log_cb:
                log_cb(f"Studio post-process live preview skipped: {exc}")

    if progress_cb:
        progress_cb(91, f"Studio post E1 · Merge Vertices by Distance · {merge_distance:g}")
    emit_preview(e1_polygons, f"STUDIO POST E1 · MERGE VERTICES BY DISTANCE · {merge_distance:g}")
    if log_cb:
        log_cb(
            f"Studio post E1 · Merge Vertices by Distance: {len(original_records):,} → {len(compact_vertices):,} vertices · "
            f"merged {merged_count:,} · rejected transitive chain welds {rejected_chain_merges:,} · "
            f"removed {removed_degenerate:,} degenerate/duplicate polygons"
        )

    before_dissolve = len(e1_polygons)
    scale = max(float(np.linalg.norm(np.ptp(compact_vertices[:, :3], axis=0))), 1.0)
    effective_planar_tolerance = (
        max(scale * 1e-7, 1e-9)
        if planar_tolerance is None
        else max(float(planar_tolerance), scale * 1e-7, 1e-9)
    )
    normal_cosine = float(np.cos(np.deg2rad(np.clip(max_normal_angle_degrees, 0.01, 10.0))))
    e2_polygons, dissolved_count = _dissolve_coplanar_faces(
        compact_vertices, e1_polygons, texcoords, normals,
        planar_tolerance=effective_planar_tolerance,
        normal_cosine=normal_cosine,
    )
    if progress_cb:
        progress_cb(95, f"Studio post E2 · Face Dissolve · {dissolved_count:,} coplanar joins")
    emit_preview(e2_polygons, f"STUDIO POST E2 · FACE DISSOLVE · {dissolved_count:,} coplanar joins")
    if log_cb:
        log_cb(
            f"Studio post E2 · Face Dissolve: {before_dissolve:,} → {len(e2_polygons):,} polygon faces · "
            f"dissolved {dissolved_count:,} coplanar shared edges · "
            f"tolerance {effective_planar_tolerance:.4g} / angle {max_normal_angle_degrees:.3g}° · "
            "UV/material/sharp/boundary edges preserved"
        )

    if not e2_polygons:
        raise ValueError("Studio post-processing removed every polygon face.")

    temporary_path = obj_path.with_name(obj_path.name + ".e-stage.tmp")
    try:
        _write_obj(
            temporary_path, compact_vertices, original_records, texcoords, normals,
            compact_extras, e2_polygons, passthrough,
        )
        temporary_path.replace(obj_path)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
    emit_preview(e2_polygons, f"STUDIO POSTPROCESS COMPLETE · {len(e2_polygons):,} polygons · cleaned output")
    return {
        "e1_input_vertices": int(len(original_records)),
        "e1_output_vertices": int(len(compact_vertices)),
        "e1_merged_vertices": int(merged_count),
        "e1_removed_polygons": int(removed_degenerate),
        "e1_rejected_chain_merges": int(rejected_chain_merges),
        "e2_input_polygons": int(before_dissolve),
        "e2_output_polygons": int(len(e2_polygons)),
        "e2_dissolved_edges": int(dissolved_count),
        "e2_planar_tolerance": float(effective_planar_tolerance),
        "e2_max_normal_angle_degrees": float(max_normal_angle_degrees),
        "merge_distance": float(merge_distance),
    }
