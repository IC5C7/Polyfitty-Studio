from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional
import csv
import re
import json
import struct

import numpy as np

_PREVIEW_NORMALS_METADATA_KEY = "_polyfitty_preview_vertex_normals"

def _unit_vertex_normals(values, count: int) -> np.ndarray | None:
    """Validate and normalize an (N, 3) array of source vertex normals."""
    try:
        normals = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if normals.shape != (int(count), 3) or not np.isfinite(normals).all():
        return None
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12):
        return None
    return np.ascontiguousarray(normals / lengths[:, None], dtype=np.float32)

def _transform_vertex_normals(normals, matrix) -> np.ndarray | None:
    """Transform normals by the inverse-transpose linear transform."""
    values = np.asarray(normals, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        return None
    try:
        normal_matrix = np.linalg.inv(transform[:3, :3]).T
    except np.linalg.LinAlgError:
        return None
    transformed = values @ normal_matrix.T
    return _unit_vertex_normals(transformed, len(values))

def _mesh_preview_vertex_normals(mesh) -> np.ndarray | None:
    """Return preserved source normals, falling back to Trimesh point normals."""
    count = len(getattr(mesh, "vertices", ()))
    metadata = getattr(mesh, "metadata", {})
    preserved = metadata.get(_PREVIEW_NORMALS_METADATA_KEY) if isinstance(metadata, dict) else None
    normals = _unit_vertex_normals(preserved, count)
    if normals is not None:
        return normals
    try:
        return _unit_vertex_normals(mesh.vertex_normals, count)
    except Exception:
        return None

def _material_image_attribute(material, *names):
    """Return a material image attribute as a PIL RGBA image when available."""
    if material is None:
        return None
    from PIL import Image
    for name in names:
        image = getattr(material, name, None)
        if image is None:
            continue
        try:
            if not hasattr(image, "convert"):
                image = Image.fromarray(np.asarray(image))
            return image.convert("RGBA")
        except Exception:
            continue
    return None

def _material_orm_preview_image(material):
    """Pack source AO/Roughness/Metallic images into one linear-data ORM image.

    VTK's PBR shader accepts a single glTF-style ORM texture. Source materials
    may store AO separately from the metallic-roughness image, so merge their
    channels on import while leaving scalar material factors on the actor.
    """
    from PIL import Image

    mr_image = _material_image_attribute(material, "metallicRoughnessTexture")
    roughness_image = _material_image_attribute(material, "roughnessTexture")
    metallic_image = _material_image_attribute(material, "metallicTexture")
    ao_image = _material_image_attribute(material, "occlusionTexture", "aoTexture")
    images = [image for image in (mr_image, roughness_image, metallic_image, ao_image) if image is not None]
    if not images:
        return None

    base_size = mr_image.size if mr_image is not None else max(
        (image.size for image in images), key=lambda size: int(size[0]) * int(size[1])
    )
    width, height = (max(1, int(base_size[0])), max(1, int(base_size[1])))
    scale = min(1.0, 4096.0 / max(width, height))
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resampling = getattr(Image, "Resampling", Image)

    def _pixels(image):
        if image is None:
            return None
        converted = image.convert("RGBA")
        if converted.size != size:
            converted = converted.resize(size, resampling.BILINEAR)
        return np.asarray(converted, dtype=np.uint8)

    packed = np.empty((size[1], size[0], 4), dtype=np.uint8)
    packed[..., 0] = 255
    packed[..., 1] = 255
    packed[..., 2] = 0
    packed[..., 3] = 255

    mr = _pixels(mr_image)
    if mr is not None:

        packed[..., 1] = mr[..., 1]
        packed[..., 2] = mr[..., 2]
    rough = _pixels(roughness_image)
    if rough is not None:
        packed[..., 1] = rough[..., 0]
    metal = _pixels(metallic_image)
    if metal is not None:
        packed[..., 2] = metal[..., 0]
    ao = _pixels(ao_image)
    if ao is not None:
        packed[..., 0] = ao[..., 0]
    return Image.fromarray(packed, mode="RGBA")

def _material_occlusion_strength(material) -> float:
    """Return the glTF occlusion strength, defaulting to the standard value."""
    try:
        value = float(getattr(material, "occlusionStrength", 1.0))
    except (TypeError, ValueError):
        value = 1.0
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 1.0

def _material_pbr_factors(material) -> dict:
    """Read imported metallic/roughness factors without replacing source values."""
    is_gltf_pbr = material is not None and (
        type(material).__name__ == "PBRMaterial"
        or hasattr(material, "metallicFactor")
        or hasattr(material, "roughnessFactor")
    )
    default_roughness, default_metallic = (1.0, 1.0) if is_gltf_pbr else (0.72, 0.0)
    try:
        roughness = float(getattr(material, "roughnessFactor", default_roughness))
    except (TypeError, ValueError):
        roughness = default_roughness
    try:
        metallic = float(getattr(material, "metallicFactor", default_metallic))
    except (TypeError, ValueError):
        metallic = default_metallic
    if not np.isfinite(roughness):
        roughness = default_roughness
    if not np.isfinite(metallic):
        metallic = default_metallic
    return {
        "roughness": float(np.clip(roughness, 0.0, 1.0)),
        "metallic": float(np.clip(metallic, 0.0, 1.0)),
    }

def merge_mesh_vertices_by_distance(
    vertices: np.ndarray, faces: np.ndarray, distance: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Merge mesh vertices whose Euclidean separation is within ``distance``.

    Nearby vertices are clustered with a strict cluster-diameter guard: a
    chain of pairwise-close vertices cannot bridge endpoints farther apart
    than the requested distance. Clusters are collapsed to their centroid,
    and faces that become degenerate or duplicated are removed.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    threshold = float(distance)
    if verts.ndim != 2 or verts.shape[1] < 3:
        raise ValueError("Mesh vertices must have shape (N, 3).")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("Mesh faces must have shape (M, 3).")
    if not np.isfinite(verts[:, :3]).all():
        raise ValueError("Mesh contains non-finite vertex coordinates.")
    if threshold <= 0.0:
        return verts[:, :3].copy(), tris.copy(), {
            "input_vertices": int(len(verts)), "output_vertices": int(len(verts)),
            "merged_vertices": 0, "removed_faces": 0, "rejected_chain_merges": 0,
        }
    if len(verts) == 0 or len(tris) == 0:
        return verts[:, :3].copy(), tris.copy(), {
            "input_vertices": int(len(verts)), "output_vertices": int(len(verts)),
            "merged_vertices": 0, "removed_faces": 0, "rejected_chain_merges": 0,
        }
    if int(tris.min()) < 0 or int(tris.max()) >= len(verts):
        raise ValueError("Mesh face indices are outside the vertex array.")

    try:
        from scipy.spatial import cKDTree
        pairs = cKDTree(verts[:, :3]).query_pairs(threshold, output_type="ndarray")
    except ImportError:
        from itertools import product
        origin = verts[:, :3].min(axis=0)
        cells = np.floor((verts[:, :3] - origin) / threshold).astype(np.int64)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        pair_list: list[tuple[int, int]] = []
        offsets = tuple(product((-1, 0, 1), repeat=3))
        limit2 = threshold * threshold
        for index, cell in enumerate(cells):
            key = (int(cell[0]), int(cell[1]), int(cell[2]))
            point = verts[index, :3]
            for dx, dy, dz in offsets:
                neighbor = (key[0] + dx, key[1] + dy, key[2] + dz)
                for other in buckets.get(neighbor, ()):
                    delta = point - verts[other, :3]
                    if float(np.dot(delta, delta)) <= limit2:
                        pair_list.append((other, index))
            buckets.setdefault(key, []).append(index)
        pairs = np.asarray(pair_list, dtype=np.int64).reshape(-1, 2)

    if len(pairs) == 0:
        return verts[:, :3].copy(), tris.copy(), {
            "input_vertices": int(len(verts)), "output_vertices": int(len(verts)),
            "merged_vertices": 0, "removed_faces": 0, "rejected_chain_merges": 0,
        }

    parent = np.arange(len(verts), dtype=np.int64)
    rank = np.zeros(len(verts), dtype=np.uint8)
    cluster_size = np.ones(len(verts), dtype=np.int64)
    cluster_center = verts[:, :3].copy()
    cluster_radius = np.zeros(len(verts), dtype=np.float64)
    rejected_chain_merges = 0

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            nxt = int(parent[item])
            parent[item] = root
            item = nxt
        return root

    radius_limit = threshold * 0.5 + max(threshold * 1e-12, 1e-15)
    for left, right in pairs:
        a, b = find(int(left)), find(int(right))
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

    roots = np.fromiter((find(i) for i in range(len(verts))), dtype=np.int64, count=len(verts))
    _, inverse = np.unique(roots, return_inverse=True)
    cluster_count = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=cluster_count).astype(np.float64)
    merged_vertices = np.zeros((cluster_count, 3), dtype=np.float64)
    np.add.at(merged_vertices, inverse, verts[:, :3])
    merged_vertices /= counts[:, None]

    mapped_faces = inverse[tris]
    keep = (mapped_faces[:, 0] != mapped_faces[:, 1]) & (
        mapped_faces[:, 1] != mapped_faces[:, 2]
    ) & (mapped_faces[:, 2] != mapped_faces[:, 0])
    removed_degenerate = int((~keep).sum())
    mapped_faces = mapped_faces[keep]
    if len(mapped_faces):

        _, first = np.unique(np.sort(mapped_faces, axis=1), axis=0, return_index=True)
        first.sort()
        mapped_faces = mapped_faces[first]
    removed_faces = int(len(tris) - len(mapped_faces))

    used = np.unique(mapped_faces) if len(mapped_faces) else np.empty(0, dtype=np.int64)
    compact = np.full(cluster_count, -1, dtype=np.int64)
    compact[used] = np.arange(len(used), dtype=np.int64)
    final_vertices = merged_vertices[used] if len(used) else merged_vertices[:0]
    final_faces = compact[mapped_faces] if len(mapped_faces) else np.empty((0, 3), dtype=np.int64)
    return final_vertices, final_faces, {
        "input_vertices": int(len(verts)),
        "output_vertices": int(len(final_vertices)),
        "merged_vertices": int(len(verts) - cluster_count),
        "removed_faces": removed_faces,
        "removed_degenerate_faces": removed_degenerate,
        "rejected_chain_merges": int(rejected_chain_merges),
    }

def weld_coincident_mesh_vertices(mesh):
    """Restore shared vertex indices for coincident triangle-soup vertices.

    GLB/GLTF exporters sometimes duplicate each triangle corner in the vertex
    buffer (often to carry independent normals/UVs). That makes a visually
    continuous surface look like thousands of disconnected components. This
    operation welds positions rounded to 12 decimal places only; it does not
    apply a user-scale distance tolerance or move the geometry intentionally.
    """
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return mesh, {"welded_vertices": 0, "removed_faces": 0}

    result = mesh.copy()
    before_vertices = int(len(result.vertices))
    before_faces = int(len(result.faces))

    result.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=12)
    welded = max(0, before_vertices - int(len(result.vertices)))
    removed_faces = 0
    if welded:
        result.update_faces(result.nondegenerate_faces())
        result.update_faces(result.unique_faces())
        result.remove_unreferenced_vertices()
        removed_faces = max(0, before_faces - int(len(result.faces)))
    return result, {"welded_vertices": welded, "removed_faces": removed_faces}

def _nearest_vertex_colors(
    target_vertices: np.ndarray, source_vertices: np.ndarray, source_colors: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Transfer sampled source appearance when a geometric weld rebuilds the mesh."""
    if source_colors is None:
        return None
    target = np.asarray(target_vertices, dtype=np.float64).reshape(-1, 3)
    source = np.asarray(source_vertices, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(source_colors, dtype=np.float32)
    if len(target) == 0 or len(source) == 0 or len(colors) != len(source):
        return None
    try:
        from scipy.spatial import cKDTree
        _distance, indices = cKDTree(source).query(target, k=1, workers=-1)
    except Exception:
        indices = np.empty(len(target), dtype=np.int64)
        chunk = max(1, min(1024, 1_000_000 // max(1, len(source))))
        for start in range(0, len(target), chunk):
            block = target[start:start + chunk]
            d2 = np.sum((block[:, None, :] - source[None, :, :]) ** 2, axis=2)
            indices[start:start + len(block)] = np.argmin(d2, axis=1)
    return np.asarray(colors[indices], dtype=np.float32)

def _normalise_colors(colors: Optional[np.ndarray], count: int) -> np.ndarray:
    if colors is None or len(colors) != count:
        return np.full((count, 3), 0.66, dtype=np.float32)
    arr = np.asarray(colors)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.full((count, 3), 0.66, dtype=np.float32)
    arr = arr[:, :3].astype(np.float32, copy=False)
    if arr.size and float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    return np.clip(np.nan_to_num(arr, nan=0.66, posinf=1.0, neginf=0.0), 0, 1)

def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode display-encoded sRGB channel values into linear-light RGB."""
    values = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32, copy=False)

def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Encode linear-light RGB for a normal sRGB display / image texture."""
    values = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    ).astype(np.float32, copy=False)

def _apply_linear_base_color_factor(srgb_rgba: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Multiply glTF/linear material factors without darkening sRGB texels.

    Base-color images are display-encoded sRGB, while glTF baseColorFactor is
    a linear-light multiplier. Multiplying their byte values directly applies
    the factor in the wrong space and makes both texture previews too dark.
    Return display-encoded sRGB so Diffuse can show the exact resulting color;
    the PBR renderer decodes it back to linear before lighting.
    """
    rgba = np.asarray(srgb_rgba, dtype=np.float32).copy()
    material_factor = np.asarray(factor, dtype=np.float32).reshape(-1)
    if rgba.shape[-1] < 3 or material_factor.size < 3:
        return rgba
    rgb_linear = _srgb_to_linear(rgba[..., :3])
    rgba[..., :3] = _linear_to_srgb(
        rgb_linear * np.clip(material_factor[:3], 0.0, 1.0)
    )
    if rgba.shape[-1] >= 4 and material_factor.size >= 4:
        rgba[..., 3] *= np.clip(material_factor[3], 0.0, 1.0)
    return np.clip(rgba, 0.0, 1.0).astype(np.float32, copy=False)

def _sample_texture_rgba(image, uv: np.ndarray) -> np.ndarray | None:
    """Sample a trimesh material image at normalized UV coordinates."""
    if image is None or uv is None:
        return None
    try:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        coords = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if rgba.ndim != 3 or len(coords) == 0:
            return None
        height, width = rgba.shape[:2]

        u = np.mod(coords[:, 0], 1.0) * max(0, width - 1)
        v = (1.0 - np.mod(coords[:, 1], 1.0)) * max(0, height - 1)
        x0, y0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        fx, fy = (u - x0)[:, None], (v - y0)[:, None]
        c00 = rgba[y0, x0].astype(np.float32)
        c10 = rgba[y0, x1].astype(np.float32)
        c01 = rgba[y1, x0].astype(np.float32)
        c11 = rgba[y1, x1].astype(np.float32)
        top = c00 * (1.0 - fx) + c10 * fx
        bottom = c01 * (1.0 - fx) + c11 * fx
        return np.clip((top * (1.0 - fy) + bottom * fy) / 255.0, 0.0, 1.0).astype(np.float32)
    except Exception:
        return None

def _material_base_color(material) -> np.ndarray:
    """Return a normalized glTF/MTL base-color factor, defaulting to white."""
    if material is None:
        return np.ones(4, dtype=np.float32)
    for attribute in ("baseColorFactor", "diffuse", "main_color"):
        value = getattr(material, attribute, None)
        if value is None:
            continue
        try:
            rgba = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if rgba.size < 3 or not np.isfinite(rgba[:3]).all():
            continue
        if rgba.size == 3:
            rgba = np.append(rgba, 255.0 if float(np.nanmax(rgba)) > 1.5 else 1.0)
        rgba = rgba[:4].copy()
        if float(np.nanmax(rgba)) > 1.5:
            rgba /= 255.0
        return np.clip(np.nan_to_num(rgba, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return np.ones(4, dtype=np.float32)

def _material_render_settings(material, source_modes=None):
    """Resolve source glTF render modes by material name, then standard fields."""
    name = str(getattr(material, "name", "") or "")
    for entry in source_modes or ():
        if name and str(entry.get("name", "")) == name:
            return {
                "alpha_mode": str(entry.get("alpha_mode", "OPAQUE")).upper(),
                "alpha_cutoff": float(entry.get("alpha_cutoff", 0.5)),
                "shadow_mode": str(entry.get("shadow_mode", "OPAQUE")).upper(),
            }
    if source_modes:
        mode_keys = {
            (
                str(entry.get("alpha_mode", "OPAQUE")).upper(),
                round(float(entry.get("alpha_cutoff", 0.5)), 8),
                str(entry.get("shadow_mode", "OPAQUE")).upper(),
            )
            for entry in source_modes
        }
        if len(mode_keys) == 1:
            alpha, cutoff, shadow = next(iter(mode_keys))
            return {"alpha_mode": alpha, "alpha_cutoff": cutoff, "shadow_mode": shadow}
    standard = str(getattr(material, "alphaMode", "OPAQUE") or "OPAQUE").upper()
    alpha = {"OPAQUE": "OPAQUE", "MASK": "ALPHA_CLIP", "BLEND": "ALPHA_BLEND"}.get(standard, "OPAQUE")
    try:
        cutoff = float(getattr(material, "alphaCutoff", 0.5))
    except (TypeError, ValueError):
        cutoff = 0.5
    return {
        "alpha_mode": alpha,
        "alpha_cutoff": float(np.clip(cutoff, 0.0, 1.0)),
        "shadow_mode": "OPAQUE",
    }

def _apply_preview_alpha_mode(image, alpha_mode: str, alpha_cutoff: float):
    """Apply the preview-compatible opacity behavior to a copied RGBA image."""
    if image is None:
        return None
    try:
        from PIL import Image
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
        mode = str(alpha_mode or "OPAQUE").upper()
        if mode == "OPAQUE":
            rgba[..., 3] = 255
        elif mode in {"ALPHA_CLIP", "ALPHA_HASHED"}:
            threshold = int(round(float(np.clip(alpha_cutoff, 0.0, 1.0)) * 255.0))
            rgba[..., 3] = np.where(rgba[..., 3] >= threshold, 255, 0).astype(np.uint8)
        return Image.fromarray(rgba, mode="RGBA")
    except Exception:
        return image

def _apply_preview_alpha_to_colors(colors, alpha_mode: str, alpha_cutoff: float):
    values = np.asarray(colors, dtype=np.float32).copy()
    mode = str(alpha_mode or "OPAQUE").upper()
    if values.ndim == 2 and values.shape[1] >= 4:
        if mode == "OPAQUE":
            values[:, 3] = 1.0
        elif mode in {"ALPHA_CLIP", "ALPHA_HASHED"}:
            values[:, 3] = (values[:, 3] >= float(np.clip(alpha_cutoff, 0.0, 1.0))).astype(np.float32)
    return values

def _material_base_color_image(material, *, include_material_factor: bool = False):
    """Return the base-color image, optionally resolved to final display albedo.

    When ``include_material_factor`` is true, apply Base Color Factor in linear
    RGB and return display-encoded sRGB. Import previews and baking use this
    same operation so a baked atlas does not become brighter or darker merely
    because its material factor moved from GLB metadata into texture pixels.
    """
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    if image is None:
        return None
    try:
        from PIL import Image
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
        if include_material_factor:
            factor = _material_base_color(material)
            normalized = _apply_linear_base_color_factor(
                rgba.astype(np.float32) / 255.0, factor
            )
            rgba = np.rint(normalized * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(rgba, mode="RGBA")
    except Exception:
        return None

def _load_polyfitty_diffuse_preview_image(glb_path: str | Path):
    """Read the color-matched Diffuse atlas embedded in a baked GLB.

    Older GLB masters may contain the historical untinted atlas. The metadata
    remains readable for backward compatibility; newly baked files store the
    same final albedo in the standard Base Color and Diffuse preview images.
    """
    import json
    import struct
    from io import BytesIO
    from PIL import Image

    try:
        raw = Path(glb_path).read_bytes()
        if len(raw) < 20 or raw[:4] != b"glTF":
            return None
        version, total_length = struct.unpack_from("<II", raw, 4)
        if version != 2 or total_length > len(raw):
            return None

        document = None
        binary = None
        offset = 12
        while offset + 8 <= total_length:
            length, chunk_type = struct.unpack_from("<II", raw, offset)
            offset += 8
            payload = raw[offset:offset + length]
            if len(payload) != length:
                return None
            offset += length
            if chunk_type == 0x4E4F534A:
                document = json.loads(payload.rstrip(b" \t\r\n\x00").decode("utf-8"))
            elif chunk_type == 0x004E4942:
                binary = payload

        if not isinstance(document, dict) or binary is None:
            return None
        extras = document.get("asset", {}).get("extras", {})

        texture_index = extras.get("polyfittyDiffusePreviewTexture")
        if texture_index is None:
            texture_index = extras.get("polyfitDiffusePreviewTexture")
        if texture_index is None:
            return None
        textures = document.get("textures") or []
        images = document.get("images") or []
        texture_index = int(texture_index)
        if texture_index < 0 or texture_index >= len(textures):
            return None
        image_index = int(textures[texture_index].get("source", -1))
        if image_index < 0 or image_index >= len(images):
            return None
        image_info = images[image_index]
        buffer_view_index = image_info.get("bufferView")
        if buffer_view_index is None:
            return None
        views = document.get("bufferViews") or []
        buffer_view_index = int(buffer_view_index)
        if buffer_view_index < 0 or buffer_view_index >= len(views):
            return None
        view = views[buffer_view_index]
        if int(view.get("buffer", 0)) != 0:
            return None
        start = int(view.get("byteOffset", 0))
        size = int(view.get("byteLength", 0))
        if start < 0 or size <= 0 or start + size > len(binary):
            return None
        with Image.open(BytesIO(binary[start:start + size])) as image:
            return image.convert("RGBA").copy()
    except Exception:
        return None

def _mesh_has_base_color_texture(mesh) -> bool:
    """Whether the mesh carries an actual base-color image, not only a tint."""
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    return _material_base_color_image(material) is not None

def _store_mesh_vertex_colors(mesh, colors: np.ndarray) -> None:
    """Normalize material/texture appearance into vertex colors before joining meshes."""
    import trimesh

    rgba = np.asarray(colors, dtype=np.float32)
    if rgba.ndim != 2 or len(rgba) != len(mesh.vertices) or rgba.shape[1] < 3:
        rgba = np.ones((len(mesh.vertices), 4), dtype=np.float32)
    if rgba.shape[1] == 3:
        rgba = np.column_stack((rgba, np.ones(len(rgba), dtype=np.float32)))
    rgba = np.clip(np.nan_to_num(rgba[:, :4], nan=1.0), 0.0, 1.0)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.rint(rgba * 255.0).astype(np.uint8),
    )

def mesh_vertex_display_colors(
    mesh, *, vertex_color_space: str = "srgb"
) -> np.ndarray | None:
    """Resolve actual mesh appearance to display-encoded sRGB vertex RGBA.

    The returned RGB values always use sRGB display encoding. This lets the
    Diffuse pass display source colors without lighting, while the PBR backend
    converts the same values back to linear before applying physically-based
    lighting. glTF vertex-color attributes are linear by specification; most
    other imported byte-color visuals are already display-encoded sRGB.
    """
    visual = getattr(mesh, "visual", None)
    count = len(getattr(mesh, "vertices", ()))
    if count <= 0:
        return None
    if visual is None:
        return np.ones((count, 4), dtype=np.float32)

    kind = str(getattr(visual, "kind", "") or "").lower()
    material = getattr(visual, "material", None)
    factor = _material_base_color(material)
    if kind == "texture":
        uv = getattr(visual, "uv", None)
        image = getattr(material, "baseColorTexture", None)
        if image is None:
            image = getattr(material, "image", None)
        sampled = _sample_texture_rgba(image, uv)
        if sampled is not None and len(sampled) == count:
            return np.ascontiguousarray(
                _apply_linear_base_color_factor(sampled, factor), dtype=np.float32
            )

        display_factor = np.ones(4, dtype=np.float32)
        display_factor[:3] = _linear_to_srgb(factor[:3])
        display_factor[3] = factor[3]
        return np.tile(display_factor, (count, 1)).astype(np.float32)

    if kind in {"vertex", "face"}:
        vertex_colors = getattr(visual, "vertex_colors", None)
        if vertex_colors is not None and len(vertex_colors) == count:
            arr = np.asarray(vertex_colors, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                if float(np.nanmax(arr)) > 1.5:
                    arr = arr / 255.0
                if arr.shape[1] == 3:
                    arr = np.column_stack((arr, np.ones(len(arr), dtype=np.float32)))
                arr = np.clip(np.nan_to_num(arr[:, :4], nan=1.0), 0.0, 1.0)

                if str(vertex_color_space or "srgb").strip().lower() == "linear":
                    arr[:, :3] = _linear_to_srgb(arr[:, :3] * factor[:3])
                    arr[:, 3] *= factor[3]
                else:
                    arr = _apply_linear_base_color_factor(arr, factor)
                return np.ascontiguousarray(np.clip(arr, 0.0, 1.0), dtype=np.float32)

    display_factor = np.ones(4, dtype=np.float32)
    display_factor[:3] = _linear_to_srgb(factor[:3])
    display_factor[3] = factor[3]
    return np.tile(display_factor, (count, 1)).astype(np.float32)

def _read_text_xyz(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(("#", "//")):
                continue
            line = line.replace(",", " ").replace(";", " ")
            vals: list[float] = []
            for token in line.split():
                try:
                    vals.append(float(token))
                except ValueError:
                    continue
            if len(vals) >= 3:
                rows.append(vals)
    if not rows:
        raise ValueError("No XYZ coordinates were found in the text file.")
    width = min(6, min(len(r) for r in rows))
    data = np.asarray([r[:width] for r in rows if len(r) >= width], dtype=np.float64)

    if path.suffix.lower() == ".xyzn":
        colors = None
    else:
        colors = data[:, 3:6] if width >= 6 else None
    return data[:, :3], colors

def read_vg_ascii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read coordinates and optional colors from PolyFit's ASCII VG format.

    The official C++ serializer may place a whole coordinate/color block on one
    line, while hand-authored examples often place one vector per line. NumPy's
    whitespace parser handles both layouts without building millions of Python
    token strings for large point clouds.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"//.*?$|#.*?$", "", text, flags=re.MULTILINE)
    points_match = re.search(
        r"\bnum_points\s*:\s*(\d+)(.*?)(?=\bnum_colors\s*:|\bnum_normals\s*:|\bnum_groups\s*:|\Z)",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    if not points_match:
        raise ValueError("This .vg file does not contain an ASCII num_points section.")
    n_points = int(points_match.group(1))
    coord_values = np.fromstring(points_match.group(2), sep=" ", dtype=np.float64)
    if n_points < 3 or len(coord_values) != 3 * n_points:
        raise ValueError("The .vg coordinate block is incomplete or contains fewer than three points.")
    points = coord_values.reshape(n_points, 3)

    colors = np.full((n_points, 3), 0.66, dtype=np.float32)
    color_match = re.search(
        r"\bnum_colors\s*:\s*(\d+)(.*?)(?=\bnum_normals\s*:|\bnum_groups\s*:|\Z)",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    if color_match:
        try:
            n_colors = int(color_match.group(1))
            if n_colors == n_points:
                color_values = np.fromstring(color_match.group(2), sep=" ", dtype=np.float32)
                if len(color_values) == 3 * n_colors:
                    colors = _normalise_colors(color_values.reshape(n_colors, 3), n_points)
        except (ValueError, OverflowError):

            pass
    return points, colors

def _read_pcd(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Read PCD ASCII or uncompressed binary data (binary_compressed is not supported)."""
    with path.open("rb") as f:
        header_lines = []
        data_mode = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PCD header is incomplete: DATA line not found.")
            text = line.decode("ascii", errors="ignore").strip()
            header_lines.append(text)
            if text.upper().startswith("DATA "):
                data_mode = text.split(None, 1)[1].strip().lower()
                break
        header = {}
        for line in header_lines:
            bits = line.split()
            if bits:
                header[bits[0].upper()] = bits[1:]
        fields = header.get("FIELDS", header.get("FIELD", []))
        if not fields or not all(k in fields for k in ("x", "y", "z")):
            raise ValueError("PCD needs x, y and z fields.")
        sizes = [int(x) for x in header.get("SIZE", ["4"] * len(fields))]
        types = header.get("TYPE", ["F"] * len(fields))
        counts = [int(x) for x in header.get("COUNT", ["1"] * len(fields))]
        points_n = int(header.get("POINTS", [str(int(header.get("WIDTH", ["0"])[0]) * int(header.get("HEIGHT", ["1"])[0]))])[0])
        if data_mode == "ascii":
            import io
            data = np.loadtxt(io.BytesIO(f.read()), dtype=np.float64, ndmin=2)
            if data.shape[0] != points_n:
                points_n = data.shape[0]
            offsets = {}
            offset = 0
            for field, count in zip(fields, counts):
                offsets[field] = offset
                offset += count
            xyz = np.column_stack([data[:, offsets[k]] for k in ("x", "y", "z")])
            colors = None
            if all(k in offsets for k in ("r", "g", "b")):
                colors = np.column_stack([data[:, offsets[k]] for k in ("r", "g", "b")])
            elif "rgb" in offsets or "rgba" in offsets:
                color_field = "rgb" if "rgb" in offsets else "rgba"
                color_col = data[:, offsets[color_field]]
                field_type = types[fields.index(color_field)].upper()
                if field_type == "F":
                    packed = color_col.astype(np.float32).view(np.uint32)
                else:
                    packed = color_col.astype(np.uint32)
                colors = np.column_stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255])
            return xyz, colors
        if data_mode != "binary":
            raise ValueError("This PCD is binary_compressed. Use ASCII or uncompressed binary PCD, or convert it to PLY first.")

        type_map = {
            ("F", 4): np.dtype("<f4"), ("F", 8): np.dtype("<f8"),
            ("I", 1): np.dtype("i1"), ("I", 2): np.dtype("<i2"),
            ("I", 4): np.dtype("<i4"), ("I", 8): np.dtype("<i8"),
            ("U", 1): np.dtype("u1"), ("U", 2): np.dtype("<u2"),
            ("U", 4): np.dtype("<u4"), ("U", 8): np.dtype("<u8"),
        }
        dtype_fields = []
        for field, size, kind, count in zip(fields, sizes, types, counts):
            base = type_map.get((kind.upper(), size))
            if base is None:
                raise ValueError(f"Unsupported PCD field type: {field} {kind}{size}")
            dtype_fields.append((field, base) if count == 1 else (field, base, (count,)))
        dtype = np.dtype(dtype_fields, align=False)
        raw = f.read(points_n * dtype.itemsize)
        if len(raw) < points_n * dtype.itemsize:
            raise ValueError("PCD binary payload is truncated.")
        records = np.frombuffer(raw, dtype=dtype, count=points_n)
        xyz = np.column_stack([records[k] for k in ("x", "y", "z")]).astype(np.float64)
        colors = None
        if all(k in fields for k in ("r", "g", "b")):
            colors = np.column_stack([records[k] for k in ("r", "g", "b")])
        elif "rgb" in fields or "rgba" in fields:
            packed = records["rgb"] if "rgb" in fields else records["rgba"]
            if packed.dtype.kind == "f":
                packed = packed.astype(np.float32, copy=False).view(np.uint32)
            else:
                packed = packed.astype(np.uint32, copy=False)
            colors = np.column_stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255])
        return xyz, colors

def _allocate_component_sample_counts(
    areas: np.ndarray | list[float], total_samples: int, min_per_component: int = 128
) -> np.ndarray:
    """Allocate a surface-sampling budget while guaranteeing a floor per component.

    The remaining budget is distributed by surface area. This protects small
    detached roof/eave/accessory meshes from receiving zero or only a handful
    of samples when a much larger building shell is present.
    """
    area_values = np.asarray(areas, dtype=np.float64).reshape(-1)
    if len(area_values) == 0:
        return np.zeros(0, dtype=np.int64)
    area_values = np.where(np.isfinite(area_values) & (area_values > 0.0), area_values, 0.0)
    total = max(int(total_samples), len(area_values))
    minimum = max(1, min(int(min_per_component), total // len(area_values)))
    counts = np.full(len(area_values), minimum, dtype=np.int64)
    remaining = total - int(counts.sum())
    if remaining <= 0:
        return counts
    weights = area_values
    if float(weights.sum()) <= 1e-15:
        weights = np.ones(len(area_values), dtype=np.float64)
    weights = weights / weights.sum()
    exact = weights * remaining
    extra = np.floor(exact).astype(np.int64)
    counts += extra
    leftover = remaining - int(extra.sum())
    if leftover > 0:
        order = np.argsort(-(exact - extra))
        counts[order[:leftover]] += 1
    return counts

def _connected_sampling_components(geom) -> list:
    """Split one triangle mesh into face-connected components for sampling."""
    try:
        components = list(geom.split(only_watertight=False))
    except Exception:
        components = []
    if not components:
        components = [geom]
    return [
        component for component in components
        if len(getattr(component, "faces", [])) > 0
        and np.isfinite(float(getattr(component, "area", 0.0)))
        and float(getattr(component, "area", 0.0)) > 1e-12
    ]

def _load_with_trimesh(
    path: Path, sample_count: int = 180000, merge_distance: float = 0.0
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load point clouds as-is and sample actual mesh surfaces for mesh files.

    Earlier versions treated every OBJ/STL/mesh as a point set of its vertices.
    That is very sparse for low-poly character meshes, and the
    vertices are not distributed uniformly over the surface. Mesh inputs are
    now sampled by triangle area; point-cloud inputs still preserve their points.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is not installed. Run INSTALL.bat.") from exc

    loaded = trimesh.load(str(path), process=False)
    source_vertex_color_space = "linear" if Path(path).suffix.lower() in {".glb", ".gltf"} else "srgb"
    if isinstance(loaded, trimesh.Scene):
        geometries = []

        for node in loaded.graph.nodes_geometry:
            try:
                transform, geometry_name = loaded.graph[node]
                geom = loaded.geometry[geometry_name].copy()
                if hasattr(geom, "apply_transform"):
                    geom.apply_transform(transform)
                elif hasattr(geom, "vertices"):
                    xyz = np.asarray(geom.vertices, dtype=np.float64)
                    hom = np.column_stack([xyz[:, :3], np.ones(len(xyz))])
                    geom.vertices = (hom @ np.asarray(transform).T)[:, :3]
                geometries.append(geom)
            except Exception:
                continue
        if not geometries:
            geometries = list(loaded.geometry.values())
    else:
        geometries = [loaded]
    mesh_geometries = [g for g in geometries if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0 and float(getattr(g, "area", 0.0)) > 1e-12]
    point_geometries = [g for g in geometries if not (isinstance(g, trimesh.Trimesh) and len(g.faces) > 0)]

    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    has_any_color = False

    if mesh_geometries:
        if merge_distance > 0.0:

            normalized_meshes = []
            for original in mesh_geometries:
                part = original.copy()
                part, _ = weld_coincident_mesh_vertices(part)
                appearance = mesh_vertex_display_colors(
                    part, vertex_color_space=source_vertex_color_space
                )
                if appearance is None:
                    appearance = np.ones((len(part.vertices), 4), dtype=np.float32)
                _store_mesh_vertex_colors(part, appearance)
                normalized_meshes.append(part)
            combined = trimesh.util.concatenate(normalized_meshes) if len(normalized_meshes) > 1 else normalized_meshes[0]
            source_vertices = np.asarray(combined.vertices, dtype=np.float64).copy()
            source_colors = mesh_vertex_display_colors(combined, vertex_color_space="srgb")
            merged_vertices, merged_faces, merge_stats = merge_mesh_vertices_by_distance(
                source_vertices, np.asarray(combined.faces), merge_distance
            )
            if len(merged_faces) == 0:
                raise ValueError(
                    "Merge by Distance removed every triangle. Reduce the merge distance "
                    "or disable vertex merging."
                )
            combined = trimesh.Trimesh(vertices=merged_vertices, faces=merged_faces, process=False)
            transferred = _nearest_vertex_colors(merged_vertices, source_vertices, source_colors)
            if transferred is not None:
                values = np.rint(np.clip(transferred, 0.0, 1.0) * 255.0).astype(np.uint8)
                if values.shape[1] == 3:
                    values = np.column_stack((values, np.full(len(values), 255, dtype=np.uint8)))
                combined.visual.vertex_colors = values
            mesh_geometries = [combined]
        else:

            mesh_geometries = [weld_coincident_mesh_vertices(part.copy())[0] for part in mesh_geometries]

        sampling_components = []
        for geom in mesh_geometries:
            sampling_components.extend(_connected_sampling_components(geom))
        if not sampling_components:
            raise ValueError("No non-degenerate triangle components were found in the input mesh.")

        count_total = max(3000, int(sample_count), len(sampling_components))
        sample_counts = _allocate_component_sample_counts(
            [float(component.area) for component in sampling_components],
            count_total,
            min_per_component=128,
        )
        for component_index, (geom, count) in enumerate(zip(sampling_components, sample_counts)):
            count = int(count)
            if count <= 0:
                continue
            try:
                sampled, face_ids = trimesh.sample.sample_surface(geom, count, seed=19 + component_index)
            except TypeError:
                sampled, face_ids = trimesh.sample.sample_surface(geom, count)
            sampled = np.asarray(sampled, dtype=np.float64)
            point_chunks.append(sampled)

            visual = getattr(geom, "visual", None)
            kind = str(getattr(visual, "kind", "") or "").lower()
            material = getattr(visual, "material", None)
            factor = _material_base_color(material)
            vertex_colors = getattr(visual, "vertex_colors", None)
            try:
                tri_vertices = np.asarray(geom.triangles[face_ids], dtype=np.float64)
                bary = trimesh.triangles.points_to_barycentric(tri_vertices, sampled)
                sampled_colors = None
                if kind == "texture":
                    uv = getattr(visual, "uv", None)
                    image = getattr(material, "baseColorTexture", None)
                    if image is None:
                        image = getattr(material, "image", None)
                    if uv is not None and image is not None:
                        tri_uv = np.asarray(uv, dtype=np.float64)[np.asarray(geom.faces)[face_ids]]
                        sampled_uv = np.einsum("ij,ijk->ik", bary, tri_uv)
                        sampled_colors = _sample_texture_rgba(image, sampled_uv)
                    if sampled_colors is not None:
                        sampled_colors = sampled_colors * factor[None, :]
                    else:
                        sampled_colors = np.tile(factor, (len(sampled), 1))
                elif kind == "face":
                    face_colors = np.asarray(visual.face_colors, dtype=np.float32)[face_ids]
                    if float(np.nanmax(face_colors)) > 1.5:
                        face_colors /= 255.0
                    if face_colors.shape[1] == 3:
                        face_colors = np.column_stack((face_colors, np.ones(len(face_colors))))
                    sampled_colors = face_colors * factor[None, :]
                elif vertex_colors is not None and len(vertex_colors) == len(geom.vertices):
                    tri_colors = np.asarray(vertex_colors, dtype=np.float32)[np.asarray(geom.faces)[face_ids]]
                    if float(np.nanmax(tri_colors)) > 1.5:
                        tri_colors /= 255.0
                    if tri_colors.shape[2] == 3:
                        tri_colors = np.concatenate(
                            (tri_colors, np.ones((*tri_colors.shape[:2], 1), dtype=np.float32)), axis=2
                        )
                    sampled_colors = np.einsum("ij,ijk->ik", bary, tri_colors) * factor[None, :]
                else:
                    sampled_colors = np.tile(factor, (len(sampled), 1))
                color_chunks.append(np.clip(sampled_colors[:, :3], 0.0, 1.0) * 255.0)
                has_any_color = True
            except Exception:
                color_chunks.append(np.full((len(sampled), 3), 255.0, dtype=np.float32))
                has_any_color = True

    for geom in point_geometries:
        if not hasattr(geom, "vertices"):
            continue
        vertices = np.asarray(geom.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] < 3 or len(vertices) == 0:
            continue
        point_chunks.append(vertices[:, :3])
        colors = getattr(getattr(geom, "visual", None), "vertex_colors", None)
        if colors is not None and len(colors) == len(vertices):
            color_chunks.append(np.asarray(colors)[:, :3])
            has_any_color = True
        else:
            color_chunks.append(np.full((len(vertices), 3), 255, dtype=np.uint8))

    if not point_chunks:
        raise ValueError(f"No point or surface geometry could be loaded from {path.name}.")
    points = np.vstack(point_chunks)
    colors = np.vstack(color_chunks) if has_any_color else None
    return points, colors

def convert_up_axis(points: np.ndarray, source_up_axis: str = "y") -> np.ndarray:
    """Rotate coordinates from a source up-axis (X/Y/Z) into Studio's Z-up frame.

    The transforms are proper rotations (determinant +1), so triangle winding
    and handedness are preserved.
    """
    pts = np.asarray(points)
    axis = str(source_up_axis or "y").lower()
    if axis == "z":
        return pts.copy()
    if axis == "y":

        return np.column_stack((pts[:, 0], -pts[:, 2], pts[:, 1])).astype(pts.dtype, copy=False)
    if axis == "x":

        return np.column_stack((-pts[:, 2], pts[:, 1], pts[:, 0])).astype(pts.dtype, copy=False)
    raise ValueError(f"Unsupported source up-axis: {source_up_axis!r}; choose x, y or z.")

def mesh_topology_summary(vertices, faces) -> dict:
    import trimesh

    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(verts) == 0 or len(tris) == 0:
        return {
            "vertices": int(len(verts)), "triangles": int(len(tris)),
            "unique_edges": 0, "boundary_edges": 0, "nonmanifold_edges": 0,
            "components": 0, "watertight": False, "surface_area": 0.0,
        }
    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    edge_counts = np.bincount(
        mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
    )
    try:
        components = len(mesh.split(only_watertight=False))
    except Exception:
        components = -1
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "unique_edges": int(len(mesh.edges_unique)),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(edge_counts > 2)),
        "components": int(components),
        "watertight": bool(mesh.is_watertight),
        "surface_area": float(mesh.area),
    }

def read_gltf_material_render_modes(path: str | Path) -> list[dict]:
    """Read glTF alpha settings and Polyfitty render-mode extras, when present."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in {".glb", ".gltf"}:
        return []
    try:
        if suffix == ".gltf":
            document = json.loads(source.read_text(encoding="utf-8-sig"))
        else:
            raw = source.read_bytes()
            if len(raw) < 20 or raw[:4] != b"glTF":
                return []
            version, total_length = struct.unpack_from("<II", raw, 4)
            if version != 2 or total_length > len(raw):
                return []
            offset = 12
            document = None
            while offset + 8 <= total_length:
                chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
                offset += 8
                payload = raw[offset:offset + chunk_length]
                if len(payload) != chunk_length:
                    return []
                offset += chunk_length
                if chunk_type == 0x4E4F534A:
                    document = json.loads(payload.rstrip(b" \t\r\n\x00").decode("utf-8"))
                    break
            if not isinstance(document, dict):
                return []
        results = []
        for index, material in enumerate(document.get("materials", []) or []):
            extras = material.get("extras") if isinstance(material.get("extras"), dict) else {}
            studio = extras.get("polyfittyStudio") if isinstance(extras, dict) else {}
            if not isinstance(studio, dict):
                studio = {}
            standard_alpha = str(material.get("alphaMode", "OPAQUE")).upper()
            fallback = {"OPAQUE": "OPAQUE", "MASK": "ALPHA_CLIP", "BLEND": "ALPHA_BLEND"}.get(standard_alpha, "OPAQUE")
            alpha = str(studio.get("alphaBlendMode", fallback)).upper()
            if alpha not in {"OPAQUE", "ALPHA_CLIP", "ALPHA_HASHED", "ALPHA_BLEND"}:
                alpha = fallback
            shadow = str(studio.get("shadowMode", "OPAQUE")).upper()
            if shadow not in {"NONE", "OPAQUE", "ALPHA_CLIP", "ALPHA_HASHED"}:
                shadow = "OPAQUE"
            cutoff = studio.get("alphaCutoff", material.get("alphaCutoff", 0.5))
            try:
                cutoff = float(cutoff)
            except (TypeError, ValueError):
                cutoff = 0.5
            cutoff = float(np.clip(cutoff, 0.0, 1.0))
            results.append({
                "index": int(index),
                "name": str(material.get("name") or f"Material {index + 1}"),
                "alpha_mode": alpha,
                "standard_alpha_mode": standard_alpha,
                "alpha_cutoff": cutoff,
                "shadow_mode": shadow,
            })
        return results
    except (OSError, UnicodeError, ValueError, TypeError, struct.error):
        return []

def load_preview_asset(
    path: str | Path, mesh_sample_count: int = 60000, merge_distance: float = 0.0,
    *, alpha_mode_override: str | None = None, alpha_cutoff_override: float | None = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """Load a display-ready source asset for the unified viewport.

    Mesh inputs are shown as their original triangle surface when feasible;
    point-cloud/VG inputs are shown as points. Large meshes are still displayed
    as meshes but the viewport may decimate their faces for interactive speed.
    When ``merge_distance`` is positive, the returned display geometry uses the
    same distance-weld helper and tolerance as Reconstruction. The file on disk
    is never modified; the original file remains available for high-poly baking.
    """
    p = Path(path)
    ext = p.suffix.lower()

    def _progress(value: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(int(value), str(message))

    _progress(-1, f"Inspecting {p.name} · reading geometry and material data…")
    source_frame_transform = np.eye(4, dtype=np.float64)
    source_mesh_frame_found = False
    source_point_frame_found = False
    mesh_exts = {".obj", ".stl", ".off", ".glb", ".gltf", ".3mf"}
    source_material_modes = read_gltf_material_render_modes(p) if ext in {".glb", ".gltf"} else []
    if ext in mesh_exts or ext == ".ply":
        try:
            import trimesh
            _progress(-1, f"Parsing {p.name} · decoding mesh, UVs, textures and scene nodes…")
            loaded = trimesh.load(str(p), process=False)
            _progress(18, "Geometry decoded · extracting transformed mesh parts…")
            meshes = []
            clouds = []
            if isinstance(loaded, trimesh.Scene):
                geometry_nodes = list(loaded.graph.nodes_geometry)
                node_count = max(1, len(geometry_nodes))
                for node_index, node in enumerate(geometry_nodes, start=1):
                    _progress(18 + int(20 * node_index / node_count),
                              f"Extracting geometry parts · {node_index}/{len(geometry_nodes)}…")
                    try:
                        transform, geometry_name = loaded.graph[node]
                        geom = loaded.geometry[geometry_name]
                        if isinstance(geom, trimesh.Trimesh) and len(geom.faces):
                            if not source_mesh_frame_found:
                                source_frame_transform = np.asarray(
                                    transform, dtype=np.float64
                                ).reshape(4, 4).copy()
                                source_mesh_frame_found = True

                            source_normals = _unit_vertex_normals(
                                geom.vertex_normals, len(geom.vertices)
                            )
                            part = geom.copy()
                            if source_normals is not None:
                                part.metadata[_PREVIEW_NORMALS_METADATA_KEY] = _transform_vertex_normals(
                                    source_normals, transform
                                )
                            part.apply_transform(transform)
                            meshes.append(part)
                        elif hasattr(geom, "vertices"):
                            if not source_mesh_frame_found and not source_point_frame_found:
                                source_frame_transform = np.asarray(
                                    transform, dtype=np.float64
                                ).reshape(4, 4).copy()
                                source_point_frame_found = True
                            vertices = np.asarray(geom.vertices, dtype=np.float64)
                            if len(vertices):
                                hom = np.column_stack([vertices[:, :3], np.ones(len(vertices))])
                                clouds.append((hom @ np.asarray(transform).T)[:, :3])
                    except Exception:
                        continue
            elif isinstance(loaded, trimesh.Trimesh) and len(loaded.faces):
                _progress(38, "Extracting mesh topology and authored normals…")
                source_normals = _unit_vertex_normals(
                    loaded.vertex_normals, len(loaded.vertices)
                )
                direct_mesh = loaded.copy()
                if source_normals is not None:
                    direct_mesh.metadata[_PREVIEW_NORMALS_METADATA_KEY] = source_normals
                meshes.append(direct_mesh)
            elif hasattr(loaded, "vertices"):
                _progress(38, "Extracting point geometry…")
                clouds.append(np.asarray(loaded.vertices, dtype=np.float64)[:, :3])

            if meshes:
                _progress(40, f"Preparing material previews · {len(meshes)} mesh parts…")

                source_vertex_color_space = "linear" if ext in {".glb", ".gltf"} else "srgb"
                normalized_meshes = []
                source_has_texture = False
                material_parts = []
                mesh_count = max(1, len(meshes))
                for mesh_index, original in enumerate(meshes, start=1):
                    _progress(40 + int(38 * (mesh_index - 1) / mesh_count),
                              f"Parsing material and texture data · {mesh_index}/{len(meshes)}…")
                    part = original.copy()
                    visual = getattr(part, "visual", None)
                    material = getattr(visual, "material", None)

                    texture_image = _material_base_color_image(
                        material, include_material_factor=True
                    )
                    diffuse_texture_image = (
                        texture_image.copy() if texture_image is not None else None
                    )
                    render_settings = (
                        _material_render_settings(material, source_material_modes)
                        if ext in {".glb", ".gltf"} else
                        {"alpha_mode": "NATIVE", "alpha_cutoff": 0.5, "shadow_mode": "OPAQUE"}
                    )
                    if alpha_mode_override is not None:
                        override_mode = str(alpha_mode_override).strip().upper()
                        if override_mode not in {"OPAQUE", "ALPHA_CLIP", "ALPHA_HASHED", "ALPHA_BLEND"}:
                            raise ValueError(f"Unsupported source alpha mode override: {override_mode}")
                        try:
                            override_cutoff = float(
                                0.5 if alpha_cutoff_override is None else alpha_cutoff_override
                            )
                        except (TypeError, ValueError) as exc:
                            raise ValueError("Source alpha cutoff must be between 0 and 1.") from exc
                        if not np.isfinite(override_cutoff) or not 0.0 <= override_cutoff <= 1.0:
                            raise ValueError("Source alpha cutoff must be a finite value between 0 and 1.")
                        render_settings["alpha_mode"] = override_mode
                        render_settings["alpha_cutoff"] = override_cutoff
                    if render_settings["alpha_mode"] != "NATIVE":
                        texture_image = _apply_preview_alpha_mode(
                            texture_image, render_settings["alpha_mode"], render_settings["alpha_cutoff"]
                        )
                        diffuse_texture_image = _apply_preview_alpha_mode(
                            diffuse_texture_image, render_settings["alpha_mode"], render_settings["alpha_cutoff"]
                        )
                    source_has_texture = source_has_texture or diffuse_texture_image is not None
                    appearance = mesh_vertex_display_colors(
                        part, vertex_color_space=source_vertex_color_space
                    )
                    if appearance is None:
                        appearance = np.ones((len(part.vertices), 4), dtype=np.float32)
                    source_normals = _mesh_preview_vertex_normals(part)
                    pbr_factors = _material_pbr_factors(material)
                    orm_texture_image = _material_orm_preview_image(material)
                    occlusion_strength = _material_occlusion_strength(material)
                    if render_settings["alpha_mode"] != "NATIVE":
                        appearance = _apply_preview_alpha_to_colors(
                            appearance, render_settings["alpha_mode"], render_settings["alpha_cutoff"]
                        )
                    uv = getattr(visual, "uv", None)
                    uv_values = None
                    if (texture_image is not None or orm_texture_image is not None) and uv is not None:
                        candidate_uv = np.asarray(uv, dtype=np.float32)
                        if candidate_uv.shape == (len(part.vertices), 2) and np.isfinite(candidate_uv).all():
                            uv_values = np.ascontiguousarray(candidate_uv)
                    material_parts.append({
                        "vertices": np.asarray(part.vertices, dtype=np.float32).copy(),
                        "faces": np.asarray(part.faces, dtype=np.uint32).copy(),
                        "colors": np.asarray(appearance, dtype=np.float32).copy(),
                        "normals": None if source_normals is None else source_normals.copy(),
                        "roughness": pbr_factors["roughness"],
                        "metallic": pbr_factors["metallic"],
                        "uvs": uv_values,
                        "texture_image": texture_image if uv_values is not None else None,
                        "diffuse_texture_image": diffuse_texture_image if uv_values is not None else None,
                        "orm_texture_image": orm_texture_image if uv_values is not None else None,
                        "occlusion_strength": occlusion_strength,
                        "alpha_mode": render_settings["alpha_mode"],
                        "alpha_cutoff": render_settings["alpha_cutoff"],
                        "shadow_mode": render_settings["shadow_mode"],
                    })
                    _store_mesh_vertex_colors(part, appearance)
                    normalized_meshes.append(part)
                _progress(80, "Combining material parts and calculating mesh statistics…")
                mesh = trimesh.util.concatenate(normalized_meshes) if len(normalized_meshes) > 1 else normalized_meshes[0]
                original_mesh_stats = mesh_topology_summary(mesh.vertices, mesh.faces)
                merge_note = ""
                if merge_distance > 0.0:
                    _progress(-1, "Applying optional vertex merge · this can take a moment for dense meshes…")
                    source_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
                    source_colors = mesh_vertex_display_colors(mesh, vertex_color_space="srgb")
                    vertices, faces, merge_stats = merge_mesh_vertices_by_distance(
                        source_vertices, np.asarray(mesh.faces), merge_distance
                    )
                    if len(faces) == 0:
                        raise ValueError(
                            "Merge by Distance removed every triangle. Reduce the merge distance "
                            "or disable vertex merging."
                        )
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                    transferred = _nearest_vertex_colors(vertices, source_vertices, source_colors)
                    if transferred is not None:
                        values = np.rint(np.clip(transferred, 0.0, 1.0) * 255.0).astype(np.uint8)
                        if values.shape[1] == 3:
                            values = np.column_stack((values, np.full(len(values), 255, dtype=np.uint8)))
                        mesh.visual.vertex_colors = values
                    merge_note = f" · merged {merge_stats['merged_vertices']:,} verts @ {merge_distance:g}"
                _progress(94, "Finalizing preview mesh and material metadata…")
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                faces = np.asarray(mesh.faces, dtype=np.uint32)
                vertex_colors = mesh_vertex_display_colors(mesh, vertex_color_space="srgb")
                if len(vertices) >= 3 and len(faces):
                    asset = {"kind": "mesh", "vertices": vertices, "faces": faces,
                             "label": f"source mesh · {len(vertices):,} vertices / {len(faces):,} faces{merge_note}"}
                    if vertex_colors is not None and len(vertex_colors) == len(vertices):
                        asset["colors"] = vertex_colors.astype(np.float32)
                    asset["has_texture"] = bool(source_has_texture)
                    asset["mesh_stats"] = original_mesh_stats

                    asset["material_parts"] = material_parts
                    asset_normals = _mesh_preview_vertex_normals(mesh)
                    if asset_normals is not None and len(asset_normals) == len(vertices):
                        asset["normals"] = asset_normals.astype(np.float32, copy=True)
                    asset["material_render_modes"] = source_material_modes
                    asset["source_frame_transform"] = source_frame_transform.copy()
                    _progress(100, f"Import ready · {len(vertices):,} vertices / {len(faces):,} faces")
                    return asset
            if clouds:
                points = np.vstack(clouds).astype(np.float32)
                if len(points) > mesh_sample_count:
                    rng = np.random.default_rng(71)
                    idx = np.sort(rng.choice(len(points), mesh_sample_count, replace=False))
                    points = points[idx]
                _progress(100, f"Import ready · {len(points):,} points")
                return {"kind": "points", "points": points,
                        "colors": np.ones((len(points), 3), dtype=np.float32),
                        "source_frame_transform": source_frame_transform.copy(),
                        "label": f"source cloud · {len(points):,} points"}
        except Exception:

            if merge_distance > 0.0:
                raise

            pass

    _progress(-1, f"Reading point-cloud data · {p.name}…")
    points, colors = load_points(p, mesh_sample_count=mesh_sample_count)
    _progress(94, "Filtering non-finite points and normalizing colors…")
    if len(points) > mesh_sample_count:
        rng = np.random.default_rng(71)
        idx = np.sort(rng.choice(len(points), mesh_sample_count, replace=False))
        points, colors = points[idx], colors[idx]
    _progress(100, f"Import ready · {len(points):,} points")
    return {"kind": "points", "points": np.asarray(points, dtype=np.float32),
            "colors": np.asarray(colors, dtype=np.float32),
            "source_frame_transform": np.eye(4, dtype=np.float64),
            "label": f"source cloud · {len(points):,} points"}

def load_points(
    path: str | Path, mesh_sample_count: int = 180000, merge_distance: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".vg":
        pts, cols = read_vg_ascii(p)
    elif ext in {".xyz", ".xyzn", ".xyzrgb", ".txt", ".pts", ".csv", ".asc"}:
        pts, cols = _read_text_xyz(p)
    elif ext == ".pcd":
        pts, cols = _read_pcd(p)
    elif ext == ".bvg":
        raise ValueError("Binary .bvg is a segmented PolyFit input; it can be reconstructed directly but cannot be previewed as raw points by this lightweight reader.")
    else:
        pts, cols = _load_with_trimesh(p, sample_count=mesh_sample_count, merge_distance=merge_distance)
    finite = np.isfinite(pts).all(axis=1)
    pts = np.asarray(pts[finite], dtype=np.float64)
    if cols is not None and len(cols) == len(finite):
        cols = np.asarray(cols)[finite]
    if len(pts) < 3:
        raise ValueError("At least three finite points are required.")
    return pts, _normalise_colors(cols, len(pts))

def reduce_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    voxel_size: float = 0.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size > 0:
        origin = points.min(axis=0)
        keys = np.floor((points - origin) / voxel_size).astype(np.int64)
        _, indices = np.unique(keys, axis=0, return_index=True)
        indices.sort()
        points, colors = points[indices], colors[indices]
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points, colors = points[indices], colors[indices]
    return points, colors

def write_vg_ascii(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray,
    groups: list[dict],
    normals: np.ndarray,
    progress: Optional[Callable[[int], None]] = None,
) -> None:
    """Write the ASCII Vertex Group file expected by PolyFit's point-set serializer."""
    path = Path(path)
    n = len(points)
    if normals.shape != (n, 3):
        raise ValueError("Normals must have shape (number_of_points, 3).")
    if colors.shape != (n, 3):
        colors = _normalise_colors(colors, n)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(f"num_points: {n}\n")
        for i, p in enumerate(points):
            f.write(f"{p[0]:.16g} {p[1]:.16g} {p[2]:.16g}\n")
            if progress and i % 50000 == 0:
                progress(5 + int(15 * i / max(1, n)))
        f.write(f"num_colors: {n}\n")
        for i, c in enumerate(colors):
            f.write(f"{c[0]:.6g} {c[1]:.6g} {c[2]:.6g}\n")
            if progress and i % 50000 == 0:
                progress(20 + int(8 * i / max(1, n)))
        f.write(f"num_normals: {n}\n")
        for i, normal in enumerate(normals):
            f.write(f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n")
            if progress and i % 50000 == 0:
                progress(28 + int(8 * i / max(1, n)))
        f.write(f"num_groups: {len(groups)}\n")
        for gi, group in enumerate(groups):
            a, b, c, d = group["plane"]
            idx = np.asarray(group["indices"], dtype=np.int64)
            color = group["color"]
            f.write("group_type: 0\n")
            f.write("num_group_parameters: 4\n")
            f.write(f"group_parameters: {a:.16g} {b:.16g} {c:.16g} {d:.16g}\n")
            f.write(f"group_label: auto_plane_{gi + 1:02d}\n")
            f.write(f"group_color: {color[0]:.6g} {color[1]:.6g} {color[2]:.6g}\n")

            f.write(f"group_num_point: {len(idx)}\n")
            chunk_count = max(1, (len(idx) + 24 - 1) // 24)
            for chunk_index, offset in enumerate(range(0, len(idx), 24)):
                f.write(" ".join(str(int(v)) for v in idx[offset:offset + 24]) + "\n")
                if progress and chunk_index % 1000 == 0:
                    fraction = (gi + (chunk_index + 1) / chunk_count) / max(1, len(groups))
                    progress(36 + int(4 * fraction))
            f.write("num_children: 0\n")
            if progress:
                progress(36 + int(4 * (gi + 1) / max(1, len(groups))))

def load_baked_glb_asset(path: str | Path) -> dict:
    """Load the final baked GLB, retaining its actual UV atlas for VTK display."""
    import trimesh

    p = Path(path)
    if not p.is_file() or p.suffix.lower() != ".glb":
        raise ValueError(f"A baked GLB result is required for result inspection: {p}")
    loaded = trimesh.load(str(p), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError("The baked GLB does not contain a valid triangle mesh.")
    vertices = np.asarray(loaded.vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(loaded.faces, dtype=np.uint32).reshape(-1, 3)
    visual = getattr(loaded, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = _material_base_color_image(material)
    orm_image = _material_orm_preview_image(material)
    pbr_factors = _material_pbr_factors(material)
    occlusion_strength = _material_occlusion_strength(material)
    diffuse_image = _load_polyfitty_diffuse_preview_image(p)
    if diffuse_image is None:
        diffuse_image = _material_base_color_image(material, include_material_factor=False)
    uvs = None
    if uv is not None:
        candidate = np.asarray(uv, dtype=np.float32)
        if candidate.shape == (len(vertices), 2):
            uvs = candidate
    colors = None
    if image is None:
        colors = mesh_vertex_display_colors(loaded, vertex_color_space="linear")
    mesh_stats = mesh_topology_summary(vertices, faces)
    try:
        mesh_stats["file_size_mb"] = p.stat().st_size / (1024.0 * 1024.0)
    except OSError:
        pass
    return {
        "kind": "mesh",
        "vertices": vertices,
        "faces": faces,
        "uvs": uvs,
        "texture_image": image,
        "diffuse_texture_image": diffuse_image,
        "orm_texture_image": orm_image,
        "pbr_factors": pbr_factors,
        "occlusion_strength": occlusion_strength,
        "colors": colors,
        "has_texture": image is not None and uvs is not None,
        "mesh_stats": mesh_stats,
        "label": f"Baked GLB · {len(vertices):,} vertices / {len(faces):,} faces",
        "path": str(p),
    }
