from __future__ import annotations

"""Export baked GLB master assets to common interchange formats."""

import json
import struct
import base64
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from .coordinate_frame import as_transform_matrix, extract_scene_source_frame
except ImportError:
    from coordinate_frame import as_transform_matrix, extract_scene_source_frame

SPECULAR_EXTENSION = "KHR_materials_specular"
PBR_MAP_KEYS = (
    "base_color", "normal", "roughness", "metallic", "ao",
    "height", "opacity", "emissive", "orm",
)

def _normalize_pbr_maps(values) -> tuple[str, ...]:
    selected = {str(value).strip().lower() for value in (values or ())}
    return tuple(key for key in PBR_MAP_KEYS if key in selected)

def supported_pbr_maps_for_format(suffix: str) -> tuple[str, ...]:
    """Return PBR maps the selected interchange format can actually preserve."""
    extension = str(suffix or "").lower()
    if not extension.startswith("."):
        extension = "." + extension if extension else ""
    if extension in {".glb", ".gltf"}:
        return PBR_MAP_KEYS
    if extension in {".obj", ".ply", ".fbx"}:
        return ("base_color",)
    return ()

ALPHA_BLEND_MODES = ("OPAQUE", "ALPHA_CLIP", "ALPHA_HASHED", "ALPHA_BLEND")
SHADOW_MODES = ("NONE", "OPAQUE", "ALPHA_CLIP", "ALPHA_HASHED")

def _normalize_render_modes(alpha_mode: str, alpha_cutoff: float, shadow_mode: str):
    alpha = str(alpha_mode or "ALPHA_CLIP").strip().upper()
    shadow = str(shadow_mode or "OPAQUE").strip().upper()
    if alpha not in ALPHA_BLEND_MODES:
        raise ValueError(f"Unsupported alpha blend mode: {alpha}")
    if shadow not in SHADOW_MODES:
        raise ValueError(f"Unsupported shadow mode: {shadow}")
    try:
        cutoff = float(alpha_cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("Alpha cutoff must be a number between 0 and 1.") from exc
    if not np.isfinite(cutoff) or not 0.0 <= cutoff <= 1.0:
        raise ValueError("Alpha cutoff must be a finite number between 0 and 1.")
    return alpha, cutoff, shadow

def _apply_gltf_material_render_modes(document: dict, alpha_mode: str, alpha_cutoff: float, shadow_mode: str) -> dict:
    """Apply standard glTF alpha fields and retain Studio-only render metadata.

    Alpha Hashed and shadow casting modes have no core glTF equivalent. They
    are preserved in material.extras.polyfittyStudio. Alpha Hashed uses MASK
    as a standards-compatible fallback for viewers that ignore the extras.
    """
    alpha, cutoff, shadow = _normalize_render_modes(alpha_mode, alpha_cutoff, shadow_mode)
    materials = document.setdefault("materials", [])
    if not materials:
        materials.append({"name": "Polyfitty Material"})
        for mesh in document.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                primitive.setdefault("material", 0)
    standard_alpha = {
        "OPAQUE": "OPAQUE",
        "ALPHA_CLIP": "MASK",
        "ALPHA_HASHED": "MASK",
        "ALPHA_BLEND": "BLEND",
    }[alpha]
    for material in materials:
        material["alphaMode"] = standard_alpha
        if standard_alpha == "MASK":
            material["alphaCutoff"] = cutoff
        else:
            material.pop("alphaCutoff", None)
        extras = material.setdefault("extras", {})
        if not isinstance(extras, dict):
            extras = {}
            material["extras"] = extras
        studio = extras.setdefault("polyfittyStudio", {})
        if not isinstance(studio, dict):
            studio = {}
            extras["polyfittyStudio"] = studio
        studio.update({
            "alphaBlendMode": alpha,
            "alphaCutoff": cutoff,
            "shadowMode": shadow,
        })
    return document

def patch_glb_material_render_modes(
    data: bytes, alpha_mode: str = "ALPHA_CLIP", alpha_cutoff: float = 0.5,
    shadow_mode: str = "OPAQUE",
) -> bytes:
    """Set glTF alpha fields and Polyfitty shadow/hashed metadata in a GLB."""
    document, binary, other_chunks = _glb_parts(data)
    _apply_gltf_material_render_modes(document, alpha_mode, alpha_cutoff, shadow_mode)
    return _pack_glb_parts(document, binary, other_chunks)

def patch_gltf_material_render_modes(
    data: bytes, alpha_mode: str = "ALPHA_CLIP", alpha_cutoff: float = 0.5,
    shadow_mode: str = "OPAQUE",
) -> bytes:
    """Set glTF alpha fields and Polyfitty shadow/hashed metadata in JSON."""
    document = json.loads(bytes(data).decode("utf-8-sig"))
    _apply_gltf_material_render_modes(document, alpha_mode, alpha_cutoff, shadow_mode)
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def _pbr_texture_keys_for_selection(selected_maps) -> tuple[str, ...]:
    """Include only selected images plus structural glTF texture dependencies."""
    selected = set(_normalize_pbr_maps(selected_maps))
    needed = set(selected)
    if "opacity" in selected:
        needed.add("base_color")
    if "roughness" in selected or "metallic" in selected:
        needed.add("metallic_roughness")
    return tuple(key for key in PBR_MAP_KEYS + ("metallic_roughness",) if key in needed)

def _glb_parts(data: bytes):
    raw = bytes(data)
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise ValueError("The exported data is not a valid GLB container.")
    version, total_length = struct.unpack_from("<II", raw, 4)
    if version != 2 or total_length > len(raw):
        raise ValueError("Unsupported or truncated GLB container.")
    document = None
    binary = None
    other_chunks = []
    offset = 12
    while offset + 8 <= total_length:
        length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        payload = raw[offset:offset + length]
        if len(payload) != length:
            raise ValueError("GLB chunk length is invalid.")
        offset += length
        if chunk_type == 0x4E4F534A:
            document = json.loads(payload.rstrip(b" \t\r\n\x00").decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = bytearray(payload)
        else:
            other_chunks.append((chunk_type, payload))
    if not isinstance(document, dict):
        raise ValueError("GLB JSON chunk is missing.")
    if binary is None:
        binary = bytearray()
    return document, binary, other_chunks

def _pack_glb_parts(document: dict, binary: bytearray, other_chunks=()) -> bytes:
    encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    chunks = [(0x4E4F534A, encoded)]
    if binary:
        chunks.append((0x004E4942, bytes(binary) + b"\x00" * ((4 - len(binary) % 4) % 4)))
    chunks.extend(other_chunks)
    body = bytearray()
    for chunk_type, payload in chunks:
        body.extend(struct.pack("<II", len(payload), chunk_type))
        body.extend(payload)
    total = 12 + len(body)
    return b"glTF" + struct.pack("<II", 2, total) + bytes(body)

def _png_bytes(image) -> bytes:
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    stream = BytesIO()
    image.convert("RGBA").save(stream, format="PNG", optimize=True)
    payload = stream.getvalue()
    if not payload:
        raise ValueError("A PBR texture image could not be encoded as PNG.")
    return payload

def _add_glb_image(document: dict, binary: bytearray, name: str, image) -> int:
    payload = _png_bytes(image)
    binary.extend(b"\x00" * ((4 - len(binary) % 4) % 4))
    byte_offset = len(binary)
    binary.extend(payload)
    buffers = document.setdefault("buffers", [{"byteLength": len(binary)}])
    if not buffers:
        buffers.append({"byteLength": len(binary)})
    if buffers[0].get("uri"):
        raise ValueError("PBR map embedding requires a GLB binary buffer.")
    buffers[0]["byteLength"] = len(binary)
    views = document.setdefault("bufferViews", [])
    view_index = len(views)
    views.append({"buffer": 0, "byteOffset": byte_offset, "byteLength": len(payload)})
    images = document.setdefault("images", [])
    image_index = len(images)
    images.append({"name": f"Polyfitty_{name}", "bufferView": view_index, "mimeType": "image/png"})
    textures = document.setdefault("textures", [])
    texture_index = len(textures)
    textures.append({"name": f"Polyfitty_{name}", "source": image_index})
    return texture_index

def _add_gltf_image(document: dict, name: str, image) -> int:
    payload = base64.b64encode(_png_bytes(image)).decode("ascii")
    images = document.setdefault("images", [])
    image_index = len(images)
    images.append({
        "name": f"Polyfitty_{name}",
        "uri": f"data:image/png;base64,{payload}",
    })
    textures = document.setdefault("textures", [])
    texture_index = len(textures)
    textures.append({"name": f"Polyfitty_{name}", "source": image_index})
    return texture_index

def _apply_pbr_map_connections(document: dict, texture_indices: dict[str, int], selected_maps, baked_maps=None) -> dict:
    selected = _normalize_pbr_maps(selected_maps)
    materials = document.setdefault("materials", [])
    if not materials:
        materials.append({"name": "Polyfitty_Baked_PBR", "pbrMetallicRoughness": {}})
    for material in materials:
        pbr = material.setdefault("pbrMetallicRoughness", {})
        pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        has_base_slot = "base_color" in selected or "opacity" in selected
        if has_base_slot and "base_color" in texture_indices:
            pbr["baseColorTexture"] = {"index": int(texture_indices["base_color"])}
        else:
            pbr.pop("baseColorTexture", None)

        if "normal" in selected and "normal" in texture_indices:
            material["normalTexture"] = {"index": int(texture_indices["normal"])}
        else:
            material.pop("normalTexture", None)

        if "orm" in selected and "orm" in texture_indices:
            packed_index = int(texture_indices["orm"])
            pbr["metallicRoughnessTexture"] = {"index": packed_index}
            material["occlusionTexture"] = {"index": packed_index}
            pbr["metallicFactor"] = 1.0
            pbr["roughnessFactor"] = 1.0
        elif ("roughness" in selected or "metallic" in selected) and "metallic_roughness" in texture_indices:
            pbr["metallicRoughnessTexture"] = {
                "index": int(texture_indices["metallic_roughness"])
            }
            if "metallic" in selected:
                pbr["metallicFactor"] = 1.0
            if "roughness" in selected:
                pbr["roughnessFactor"] = 1.0
            if "ao" in selected and "ao" in texture_indices:
                material["occlusionTexture"] = {"index": int(texture_indices["ao"])}
            else:
                material.pop("occlusionTexture", None)
        else:
            pbr.pop("metallicRoughnessTexture", None)
            if "ao" in selected and "ao" in texture_indices:
                material["occlusionTexture"] = {"index": int(texture_indices["ao"])}
            else:
                material.pop("occlusionTexture", None)

        if "emissive" in selected and "emissive" in texture_indices:
            material["emissiveTexture"] = {"index": int(texture_indices["emissive"])}
            material["emissiveFactor"] = [1.0, 1.0, 1.0]
        else:
            material.pop("emissiveTexture", None)
            material["emissiveFactor"] = [0.0, 0.0, 0.0]

        if "opacity" in selected:
            material["alphaMode"] = "BLEND"
            material.pop("alphaCutoff", None)
        else:
            material.pop("alphaMode", None)
            material.pop("alphaCutoff", None)

    map_texture_indices = {
        key: int(texture_indices[key])
        for key in selected if key in texture_indices
    }
    if "opacity" in selected and "base_color" in texture_indices:
        map_texture_indices["base_color"] = int(texture_indices["base_color"])

    if "metallic_roughness" in texture_indices and (
        "roughness" in selected or "metallic" in selected
    ):
        map_texture_indices["metallic_roughness"] = int(texture_indices["metallic_roughness"])
    asset = document.setdefault("asset", {})
    extras = asset.setdefault("extras", {})
    previous = extras.get("polyfittyPBRMaps")
    baked = _normalize_pbr_maps(baked_maps) if baked_maps is not None else (previous.get("baked", selected) if isinstance(previous, dict) else selected)
    extras["polyfittyPBRMaps"] = {
        "baked": list(_normalize_pbr_maps(baked)),
        "selected": list(selected),
        "textures": map_texture_indices,
    }
    return texture_indices

def patch_glb_pbr_maps(data: bytes, images_by_key: dict, selected_maps, *, baked_maps=None) -> bytes:
    """Embed individual PBR maps and connect only the selected glTF material slots."""
    document, binary, other_chunks = _glb_parts(data)
    texture_indices = {}
    image_keys = (PBR_MAP_KEYS + ("metallic_roughness",)) if baked_maps is None else _pbr_texture_keys_for_selection(baked_maps)
    for key in image_keys:
        image = images_by_key.get(key)
        if image is not None:
            texture_indices[key] = _add_glb_image(document, binary, key, image)
    _apply_pbr_map_connections(document, texture_indices, selected_maps, baked_maps=baked_maps)
    return _pack_glb_parts(document, binary, other_chunks)

def patch_gltf_pbr_maps(data: bytes, images_by_key: dict, selected_maps, *, baked_maps=None) -> bytes:
    """Add PBR map data-URI images and connect selected glTF material slots."""
    document = json.loads(bytes(data).decode("utf-8-sig"))
    texture_indices = {}
    image_keys = (PBR_MAP_KEYS + ("metallic_roughness",)) if baked_maps is None else _pbr_texture_keys_for_selection(baked_maps)
    for key in image_keys:
        image = images_by_key.get(key)
        if image is not None:
            texture_indices[key] = _add_gltf_image(document, key, image)
    _apply_pbr_map_connections(document, texture_indices, selected_maps, baked_maps=baked_maps)
    return json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")

def _read_glb_image(document: dict, binary: bytes, texture_index: int):
    from PIL import Image

    textures = document.get("textures") or []
    images = document.get("images") or []
    if texture_index < 0 or texture_index >= len(textures):
        return None
    image_index = int(textures[texture_index].get("source", -1))
    if image_index < 0 or image_index >= len(images):
        return None
    image_info = images[image_index]
    view_index = image_info.get("bufferView")
    if view_index is None:
        uri = str(image_info.get("uri", ""))
        if uri.startswith("data:image/") and "," in uri:
            return Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGBA")
        return None
    views = document.get("bufferViews") or []
    view_index = int(view_index)
    if view_index < 0 or view_index >= len(views):
        return None
    view = views[view_index]
    start = int(view.get("byteOffset", 0))
    size = int(view.get("byteLength", 0))
    if start < 0 or size <= 0 or start + size > len(binary):
        return None
    return Image.open(BytesIO(binary[start:start + size])).convert("RGBA").copy()

def extract_glb_pbr_map_images(source_glb: str | Path) -> dict:
    """Extract images recorded in Polyfitty's baked-map metadata from a GLB."""
    path = Path(source_glb)
    document, binary, _other = _glb_parts(path.read_bytes())
    extras = document.get("asset", {}).get("extras", {})
    metadata = extras.get("polyfittyPBRMaps") or {}
    textures = metadata.get("textures") or {}
    result = {}
    for key, texture_index in textures.items():
        try:
            image = _read_glb_image(document, binary, int(texture_index))
            if image is not None:
                result[str(key)] = image
        except Exception:
            continue
    if "base_color" not in result:
        materials = document.get("materials") or []
        if materials:
            texture_info = materials[0].get("pbrMetallicRoughness", {}).get("baseColorTexture") or {}
            try:
                image = _read_glb_image(document, binary, int(texture_info.get("index", -1)))
                if image is not None:
                    result["base_color"] = image
            except Exception:
                pass
    return result

def get_glb_baked_pbr_maps(source_glb: str | Path) -> tuple[str, ...]:
    """Return the actual PBR map selection saved in a baked GLB master."""
    try:
        document, _binary, _other = _glb_parts(Path(source_glb).read_bytes())
        metadata = document.get("asset", {}).get("extras", {}).get("polyfittyPBRMaps") or {}
        baked = metadata.get("baked")
        if baked is not None:
            return _normalize_pbr_maps(baked)
        materials = document.get("materials") or []
        if materials and materials[0].get("pbrMetallicRoughness", {}).get("baseColorTexture"):
            return ("base_color",)
    except Exception:
        pass
    return ()

def studio_to_output_matrix(up_axis: str = "y") -> np.ndarray:
    """Return a proper rotation from Studio Z-up to the requested output up-axis."""
    axis = str(up_axis or "y").lower()
    matrices = {
        "z": np.eye(4, dtype=np.float64),

        "y": np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        ),

        "x": np.array(
            [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        ),
    }
    if axis not in matrices:
        raise ValueError("Output up-axis must be X, Y, or Z.")
    return matrices[axis]

def default_export_path(source_glb: str | Path, suffix: str) -> Path:
    """Choose a distinct sibling path so the canonical baked GLB is never overwritten."""
    source = Path(source_glb)
    extension = str(suffix or ".glb").lower()
    if not extension.startswith("."):
        extension = "." + extension
    allowed = {".glb", ".gltf", ".fbx", ".obj", ".ply", ".stl", ".3mf", ".off"}
    if extension not in allowed:
        raise ValueError(f"Unsupported export format: {extension}")
    return source.with_name(f"{source.stem}_export{extension}")

def _patch_gltf_document(document: dict, specular: float) -> dict:
    materials = document.setdefault("materials", [])
    for material in materials:
        extensions = material.setdefault("extensions", {})
        spec = extensions.setdefault(SPECULAR_EXTENSION, {})
        spec["specularFactor"] = float(np.clip(specular, 0.0, 1.0))
        spec.setdefault("specularColorFactor", [1.0, 1.0, 1.0])
    used = document.setdefault("extensionsUsed", [])
    if SPECULAR_EXTENSION not in used:
        used.append(SPECULAR_EXTENSION)
    return document

def patch_glb_specular(data: bytes, specular: float) -> bytes:
    """Patch KHR_materials_specular into the JSON chunk of a GLB 2.0 file."""
    raw = bytes(data)
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise ValueError("The exported data is not a valid GLB container.")
    version, total_length = struct.unpack_from("<II", raw, 4)
    if version != 2 or total_length > len(raw):
        raise ValueError("Unsupported or truncated GLB container.")
    chunks = []
    offset = 12
    json_found = False
    while offset + 8 <= total_length:
        length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        payload = raw[offset:offset + length]
        if len(payload) != length:
            raise ValueError("GLB chunk length is invalid.")
        offset += length
        if chunk_type == 0x4E4F534A:
            document = json.loads(payload.rstrip(b" \t\r\n\x00").decode("utf-8"))
            document = _patch_gltf_document(document, specular)
            encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            encoded += b" " * ((4 - len(encoded) % 4) % 4)
            payload = encoded
            json_found = True
        chunks.append((chunk_type, payload))
    if not json_found:
        raise ValueError("GLB JSON chunk is missing.")
    body = bytearray()
    for chunk_type, payload in chunks:
        body.extend(struct.pack("<II", len(payload), chunk_type))
        body.extend(payload)
    total = 12 + len(body)
    return b"glTF" + struct.pack("<II", 2, total) + bytes(body)

def patch_glb_diffuse_preview(data: bytes, image) -> bytes:
    """Embed an unmodified, image-only Diffuse preview atlas in GLB extras.

    The standard glTF baseColorTexture is baked from the raw Base Color image
    with a neutral white material factor. PolyFit's unlit Diffuse (Albedo) view
    can load this image-only preview without applying material multiplication.
    """
    raw = bytes(data)
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise ValueError("The exported data is not a valid GLB container.")
    version, total_length = struct.unpack_from("<II", raw, 4)
    if version != 2 or total_length > len(raw):
        raise ValueError("Unsupported or truncated GLB container.")

    png_buffer = BytesIO()
    image.convert("RGBA").save(png_buffer, format="PNG", optimize=True)
    png_bytes = png_buffer.getvalue()
    if not png_bytes:
        raise ValueError("The Diffuse preview image could not be encoded as PNG.")

    chunks: list[tuple[int, bytes]] = []
    document = None
    bin_payload = None
    offset = 12
    while offset + 8 <= total_length:
        length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        payload = raw[offset:offset + length]
        if len(payload) != length:
            raise ValueError("GLB chunk length is invalid.")
        offset += length
        if chunk_type == 0x4E4F534A:
            document = json.loads(payload.rstrip(b" \t\r\n\x00").decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_payload = bytearray(payload)
        else:
            chunks.append((chunk_type, payload))

    if document is None:
        raise ValueError("GLB JSON chunk is missing.")
    if bin_payload is None:
        raise ValueError("GLB binary buffer chunk is missing.")

    buffers = document.setdefault("buffers", [{"byteLength": len(bin_payload)}])
    if not buffers:
        buffers.append({"byteLength": len(bin_payload)})
    if buffers[0].get("uri"):
        raise ValueError("A GLB diffuse preview requires an embedded binary buffer.")

    bin_payload.extend(b"\x00" * ((4 - (len(bin_payload) % 4)) % 4))
    byte_offset = len(bin_payload)
    bin_payload.extend(png_bytes)
    buffers[0]["byteLength"] = byte_offset + len(png_bytes)

    buffer_views = document.setdefault("bufferViews", [])
    buffer_view_index = len(buffer_views)
    buffer_views.append({
        "buffer": 0,
        "byteOffset": int(byte_offset),
        "byteLength": int(len(png_bytes)),
    })

    images = document.setdefault("images", [])
    image_index = len(images)
    images.append({
        "name": "Polyfitty_Unmodified_Diffuse_Albedo_Preview",
        "bufferView": buffer_view_index,
        "mimeType": "image/png",
    })

    textures = document.setdefault("textures", [])
    texture_index = len(textures)
    textures.append({
        "name": "Polyfitty_Unmodified_Diffuse_Albedo_Preview",
        "source": image_index,
    })
    asset = document.setdefault("asset", {})
    extras = asset.setdefault("extras", {})
    extras["polyfittyDiffusePreviewTexture"] = int(texture_index)

    encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - (len(encoded) % 4)) % 4)
    chunks.insert(0, (0x4E4F534A, encoded))
    chunks.append((0x004E4942, bytes(bin_payload) + b"\x00" * ((4 - (len(bin_payload) % 4)) % 4)))

    body = bytearray()
    for chunk_type, payload in chunks:
        body.extend(struct.pack("<II", len(payload), chunk_type))
        body.extend(payload)
    new_total = 12 + len(body)
    return b"glTF" + struct.pack("<II", 2, new_total) + bytes(body)

def _report(callback: Callable | None, fraction: float | None, label: str) -> None:
    if callback is None:
        return
    try:
        callback(fraction, label)
    except Exception:
        pass

def _set_material(mesh, metallic: float, roughness: float) -> None:
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    if material is None:
        return
    if hasattr(material, "metallicFactor"):
        material.metallicFactor = float(np.clip(metallic, 0.0, 1.0))
    if hasattr(material, "roughnessFactor"):
        material.roughnessFactor = float(np.clip(roughness, 0.0, 1.0))

def _fbx_escape(value) -> str:
    """Escape a string for an FBX ASCII quoted field."""
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")

def _fbx_number(value) -> str:
    number = float(value)
    if not np.isfinite(number):
        number = 0.0
    return format(number, ".9g")

def _fbx_values(values) -> str:
    return ",".join(_fbx_number(value) for value in np.asarray(values).reshape(-1))

def _fbx_material_factor(material) -> np.ndarray:
    """Read a normalized RGBA base/diffuse factor from a trimesh material."""
    for name in ("baseColorFactor", "diffuse", "main_color"):
        value = getattr(material, name, None) if material is not None else None
        if value is None:
            continue
        try:
            rgba = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if rgba.size < 3 or not np.isfinite(rgba[:3]).all():
            continue
        if rgba.size == 3:
            rgba = np.r_[rgba, 255.0 if float(np.max(rgba)) > 1.5 else 1.0]
        rgba = rgba[:4]
        if float(np.nanmax(rgba)) > 1.5:
            rgba = rgba / 255.0
        return np.clip(np.nan_to_num(rgba, nan=1.0), 0.0, 1.0)
    return np.ones(4, dtype=np.float64)

def _fbx_export_texture_image(image, alpha_mode: str, alpha_cutoff: float):
    """Prepare the Base Color image with the selected opacity semantics."""
    if image is None:
        return None
    from PIL import Image
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    mode = str(alpha_mode or "ALPHA_CLIP").upper()
    if mode == "OPAQUE":
        rgba[..., 3] = 255
    elif mode in {"ALPHA_CLIP", "ALPHA_HASHED"}:
        threshold = int(round(float(np.clip(alpha_cutoff, 0.0, 1.0)) * 255.0))
        rgba[..., 3] = np.where(rgba[..., 3] >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")

def export_fbx_mesh(
    mesh,
    output_path: str | Path,
    *,
    base_color_image=None,
    alpha_mode: str = "ALPHA_CLIP",
    alpha_cutoff: float = 0.5,
    specular: float = 0.25,
    roughness: float = 1.0,
    metallic: float = 0.0,
    output_up_axis: str = "y",
    allow_material_texture: bool = True,
    progress_callback: Callable | None = None,
) -> dict:
    """Write an FBX 7.4 ASCII mesh with a copied Base Color texture sidecar.

    FBX is written directly so FBX export does not depend on an external SDK,
    Blender, or an Assimp executable. The material uses the broadly compatible
    FBX Phong model; Base Color, UVs, normals, vertex colors and opacity are
    retained, while metallic/roughness are approximated by specular/shininess.
    """
    from PIL import Image

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    alpha_mode, alpha_cutoff, _shadow = _normalize_render_modes(
        alpha_mode, alpha_cutoff, "OPAQUE"
    )
    vertices = np.asarray(getattr(mesh, "vertices", ()), dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(getattr(mesh, "faces", ()), dtype=np.int64).reshape(-1, 3)
    if len(vertices) < 3 or len(faces) == 0 or not np.isfinite(vertices).all():
        raise ValueError("FBX export requires a finite triangle mesh with at least one face.")
    valid_faces = (faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1)
    faces = faces[valid_faces]
    if len(faces) == 0:
        raise ValueError("FBX export found no valid triangle faces.")

    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    uv_values = getattr(visual, "uv", None)
    uvs = None
    if uv_values is not None:
        candidate = np.asarray(uv_values, dtype=np.float64)
        if candidate.shape == (len(vertices), 2) and np.isfinite(candidate).all():
            uvs = candidate
    if base_color_image is None and allow_material_texture and material is not None:
        base_color_image = getattr(material, "baseColorTexture", None)
        if base_color_image is None:
            base_color_image = getattr(material, "image", None)
    texture_image = _fbx_export_texture_image(base_color_image, alpha_mode, alpha_cutoff)
    if texture_image is not None and uvs is None:

        texture_image = None

    factor = _fbx_material_factor(material)
    if alpha_mode == "OPAQUE":
        factor[3] = 1.0
    opacity = float(np.clip(factor[3], 0.0, 1.0))

    vertex_colors = None
    if texture_image is None and visual is not None:
        color_values = getattr(visual, "vertex_colors", None)
        if color_values is not None:
            color_values = np.asarray(color_values, dtype=np.float64)
            if color_values.ndim == 2 and len(color_values) == len(vertices) and color_values.shape[1] >= 3:
                if float(np.nanmax(color_values)) > 1.5:
                    color_values = color_values / 255.0
                if color_values.shape[1] == 3:
                    color_values = np.column_stack((color_values, np.ones(len(color_values))))
                vertex_colors = np.clip(np.nan_to_num(color_values[:, :4], nan=1.0), 0.0, 1.0)
                if alpha_mode == "OPAQUE":
                    vertex_colors[:, 3] = 1.0
                elif alpha_mode in {"ALPHA_CLIP", "ALPHA_HASHED"}:
                    vertex_colors[:, 3] = (vertex_colors[:, 3] >= alpha_cutoff).astype(np.float64)

    texture_path = None
    if texture_image is not None:
        texture_path = destination.with_name(f"{destination.stem}_BaseColor.png")
        texture_image.save(texture_path, format="PNG", optimize=True)

    try:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        if normals.shape != vertices.shape or not np.isfinite(normals).all():
            raise ValueError("invalid vertex normals")
    except Exception:
        normals = np.zeros_like(vertices)
        for face in faces:
            a, b, c = vertices[face]
            n = np.cross(b - a, c - a)
            length = float(np.linalg.norm(n))
            if length > 1e-15:
                n /= length
            normals[face] += n
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals / np.maximum(lengths[:, None], 1e-15)

    polygon_vertices = faces.reshape(-1)
    polygon_normals = normals[polygon_vertices]
    polygon_uvs = uvs[polygon_vertices] if texture_path is not None and uvs is not None else None
    polygon_colors = vertex_colors[polygon_vertices] if vertex_colors is not None else None
    polygon_indices = []
    for face in faces:
        polygon_indices.extend((int(face[0]), int(face[1]), -int(face[2]) - 1))

    ids = {"geometry": 1000001, "model": 1000002, "material": 1000003}
    if texture_path is not None:
        ids.update(texture=1000004, video=1000005)
    up_axis = str(output_up_axis or "y").strip().lower()
    up_axis_index = {"x": 0, "y": 1, "z": 2}.get(up_axis, 1)
    coord_axis_index = (up_axis_index + 2) % 3
    front_axis_index = (up_axis_index + 1) % 3
    shininess = float(np.clip((1.0 - float(roughness)) * 100.0 + float(metallic) * 20.0, 0.0, 100.0))
    specular_amount = float(np.clip(float(specular) + float(metallic) * 0.5, 0.0, 1.0))
    diffuse_rgb = factor[:3]

    lines = [
        "; FBX 7.4.0 project file",
        "; Generated by Polyfitty Studio",
        "FBXHeaderExtension:  {",
        "    FBXHeaderVersion: 1003",
        "    FBXVersion: 7400",
        "    EncryptionType: 0",
        "}",
        "GlobalSettings:  {",
        "    Version: 1000",
        "    Properties70:  {",
        f'        P: "UpAxis", "int", "Integer", "",{up_axis_index}',
        '        P: "UpAxisSign", "int", "Integer", "",1',
        f'        P: "FrontAxis", "int", "Integer", "",{front_axis_index}',
        '        P: "FrontAxisSign", "int", "Integer", "",1',
        f'        P: "CoordAxis", "int", "Integer", "",{coord_axis_index}',
        '        P: "CoordAxisSign", "int", "Integer", "",1',
        '        P: "OriginalUpAxis", "int", "Integer", "",-1',
        '        P: "UnitScaleFactor", "double", "Number", "",1',
        '        P: "OriginalUnitScaleFactor", "double", "Number", "",1',
        "    }",
        "}",
        "Documents:  {",
        "    Count: 1",
        '    Document: 1000000, "Scene", "Scene" {',
        '        Properties70:  { P: "SourceObject", "object", "", "" }',
        "        RootNode: 0",
        "    }",
        "}",
        "References:  {",
        "}",
        "Definitions:  {",
        "    Version: 100",
        f"    Count: {3 + (2 if texture_path is not None else 0)}",
        '    ObjectType: "Geometry" { Count: 1 }',
        '    ObjectType: "Model" { Count: 1 }',
        '    ObjectType: "Material" { Count: 1 }',
    ]
    if texture_path is not None:
        lines.extend([
            '    ObjectType: "Texture" { Count: 1 }',
            '    ObjectType: "Video" { Count: 1 }',
        ])
    lines.extend([
        "}",
        "Objects:  {",
        f'    Geometry: {ids["geometry"]}, "Geometry::PolyfittyMesh", "Mesh" {{',
        "        GeometryVersion: 124",
        f"        Vertices: *{vertices.size} {{ a: {_fbx_values(vertices)} }}",
        f"        PolygonVertexIndex: *{len(polygon_indices)} {{ a: {_fbx_values(polygon_indices)} }}",
        "        LayerElementNormal: 0 {",
        '            Version: 101',
        '            Name: ""',
        '            MappingInformationType: "ByPolygonVertex"',
        '            ReferenceInformationType: "Direct"',
        f"            Normals: *{polygon_normals.size} {{ a: {_fbx_values(polygon_normals)} }}",
        "        }",
    ])
    if polygon_uvs is not None:
        lines.extend([
            "        LayerElementUV: 0 {",
            '            Version: 101',
            '            Name: "UVChannel_1"',
            '            MappingInformationType: "ByPolygonVertex"',
            '            ReferenceInformationType: "Direct"',
            f"            UV: *{polygon_uvs.size} {{ a: {_fbx_values(polygon_uvs)} }}",
            "        }",
        ])
    if polygon_colors is not None:
        lines.extend([
            "        LayerElementColor: 0 {",
            '            Version: 101',
            '            Name: "PolyfittyVertexColor"',
            '            MappingInformationType: "ByPolygonVertex"',
            '            ReferenceInformationType: "Direct"',
            f"            Colors: *{polygon_colors.size} {{ a: {_fbx_values(polygon_colors)} }}",
            "        }",
        ])
    lines.extend([
        "        LayerElementMaterial: 0 {",
        '            Version: 101',
        '            Name: ""',
        '            MappingInformationType: "AllSame"',
        '            ReferenceInformationType: "IndexToDirect"',
        "            Materials: *1 { a: 0 }",
        "        }",
        "        Layer: 0 {",
        "            Version: 100",
        '            LayerElement: { Type: "LayerElementNormal" TypedIndex: 0 }',
        '            LayerElement: { Type: "LayerElementMaterial" TypedIndex: 0 }',
    ])
    if polygon_uvs is not None:
        lines.append('            LayerElement: { Type: "LayerElementUV" TypedIndex: 0 }')
    if polygon_colors is not None:
        lines.append('            LayerElement: { Type: "LayerElementColor" TypedIndex: 0 }')
    lines.extend([
        "        }",
        "    }",
        f'    Model: {ids["model"]}, "Model::PolyfittyMesh", "Mesh" {{',
        "        Version: 232",
        '        Properties70:  {',
        '            P: "InheritType", "enum", "", "",1',
        '            P: "Lcl Translation", "Lcl Translation", "A", ",,",0,0,0',
        '            P: "Lcl Rotation", "Lcl Rotation", "A", ",,",0,0,0',
        '            P: "Lcl Scaling", "Lcl Scaling", "A", ",,",1,1,1',
        '            P: "Visibility", "Visibility", "", "A",1',
        "        }",
        "        Shading: T",
        '        Culling: "CullingOff"',
        "    }",
        f'    Material: {ids["material"]}, "Material::PolyfittyMaterial", "" {{',
        "        Version: 102",
        '        ShadingModel: "phong"',
        "        MultiLayer: 0",
        '        Properties70:  {',
        f'            P: "DiffuseColor", "Color", "", "A",{_fbx_values(diffuse_rgb)}',
        '            P: "DiffuseFactor", "double", "Number", "A",1',
        f'            P: "SpecularColor", "Color", "", "A",{specular_amount:.9g},{specular_amount:.9g},{specular_amount:.9g}',
        f'            P: "SpecularFactor", "double", "Number", "A",{specular_amount:.9g}',
        f'            P: "Shininess", "double", "Number", "A",{shininess:.9g}',
        f'            P: "Opacity", "double", "Number", "A",{opacity:.9g}',
        f'            P: "TransparencyFactor", "double", "Number", "A",{1.0 - opacity:.9g}',
        '            P: "TransparentColor", "Color", "", "A",1,1,1',
        "        }",
        "    }",
    ])
    if texture_path is not None:
        abs_path = str(texture_path.resolve())
        rel_path = texture_path.name
        alpha_source = "Black" if texture_image.getchannel("A").getextrema() != (255, 255) else "None"
        lines.extend([
            f'    Texture: {ids["texture"]}, "Texture::BaseColor", "" {{',
            '        Type: "TextureVideoClip"',
            "        Version: 202",
            '        TextureName: "Texture::BaseColor"',
            '        Media: "Video::BaseColor"',
            f'        FileName: "{_fbx_escape(abs_path)}"',
            f'        RelativeFilename: "{_fbx_escape(rel_path)}"',
            '        ModelUVTranslation: 0,0',
            '        ModelUVScaling: 1,1',
            f'        Texture_Alpha_Source: "{alpha_source}"',
            '        Cropping: 0,0,0,0',
            '        currentTextureBlendMode: "Modulate"',
            '        UVSet: "UVChannel_1"',
            '        UseMaterial: 1',
            '        Texture_Type: "TextureVideoClip"',
            '    }',
            f'    Video: {ids["video"]}, "Video::BaseColor", "Clip" {{',
            '        Type: "Clip"',
            f'        Properties70:  {{ P: "Path", "KString", "XRefUrl", "", "{_fbx_escape(rel_path)}" }}',
            '        UseMipMap: 0',
            f'        Filename: "{_fbx_escape(abs_path)}"',
            f'        RelativeFilename: "{_fbx_escape(rel_path)}"',
            '    }',
        ])
    lines.extend([
        "}",
        "Connections:  {",
        f'    C: "OO",{ids["geometry"]},{ids["model"]}',
        f'    C: "OO",{ids["model"]},0',
        f'    C: "OO",{ids["material"]},{ids["model"]}',
    ])
    if texture_path is not None:
        lines.extend([
            f'    C: "OP",{ids["texture"]},{ids["material"]},"DiffuseColor"',
            f'    C: "OO",{ids["video"]},{ids["texture"]}',
        ])
    lines.extend([
        "}",
        'Takes:  { Current: "" }',
    ])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if destination.stat().st_size <= 0:
        raise RuntimeError("FBX exporter produced an empty file.")
    written = [destination]
    if texture_path is not None:
        written.append(texture_path)
    _report(progress_callback, 1.0, f"FBX export complete · {destination.name}")
    return {
        "path": str(destination),
        "format": "fbx",
        "file_bytes": int(destination.stat().st_size),
        "written_files": [str(path) for path in written],
        "texture_path_mode": "COPY",
        "texture_path": str(texture_path) if texture_path is not None else None,
        "pbr_maps_exported": ["base_color"] if texture_path is not None else [],
        "alpha_mode": alpha_mode,
        "alpha_cutoff": float(alpha_cutoff),
        "render_modes_preserved": False,
        "render_mode_limitations": (
            "FBX transparency is represented through the Base Color image alpha and Phong opacity. "
            "Alpha Hashed is approximated as Alpha Clip; shadow casting mode has no portable FBX equivalent."
        ),
        "material_limitations": (
            "FBX uses a Phong-compatible material approximation. Metallic and roughness are mapped to "
            "specular strength and shininess; non-Base-Color PBR maps are not exported."
        ),
    }

def export_baked_asset(
    source_glb: str | Path,
    output_path: str | Path,
    *,
    output_up_axis: str = "y",
    specular: float = 0.25,
    roughness: float = 1.0,
    metallic: float = 0.0,
    pbr_maps=None,
    alpha_mode: str = "ALPHA_CLIP",
    alpha_cutoff: float = 0.5,
    shadow_mode: str = "OPAQUE",
    progress_callback: Callable | None = None,
) -> dict:
    """Convert the automatically baked GLB master to the selected format.

    GLB/GLTF retain UV textures and glTF PBR. FBX writes an ASCII mesh and
    copies Base Color to a relative-path sidecar. OBJ writes MTL + image sidecars.
    PLY attempts texture-to-vertex-color conversion. STL/OFF/3MF retain geometry,
    but these formats do not have portable support for full glTF PBR textures.
    """
    import trimesh

    source = Path(source_glb)
    destination = Path(output_path)
    suffix = destination.suffix.lower()
    supported = {".glb", ".gltf", ".fbx", ".obj", ".ply", ".stl", ".3mf", ".off"}
    if suffix not in supported:
        raise ValueError(f"Unsupported export format: {suffix or '(no extension)'}")
    if not source.is_file():
        raise FileNotFoundError(f"Baked GLB master was not found: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError(
            "The baked GLB master is read-only. Choose a separate export destination "
            "so later exports always start from the unchanged master asset."
        )

    alpha_mode, alpha_cutoff, shadow_mode = _normalize_render_modes(
        alpha_mode, alpha_cutoff, shadow_mode
    )
    baked_pbr_maps = get_glb_baked_pbr_maps(source)
    requested_pbr_maps = _normalize_pbr_maps(
        baked_pbr_maps if pbr_maps is None else pbr_maps
    )
    format_supported_maps = set(supported_pbr_maps_for_format(suffix))
    selected_pbr_maps = tuple(
        key for key in requested_pbr_maps
        if key in baked_pbr_maps and key in format_supported_maps
    )
    pbr_images = extract_glb_pbr_map_images(source)

    _report(progress_callback, 0.05, "Loading baked GLB master · preserving geometry, UVs, and texture")
    loaded_scene = trimesh.load(str(source), force="scene", process=False)
    source_frame = extract_scene_source_frame(loaded_scene)
    loaded = trimesh.load(str(source), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError("The baked GLB does not contain a triangle mesh.")
    world_mesh = loaded.copy()
    mesh = world_mesh.copy()

    _report(progress_callback, 0.20, f"Applying {str(output_up_axis).upper()}-up coordinate conversion")
    matrix = studio_to_output_matrix(output_up_axis)
    gltf_output = suffix in {".glb", ".gltf"}
    output_frame = matrix @ source_frame
    if gltf_output:
        if not np.allclose(source_frame, np.eye(4), atol=1e-10):
            mesh.apply_transform(np.linalg.inv(source_frame))
    elif not np.allclose(matrix, np.eye(4)):
        mesh.apply_transform(matrix)

    _report(progress_callback, 0.30, "Applying export material · Specular / Roughness / Metallic")
    _set_material(mesh, metallic, roughness)
    if suffix in {".glb", ".gltf"} and "base_color" not in selected_pbr_maps:
        visual = getattr(mesh, "visual", None)
        material = getattr(visual, "material", None)
        if material is not None and hasattr(material, "baseColorTexture"):
            material.baseColorTexture = None
    destination.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".fbx":
        _report(progress_callback, 0.50, "Writing FBX 7.4 ASCII mesh · copying Base Color texture")
        base_color_image = pbr_images.get("base_color") if "base_color" in selected_pbr_maps else None
        fbx_info = export_fbx_mesh(
            mesh, destination, base_color_image=base_color_image,
            alpha_mode=alpha_mode, alpha_cutoff=alpha_cutoff,
            specular=specular, roughness=roughness, metallic=metallic,
            output_up_axis=output_up_axis,
            allow_material_texture=("base_color" in selected_pbr_maps),
            progress_callback=progress_callback,
        )
        written = [Path(item) for item in fbx_info.get("written_files", [])]
    elif suffix in {".glb", ".gltf"}:
        export_scene = trimesh.Scene()
        export_scene.add_geometry(
            mesh,
            node_name="Polyfitty_Source_Origin",
            geom_name="Polyfitty_Exported_Reconstruction",
            transform=output_frame,
        )
        _report(progress_callback, 0.50, f"Serializing textured {suffix[1:].upper()} asset")
        if suffix == ".glb":
            data = export_scene.export(file_type="glb")
            data = patch_glb_specular(data, specular)
            data = patch_glb_pbr_maps(data, pbr_images, selected_pbr_maps, baked_maps=selected_pbr_maps)
            data = patch_glb_material_render_modes(
                data, alpha_mode, alpha_cutoff, shadow_mode
            )
            destination.write_bytes(data)
            written = [destination]
        else:
            assets = export_scene.export(file_type="gltf")
            if not isinstance(assets, dict):
                raise RuntimeError("The glTF exporter did not return its external asset files.")
            written = []
            for name, data in assets.items():
                target_name = destination.name if str(name).lower().endswith(".gltf") else Path(name).name
                target = destination.parent / target_name
                payload = data
                if target.suffix.lower() == ".gltf":
                    document = json.loads(bytes(payload).decode("utf-8"))
                    document = _patch_gltf_document(document, specular)
                    payload = patch_gltf_pbr_maps(
                        json.dumps(document, ensure_ascii=False).encode("utf-8"),
                        pbr_images, selected_pbr_maps, baked_maps=selected_pbr_maps,
                    )
                    payload = patch_gltf_material_render_modes(
                        payload, alpha_mode, alpha_cutoff, shadow_mode
                    )
                target.write_bytes(payload if isinstance(payload, bytes) else str(payload).encode("utf-8"))
                written.append(target)
    elif suffix == ".obj":
        _report(progress_callback, 0.50, "Writing Wavefront OBJ, MTL, and texture image")
        if "base_color" not in selected_pbr_maps:
            material = getattr(getattr(mesh, "visual", None), "material", None)
            if material is not None and hasattr(material, "baseColorTexture"):
                material.baseColorTexture = None
        from trimesh.exchange.obj import export_obj
        obj_text, sidecars = export_obj(
            mesh, include_normals=True, include_color=True,
            include_texture=True, return_texture=True,
            write_texture=False, mtl_name=destination.stem + ".mtl",
        )
        destination.write_text(obj_text, encoding="utf-8")
        written = [destination]
        for name, payload in sidecars.items():
            target = destination.parent / Path(name).name
            if target.suffix.lower() == ".mtl":
                text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else str(payload)
                ns = max(0.0, min(1000.0, (1.0 - float(roughness)) * 1000.0))
                replacement = f"Ks {float(specular):.6g} {float(specular):.6g} {float(specular):.6g}\nNs {ns:.6g}"
                import re
                if re.search(r"(?m)^Ks\s+", text):
                    text = re.sub(r"(?m)^Ks\s+[^\n]*", replacement.splitlines()[0], text)
                else:
                    text += "\n" + replacement.splitlines()[0] + "\n"
                if re.search(r"(?m)^Ns\s+", text):
                    text = re.sub(r"(?m)^Ns\s+[^\n]*", replacement.splitlines()[1], text)
                else:
                    text += replacement.splitlines()[1] + "\n"
                payload = text.encode("utf-8")
            target.write_bytes(payload if isinstance(payload, bytes) else str(payload).encode("utf-8"))
            written.append(target)
    else:
        _report(progress_callback, 0.50, f"Preparing {suffix.upper()} geometry")
        if suffix == ".ply" and "base_color" in selected_pbr_maps:
            try:
                colored = mesh.visual.to_color()
                mesh.visual = colored
            except Exception:
                pass
        data = mesh.export(file_type=suffix[1:])
        if isinstance(data, str):
            destination.write_text(data, encoding="utf-8")
        else:
            destination.write_bytes(bytes(data))
        written = [destination]

    _report(progress_callback, 1.0, f"Export complete · {destination.name}")
    return {
        "path": str(destination),
        "format": suffix.lstrip("."),
        "output_up_axis": str(output_up_axis).lower(),
        "specular": float(np.clip(specular, 0.0, 1.0)),
        "roughness": float(np.clip(roughness, 0.0, 1.0)),
        "metallic": float(np.clip(metallic, 0.0, 1.0)),
        "alpha_mode": alpha_mode,
        "alpha_cutoff": float(alpha_cutoff),
        "shadow_mode": shadow_mode,
        "alpha_mode_standard_mapping": {
            "OPAQUE": "OPAQUE", "ALPHA_CLIP": "MASK",
            "ALPHA_HASHED": "MASK", "ALPHA_BLEND": "BLEND",
        }[alpha_mode],
        "shadow_mode_storage": "Polyfitty material extras (not a glTF core feature)" if suffix in {".glb", ".gltf"} else None,
        "render_modes_preserved": bool(suffix in {".glb", ".gltf"}),
        "render_mode_limitations": (
            fbx_info.get("render_mode_limitations") if suffix == ".fbx" else
            None if suffix in {".glb", ".gltf"} else
            f"{suffix.upper()} does not preserve the selectable alpha blend or shadow mode settings."
        ),
        "texture_path_mode": fbx_info.get("texture_path_mode") if suffix == ".fbx" else None,
        "pbr_maps_baked": list(baked_pbr_maps),
        "pbr_maps_requested": list(requested_pbr_maps),
        "pbr_maps_exported": (
            list(fbx_info.get("pbr_maps_exported", [])) if suffix == ".fbx" else list(selected_pbr_maps)
        ),
        "file_bytes": int(destination.stat().st_size),
        "written_files": [str(path) for path in written],
        "material_limitations": (
            "This format does not preserve full glTF PBR texture/material data. "
            "OBJ keeps diffuse texture via MTL but approximates specular/roughness and drops metallic; "
            "PLY converts the texture to vertex colors; STL/OFF/3MF are geometry-focused exports."
            if suffix in {".obj", ".ply", ".stl", ".off", ".3mf"} else (
                fbx_info.get("material_limitations") if suffix == ".fbx" else None
            )
        ),
    }
