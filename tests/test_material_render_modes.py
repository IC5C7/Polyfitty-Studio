from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from app.asset_export import (
    _normalize_render_modes,
    default_export_path,
    export_baked_asset,
    export_fbx_mesh,
    patch_glb_material_render_modes,
    patch_glb_pbr_maps,
    patch_gltf_material_render_modes,
)
from app.export_runtime import required_export_modules
from app.pointio import (
    _apply_preview_alpha_mode,
    _apply_preview_alpha_to_colors,
    _material_render_settings,
    load_preview_asset,
    mesh_vertex_display_colors,
    read_gltf_material_render_modes,
)

def _minimal_glb() -> bytes:
    scene = trimesh.Scene(trimesh.creation.box())
    return scene.export(file_type="glb")

def test_render_mode_defaults_and_validation():
    assert _normalize_render_modes("ALPHA_CLIP", 0.5, "OPAQUE") == (
        "ALPHA_CLIP", 0.5, "OPAQUE"
    )
    for alpha, cutoff, shadow in (
        ("UNKNOWN", 0.5, "OPAQUE"),
        ("ALPHA_CLIP", -0.01, "OPAQUE"),
        ("ALPHA_CLIP", 1.01, "OPAQUE"),
        ("ALPHA_CLIP", 0.5, "UNKNOWN"),
    ):
        try:
            _normalize_render_modes(alpha, cutoff, shadow)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid render mode values must be rejected.")

def test_glb_patches_standard_alpha_and_polyfitty_shadow_metadata(tmp_path: Path):
    path = tmp_path / "render_modes.glb"
    path.write_bytes(patch_glb_material_render_modes(
        _minimal_glb(), "ALPHA_HASHED", 0.37, "NONE"
    ))
    modes = read_gltf_material_render_modes(path)
    assert len(modes) == 1
    assert modes[0]["alpha_mode"] == "ALPHA_HASHED"
    assert modes[0]["standard_alpha_mode"] == "MASK"
    assert abs(modes[0]["alpha_cutoff"] - 0.37) < 1e-8
    assert modes[0]["shadow_mode"] == "NONE"

def test_gltf_json_patches_blend_as_standard_and_keeps_polyfitty_shadow_mode():
    source = json.dumps({
        "asset": {"version": "2.0"},
        "materials": [{"name": "Glass"}],
    }).encode("utf-8")
    result = patch_gltf_material_render_modes(source, "ALPHA_BLEND", 0.5, "ALPHA_HASHED")
    document = json.loads(result.decode("utf-8"))
    material = document["materials"][0]
    assert material["alphaMode"] == "BLEND"
    assert "alphaCutoff" not in material
    assert material["extras"]["polyfittyStudio"] == {
        "alphaBlendMode": "ALPHA_BLEND",
        "alphaCutoff": 0.5,
        "shadowMode": "ALPHA_HASHED",
    }

def test_preview_alpha_clip_and_hashed_use_cutoff_while_blend_preserves_alpha():
    source = Image.fromarray(np.array([[[255, 255, 255, 0], [255, 255, 255, 127],
                                        [255, 255, 255, 128], [255, 255, 255, 255]]], dtype=np.uint8))
    clip = np.asarray(_apply_preview_alpha_mode(source, "ALPHA_CLIP", 0.5))
    hashed = np.asarray(_apply_preview_alpha_mode(source, "ALPHA_HASHED", 0.5))
    blend = np.asarray(_apply_preview_alpha_mode(source, "ALPHA_BLEND", 0.5))
    opaque = np.asarray(_apply_preview_alpha_mode(source, "OPAQUE", 0.5))
    assert clip[0, :, 3].tolist() == [0, 0, 255, 255]
    assert hashed[0, :, 3].tolist() == [0, 0, 255, 255]
    assert blend[0, :, 3].tolist() == [0, 127, 128, 255]
    assert opaque[0, :, 3].tolist() == [255, 255, 255, 255]

def test_vertex_alpha_uses_same_clip_cutoff_and_preserves_blend():
    colors = np.array([[1.0, 1.0, 1.0, 0.49], [1.0, 1.0, 1.0, 0.5],
                       [1.0, 1.0, 1.0, 0.9]], dtype=np.float32)
    clipped = _apply_preview_alpha_to_colors(colors, "ALPHA_CLIP", 0.5)
    blended = _apply_preview_alpha_to_colors(colors, "ALPHA_BLEND", 0.5)
    opaque = _apply_preview_alpha_to_colors(colors, "OPAQUE", 0.5)
    assert clipped[:, 3].tolist() == [0.0, 1.0, 1.0]
    assert np.allclose(blended[:, 3], [0.49, 0.5, 0.9])
    assert opaque[:, 3].tolist() == [1.0, 1.0, 1.0]

def test_source_material_modes_prefer_polyfitty_extras_and_accept_standard_gltf_modes():
    modes = [{
        "name": "Transparent Material",
        "alpha_mode": "ALPHA_HASHED",
        "alpha_cutoff": 0.25,
        "shadow_mode": "ALPHA_CLIP",
    }]
    material = type("Material", (), {"name": "Transparent Material", "alphaMode": "BLEND"})()
    settings = _material_render_settings(material, modes)
    assert settings == {
        "alpha_mode": "ALPHA_HASHED",
        "alpha_cutoff": 0.25,
        "shadow_mode": "ALPHA_CLIP",
    }
    standard = type("Material", (), {"name": "Cutout", "alphaMode": "MASK", "alphaCutoff": 0.61})()
    settings = _material_render_settings(standard, [])
    assert settings["alpha_mode"] == "ALPHA_CLIP"
    assert abs(settings["alpha_cutoff"] - 0.61) < 1e-8
    unnamed = type("Material", (), {"name": None, "alphaMode": "MASK"})()
    shared_modes = [
        {"name": "Material 1", "alpha_mode": "ALPHA_HASHED", "alpha_cutoff": 0.31, "shadow_mode": "NONE"},
        {"name": "Material 2", "alpha_mode": "ALPHA_HASHED", "alpha_cutoff": 0.31, "shadow_mode": "NONE"},
    ]
    assert _material_render_settings(unnamed, shared_modes) == {
        "alpha_mode": "ALPHA_HASHED", "alpha_cutoff": 0.31, "shadow_mode": "NONE"
    }

def test_advertised_interchange_formats_export_and_report_material_capabilities(tmp_path: Path):
    source = tmp_path / "master.glb"
    source.write_bytes(_minimal_glb())
    for suffix in (".glb", ".gltf", ".obj", ".ply", ".stl", ".3mf", ".off"):
        output = tmp_path / f"converted{suffix}"
        info = export_baked_asset(
            source, output, alpha_mode="ALPHA_CLIP", alpha_cutoff=0.5,
            shadow_mode="OPAQUE", pbr_maps=[],
        )
        assert output.is_file(), suffix
        assert info["render_modes_preserved"] is (suffix in {".glb", ".gltf"})

def test_fbx_export_writes_ascii_mesh_and_reports_copy_path_mode(tmp_path: Path):
    assert "PIL" in required_export_modules(".fbx")
    source = tmp_path / "master.glb"
    source.write_bytes(_minimal_glb())
    output = tmp_path / "converted.fbx"
    info = export_baked_asset(
        source, output, alpha_mode="ALPHA_CLIP", alpha_cutoff=0.5,
        shadow_mode="OPAQUE", pbr_maps=[],
    )
    assert output.is_file()
    content = output.read_text(encoding="utf-8")
    assert content.startswith("; FBX 7.4.0 project file")
    assert 'ObjectType: "Geometry"' in content
    assert "PolygonVertexIndex" in content
    assert 'ShadingModel: "phong"' in content
    assert info["format"] == "fbx"
    assert info["texture_path_mode"] == "COPY"
    assert info["render_modes_preserved"] is False
    assert "shadow casting mode has no portable" in info["render_mode_limitations"]
    assert default_export_path(source, ".fbx").name == "master_export.fbx"

def test_baked_fbx_export_copies_the_selected_base_color_map(tmp_path: Path):
    image = Image.fromarray(np.asarray([
        [[220, 60, 25, 255], [25, 140, 220, 128]],
        [[35, 200, 90, 255], [230, 230, 230, 255]],
    ], dtype=np.uint8))
    mesh = trimesh.creation.box()
    uv = np.asarray([
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=np.float64)
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(name="BakedBaseColor", baseColorTexture=image),
    )
    baked = trimesh.Scene(mesh).export(file_type="glb")
    baked = patch_glb_pbr_maps(
        baked, {"base_color": image}, ["base_color"], baked_maps=["base_color"]
    )
    source = tmp_path / "baked_master.glb"
    output = tmp_path / "baked_master_export.fbx"
    source.write_bytes(baked)
    info = export_baked_asset(source, output, pbr_maps=["base_color"])
    assert output.is_file()
    sidecar = tmp_path / "baked_master_export_BaseColor.png"
    assert sidecar.is_file()
    assert "baked_master_export_BaseColor.png" in output.read_text(encoding="utf-8")
    assert info["pbr_maps_exported"] == ["base_color"]
    assert info["texture_path_mode"] == "COPY"

def test_fbx_export_copies_base_color_texture_with_relative_path(tmp_path: Path):
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.asarray([
            [0, 0], [1, 0], [1, 1], [0, 1],
            [0, 0], [1, 0], [1, 1], [0, 1],
        ], dtype=np.float64),
        material=trimesh.visual.material.PBRMaterial(
            name="Paint", baseColorTexture=Image.fromarray(np.asarray([
                [[240, 20, 30, 0], [20, 220, 30, 180]],
                [[20, 30, 240, 255], [220, 220, 220, 255]],
            ], dtype=np.uint8)), alphaMode="BLEND",
        ),
    )
    output = tmp_path / "textured.fbx"
    info = export_fbx_mesh(mesh, output, alpha_mode="ALPHA_BLEND")
    sidecar = tmp_path / "textured_BaseColor.png"
    assert output.is_file() and sidecar.is_file()
    content = output.read_text(encoding="utf-8")
    assert 'RelativeFilename: "textured_BaseColor.png"' in content
    assert 'UVSet: "UVChannel_1"' in content
    assert 'Texture_Alpha_Source: "Black"' in content
    assert info["texture_path_mode"] == "COPY"
    assert info["written_files"] == [str(output), str(sidecar)]
    exported = np.asarray(Image.open(sidecar).convert("RGBA"))
    assert exported[..., 3].tolist() == [[0, 180], [255, 255]]

def test_opaque_source_alpha_keeps_texture_rgb_and_forces_opaque_preview(tmp_path: Path):
    image = Image.fromarray(np.asarray([[[190, 80, 35, 0], [35, 150, 190, 90]]], dtype=np.uint8))
    mesh = trimesh.creation.box()
    uv = np.asarray([
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=np.float64)
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            name="Opaque", baseColorTexture=image, baseColorFactor=[1.0, 0.8, 0.5, 0.0],
            alphaMode="OPAQUE",
        ),
    )
    source = tmp_path / "opaque_source.glb"
    source.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
    asset = load_preview_asset(source)
    part = asset["material_parts"][0]
    assert part["alpha_mode"] == "OPAQUE"
    preview = np.asarray(part["texture_image"].convert("RGBA"))
    assert np.all(preview[..., 3] == 255)

    assert tuple(preview[0, 0, :3]) == (190, 72, 23)

def test_material_factor_only_is_encoded_for_unlit_diffuse_preview():
    from types import SimpleNamespace

    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    )
    mesh = SimpleNamespace(
        vertices=np.zeros((3, 3), dtype=np.float32),
        visual=SimpleNamespace(kind="texture", material=material, uv=None),
    )
    colors = mesh_vertex_display_colors(mesh)
    expected = np.asarray([137, 188, 225], dtype=np.uint8)
    np.testing.assert_allclose(np.rint(colors[0, :3] * 255.0), expected, atol=1.0)

def test_gltf_linear_vertex_colors_are_encoded_for_diffuse_display():
    from types import SimpleNamespace

    mesh = SimpleNamespace(
        vertices=np.zeros((1, 3), dtype=np.float32),
        visual=SimpleNamespace(
            kind="vertex",
            material=None,
            vertex_colors=np.asarray([[64, 128, 192, 255]], dtype=np.uint8),
        ),
    )
    colors = mesh_vertex_display_colors(mesh, vertex_color_space="linear")
    expected = np.asarray([137, 188, 225], dtype=np.uint8)
    np.testing.assert_allclose(np.rint(colors[0, :3] * 255.0), expected, atol=1.0)

def test_imported_glb_material_alpha_can_use_file_or_source_preview_override(tmp_path: Path):
    image = Image.fromarray(np.array([
        [[255, 0, 0, 0], [0, 255, 0, 127]],
        [[0, 0, 255, 128], [255, 255, 0, 255]],
    ], dtype=np.uint8))
    mesh = trimesh.creation.box()
    uv = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=np.float64)
    material = trimesh.visual.material.PBRMaterial(
        name="Cutout", baseColorTexture=image, alphaMode="BLEND"
    )
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    source = tmp_path / "source.glb"
    raw = trimesh.Scene(mesh).export(file_type="glb")
    source.write_bytes(patch_glb_material_render_modes(raw, "ALPHA_BLEND", 0.5, "NONE"))

    from_file = load_preview_asset(source)
    assert from_file["material_parts"][0]["alpha_mode"] == "ALPHA_BLEND"
    assert np.asarray(from_file["material_parts"][0]["texture_image"].getchannel("A")).tolist() == [
        [0, 127], [128, 255]
    ]

    overridden = load_preview_asset(
        source, alpha_mode_override="ALPHA_CLIP", alpha_cutoff_override=0.5
    )
    assert overridden["material_parts"][0]["alpha_mode"] == "ALPHA_CLIP"
    assert np.asarray(overridden["material_parts"][0]["texture_image"].getchannel("A")).tolist() == [
        [0, 0], [255, 255]
    ]
