from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.texture_baking import (
    _SourceSurfaceAppearanceSampler,
    _recent_face_edge_segments,
    bake_triangle_atlas,
    bake_unwrapped_mesh_atlas,
    _build_metallic_roughness_atlas,
)

def _ten_face_fixture():

    vertices = np.asarray([
        point
        for face in range(12)
        for point in (
            (float(face * 2), 0.0, 0.0),
            (float(face * 2 + 1), 0.0, 0.0),
            (float(face * 2), 1.0, 0.0),
        )
    ], dtype=np.float32)
    return vertices.reshape(-1, 3), np.arange(36, dtype=np.int64).reshape(-1, 3)

def test_bake_onion_trail_contains_only_latest_ten_faces_and_current_face():
    vertices, faces = _ten_face_fixture()
    trail, current = _recent_face_edge_segments(vertices, faces, 12, trail_faces=10)

    assert trail.shape == (30, 2, 3)
    assert current.shape == (3, 2, 3)

    assert float(trail[:, :, 0].min()) == 4.0
    assert float(trail[:, :, 0].max()) == 23.0
    assert float(current[:, :, 0].min()) == 22.0

def test_unselected_metallic_roughness_channels_remain_white_for_factor_passthrough():
    roughness_atlas = np.full((4, 4, 4), 77, dtype=np.uint8)
    metallic_atlas = np.full((4, 4, 4), 123, dtype=np.uint8)

    roughness_only = _build_metallic_roughness_atlas(
        4, {"roughness": roughness_atlas}, ("roughness",)
    )
    assert np.all(roughness_only[..., 1] == 77)
    assert np.all(roughness_only[..., 2] == 255)

    metallic_only = _build_metallic_roughness_atlas(
        4, {"metallic": metallic_atlas}, ("metallic",)
    )
    assert np.all(metallic_only[..., 1] == 255)
    assert np.all(metallic_only[..., 2] == 123)

def test_source_sampler_returns_display_equivalent_base_color_and_independent_pbr_channels():
    sampler = object.__new__(_SourceSurfaceAppearanceSampler)
    sampler.pbr_map_keys = ("base_color", "roughness", "orm")
    base_image = object()
    roughness_image = object()
    ao_image = object()

    def sample_texture(image, uv):
        if image is base_image:
            color = (0.8, 0.6, 0.4, 1.0)
        elif image is roughness_image:
            color = (0.35, 0.35, 0.35, 1.0)
        else:
            color = (0.4, 0.4, 0.4, 1.0)
        return np.tile(np.asarray([color], dtype=np.float32), (len(uv), 1))

    sampler._sample_texture_rgba = sample_texture
    from app.pointio import _apply_linear_base_color_factor
    sampler._apply_linear_base_color_factor = _apply_linear_base_color_factor
    record = {
        "faces": np.asarray([[0, 1, 2]], dtype=np.int64),
        "uv": np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
        "pbr_images": {"base_color": base_image, "roughness": roughness_image, "ao": ao_image},
        "vertex_colors": np.ones((3, 4), dtype=np.float32),
        "roughness_factor": 0.15,
        "metallic_factor": 0.0,
        "occlusion_strength": 0.5,
        "emissive_factor": np.zeros(3, dtype=np.float32),
    }
    sampled = sampler._sample_record_faces(
        record,
        np.asarray([0], dtype=np.int64),
        np.asarray([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float64),
    )

    assert sampled.shape == (1, 12)
    np.testing.assert_allclose(sampled[0, :4], [0.8, 0.6, 0.4, 1.0], atol=1e-6)
    np.testing.assert_allclose(sampled[0, 4:8], [0.0525, 0.0525, 0.0525, 1.0], atol=1e-6)

    np.testing.assert_allclose(sampled[0, 8:], [0.7, 0.0525, 0.0, 1.0], atol=1e-6)

def test_direct_uv_bake_builds_separate_raw_diffuse_atlas_and_emits_edge_trail():
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    uv = np.asarray(((0.10, 0.10), (0.12, 0.10), (0.10, 0.12)), dtype=np.float32)
    diffuse_result: dict = {}
    edge_events: list[dict] = []

    def sample(points, _normals):
        row = np.asarray([[0.20, 0.30, 0.40, 1.0, 0.85, 0.75, 0.65, 1.0]], dtype=np.float32)
        return np.tile(row, (len(points), 1))

    pbr_atlas, _uvs, _expanded = bake_unwrapped_mesh_atlas(
        vertices,
        faces,
        uv,
        faces,
        colors=None,
        uv_mesh_vertices=vertices,
        resolution=1024,
        color_sampler=sample,
        edge_preview_callback=edge_events.append,
        diffuse_atlas_out=diffuse_result,
        chart_padding=4,
    )

    raw_atlas = diffuse_result.get("image")
    assert raw_atlas is not None and raw_atlas.shape == pbr_atlas.shape
    painted = (pbr_atlas[..., 0] != 255) | (raw_atlas[..., 0] != 255)
    assert int(painted.sum()) > 0
    np.testing.assert_array_equal(pbr_atlas[painted], raw_atlas[painted])
    assert edge_events
    assert edge_events[-1]["completed_faces"] == 1
    assert edge_events[-1]["recent_edge_segments"].shape == (3, 2, 3)

def test_per_triangle_fallback_bake_does_not_apply_legacy_multiply_channels():
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    colors = np.tile(
        np.asarray([[0.20, 0.30, 0.40, 1.0, 0.85, 0.75, 0.65, 1.0]], dtype=np.float32),
        (3, 1),
    )
    diffuse_result: dict = {}
    pbr_atlas, _uvs, _expanded = bake_triangle_atlas(
        vertices,
        faces,
        colors,
        resolution=1024,
        tile_size=32,
        diffuse_atlas_out=diffuse_result,
    )
    raw_atlas = diffuse_result.get("image")
    assert raw_atlas is not None
    painted = (pbr_atlas[..., 0] != 255) | (raw_atlas[..., 0] != 255)
    assert int(painted.sum()) > 0
    np.testing.assert_array_equal(pbr_atlas[painted], raw_atlas[painted])

def test_diffuse_preview_glb_extra_roundtrips_without_replacing_pbr_texture():
    trimesh = pytest.importorskip("trimesh")
    Image = pytest.importorskip("PIL.Image")
    from app.asset_export import patch_glb_diffuse_preview, patch_glb_specular
    from app.pointio import load_baked_glb_asset

    mesh = trimesh.creation.box()
    uv = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
    uv[:, 0] = mesh.vertices[:, 0] + 0.5
    uv[:, 1] = mesh.vertices[:, 1] + 0.5
    pbr_image = Image.new("RGBA", (8, 8), (50, 80, 120, 255))
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=pbr_image,
            baseColorFactor=np.asarray([255, 255, 255, 255], dtype=np.uint8),
        ),
    )
    glb_data = mesh.export(file_type="glb")
    if isinstance(glb_data, str):
        glb_data = glb_data.encode("latin1")
    glb_data = patch_glb_specular(bytes(glb_data), 0.25)
    raw_diffuse = Image.new("RGBA", (8, 8), (200, 150, 100, 255))
    glb_data = patch_glb_diffuse_preview(glb_data, raw_diffuse)

    with tempfile.TemporaryDirectory() as temp_dir:
        glb_path = Path(temp_dir) / "baked.glb"
        glb_path.write_bytes(glb_data)
        asset = load_baked_glb_asset(glb_path)

    assert asset["has_texture"] is True
    assert asset["texture_image"].getpixel((2, 2))[:3] == (50, 80, 120)
    assert asset["diffuse_texture_image"].getpixel((2, 2))[:3] == (200, 150, 100)
    assert asset["orm_texture_image"] is None

def test_viewport_routes_bake_edge_trail_through_real_pixel_inversion():
    view_source = (ROOT / "app" / "view3d.py").read_text(encoding="utf-8")
    assert "self._bake_recent_edge_item" in view_source
    assert "self._bake_recent_edge_item," in view_source
    assert "item.setInvertOverlay(True)" in view_source
    assert "self._bake_recent_edge_item" in view_source[view_source.index("def _invertible_line_items"):]
    clear_method = view_source[view_source.index("def clear_bake_process_visualization"):]
    assert '"_bake_recent_edge_item"' in clear_method

@pytest.mark.parametrize("projection_mode", ["direct", "cage"])
def test_end_to_end_bake_applies_base_color_factor_and_keeps_albedo_parity(projection_mode):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    Image = pytest.importorskip("PIL.Image")
    from app.pointio import load_baked_glb_asset
    from app.texture_baking import export_textured_glb

    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    source_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    source_image = Image.new("RGBA", (16, 16), (40, 120, 220, 255))
    source_mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float32),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=source_image,
            baseColorFactor=np.asarray((96, 160, 220, 255), dtype=np.uint8),
            metallicFactor=0.45,
            roughnessFactor=0.33,
        ),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.glb"
        target_path = root / "target.obj"
        output_path = root / "baked.glb"
        source_mesh.export(source_path)
        target_path.write_text(
            "o target\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            encoding="utf-8",
        )
        edge_events: list[dict] = []
        info = export_textured_glb(
            output_path,
            resolution=1024,
            source_mesh_path=source_path,
            source_up_axis="z",
            projection_mode=projection_mode,
            target_mesh_path=target_path,
            compute_device="cpu",
            edge_preview_callback=edge_events.append,
        )
        asset = load_baked_glb_asset(output_path)
        from app.asset_export import extract_glb_pbr_map_images
        map_images = extract_glb_pbr_map_images(output_path)

    assert info.get("diffuse_preview_embedded") is True
    pbr_pixels = np.asarray(asset["texture_image"], dtype=np.uint8)
    diffuse_pixels = np.asarray(asset["diffuse_texture_image"], dtype=np.uint8)
    pbr_mask = pbr_pixels[..., 0] != 255
    diffuse_mask = diffuse_pixels[..., 0] != 255
    assert pbr_mask.any() and diffuse_mask.any()
    from app.pointio import _apply_linear_base_color_factor
    expected = _apply_linear_base_color_factor(
        np.asarray([[40, 120, 220, 255]], dtype=np.float32) / 255.0,
        np.asarray([96, 160, 220, 255], dtype=np.float32) / 255.0,
    )
    expected_red = int(round(float(expected[0, 0]) * 255.0))
    assert int(np.median(diffuse_pixels[..., 0][diffuse_mask])) == expected_red
    assert int(np.median(pbr_pixels[..., 0][pbr_mask])) == expected_red
    np.testing.assert_array_equal(pbr_pixels[pbr_mask], diffuse_pixels[diffuse_mask])
    assert info["pbr_maps_requested"] == ["base_color"]
    assert info["pbr_maps_baked"] == ["base_color", "orm"]
    assert info["pbr_material_response_auto_preserved"] is True
    assert asset["orm_texture_image"] is not None
    assert asset["pbr_factors"] == {"roughness": 1.0, "metallic": 1.0}
    loaded_orm_pixels = np.asarray(asset["orm_texture_image"], dtype=np.uint8)
    loaded_roughness = loaded_orm_pixels[..., 1] < 255
    assert loaded_roughness.any()
    assert int(np.median(loaded_orm_pixels[..., 1][loaded_roughness])) in range(82, 86)
    assert int(np.median(loaded_orm_pixels[..., 2][loaded_roughness])) in range(113, 117)
    assert "orm" in map_images
    orm_pixels = np.asarray(map_images["orm"], dtype=np.uint8)
    painted_roughness = orm_pixels[..., 1] < 255
    assert painted_roughness.any()
    assert int(np.median(orm_pixels[..., 1][painted_roughness])) in range(82, 86)
    assert int(np.median(orm_pixels[..., 2][painted_roughness])) in range(113, 117)
    assert edge_events and edge_events[-1]["completed_faces"] == 1

def test_bake_edge_trail_respects_one_pixel_overlay_width_contract():
    view_source = (ROOT / "app" / "view3d.py").read_text(encoding="utf-8")
    start = view_source.index("def set_bake_recent_edges")
    end = view_source.index("def _apply_forced_infront_to_item", start)
    method = view_source[start:end]
    assert "GLLinePlotItem(" in method
    assert "width=" not in method
    assert "self._sync_invert_overlay_visibility(refresh_desired=True)" in method
    apply_visibility = view_source[view_source.index("def _apply_visibility"):]
    assert "self._bake_recent_edge_item.setVisible(self._bake_recent_edge_visible)" in apply_visibility

def test_diffuse_preview_metadata_uses_polyfitty_brand_and_reads_legacy_key():
    source = (ROOT / "app" / "asset_export.py").read_text(encoding="utf-8")
    reader = (ROOT / "app" / "pointio.py").read_text(encoding="utf-8")
    assert 'extras["polyfittyDiffusePreviewTexture"]' in source
    assert 'extras.get("polyfittyDiffusePreviewTexture")' in reader

    assert 'extras.get("polyfitDiffusePreviewTexture")' in reader

def test_glb_pbr_maps_are_embedded_connected_and_exportable_as_a_subset():
    trimesh = pytest.importorskip("trimesh")
    Image = pytest.importorskip("PIL.Image")
    from app.asset_export import (
        extract_glb_pbr_map_images,
        get_glb_baked_pbr_maps,
        patch_glb_pbr_maps,
        supported_pbr_maps_for_format,
    )
    from app.asset_export import export_baked_asset

    mesh = trimesh.creation.box()
    uv = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGBA", (8, 8), (32, 64, 96, 255)),
            baseColorFactor=np.asarray((64, 128, 192, 255), dtype=np.uint8),
        ),
    )
    glb_data = mesh.export(file_type="glb")
    if isinstance(glb_data, str):
        glb_data = glb_data.encode("latin1")
    map_keys = ("base_color", "normal", "roughness", "metallic", "ao", "height", "opacity", "emissive", "orm")
    assert supported_pbr_maps_for_format(".glb") == map_keys
    assert supported_pbr_maps_for_format(".gltf") == map_keys
    assert supported_pbr_maps_for_format(".obj") == ("base_color",)
    assert supported_pbr_maps_for_format(".ply") == ("base_color",)
    assert supported_pbr_maps_for_format(".stl") == ()
    map_colors = {
        "base_color": (90, 100, 110, 255),
        "normal": (128, 128, 255, 255),
        "roughness": (150, 150, 150, 255),
        "metallic": (40, 40, 40, 255),
        "ao": (220, 220, 220, 255),
        "height": (128, 128, 128, 255),
        "opacity": (200, 200, 200, 255),
        "emissive": (4, 8, 12, 255),
        "orm": (220, 150, 40, 255),
    }
    images = {key: Image.new("RGBA", (8, 8), color) for key, color in map_colors.items()}
    master_data = patch_glb_pbr_maps(glb_data, images, map_keys)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        master_path = root / "master.glb"
        output_path = root / "selected.glb"
        master_path.write_bytes(master_data)
        assert get_glb_baked_pbr_maps(master_path) == map_keys
        extracted = extract_glb_pbr_map_images(master_path)
        assert set(extracted) == set(map_keys)
        assert extracted["height"].getpixel((2, 2)) == map_colors["height"]

        result = export_baked_asset(
            master_path,
            output_path,
            pbr_maps=("normal", "height"),
        )
        assert result["pbr_maps_exported"] == ["normal", "height"]

        raw = output_path.read_bytes()
        offset = 12
        document = None
        while offset + 8 <= len(raw):
            chunk_length, chunk_type = __import__("struct").unpack_from("<II", raw, offset)
            offset += 8
            chunk = raw[offset:offset + chunk_length]
            offset += chunk_length
            if chunk_type == 0x4E4F534A:
                document = __import__("json").loads(chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
        assert document is not None
        assert get_glb_baked_pbr_maps(output_path) == ("normal", "height")
        subset_images = extract_glb_pbr_map_images(output_path)
        assert set(subset_images) == {"normal", "height"}
        material = document["materials"][0]
        assert "baseColorTexture" not in material.get("pbrMetallicRoughness", {})
        assert "normalTexture" in material
        extras = document["asset"]["extras"]["polyfittyPBRMaps"]
        assert extras["selected"] == ["normal", "height"]
        assert "height" in extras["textures"]
