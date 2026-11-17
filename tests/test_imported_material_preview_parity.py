from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import vtk
import pytest
from vtk.util.numpy_support import vtk_to_numpy

from app.pointio import load_preview_asset, _transform_vertex_normals, _material_pbr_factors
from app.vtk_geometry import (
    apply_studio_environment_intensity,
    configure_studio_environment,
    mesh_to_polydata,
)

def test_gltf_import_preserves_authored_vertex_normals_and_pbr_factors(tmp_path: Path):
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    authored_normals = np.asarray(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]], dtype=np.float64
    )
    material = trimesh.visual.material.PBRMaterial(
        name="Imported PBR",
        baseColorFactor=[0.8, 0.9, 1.0, 1.0],
        metallicFactor=0.63,
        roughnessFactor=0.27,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=authored_normals,
        process=False,
    )
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.zeros((len(vertices), 2), dtype=np.float64), material=material
    )
    source = tmp_path / "authored_normals.glb"
    source.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))

    asset = load_preview_asset(source)
    assert asset["kind"] == "mesh"
    part = asset["material_parts"][0]
    np.testing.assert_allclose(part["normals"], authored_normals, atol=1e-5)
    assert part["metallic"] == 0.63
    assert part["roughness"] == 0.27

def test_normal_transform_uses_inverse_transpose_for_nonuniform_scale():
    normals = np.asarray([[1.0, 1.0, 0.0]], dtype=np.float64)
    transform = np.diag([2.0, 1.0, 1.0, 1.0])
    transformed = _transform_vertex_normals(normals, transform)
    expected = np.asarray([[0.5, 1.0, 0.0]], dtype=np.float64)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(transformed, expected, atol=1e-6)

def test_pbr_material_factors_are_not_replaced_by_preview_defaults():
    material = trimesh.visual.material.PBRMaterial(
        metallicFactor=0.84, roughnessFactor=0.16
    )
    assert _material_pbr_factors(material) == {
        "metallic": 0.84,
        "roughness": 0.16,
    }

def test_vtk_polydata_keeps_valid_source_point_normals():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
    normals = np.asarray(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32
    )
    polydata = mesh_to_polydata(vertices, faces, vertex_normals=normals)
    attached = polydata.GetPointData().GetNormals()
    assert attached is not None
    np.testing.assert_allclose(vtk_to_numpy(attached), normals, atol=1e-6)

def test_environment_control_scales_real_linear_ibl_independently():
    renderer = vtk.vtkRenderer()
    texture = configure_studio_environment(renderer)
    scalars = texture.GetInput().GetPointData().GetScalars()
    current = vtk_to_numpy(scalars)
    base = current.copy()

    apply_studio_environment_intensity(texture, base, 20.0)
    np.testing.assert_allclose(current, base * 0.5, atol=1e-6)

    apply_studio_environment_intensity(texture, base, 80.0, (1.0, 0.5, 0.25))
    expected = base * np.asarray([2.0, 1.0, 0.5], dtype=np.float32)
    np.testing.assert_allclose(current, expected, atol=1e-6)

    apply_studio_environment_intensity(texture, base, 40.0, enabled=False)
    assert float(np.max(current)) == 0.0

def test_import_preview_applies_vertex_color_material_factor_in_linear_light():
    from PIL import Image
    from app.pointio import mesh_vertex_display_colors, _apply_linear_base_color_factor

    mesh = trimesh.creation.box()
    colors = np.tile(np.asarray([[128, 160, 192, 255]], dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=np.asarray([128, 200, 255, 255], dtype=np.uint8)
    )

    actual = mesh_vertex_display_colors(mesh, vertex_color_space="srgb")
    expected = _apply_linear_base_color_factor(
        colors.astype(np.float32) / 255.0,
        np.asarray([128, 200, 255, 255], dtype=np.float32) / 255.0,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)

def test_import_preview_reports_real_parse_stages(tmp_path: Path):
    from app.pointio import load_preview_asset

    mesh = trimesh.creation.box()
    source = tmp_path / "progress.glb"
    source.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
    events = []
    asset = load_preview_asset(source, progress_callback=lambda value, message: events.append((value, message)))

    assert asset["kind"] == "mesh"
    assert events
    assert events[0][0] == -1
    assert any(value == 18 for value, _ in events)
    assert events[-1][0] == 100
    determinate = [value for value, _ in events if value >= 0]
    assert determinate == sorted(determinate)

def test_imported_material_orm_images_are_packed_as_linear_glTF_channels():
    from types import SimpleNamespace
    from PIL import Image
    from app.pointio import _material_orm_preview_image

    material = SimpleNamespace(
        metallicRoughnessTexture=Image.new("RGBA", (4, 4), (12, 96, 144, 255)),
        occlusionTexture=Image.new("RGBA", (4, 4), (200, 0, 0, 255)),
    )
    orm = _material_orm_preview_image(material)
    assert orm is not None
    assert orm.mode == "RGBA"
    assert orm.getpixel((1, 1)) == (200, 96, 144, 255)

def test_vtk_property_supports_glTF_orm_and_occlusion_strength():
    prop = vtk.vtkProperty()
    assert hasattr(prop, "SetORMTexture")
    assert hasattr(prop, "SetOcclusionStrength")
    texture = vtk.vtkTexture()
    prop.SetORMTexture(texture)
    prop.SetOcclusionStrength(0.35)
    assert prop.GetTexture("materialTex") == texture
    assert prop.GetOcclusionStrength() == pytest.approx(0.35)

def test_material_preview_backend_binds_orm_only_during_pbr():
    backend = (Path(__file__).resolve().parents[1] / "app" / "vtk_backend.py").read_text(encoding="utf-8")
    assert "def _apply_orm_texture_image(self, image):" in backend
    assert "prop.SetORMTexture(texture)" in backend
    assert "texture = self._vtk_orm_texture if self._pbr_enabled else None" in backend
    assert "set_texture_srgb_mode(texture, False)" in backend

def test_gltf_source_import_keeps_orm_map_and_material_scalar_factors(tmp_path: Path):
    from PIL import Image

    mesh = trimesh.creation.box()
    uv = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGBA", (4, 4), (128, 128, 128, 255)),
        metallicRoughnessTexture=Image.new("RGBA", (4, 4), (8, 100, 180, 255)),
        occlusionTexture=Image.new("RGBA", (4, 4), (200, 0, 0, 255)),
        metallicFactor=0.6,
        roughnessFactor=0.7,
    )
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    source = tmp_path / "source_orm.glb"
    source.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))

    asset = load_preview_asset(source)
    part = asset["material_parts"][0]
    assert part["orm_texture_image"].getpixel((1, 1)) == (200, 100, 180, 255)
    assert part["occlusion_strength"] == pytest.approx(1.0)
    assert part["roughness"] == pytest.approx(0.7)
    assert part["metallic"] == pytest.approx(0.6)
