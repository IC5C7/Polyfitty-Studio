from __future__ import annotations

import ast
import io
from pathlib import Path

import numpy as np
import pytest

from app.coordinate_frame import (
    axis_conversion_matrix,
    extract_scene_source_frame,
    points_to_source_local,
    source_frame_to_studio,
    source_frame_oriented_bounds,
    source_frame_oriented_axis_edges,
    source_frame_camera_pose,
    transform_points,
)

ROOT = Path(__file__).resolve().parents[1]

def _method_source(path: str, class_name: str, method_name: str) -> str:
    source = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""

def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = (
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    return matrix

def test_source_local_reconstruction_points_round_trip_to_source_world_frame():
    source_frame = _rotation_z(37.0)
    source_frame[:3, 3] = (12.5, -4.25, 8.0)
    source_world = np.asarray(
        ((12.5, -4.25, 8.0), (13.5, -4.25, 8.0), (12.5, -3.25, 8.0)),
        dtype=np.float64,
    )

    local = points_to_source_local(source_world, source_frame)
    restored = transform_points(local, source_frame)

    assert np.allclose(restored, source_world, atol=1e-9)
    assert np.allclose(local[0], (0.0, 0.0, 0.0), atol=1e-9)

@pytest.mark.parametrize("source_up_axis", ["x", "y", "z"])
def test_studio_source_frame_conjugation_preserves_round_trip_for_each_up_axis(source_up_axis):
    source_frame = _rotation_z(25.0)
    source_frame[:3, 3] = (3.0, 7.0, -2.0)
    conversion = axis_conversion_matrix(source_up_axis)
    studio_frame = source_frame_to_studio(source_frame, source_up_axis)
    source_point = np.asarray(((1.25, -3.0, 2.5),), dtype=np.float64)

    studio_source_world = transform_points(
        transform_points(source_point, source_frame), conversion
    )
    studio_local = transform_points(source_point, conversion)
    studio_restored = transform_points(studio_local, studio_frame)

    assert np.allclose(studio_source_world, studio_restored, atol=1e-9)

def test_first_mesh_node_transform_is_selected_as_source_object_origin():
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (10.0, 20.0, 30.0)

    class FakeGeometry:
        vertices = np.zeros((3, 3), dtype=np.float64)
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)

    class FakeGraph:
        nodes_geometry = ("mesh-node",)

        def __getitem__(self, _node):
            return transform, "mesh-geometry"

    class FakeScene:
        graph = FakeGraph()
        geometry = {"mesh-geometry": FakeGeometry()}

    assert np.allclose(extract_scene_source_frame(FakeScene()), transform)

def test_transformed_scene_glb_retains_mesh_vertices_and_node_origin():
    trimesh = pytest.importorskip("trimesh")
    vertices = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    transform = _rotation_z(90.0)
    transform[:3, 3] = (4.0, 5.0, 6.0)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="SourceOrigin", geom_name="Reconstruction", transform=transform)

    payload = scene.export(file_type="glb")
    loaded_scene = trimesh.load(io.BytesIO(payload), file_type="glb", force="scene", process=False)
    loaded_mesh = trimesh.load(io.BytesIO(payload), file_type="glb", force="mesh", process=False)

    assert np.allclose(extract_scene_source_frame(loaded_scene), transform, atol=1e-6)
    expected = transform_points(vertices, transform)
    assert np.allclose(np.asarray(loaded_mesh.vertices), expected, atol=1e-6)

def test_preview_import_and_mesh_reconstruction_keep_the_source_node_origin(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from app.pipeline import _load_source_mesh_geometry
    from app.pointio import load_preview_asset

    vertices = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    source_frame = _rotation_z(90.0)
    source_frame[:3, 3] = (11.0, -7.0, 3.5)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="OriginalObject", geom_name="SourceMesh", transform=source_frame)
    source_path = tmp_path / "transformed_source.glb"
    source_path.write_bytes(scene.export(file_type="glb"))

    preview = load_preview_asset(source_path)
    reconstructed_vertices, reconstructed_faces = _load_source_mesh_geometry(
        source_path, source_up_axis="z", source_frame_transform=preview["source_frame_transform"]
    )

    assert np.allclose(preview["source_frame_transform"], source_frame, atol=1e-6)
    assert np.allclose(reconstructed_vertices, vertices, atol=1e-6)
    assert np.array_equal(reconstructed_faces, faces)

def test_baked_glb_format_conversion_preserves_the_original_node_origin(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from app.asset_export import export_baked_asset

    vertices = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    source_frame = _rotation_z(90.0)
    source_frame[:3, 3] = (-2.0, 6.0, 9.0)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="SourceOrigin", geom_name="BakedMesh", transform=source_frame)
    source_path = tmp_path / "baked_master.glb"
    output_path = tmp_path / "converted.glb"
    source_path.write_bytes(scene.export(file_type="glb"))

    export_baked_asset(source_path, output_path, output_up_axis="z", pbr_maps=[])
    exported_scene = trimesh.load(str(output_path), force="scene", process=False)
    exported_mesh = trimesh.load(str(output_path), force="mesh", process=False)

    assert np.allclose(extract_scene_source_frame(exported_scene), source_frame, atol=1e-6)
    assert np.allclose(np.asarray(exported_mesh.vertices), transform_points(vertices, source_frame), atol=1e-6)

def test_reconstruction_pipeline_localizes_mesh_geometry_and_sampled_points_before_axis_conversion():
    pipeline = (ROOT / "app/pipeline.py").read_text(encoding="utf-8")
    pointio = (ROOT / "app/pointio.py").read_text(encoding="utf-8")
    assert "vertices = points_to_source_local(world_vertices, source_frame_transform)" in pipeline
    run_start = pipeline.index("points = points_to_source_local(")
    axis_start = pipeline.index("points = convert_up_axis(points, config.get(\"source_up_axis\", \"y\"))", run_start)
    assert run_start < axis_start
    assert '"source_frame_transform": source_frame_transform.copy()' in pointio

def test_texture_sampler_and_bake_glb_use_the_same_source_local_frame():
    texture = (ROOT / "app/texture_baking.py").read_text(encoding="utf-8")
    sampler = _method_source("app/texture_baking.py", "_SourceSurfaceAppearanceSampler", "__init__")
    assert "source_frame_transform=None" in sampler
    assert "points_to_source_local(" in texture
    assert "transform=studio_source_frame" in texture

def test_main_window_applies_source_frame_to_previews_and_direct_exports():
    main = (ROOT / "app/main.py").read_text(encoding="utf-8")
    preview_method = _method_source("app/main.py", "MainWindow", "_local_studio_points_to_scene")
    export_method = _method_source("app/main.py", "ExportWorker", "_export_reconstruction_geometry")
    assert "transform_points" in preview_method
    assert "studio_to_output_matrix(self.output_up_axis) @ self.source_frame_transform" in export_method
    assert 'source_frame_transform=self._source_frame_transform_studio' in main

def test_drop_events_are_handled_app_wide_and_accept_supported_local_3d_files():
    event_filter = _method_source("app/main.py", "MainWindow", "eventFilter")
    drop_parser = _method_source("app/main.py", "MainWindow", "_dropped_input_path")
    assert "QEvent.Type.DragEnter" in event_filter
    assert "QEvent.Type.DragMove" in event_filter
    assert "QEvent.Type.Drop" in event_filter
    assert "self._load_input_source" in event_filter
    for suffix in (".obj", ".stl", ".ply", ".glb", ".gltf", ".3mf", ".pcd", ".bvg"):
        assert suffix in drop_parser

def test_viewport_exposes_camera_axis_gizmo_and_location_rotation_hud_switch():
    gizmo = _method_source("app/vtk_backend.py", "GLViewWidget", "_initialize_orientation_gizmo")
    toggle = _method_source("app/view3d.py", "SceneView", "_toggle_rotation_display_mode")
    hud = _method_source("app/view3d.py", "SceneView", "_transform_hud_text")
    assert "vtkOrientationMarkerWidget" in gizmo
    assert "SetViewport" in gizmo and "SetLayer(3)" in gizmo
    assert "quaternion" in toggle and "euler" in toggle
    assert 'f"X {location[0]:' in hud and 'f"W {rotation[3]:.4f}  X {rotation[0]:.4f}' in hud and 'f"X {rotation[0]:.2f}°' in hud

def test_dimension_bounds_follow_the_source_origin_and_rotation_instead_of_world_aabb():
    local_vertices = np.asarray(
        ((-2.0, -1.0, 0.5), (3.0, 4.0, 7.5), (0.0, 2.0, 2.0)),
        dtype=np.float64,
    )
    source_frame = _rotation_z(63.0)
    source_frame[:3, 3] = (17.25, -8.5, 4.0)
    world_vertices = transform_points(local_vertices, source_frame)

    box = source_frame_oriented_bounds(world_vertices, source_frame)

    assert box is not None
    assert np.allclose(box["local_bounds"][0], (-2.0, -1.0, 0.5), atol=1e-9)
    assert np.allclose(box["local_bounds"][1], (3.0, 4.0, 7.5), atol=1e-9)
    assert np.allclose(box["dimensions"], (5.0, 5.0, 7.0), atol=1e-9)
    assert np.allclose(box["world_corners"], transform_points(box["local_corners"], source_frame))

def test_dimension_axis_edges_stay_fixed_at_origin_bottom_and_top_in_source_frame():
    source_frame = _rotation_z(-41.0)
    source_frame[:3, 3] = (-12.0, 6.5, 21.0)
    lower = np.asarray((-1.0, -2.0, -3.0), dtype=np.float64)
    upper = np.asarray((4.0, 5.0, 6.0), dtype=np.float64)

    world_edges = source_frame_oriented_axis_edges((lower, upper), source_frame)
    local_edges = transform_points(world_edges.reshape(-1, 3), np.linalg.inv(source_frame)).reshape(3, 2, 3)
    expected_local_edges = np.asarray(
        (
            ((-1.0, -2.0, -3.0), (4.0, -2.0, -3.0)),
            ((-1.0, -2.0, -3.0), (-1.0, 5.0, -3.0)),
            ((-1.0, -2.0, -3.0), (-1.0, -2.0, 6.0)),
        ),
        dtype=np.float64,
    )

    assert local_edges.shape == (3, 2, 3)
    assert np.allclose(local_edges, expected_local_edges, atol=1e-8)
    assert np.allclose(world_edges, transform_points(expected_local_edges.reshape(-1, 3), source_frame).reshape(3, 2, 3))

def test_initial_camera_pose_follows_source_origin_rotation_and_centers_the_mesh():
    source_frame = _rotation_z(67.0)
    source_frame[:3, 3] = (120.0, -45.0, 18.0)
    local_bounds = (
        np.asarray((-2.0, -1.0, 0.5), dtype=np.float64),
        np.asarray((6.0, 3.0, 8.5), dtype=np.float64),
    )
    pose = source_frame_camera_pose(
        local_bounds, source_frame, (1440, 900), 67.0, margin=1.28
    )

    expected_focal = transform_points(
        ((local_bounds[0] + local_bounds[1]) * 0.5).reshape(1, 3), source_frame
    )[0]
    np.testing.assert_allclose(pose["focal"], expected_focal, atol=1e-9)

    direction = (pose["position"] - pose["focal"]) / pose["distance"]
    expected_rotation = source_frame[:3, :3].copy()
    left, _singular, right = np.linalg.svd(expected_rotation)
    expected_rotation = left @ right
    if np.linalg.det(expected_rotation) < 0.0:
        left[:, -1] *= -1.0
        expected_rotation = left @ right
    local_direction = np.asarray((
        np.cos(np.deg2rad(22.0)) * np.cos(np.deg2rad(35.0)),
        np.cos(np.deg2rad(22.0)) * np.sin(np.deg2rad(35.0)),
        np.sin(np.deg2rad(22.0)),
    ))
    expected_direction = expected_rotation @ local_direction
    expected_direction /= np.linalg.norm(expected_direction)
    np.testing.assert_allclose(direction, expected_direction, atol=1e-8)
    assert np.all(np.isfinite(pose["position"]))
    assert pose["distance"] > 0.0

def test_initial_camera_pose_tracks_source_translation_and_rotation():
    local_bounds = (np.asarray((-1.0, -2.0, -0.5)), np.asarray((3.0, 2.0, 4.5)))
    base = source_frame_camera_pose(local_bounds, np.eye(4), (1280, 720), 67.0)

    translated = np.eye(4)
    translated[:3, 3] = (40.0, -12.0, 7.5)
    moved = source_frame_camera_pose(local_bounds, translated, (1280, 720), 67.0)
    np.testing.assert_allclose(moved["focal"] - base["focal"], translated[:3, 3], atol=1e-9)
    np.testing.assert_allclose(moved["position"] - base["position"], translated[:3, 3], atol=1e-9)

    rotated = _rotation_z(90.0)
    turned = source_frame_camera_pose(local_bounds, rotated, (1280, 720), 67.0)
    expected_position = transform_points(base["position"].reshape(1, 3), rotated)[0]
    expected_focal = transform_points(base["focal"].reshape(1, 3), rotated)[0]
    np.testing.assert_allclose(turned["position"], expected_position, atol=1e-8)
    np.testing.assert_allclose(turned["focal"], expected_focal, atol=1e-8)

def test_initial_camera_pose_fits_all_oriented_bounds_inside_the_perspective_frustum():
    source_frame = _rotation_z(-31.0)
    source_frame[:3, 3] = (5.0, 8.0, -3.0)
    source_frame[:3, :3] = source_frame[:3, :3] @ np.diag((2.0, 0.7, 1.4))
    lower = np.asarray((-4.0, -2.0, -1.0))
    upper = np.asarray((3.0, 5.0, 9.0))
    pose = source_frame_camera_pose((lower, upper), source_frame, (1600, 900), 67.0, margin=1.28)

    relative = pose["world_corners"] - pose["focal"]
    depth = pose["distance"] - relative @ pose["direction"]
    horizontal = np.abs(relative @ pose["right"])
    vertical = np.abs(relative @ pose["camera_up"])
    horizontal_half_tangent = np.tan(np.deg2rad(67.0) * 0.5)
    vertical_half_tangent = horizontal_half_tangent / (1600.0 / 900.0)
    assert np.all(horizontal <= depth * horizontal_half_tangent + 1e-8)
    assert np.all(vertical <= depth * vertical_half_tangent + 1e-8)
