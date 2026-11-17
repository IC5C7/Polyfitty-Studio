from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.coordinate_frame import (
    source_frame_oriented_axis_edges,
    source_frame_oriented_axis_edge_candidates,
    source_frame_opposite_camera_z_side_axis_edges,
    points_to_source_local,
    select_camera_visible_axis_edge,
)

ROOT = Path(__file__).resolve().parents[1]

def _scene_method(name: str):
    source = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    scene = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SceneView")
    method = next(
        node for node in scene.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    isolated = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    module = ast.Module(body=[isolated], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    if name != "_dimension_unit_profile":
        namespace["SceneView"] = SimpleNamespace(
            _dimension_unit_profile=_scene_method("_dimension_unit_profile")
        )
    exec(compile(module, f"scene_{name}_test", "exec"), namespace)
    return namespace[name]

def test_dimension_hud_uses_the_currently_dominant_visible_layer():
    current = _scene_method("_current_dimension_layer")
    source_bounds = (np.asarray((0, 0, 0)), np.asarray((10, 20, 30)))
    result_bounds = (np.asarray((-1, 2, 4)), np.asarray((14, 22, 35)))

    view = SimpleNamespace(
        _show_source=True, _show_result=True,
        _source_bounds=source_bounds, _result_bounds=result_bounds,
        _surface_mix=0.0, _selected_mesh_layer=None,
    )
    assert current(view) == "source"

    view._surface_mix = 0.75
    assert current(view) == "result"

    view._selected_mesh_layer = "source"
    assert current(view) == "source"

    view._show_source = False
    assert current(view) == "result"

    view._show_result = False
    assert current(view) is None

def test_dimension_hud_text_formats_axis_extents_and_hides_invalid_bounds():
    render_text = _scene_method("_dimension_hud_text")
    bounds = (np.asarray((-1.0, 2.0, 4.5)), np.asarray((14.25, 22.0, 35.0)))
    assert render_text(bounds) == "X 15.250 m    Y 20.000 m    Z 30.500 m"
    assert render_text(bounds, "cm") == "X 1,525.00 cm    Y 2,000.00 cm    Z 3,050.00 cm"
    assert render_text(bounds, "mm") == "X 15,250.00 mm    Y 20,000.00 mm    Z 30,500.00 mm"
    assert render_text(bounds, "km") == "X 0.015250 km    Y 0.020000 km    Z 0.030500 km"
    assert render_text(bounds, "invalid") == render_text(bounds)
    assert render_text(None) == ""
    assert render_text(((0, 0, 0), (np.nan, 1, 2))) == ""

def test_dimension_display_unit_profiles_and_runtime_setter():
    profile = _scene_method("_dimension_unit_profile")
    assert profile("mm") == (1000.0, 2)
    assert profile("cm") == (100.0, 2)
    assert profile("m") == (1.0, 3)
    assert profile("km") == (0.001, 6)
    assert profile("invalid") == profile("m")

    set_unit = _scene_method("set_dimension_display_unit")
    refreshes = []
    view = SimpleNamespace(_update_dimension_hud=lambda: refreshes.append(True))
    set_unit(view, "cm")
    assert view._dimension_display_unit == "cm"
    set_unit(view, "invalid")
    assert view._dimension_display_unit == "m"
    assert len(refreshes) == 2

def test_location_display_unit_setter_normalizes_values_and_refreshes_the_transform_hud():
    set_unit = _scene_method("set_location_display_unit")
    refreshes = []
    view = SimpleNamespace(_update_transform_hud=lambda: refreshes.append(True))
    set_unit(view, "cm")
    assert view._location_display_unit == "cm"
    set_unit(view, "invalid")
    assert view._location_display_unit == "m"
    assert len(refreshes) == 2

def test_item_info_option_row_switches_between_metric_units_and_two_column_rotation_formats():
    main = (ROOT / "app/main.py").read_text(encoding="utf-8")
    assert 'self.item_info_mode_group.setExclusive(True)' in main
    assert 'self.item_info_rotation_group.setExclusive(True)' in main
    assert 'button.setText(label)' in main
    assert '("euler", "[ELR]")' in main
    assert '("quaternion", "[QTN]")' in main
    assert 'item_info_rotation_row.addWidget(button, 2)' in main
    assert 'item_info_unit_row.addWidget(button, 1)' in main
    assert 'unit_row.setVisible(enabled and mode in {"LOC", "DIM"})' in main
    assert 'rotation_row.setVisible(enabled and mode == "ROT")' in main
    assert 'self._sync_item_info_unit_buttons()' in main
    assert 'self._sync_item_info_rotation_buttons()' in main
    assert 'self.scene_view.set_rotation_display_mode(saved_rotation_display_mode)' in main
    assert 'self.settings.setValue("viewport/rotation_display_mode", normalized)' in main
    assert 'def _on_item_info_rotation_changed' in main
    assert 'def _sync_item_info_rotation_buttons' in main

def test_rotation_format_setter_switches_euler_and_quaternion_without_a_floating_viewport_button():
    setter = _scene_method("set_rotation_display_mode")
    calls = []
    view = SimpleNamespace(_rotation_display_mode="euler", _update_transform_hud=lambda: calls.append(True))
    setter(view, "quaternion")
    assert view._rotation_display_mode == "quaternion"
    setter(view, "invalid")
    assert view._rotation_display_mode == "euler"
    assert calls == [True, True]
    source = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    assert "rotation_display_button" not in source

def test_dimension_axis_labels_convert_meter_geometry_to_selected_metric_unit():
    render_labels = _scene_method("_dimension_axis_texts")
    dimension_box = {"dimensions": np.asarray((1.25, 2.5, 3.75), dtype=np.float64)}
    assert render_labels(dimension_box) == {
        "X": "X 1.250 m", "Y": "Y 2.500 m", "Z": "Z 3.750 m"
    }
    assert render_labels(dimension_box, "mm") == {
        "X": "X 1,250.00 mm", "Y": "Y 2,500.00 mm", "Z": "Z 3,750.00 mm"
    }
    assert render_labels(dimension_box, "cm") == {
        "X": "X 125.00 cm", "Y": "Y 250.00 cm", "Z": "Z 375.00 cm"
    }
    assert render_labels(dimension_box, "km") == {
        "X": "X 0.001250 km", "Y": "Y 0.002500 km", "Z": "Z 0.003750 km"
    }
    assert render_labels(None, "mm") == {"X": "", "Y": "", "Z": ""}

def test_dimension_labels_are_projected_from_fixed_source_frame_edges_as_fixed_size_vtk_glyphs():
    source = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    assert 'self.dimension_axis_huds = {}' in source
    assert 'self._invert_renderer.AddViewProp(actor)' in source
    assert 'self._overlay_renderer.AddViewProp(actor)' not in source
    assert 'text_property.SetFontSize(16)' in source
    assert 'text_property.SetJustificationToCentered()' in source
    assert 'text_property.SetVerticalJustificationToCentered()' in source
    assert 'self._renderer.AddObserver(' in source
    assert 'vtk.vtkCommand.EndEvent, self._update_dimension_edge_label_positions' in source
    assert 'renderer.WorldToDisplay()' in source
    assert 'actor.SetDisplayPosition(x, y)' in source
    assert 'actor.SetVisibility(bool(text) and show_dimensions)' in source
    assert 'self.dimension_hud = self.dimension_axis_huds["X"]' in source
    assert 'self.dimension_hud = QtWidgets.QLabel(self)' not in source
    assert 'self._item_info_mode == "DIM"' in source
    assert 'self._update_transform_hud_position()' in source

    visibility = source.split('    def _apply_visibility(self):', 1)[1]
    assert 'self._update_dimension_hud()' in visibility

def test_dimension_labels_use_all_source_aligned_edges_and_depth_test_camera_nearest_candidates():
    source = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    frame_source = (ROOT / "app/coordinate_frame.py").read_text(encoding="utf-8")
    cache = source.split('    def _refresh_dimension_box_caches(self):', 1)[1].split(
        '    def _current_dimension_box(self):', 1
    )[0]
    project = source.split('    def _update_dimension_edge_label_positions(self, *_args):', 1)[1].split(
        '    def _position_view_reset_button(self):', 1
    )[0]
    assert 'source_frame_oriented_bounds(source_points, frame)' in cache
    assert 'source_frame_oriented_bounds(self._result_vertices, frame)' in cache
    assert 'source_frame_oriented_axis_edge_candidates(' in project
    assert 'select_camera_visible_axis_edge(' in project
    assert 'self._source_frame_transform' in project
    assert 'camera.GetPosition()' in project
    assert 'renderer.GetZ(' in project
    assert 'renderer.GetActiveCamera()' in project
    assert 'SetDisplayPosition(x, y)' in project
    assert 'source_frame_oriented_axis_edge_candidates(local_bounds, source_frame)' in frame_source
    assert 'select_camera_visible_axis_edge(' in frame_source
    assert 'visible_samples' in frame_source
    assert 'depth_tolerance' in frame_source
    assert 'SetFontSize(16)' in source

def test_all_twelve_oriented_box_edges_are_available_as_axis_candidates():
    bounds = (np.asarray((-1.0, -2.0, -3.0)), np.asarray((4.0, 5.0, 6.0)))
    frame = np.eye(4, dtype=np.float64)
    angle = np.deg2rad(37.0)
    frame[:3, :3] = ((np.cos(angle), -np.sin(angle), 0.0),
                     (np.sin(angle), np.cos(angle), 0.0),
                     (0.0, 0.0, 1.0))
    frame[:3, 3] = (8.0, -4.0, 3.0)
    candidates = source_frame_oriented_axis_edge_candidates(bounds, frame)
    assert candidates.shape == (3, 4, 2, 3)
    for axis in range(3):
        for edge in candidates[axis]:
            np.testing.assert_allclose(edge[1] - edge[0], frame[:3, axis] * (bounds[1][axis] - bounds[0][axis]))

def test_xy_dimension_edges_switch_to_the_side_opposite_the_camera_vertical_position():
    bounds = (np.asarray((-2.0, -3.0, -4.0)), np.asarray((5.0, 6.0, 8.0)))
    frame = np.eye(4, dtype=np.float64)
    candidates = source_frame_oriented_axis_edge_candidates(bounds, frame)

    for axis_index in (0, 1):
        selected = source_frame_opposite_camera_z_side_axis_edges(
            candidates[axis_index], np.asarray((0.0, 0.0, 12.0)), frame, bounds, axis_index
        )
        assert selected.shape == (2, 2, 3)
        local = points_to_source_local(selected.reshape(-1, 3), frame).reshape(2, 2, 3)
        np.testing.assert_allclose(local[:, :, 2], bounds[0][2])

        selected = source_frame_opposite_camera_z_side_axis_edges(
            candidates[axis_index], np.asarray((0.0, 0.0, -12.0)), frame, bounds, axis_index
        )
        assert selected.shape == (2, 2, 3)
        local = points_to_source_local(selected.reshape(-1, 3), frame).reshape(2, 2, 3)
        np.testing.assert_allclose(local[:, :, 2], bounds[1][2])

def test_xy_opposite_side_switching_uses_the_rotated_source_origin_frame_not_world_z():
    bounds = (np.asarray((-1.0, -2.0, -3.0)), np.asarray((4.0, 5.0, 7.0)))
    angle = np.deg2rad(120.0)
    frame = np.eye(4, dtype=np.float64)
    frame[:3, :3] = (
        (1.0, 0.0, 0.0),
        (0.0, np.cos(angle), -np.sin(angle)),
        (0.0, np.sin(angle), np.cos(angle)),
    )
    frame[:3, 3] = (14.0, -8.0, 21.0)
    candidates = source_frame_oriented_axis_edge_candidates(bounds, frame)

    local_above_world_camera = np.asarray((0.0, 0.0, 10.0)) @ frame[:3, :3].T + frame[:3, 3]
    assert local_above_world_camera[2] < frame[2, 3]
    selected = source_frame_opposite_camera_z_side_axis_edges(
        candidates[0], local_above_world_camera, frame, bounds, 0
    )
    local = points_to_source_local(selected.reshape(-1, 3), frame).reshape(2, 2, 3)
    np.testing.assert_allclose(local[:, :, 2], bounds[0][2])

    local_below_world_camera = np.asarray((0.0, 0.0, -10.0)) @ frame[:3, :3].T + frame[:3, 3]
    selected = source_frame_opposite_camera_z_side_axis_edges(
        candidates[1], local_below_world_camera, frame, bounds, 1
    )
    local = points_to_source_local(selected.reshape(-1, 3), frame).reshape(2, 2, 3)
    np.testing.assert_allclose(local[:, :, 2], bounds[1][2])

def test_dimension_projection_reverses_xy_vertical_side_but_keeps_z_second_nearest():
    source = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    project = source.split('    def _update_dimension_edge_label_positions(self, *_args):', 1)[1].split(
        '    def _position_view_reset_button(self):', 1
    )[0]
    assert 'if axis in {"X", "Y"}:' in project
    assert 'source_frame_opposite_camera_z_side_axis_edges(' in project
    assert 'selection_rank=0,' in project
    assert 'if selected is None:' in project
    assert '                    None,' in project
    assert 'selection_rank=1,' in project
    assert 'self._source_frame_transform' in project

def test_dimension_edge_selector_skips_near_occluded_edges_and_chooses_nearest_visible_edge():
    candidates = np.asarray((
        ((1.0, 0.0, 1.0), (1.2, 0.0, 1.0)),
        ((2.0, 0.0, 1.0), (2.2, 0.0, 1.0)),
        ((3.0, 0.0, 1.0), (3.2, 0.0, 1.0)),
        ((4.0, 0.0, 1.0), (4.2, 0.0, 1.0)),
    ), dtype=np.float64)

    def project(point):
        return (float(point[0] * 50.0), 200.0 + float(point[1]), 0.5)

    def depth_at(x, _y):
        return 0.1 if x < 100 else 1.0

    selected = select_camera_visible_axis_edge(
        candidates, np.zeros(3), project, depth_at, (800, 600)
    )
    assert selected is not None
    assert selected["edge_index"] == 1
    assert selected["visible_samples"] >= 3
    assert selected["world_point"][0] >= 2.0

def test_dimension_edge_selector_can_choose_second_nearest_visible_edge_for_z_axis():
    candidates = np.asarray((
        ((1.0, 0.0, 1.0), (1.2, 0.0, 1.0)),
        ((2.0, 0.0, 1.0), (2.2, 0.0, 1.0)),
        ((3.0, 0.0, 1.0), (3.2, 0.0, 1.0)),
        ((4.0, 0.0, 1.0), (4.2, 0.0, 1.0)),
    ), dtype=np.float64)

    selected = select_camera_visible_axis_edge(
        candidates,
        np.zeros(3),
        lambda point: (float(point[0] * 50.0), 200.0, 0.5),
        lambda _x, _y: 1.0,
        (800, 600),
        selection_rank=1,
    )
    assert selected is not None
    assert selected["edge_index"] == 1
    assert selected["selection_rank"] == 1
    assert selected["candidate_count"] == 4

def test_dimension_edge_selector_falls_back_to_the_only_visible_edge_when_second_is_unavailable():
    candidates = np.asarray((
        ((1.0, 0.0, 1.0), (1.2, 0.0, 1.0)),
        ((2.0, 0.0, 1.0), (2.2, 0.0, 1.0)),
        ((3.0, 0.0, 1.0), (3.2, 0.0, 1.0)),
    ), dtype=np.float64)
    selected = select_camera_visible_axis_edge(
        candidates,
        np.zeros(3),
        lambda point: (float(point[0] * 50.0), 200.0, 0.5),
        lambda x, _y: 1.0 if 75 <= x < 125 else 0.1,
        (800, 600),
        selection_rank=1,
    )
    assert selected is not None
    assert selected["edge_index"] == 1
    assert selected["selection_rank"] == 0
    assert selected["candidate_count"] == 1

def test_dimension_edge_selector_can_anchor_to_the_visible_part_of_a_partially_occluded_edge():
    candidates = np.asarray((
        ((1.0, 0.0, 1.0), (3.0, 0.0, 1.0)),
        ((4.0, 0.0, 1.0), (5.0, 0.0, 1.0)),
        ((6.0, 0.0, 1.0), (7.0, 0.0, 1.0)),
        ((8.0, 0.0, 1.0), (9.0, 0.0, 1.0)),
    ), dtype=np.float64)

    def project(point):
        return (float(point[0] * 100.0), 200.0, 0.5)

    selected = select_camera_visible_axis_edge(
        candidates,
        np.zeros(3),
        project,
        lambda x, _y: 0.1 if x < 200 else 1.0,
        (1200, 600),
    )
    assert selected is not None
    assert selected["edge_index"] == 0
    assert selected["fraction"] > 0.5

def test_source_dimensions_are_measured_before_viewport_sampling():
    source = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    points = source.split('    def set_source_points(', 1)[1].split('    def set_source_mesh(', 1)[0]
    assert points.index('self._source_bounds = self._bounds(points)') < points.index('if len(points) > 120000:')

    mesh = source.split('    def set_source_mesh(', 1)[1].split('    def set_reconstruction_mesh(', 1)[0]
    assert mesh.index('self._source_bounds = self._bounds(self._source_vertices)') < mesh.index('if len(faces) > 240000:')

def test_dimension_hud_uses_the_bottom_hint_live_framebuffer_inversion_glyph_color():
    update_color = _scene_method("_update_dimension_hud_color")

    class TextProperty:
        def __init__(self):
            self.color = None

        def SetColor(self, *color):
            self.color = tuple(color)

        def SetOpacity(self, _value):
            pass

        def SetBackgroundOpacity(self, _value):
            pass

        def SetFrame(self, _value):
            pass

        def SetFrameWidth(self, _value):
            pass

    class Actor:
        def __init__(self):
            self.prop = TextProperty()

        def GetTextProperty(self):
            return self.prop

        def Modified(self):
            pass

    actors = {axis: Actor() for axis in "XYZ"}
    view = SimpleNamespace(dimension_axis_huds=actors, _scene_background_rgba=(226, 233, 241, 255))
    update_color(view)
    assert all(actor.prop.color == (1.0, 1.0, 1.0) for actor in actors.values())

def test_item_info_modes_are_mutually_exclusive_and_include_scale():
    source = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    main = (ROOT / "app/main.py").read_text(encoding="utf-8")
    assert 'normalized not in {"LOC", "ROT", "SCALE", "DIM"}' in source
    assert 'self.item_info_mode_group.setExclusive(True)' in main
    assert 'self.item_info_enabled_check = QCheckBox()' in main
    assert 'self.item_info_enabled_check.setAccessibleName("ITEM INFO")' in main
    assert 'self._sync_item_info_checkbox_icon(saved_item_info_enabled)' in main
    assert 'button.setText("[SCL]" if mode == "SCALE" else f"[{mode}]")' in main
    assert 'for mode in ("LOC", "ROT", "SCALE", "DIM"):' in main
    assert 'self._item_info_enabled and self._item_info_mode in {"LOC", "ROT", "SCALE"}' in source
    assert 'self.item_info_unit_group.setExclusive(True)' in main
    assert 'self.item_info_unit_buttons[unit]' in main
    assert 'for unit in ("mm", "cm", "m", "km"):' in main
    assert 'controls_layout.addWidget(self.item_info_unit_row_widget)' in main
    assert 'def _on_item_info_unit_changed' in main
    assert 'def _sync_item_info_unit_buttons' in main
    assert 'def _update_item_info_controls_visibility' in main
    assert 'selection_rank=1,' in source
    assert "source_frame_opposite_camera_z_side_axis_edges" in source
    mode_row_position = main.index('controls_layout.addLayout(item_info_mode_row)')
    unit_row_position = main.index('controls_layout.addWidget(self.item_info_unit_row_widget)')
    assert unit_row_position > mode_row_position

def test_inline_metric_unit_button_updates_only_the_active_location_or_dimension_setting():
    handler = None
    source = (ROOT / "app/main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    method = next(node for node in main_class.body if isinstance(node, ast.FunctionDef) and node.name == "_on_item_info_unit_changed")
    isolated = ast.FunctionDef(name=method.name, args=method.args, body=method.body, decorator_list=[])
    module = ast.Module(body=[isolated], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "unit_button_handler_test", "exec"), namespace)
    handler = namespace["_on_item_info_unit_changed"]

    class Settings:
        def __init__(self): self.values = {}
        def setValue(self, key, value): self.values[key] = value

    class View:
        def __init__(self, mode):
            self._item_info_mode = mode
            self.location = []
            self.dimension = []
        def set_location_display_unit(self, unit): self.location.append(unit)
        def set_dimension_display_unit(self, unit): self.dimension.append(unit)

    location_view = View("LOC")
    location = SimpleNamespace(settings=Settings(), scene_view=location_view, _sync_item_info_unit_buttons=lambda: None)
    handler(location, "cm", True)
    assert location.settings.values == {"viewport/location_display_unit": "cm"}
    assert location_view.location == ["cm"]
    assert location_view.dimension == []

    dimension_view = View("DIM")
    dimension = SimpleNamespace(settings=Settings(), scene_view=dimension_view, _sync_item_info_unit_buttons=lambda: None)
    handler(dimension, "km", True)
    assert dimension.settings.values == {"viewport/dimension_display_unit": "km"}
    assert dimension_view.dimension == ["km"]
    assert dimension_view.location == []

def test_rotation_option_handler_persists_the_exclusive_elr_qtn_selection():
    source = (ROOT / "app/main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    method = next(node for node in main_class.body if isinstance(node, ast.FunctionDef) and node.name == "_on_item_info_rotation_changed")
    isolated = ast.FunctionDef(name=method.name, args=method.args, body=method.body, decorator_list=[])
    module = ast.Module(body=[isolated], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "rotation_option_handler_test", "exec"), namespace)
    handler = namespace["_on_item_info_rotation_changed"]

    class Settings:
        def __init__(self): self.values = {}
        def setValue(self, key, value): self.values[key] = value

    class View:
        def __init__(self): self.modes = []
        def set_rotation_display_mode(self, mode): self.modes.append(mode)

    settings = Settings()
    view = View()
    checked = SimpleNamespace(
        settings=settings,
        scene_view=view,
        _sync_item_info_rotation_buttons=lambda: None,
    )
    handler(checked, "quaternion", True)
    assert settings.values == {"viewport/rotation_display_mode": "quaternion"}
    assert view.modes == ["quaternion"]
    handler(checked, "euler", False)
    assert settings.values == {"viewport/rotation_display_mode": "quaternion"}
    assert view.modes == ["quaternion"]

def test_scale_components_are_extracted_from_the_source_transform_basis():
    extract_scale = _scene_method("_scale_components")
    angle = np.deg2rad(90.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle), 0.0),
         (np.sin(angle), np.cos(angle), 0.0),
         (0.0, 0.0, 1.0)), dtype=np.float64
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag((2.0, 3.0, 4.0))
    np.testing.assert_allclose(extract_scale(matrix), (2.0, 3.0, 4.0), atol=1e-8)

    matrix[:3, :3] = rotation @ np.diag((-2.0, 3.0, 4.0))
    np.testing.assert_allclose(extract_scale(matrix), (-2.0, 3.0, 4.0), atol=1e-8)

def test_transform_hud_text_switches_to_one_selected_item_property():
    render_text = _scene_method("_transform_hud_text")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (1.25, -2.5, 3.75)
    matrix[:3, :3] = np.diag((2.0, 3.0, 4.0))
    view = SimpleNamespace(
        _source_frame_transform=matrix,
        _item_info_mode="LOC",
        _location_display_unit="m",
        _rotation_display_mode="euler",
        _dimension_unit_profile=_scene_method("_dimension_unit_profile"),
        _scale_components=_scene_method("_scale_components"),
    )
    assert render_text(view).startswith("X 1.250m")
    view._location_display_unit = "mm"
    assert render_text(view) == "X 1,250.00mm  Y -2,500.00mm  Z 3,750.00mm"
    view._location_display_unit = "cm"
    assert render_text(view) == "X 125.00cm  Y -250.00cm  Z 375.00cm"
    view._location_display_unit = "km"
    assert render_text(view) == "X 0.001250km  Y -0.002500km  Z 0.003750km"
    view._location_display_unit = "m"
    view._item_info_mode = "SCALE"
    assert render_text(view) == "X 2.0000  Y 3.0000  Z 4.0000"
    view._item_info_mode = "ROT"
    view._rotation_display_mode = "euler"
    view._rotation_components = lambda: np.asarray((12.3, -45.6, 90.0), dtype=np.float64)
    assert render_text(view) == "X 12.30°  Y -45.60°  Z 90.00°"

    view._rotation_display_mode = "quaternion"
    view._rotation_components = lambda: np.asarray((0.1, 0.2, 0.3, 0.9), dtype=np.float64)
    assert render_text(view) == "W 0.9000  X 0.1000  Y 0.2000  Z 0.3000"

    view._item_info_mode = "DIM"
    assert render_text(view) == ""

def test_transform_hud_replaces_mode_prefixes_and_places_quaternion_w_first():
    source = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    method = source.split("    def _transform_hud_text(self):", 1)[1].split(
        "    def _update_transform_hud(self):", 1
    )[0]
    assert 'f"X {location[0]:' in method
    assert 'f"W {rotation[3]:.4f}  X {rotation[0]:.4f}' in method
    assert 'f"X {scale[0]:' in method
    assert 'LOC X' not in method
    assert 'ROT X' not in method
    assert 'QUAT X' not in method
    assert 'SCALE X' not in method

def test_dimension_projection_indexes_each_axis_and_keeps_labels_at_distinct_edge_positions():
    source = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    scene = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SceneView")
    method = next(node for node in scene.body if isinstance(node, ast.FunctionDef) and node.name == "_update_dimension_edge_label_positions")
    isolated = ast.FunctionDef(
        name=method.name, args=method.args, body=method.body,
        decorator_list=[], returns=method.returns, type_comment=method.type_comment,
    )
    module = ast.Module(body=[isolated], type_ignores=[])
    ast.fix_missing_locations(module)

    class FakeActor:
        def __init__(self):
            self.position = None
            self.visible = False
        def SetVisibility(self, value): self.visible = bool(value)
        def SetDisplayPosition(self, x, y): self.position = (int(x), int(y))
        def Modified(self): pass

    class FakeCamera:
        def GetPosition(self): return (1.0, -10.0, 1.0)

    class FakeRenderer:
        def SetWorldPoint(self, x, y, z, _w): self.world_point = np.asarray((x, y, z), dtype=np.float64)
        def WorldToDisplay(self):
            point = self.world_point
            self.display_point = (120.0 + point[0] * 50.0, 140.0 + point[1] * 40.0 + point[2] * 10.0, 0.5)
        def GetDisplayPoint(self): return self.display_point
        def GetActiveCamera(self): return FakeCamera()
        def GetZ(self, _x, _y): return 1.0

    class FakeRenderWindow:
        def GetSize(self): return (800, 600)

    actors = {axis: FakeActor() for axis in "XYZ"}
    view = SimpleNamespace(
        _item_info_enabled=True, _item_info_mode="DIM",
        _update_transform_hud_position=lambda: None, dimension_axis_huds=actors,
        _current_dimension_box=lambda: {
            "local_bounds": (np.zeros(3), np.asarray((2.0, 3.0, 4.0))),
            "world_corners": np.asarray([(x, y, z) for x in (0.0, 2.0) for y in (0.0, 3.0) for z in (0.0, 4.0)]),
        },
        _renderer=FakeRenderer(), _source_frame_transform=np.eye(4),
        GetRenderWindow=lambda: FakeRenderWindow(), width=lambda: 800, height=lambda: 600,
    )
    namespace = {
        "np": np,
        "source_frame_oriented_axis_edge_candidates": source_frame_oriented_axis_edge_candidates,
        "source_frame_opposite_camera_z_side_axis_edges": source_frame_opposite_camera_z_side_axis_edges,
        "select_camera_visible_axis_edge": select_camera_visible_axis_edge,
    }
    exec(compile(module, "dimension_projection_test", "exec"), namespace)
    namespace[method.name](view)

    assert all(actor.visible for actor in actors.values())
    assert all(actor.position is not None for actor in actors.values())
    assert len({actor.position for actor in actors.values()}) == 3
    assert view._dimension_label_projected_positions["X"] != view._dimension_label_projected_positions["Y"]
    assert view._dimension_label_projected_positions["Z"] != view._dimension_label_projected_positions["X"]

def test_dimension_projection_hides_offscreen_labels_instead_of_clamping_them_to_the_corner():
    source = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    scene = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SceneView")
    method = next(node for node in scene.body if isinstance(node, ast.FunctionDef) and node.name == "_update_dimension_edge_label_positions")
    isolated = ast.FunctionDef(
        name=method.name, args=method.args, body=method.body,
        decorator_list=[], returns=method.returns, type_comment=method.type_comment,
    )
    module = ast.Module(body=[isolated], type_ignores=[])
    ast.fix_missing_locations(module)

    class FakeActor:
        def __init__(self): self.visible = False; self.position = None
        def SetVisibility(self, value): self.visible = bool(value)
        def SetDisplayPosition(self, x, y): self.position = (x, y)
        def Modified(self): pass

    class FakeCamera:
        def GetPosition(self): return (0.0, 0.0, 5.0)

    class FakeRenderer:
        def SetWorldPoint(self, x, y, z, _w): self.point = (x, y, z)
        def WorldToDisplay(self): self.display = (-200.0, -150.0, 0.5)
        def GetDisplayPoint(self): return self.display
        def GetActiveCamera(self): return FakeCamera()
        def GetZ(self, _x, _y): return 1.0

    class FakeRenderWindow:
        def GetSize(self): return (800, 600)

    actors = {axis: FakeActor() for axis in "XYZ"}
    view = SimpleNamespace(
        _item_info_enabled=True, _item_info_mode="DIM",
        _update_transform_hud_position=lambda: None, dimension_axis_huds=actors,
        _current_dimension_box=lambda: {
            "local_bounds": (np.zeros(3), np.ones(3)),
            "world_corners": np.asarray([(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]),
        },
        _renderer=FakeRenderer(), _source_frame_transform=np.eye(4),
        GetRenderWindow=lambda: FakeRenderWindow(), width=lambda: 800, height=lambda: 600,
    )
    namespace = {
        "np": np,
        "source_frame_oriented_axis_edge_candidates": source_frame_oriented_axis_edge_candidates,
        "source_frame_opposite_camera_z_side_axis_edges": source_frame_opposite_camera_z_side_axis_edges,
        "select_camera_visible_axis_edge": select_camera_visible_axis_edge,
    }
    exec(compile(module, "dimension_projection_offscreen_test", "exec"), namespace)
    namespace[method.name](view)

    assert all(not actor.visible for actor in actors.values())
    assert all(actor.position is None for actor in actors.values())
    assert view._dimension_label_projected_positions == {}

def test_new_import_camera_is_framed_immediately_instead_of_waiting_for_an_eased_motion():
    main = (ROOT / "app/main.py").read_text(encoding="utf-8")
    reset = main.split("    def _reset_viewport_to_new_source(self):", 1)[1].split(
        "    def _set_output_workspace_display", 1
    )[0]
    assert "self.scene_view.reset_view(animate=False)" in reset

    view = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    fit = view.split("    def _fit_union_camera(self, *, animate: bool = False):", 1)[1].split(
        "    def set_source_points", 1
    )[0]
    assert "source_frame_camera_pose(" in fit
    assert "self._source_frame_transform" in fit
    assert "active_camera.SetFocalPoint(*focal)" in fit
    assert "active_camera.SetPosition(*position)" in fit
    assert "active_camera.SetViewUp(*view_up)" in fit
