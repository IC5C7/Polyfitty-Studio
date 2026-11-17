from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _source(name: str) -> str:
    return (ROOT / "app" / name).read_text(encoding="utf-8")

def _method(file_name: str, class_name: str, method_name: str) -> str:
    source = _source(file_name)
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""

def test_line_width_is_hard_locked_to_one_pixel_and_ui_slider_is_removed():
    main = _source("main.py")
    backend = _source("vtk_backend.py")
    view = _source("view3d.py")
    assert "wire_width_slider" not in main
    assert "LINE WIDTH" not in main
    assert "set_wireframe_width" not in main
    assert backend.count("SetLineWidth(1.0)") >= 2
    assert "width=" not in view
    assert "setData(self, pos=None, color=None, width=None" not in backend

def test_overlay_toggles_keep_wireframe_and_connected_faces_but_remove_sharp_checkbox():
    main = _source("main.py")
    assert 'QCheckBox("Wireframe")' in main
    assert 'QCheckBox("Connected faces")' in main
    assert 'QCheckBox("Sharp edges")' not in main
    assert "sharp_edges_toggle" not in main
    visibility = _method("main.py", "MainWindow", "_update_overlay_control_visibility")
    assert "self.wireframe_options_widget.setVisible(wire)" in visibility
    assert "self.connected_faces_options_widget.setVisible(connected)" in visibility
    assert 'selected_layer = getattr(self.scene_view, "selected_mesh_layer", None)' in visibility
    assert "self.overlay_mode_widget.setVisible(wire or sharp)" in visibility
    assert 'button.setVisible(sharp and layer == selected_layer)' in visibility

def test_reconstruction_preview_draws_only_edges_without_face_fills():
    draw = _method("view3d.py", "SceneView", "set_processing_mesh")
    assert "_polygon_wireframe_positions(vertices, triangles, quad_faces)" in draw
    assert "GLMeshItem(" not in draw
    assert "vertexColors" not in draw
    assert "width=" not in draw
    assert "highlight_segments" in draw

def test_preview_scene_has_no_grid_and_unity_scene_view_bindings():
    view = _source("view3d.py")
    backend = _source("vtk_backend.py")
    assert "GLGridItem" not in view
    assert "_grid" not in view
    assert "vtkInteractorStyleUser" in backend
    assert "viewportClicked = QtCore.Signal(float, float)" in backend
    for method in ("mousePressEvent", "mouseMoveEvent", "mouseReleaseEvent", "wheelEvent", "keyPressEvent", "keyReleaseEvent", "_step_keyboard_fly", "_pan_camera"):
        assert f"def {method}" in backend

def test_unity_viewport_mouse_bindings_and_selection_click():
    press = _method("vtk_backend.py", "GLViewWidget", "mousePressEvent")
    move = _method("vtk_backend.py", "GLViewWidget", "mouseMoveEvent")
    release = _method("vtk_backend.py", "GLViewWidget", "mouseReleaseEvent")
    wheel = _method("vtk_backend.py", "GLViewWidget", "wheelEvent")
    pan = _method("vtk_backend.py", "GLViewWidget", "_pan_camera")
    assert "_left_press_pos = event.position()" in press
    assert "event.buttons()" in move and "LeftButton" in move
    assert 'self._navigation_mode = "orbit"' in press
    assert 'self._navigation_mode = "look"' in press
    assert 'self._navigation_mode = "pan"' in press
    assert 'self._navigation_mode in {"orbit", "look"}' in move
    assert "self._queue_smooth_look(dx, dy)" in move
    smooth_step = _method("vtk_backend.py", "GLViewWidget", "_step_smooth_look")
    assert "self._set_camera_look(" in smooth_step
    assert "self._pan_camera(dx, dy)" in move
    assert "viewportClicked.emit" in release
    assert "math.exp(-delta * zoom_speed)" in wheel
    assert "self._queue_camera_motion(position + offset, focal + offset, view_up)" in pan
    look = _method("vtk_backend.py", "GLViewWidget", "_set_camera_look")
    assert "camera.SetFocalPoint" in look
    assert "camera.SetPosition" not in look
    assert "Key_W" not in wheel

def test_unity_fly_controls_use_rmb_wasd_qe_and_progressive_shift_acceleration():
    key_press = _method("vtk_backend.py", "GLViewWidget", "keyPressEvent")
    key_release = _method("vtk_backend.py", "GLViewWidget", "keyReleaseEvent")
    fly = _method("vtk_backend.py", "GLViewWidget", "_step_keyboard_fly")
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    assert "self._right_mouse_down" in key_press
    assert "self._navigation_action_for_event(event)" in key_press
    assert "self._fly_keys_by_physical.pop(key, None)" in key_release
    for action in ("forward", "backward", "left", "right", "down", "up"):
        assert f'"{action}"' in init
        assert f'"{action}" in self._fly_keys' in fly
    assert '"forward": "W"' in init and '"up": "E"' in init
    assert '"down": "Q"' in init
    assert "self._shift_down = True" in key_press
    assert "self._shift_down = False" in key_release

    assert "super().keyPressEvent(event)" not in key_press
    assert "super().keyReleaseEvent(event)" not in key_release
    assert "event.accept()" in key_press and "event.accept()" in key_release
    assert "self._fly_speed_multiplier += self._fly_shift_acceleration_per_second * dt" in fly
    assert "1.0 - math.exp(-self._fly_shift_release_easing_per_second * dt)" in fly
    assert "speed = base_speed * self._fly_speed_multiplier" in fly
    assert "self._fly_speed_multiplier = 1.0" in init
    assert "self._fly_shift_acceleration_per_second = 3.0" in init
    assert "self._fly_shift_release_easing_per_second = 5.0" in init
    assert "time.monotonic()" in fly
    assert "velocity_blend = 1.0 - math.exp(-16.0 * dt)" in fly
    assert "self._fly_velocity += (target_velocity - self._fly_velocity) * velocity_blend" in fly
    assert "self._fly_timer.stop()" in fly
    assert "self._fly_keys_by_physical[key] = action" in key_press

def test_click_selection_toggles_edges_off_when_reclicking_the_selected_mesh():
    view = _source("view3d.py")
    handler = _method("view3d.py", "SceneView", "_handle_viewport_click")
    mapping = _method("view3d.py", "SceneView", "_mesh_layer_for_actor")
    visibility = _method("view3d.py", "SceneView", "_apply_visibility")
    assert "self.pick_actor_at(x, y)" in handler
    assert "if layer is None and self._selected_mesh_layer is None:" in handler
    assert "if layer == self._selected_mesh_layer:\n            layer = None" in handler
    assert "self._selected_mesh_layer = layer" in handler
    assert "self._sharp_edges_enabled = layer is not None" in handler
    assert "self.selectionChanged.emit(layer)" in handler
    assert "self._apply_visibility()" in handler
    assert "_source_surface_item" in mapping and "_result_surface_item" in mapping
    assert 'self._selected_mesh_layer == "source"' in visibility
    assert 'self._selected_mesh_layer == "result"' in visibility
    assert "selectionChanged = QtCore.Signal(object)" in view
    assert "self._selected_mesh_layer = None" in view
    assert "self._sharp_edges_enabled = False" in view
    picker = _method("vtk_backend.py", "GLViewWidget", "pick_actor_at")
    assert "vtk.vtkCellPicker()" in picker
    assert "self._renderer" in picker

def test_connected_faces_keeps_seeded_random_component_face_colors():
    view = _source("view3d.py")
    mesh_vis = _source("mesh_visualization.py")
    builder = _method("view3d.py", "SceneView", "_rebuild_face_island_items")
    assert "connected_component_mesh_data(" in builder
    assert "seed=seed" in builder
    assert "vertexColors=expanded_c" in builder
    assert "face_component_colors" in mesh_vis
    assert "np.random.default_rng(int(seed))" in mesh_vis

def test_all_application_hover_tooltips_are_suppressed_globally():
    main = _source("main.py")
    event_filter = _method("main.py", "MainWindow", "eventFilter")
    assert "app.installEventFilter(self)" in main
    assert "QEvent.Type.ToolTip" in event_filter
    assert "return True" in event_filter
    assert "QEvent.Type.ToolTip" in main

def test_navigation_help_is_glyph_only_hardware_inverted_vtk_overlay():
    backend = _source("vtk_backend.py")
    invert = _source("gl_invert.py")
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    refresh = _method("vtk_backend.py", "GLViewWidget", "_refresh_navigation_help")
    position = _method("vtk_backend.py", "GLViewWidget", "_position_navigation_help")
    begin = _method("vtk_backend.py", "GLViewWidget", "_begin_invert_pass")
    assert "vtk.vtkTextActor()" in init
    assert "self._invert_renderer.AddViewProp(self._navigation_help)" in init
    assert "SetBackgroundOpacity(0.0)" in init
    assert "SetFrame(False)" in init
    assert "SetFrameWidth(0)" in init
    assert "help_text_property.SetColor(1.0, 1.0, 1.0)" in init
    assert "self._invert_renderer" in init and "self._navigation_help" in init
    assert "GetViewProps()" in begin
    assert "QtWidgets.QLabel(self)" not in init
    assert "WA_TranslucentBackground" not in init
    assert "setAutoFillBackground(False)" not in init
    assert "glLogicOp" in invert and "GL_INVERT" in invert
    assert "SetInput(" in refresh
    assert "RMB Camera Rotate / Fly" in refresh
    assert "Face Overlay" in refresh
    assert "WASD" not in refresh
    assert "SetDisplayPosition" in position
    assert "raise_()" not in position

def test_viewport_shortcuts_are_fixed_and_legacy_vtk_shortcuts_are_blocked():
    backend = _source("vtk_backend.py")
    main = _source("main.py")
    press = _method("vtk_backend.py", "GLViewWidget", "keyPressEvent")
    release = _method("vtk_backend.py", "GLViewWidget", "keyReleaseEvent")
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    nav_setter = _method("vtk_backend.py", "GLViewWidget", "set_navigation_shortcuts")
    action_setter = _method("vtk_backend.py", "GLViewWidget", "set_viewport_shortcuts")
    assert '"forward": "W"' in init and '"toggle_face_overlay": "Z"' in init
    assert '"down": "Q"' in init and '"up": "E"' in init
    assert "values = dict(defaults)" in nav_setter
    assert "values = dict(defaults)" in action_setter
    assert "super().keyPressEvent(event)" not in press
    assert "super().keyReleaseEvent(event)" not in release
    assert "self._navigation_action_for_event(event)" in press
    assert "self._viewport_action_for_event(event)" in press
    assert "event.isAutoRepeat()" in release
    assert "event.accept()" in press and "event.accept()" in release
    assert "QKeySequenceEdit" not in main
    assert "_clear_duplicate_saved_viewport_shortcuts" not in main
    assert 'CollapsibleSection("SUPPORTED INPUTS / HELP"' not in main
    assert main.index('CollapsibleSection("VIEWPORT SHORTCUTS"') > main.index('CollapsibleSection("ADVANCED · POLYFIT WEIGHTS"')

def test_shortcut_panel_replaces_supported_inputs_help_at_bottom_of_options():
    main = _source("main.py")
    assert 'CollapsibleSection("VIEWPORT SHORTCUTS", expanded=False)' in main
    assert 'CollapsibleSection("SUPPORTED INPUTS / HELP"' not in main
    assert "fixed_shortcuts = (" in main
    assert "key_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)" in main
    assert "QKeySequenceEdit" not in main
    assert "_on_viewport_shortcut_changed" not in main

def test_gpu_status_panel_uses_compact_summary_and_compact_rescan_button():
    main = _source("main.py")
    assert 'QPushButton("Rescan")' in main
    assert 'self.gpu_rescan_btn.setObjectName("GPURescanButton")' in main
    assert "self.gpu_rescan_btn.setFixedSize(88, 30)" in main
    assert "padding: 1px 8px" in main
    assert 'self.gpu_status_detail.setText(compact.replace(" · ", " "))' in main

def test_pipeline_previews_do_not_generate_or_serialize_random_face_colors():
    pipeline = _source("pipeline.py")
    worker = _source("worker.py")
    assert "normal_direction_colors" not in pipeline
    assert "planar_region_colors" not in pipeline
    assert "face_colors=" not in pipeline
    assert '"face_colors"' not in worker
    assert "highlight_segments" in pipeline

def test_rmb_look_uses_reduced_sensitivity_and_smooth_easing():
    backend = _source("vtk_backend.py")
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    mouse_move = _method("vtk_backend.py", "GLViewWidget", "mouseMoveEvent")
    queue = _method("vtk_backend.py", "GLViewWidget", "_queue_smooth_look")
    step = _method("vtk_backend.py", "GLViewWidget", "_step_smooth_look")
    cancel = _method("vtk_backend.py", "GLViewWidget", "_cancel_smooth_look")
    assert "self._look_sensitivity = 0.045" in init
    assert "self._look_easing = 0.28" in init
    assert "self._look_timer.timeout.connect(self._step_smooth_look)" in init
    assert "self._queue_smooth_look(dx, dy)" in mouse_move
    assert "self._look_sensitivity * (2.0 if self._shift_down else 1.0)" in queue
    assert "self._look_timer.start()" in queue
    assert "self._look_easing" in step
    assert "azimuth_error" in step and "elevation_error" in step
    assert "self._look_timer.stop()" in cancel

def test_connected_faces_uses_explicit_filled_unlit_random_face_colors():
    view = _source("view3d.py")
    builder = _method("view3d.py", "SceneView", "_rebuild_face_island_items")
    configure = _method("view3d.py", "SceneView", "_configure_face_island_fill")
    visibility = _method("view3d.py", "SceneView", "_apply_visibility")
    assert "connected_component_mesh_data(" in builder
    assert "vertexColors=expanded_c" in builder
    assert "drawEdges=False, drawFaces=True" in builder
    assert "_configure_face_island_fill(item)" in builder
    assert "SetRepresentationToSurface()" in configure
    assert "LightingOff()" in configure
    assert "SetColorModeToDirectScalars()" in configure
    assert "ScalarVisibilityOn()" in configure
    assert "colors[:, 3] = 1.0" in visibility
    assert "item.actor.GetProperty().SetOpacity(overlay_opacity)" in visibility
    assert "face_component_colors" in _source("mesh_visualization.py")
    assert "self._face_islands_opacity = 1.0" in view
    assert 'self.connected_faces_opacity_slider.setValue(100)' in _source("main.py")

def test_rotation_recenters_pointer_for_unbounded_alt_orbit_and_rmb_look():
    backend = _source("vtk_backend.py")
    begin = _method("vtk_backend.py", "GLViewWidget", "_begin_mouse_wrap")
    move = _method("vtk_backend.py", "GLViewWidget", "mouseMoveEvent")
    press = _method("vtk_backend.py", "GLViewWidget", "mousePressEvent")
    release = _method("vtk_backend.py", "GLViewWidget", "mouseReleaseEvent")
    assert "mapToGlobal(self.rect().center())" in begin
    assert "QtGui.QCursor.setPos" in begin
    assert "event.globalPosition().toPoint()" in move
    assert "self._warp_mouse_to_anchor()" in move
    assert "self._begin_mouse_wrap()" in press
    assert "self._stop_pointer_navigation(cancel_camera_motion=True)" in release
    assert "self._end_mouse_wrap()" in _method("vtk_backend.py", "GLViewWidget", "_stop_pointer_navigation")
    assert "_mouse_wrap_anchor_global" in backend

def test_qe_vertical_fly_has_eased_acceleration_and_deceleration():
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    release = _method("vtk_backend.py", "GLViewWidget", "keyReleaseEvent")
    fly = _method("vtk_backend.py", "GLViewWidget", "_step_keyboard_fly")
    assert "self._fly_velocity = np.zeros(3" in init
    assert "self._fly_shift_acceleration_per_second = 3.0" in init
    assert '"down": "Q"' in init and '"up": "E"' in init
    main = _source("main.py")
    assert '("Fly down", "Q")' in main
    assert '("Fly up", "E")' in main
    assert "self._fly_timer.stop()" not in release
    assert "self._fly_keys.discard(released_action)" in release
    assert 'int("up" in self._fly_keys) - int("down" in self._fly_keys)' in fly
    assert "target_velocity" in fly and "self._fly_velocity" in fly

def test_viewport_reset_button_and_per_task_camera_reset_are_available():
    view = _source("view3d.py")
    reset = _method("view3d.py", "SceneView", "reset_view")
    assert "self.view_reset_button = QtWidgets.QToolButton(self)" in view
    assert "self.view_reset_button.setFixedSize(28, 28)" in view
    assert "self.view_reset_button.clicked.connect(self.reset_view)" in view
    assert "self._fit_union_camera(animate=True)" in reset
    assert "self._queue_camera_motion(direction * 8.0, np.zeros(3), (0.0, 0.0, 1.0))" in reset

def test_viewport_shortcuts_are_fixed_read_only_and_z_toggles_face_overlay():
    main = _source("main.py")
    backend = _source("vtk_backend.py")
    assert '("Focus / restore camera", "F")' in main
    assert '("Toggle wireframe", "X")' in main
    assert '("Toggle connected faces", "C")' in main
    assert '("Toggle view-aligned grid", "G")' in main
    assert '("Reimport source", "R")' in main
    assert '("Toggle face overlay", "Z")' in main
    assert '("Fly down", "Q")' in main
    assert '("Fly up", "E")' in main
    assert 'self.settings.setValue(f"viewport/shortcuts/' not in main
    assert "QKeySequenceEdit" not in main
    assert "key_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)" in main
    assert "viewportShortcutTriggered.emit(action)" in backend

def test_viewport_hotkeys_dispatch_to_each_expected_ui_action():
    main = _source("main.py")
    handler = _method("main.py", "MainWindow", "_handle_viewport_shortcut")
    assert 'if action == "focus_view":' in handler and "self.scene_view.focus_view()" in handler
    assert 'elif action == "toggle_wireframe":' in handler
    assert 'self.wireframe_toggle.setChecked(not self.wireframe_toggle.isChecked())' in handler
    assert 'elif action == "toggle_connected_faces":' in handler
    assert 'self.face_islands_toggle.setChecked(not self.face_islands_toggle.isChecked())' in handler
    assert 'elif action == "toggle_face_overlay":' in handler
    assert 'self.face_overlay_toggle.setChecked(not self.face_overlay_toggle.isChecked())' in handler
    assert 'self.scene_view.viewportShortcutTriggered.connect(self._handle_viewport_shortcut)' in main

def test_direct_bake_is_the_default_projection_mode():
    main = _source("main.py")
    assert 'self.settings.value("bake_projection_mode", "direct") or "direct"' in main

def test_auto_attempts_polyfit_before_worker_supervisor_falls_back_to_li_nan():
    pipeline = _source("pipeline.py")
    main = _source("main.py")
    assert '"AUTO primary engine: PolyFit "' in pipeline
    assert 'if requested == "auto" and mesh_input_stats:' in pipeline
    auto_block = _method("main.py", "PipelineWorker", "run")
    assert 'requested_engine == "auto" and initial_failed' in auto_block
    assert '"li_nan"' in auto_block
    assert "return_code" in auto_block and "completed_result" in auto_block
    assert "resolve_polyfit_timeout_fallback_source" in auto_block
    assert "stale_candidate.unlink(missing_ok=True)" in auto_block
    assert 'requested_engine == "polyfit"' in auto_block
    assert "self._cancel_requested" in auto_block

def test_shift_is_an_independent_boost_modifier_and_navigation_uses_physical_keys():
    backend = _source("vtk_backend.py")
    navigation = _method("vtk_backend.py", "GLViewWidget", "_navigation_action_for_event")
    viewport_actions = _method("vtk_backend.py", "GLViewWidget", "_viewport_action_for_event")
    press = _method("vtk_backend.py", "GLViewWidget", "keyPressEvent")
    release = _method("vtk_backend.py", "GLViewWidget", "keyReleaseEvent")
    fly = _method("vtk_backend.py", "GLViewWidget", "_step_keyboard_fly")
    assert "event.key()" in navigation
    assert "Key_W" in navigation and "Key_A" in navigation and "Key_S" in navigation
    assert "Key_D" in navigation and "Key_Q" in navigation and "Key_E" in navigation
    assert "event.modifiers()" not in navigation
    assert "Key_F" in viewport_actions and "Key_G" in viewport_actions
    assert "Key_X" in viewport_actions and "Key_Z" in viewport_actions
    assert "self._right_mouse_down" in press
    assert "self._shift_down = bool(event.modifiers() & shift_modifier)" in press
    assert "self._shift_down = True" in press
    assert "QApplication.keyboardModifiers()" in fly
    assert "QApplication.keyboardModifiers()" in release
    assert '"down": "Q"' in backend and '"up": "E"' in backend
    assert "self._fly_speed_multiplier += self._fly_shift_acceleration_per_second * dt" in fly
    assert "speed = base_speed * self._fly_speed_multiplier" in fly
    assert "event.isAutoRepeat()" in release

def test_fixed_viewport_keymap_matches_requested_actions_and_grid_is_one_pixel():
    backend = _source("vtk_backend.py")
    main = _source("main.py")
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    keymap = _method("vtk_backend.py", "GLViewWidget", "_viewport_action_for_event")
    grid_geometry = _method("vtk_backend.py", "GLViewWidget", "_update_viewport_grid_geometry")
    assert '"focus_view": "F"' in init
    assert '"toggle_wireframe": "X"' in init
    assert '"toggle_connected_faces": "C"' in init
    assert '"toggle_grid": "G"' in init
    assert '"reimport": "R"' in init
    assert '"toggle_face_overlay": "Z"' in init
    for qt_key, action in (("Key_F", "focus_view"), ("Key_X", "toggle_wireframe"),
                           ("Key_C", "toggle_connected_faces"), ("Key_G", "toggle_grid"),
                           ("Key_R", "reimport"), ("Key_Z", "toggle_face_overlay")):
        assert f"key_map.{qt_key}:" not in keymap or f"int(key_map.{qt_key})" in keymap
        assert f'"{action}"' in keymap
    assert 'self._viewport_grid_base_spacing = 1.0' in init
    assert 'self._viewport_grid_actor = vtk.vtkActor()' in init
    assert 'grid_property.SetLineWidth(1.0)' in init
    assert 'self._renderer.AddActor(self._viewport_grid_actor)' in init
    assert 'self._viewport_grid_mapper.SetScalarModeToUseCellData()' in init
    assert 'floor_z = (zmin - max(scene_span * 0.0015, 1.0e-4)) if bounds is not None else 0.0' in grid_geometry
    assert 'plane_gap = max(scene_span * 0.0015, 1.0e-4)' in grid_geometry
    assert 'fade = 1.0 if orthographic else radial_grid_fade(radius, outer_fade_radius)' in grid_geometry
    assert 'outer_fade_radius = half_extent * 0.96' in grid_geometry
    assert 'major_rgb = (165, 178, 191)' in grid_geometry
    assert 'axis_rgb = (194, 112, 99)' in grid_geometry
    assert 'second_axis_rgb = (94, 151, 196)' in grid_geometry
    assert '"RMB + WASD/QE"' not in main

def test_ground_grid_uses_a_smooth_circular_radial_fade_with_a_rounded_edge():
    from app.viewport_grid import radial_grid_fade

    assert radial_grid_fade(0.0, 1.0) == 1.0
    assert radial_grid_fade(0.58, 1.0) == 1.0
    assert abs(radial_grid_fade(0.79, 1.0) - 0.5) <= 1.0e-6
    assert radial_grid_fade(1.0, 1.0) == 0.0
    assert radial_grid_fade(1.2, 1.0) == 0.0

    radial_distance = 0.82
    import math
    along_x = math.hypot(radial_distance, 0.0)
    diagonal = math.hypot(radial_distance / math.sqrt(2.0), radial_distance / math.sqrt(2.0))
    assert abs(along_x - diagonal) <= 1.0e-12
    assert radial_grid_fade(along_x, 1.0) == radial_grid_fade(diagonal, 1.0)
    values = [radial_grid_fade(distance, 1.0) for distance in (0.58, 0.65, 0.72, 0.79, 0.86, 0.93, 1.0)]
    assert values == sorted(values, reverse=True)

    grid = _method("vtk_backend.py", "GLViewWidget", "_update_viewport_grid_geometry")
    assert "math.hypot((first[0] + second[0]) * 0.5 - center_u" in grid
    assert "radial_grid_fade(radius, outer_fade_radius)" in grid
    assert "fade = 1.0 if orthographic else radial_grid_fade(radius, outer_fade_radius)" in grid
    assert "outer_fade_radius = half_extent * 0.96" in grid
    assert "segment_count = 48" in grid
    assert "half_extent * math.sqrt(2.0)" not in grid

def test_f_focuses_selected_asset_instead_of_resetting_camera():
    focus = _method("view3d.py", "SceneView", "focus_view")
    handler = _method("main.py", "MainWindow", "_handle_viewport_shortcut")
    assert "selected = self._selected_mesh_layer" in focus
    assert 'target_bounds = self._source_bounds' in focus
    assert 'target_bounds = self._result_bounds' in focus
    assert 'self._focus_restore_camera = {' in focus
    assert 'restore["position"]' in focus and 'restore["focal"]' in focus
    assert "self._queue_camera_motion(center + direction * distance, center, camera.GetViewUp())" in focus
    assert "self.scene_view.focus_view()" in handler
    assert "self.scene_view.reset_view()" not in handler

def test_imported_source_stays_unselected_until_f_auto_selects_and_focuses_it():
    view = _source("view3d.py")
    source_mesh = _method("view3d.py", "SceneView", "set_source_mesh")
    source_points = _method("view3d.py", "SceneView", "set_source_points")
    focus = _method("view3d.py", "SceneView", "focus_view")
    assert "import math" in view
    assert 'self._select_mesh_layer("source")' not in source_mesh
    assert 'self._select_mesh_layer("source")' not in source_points
    assert 'selected = "source"' in focus
    assert 'self._select_mesh_layer(selected)' in focus
    assert 'self._queue_camera_motion(\n                restore["position"]' in focus
    assert "self._focused_mesh_layer = None" in focus

def test_mesh_backfaces_are_culled_and_point_clouds_remain_depth_tested():
    backend = _source("vtk_backend.py")
    mesh_init = _method("vtk_backend.py", "GLMeshItem", "__init__")
    points_init = _method("vtk_backend.py", "GLScatterPlotItem", "__init__")
    point_import = _method("view3d.py", "SceneView", "set_source_points")
    infront = _source("wireframe.py")
    assert "prop.SetBackfaceCulling(True)" in mesh_init
    assert "self._renderer.SetUseDepthPeeling(True)" in backend
    assert "self._actor.GetProperty().SetRepresentationToPoints()" in points_init
    assert 'self._apply_infront_to_item(item, "other")' in point_import

    assert 'return normalized_kind in {"surface", "wireframe"}' in infront

def test_face_overlay_renderer_routing_keeps_the_ground_grid_world_anchor_stable():
    backend = _source("vtk_backend.py")
    bounds = _method("vtk_backend.py", "GLViewWidget", "_viewport_grid_scene_bounds")
    assert "getattr(item, \"_invert_overlay\", False)" in bounds
    assert "getattr(item, \"_overlay\", False)" not in bounds

def test_viewport_lens_and_clip_planes_are_editable_and_persisted():
    backend = _source("vtk_backend.py")
    main = _source("main.py")
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    focal = _method("vtk_backend.py", "GLViewWidget", "set_camera_focal_length")
    clipping = _method("vtk_backend.py", "GLViewWidget", "set_camera_clip_range")
    apply_ui = _method("main.py", "MainWindow", "_apply_viewport_camera_settings")
    assert "self._camera_focal_length_mm = 26.9915" in init
    assert "self._camera_clip_start = 0.1" in init
    assert "self._camera_clip_end = 10000.0" in init
    assert "camera.SetUseHorizontalViewAngle(True)" in focal
    assert "36.0 / (2.0 * self._camera_focal_length_mm)" in focal
    assert "camera.SetClippingRange" in _method("vtk_backend.py", "GLViewWidget", "_apply_camera_clip_range")
    assert "0 < Clip Start < Clip End" in clipping
    assert 'CollapsibleSection("VIEWPORT CAMERA", expanded=False)' in main
    assert '"viewport/camera/focal_length_mm", 26.9915' in main
    assert '"viewport/camera/clip_start_m", 0.1' in main
    assert '"viewport/camera/clip_end_m", 10000.0' in main
    assert "self.scene_view.set_camera_focal_length(focal_length)" in apply_ui
    assert "self.scene_view.set_camera_clip_range(clip_start, clip_end)" in apply_ui

def test_orbit_pan_zoom_and_focus_use_short_eased_camera_transitions():
    backend = _source("vtk_backend.py")
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    queue = _method("vtk_backend.py", "GLViewWidget", "_queue_camera_motion")
    step = _method("vtk_backend.py", "GLViewWidget", "_step_camera_motion")
    orbit = _method("vtk_backend.py", "GLViewWidget", "_set_camera_orbit")
    pan = _method("vtk_backend.py", "GLViewWidget", "_pan_camera")
    assert "self._camera_motion_easing = 0.30" in init
    assert "self._camera_motion_timer.timeout.connect(self._step_camera_motion)" in init
    assert "self._camera_motion_target_position = position.copy()" in queue
    assert "position_error * easing" in step and "focal_error * easing" in step
    assert "self._queue_camera_motion(position, focus, (0.0, 0.0, 1.0))" in orbit
    assert "self._queue_camera_motion(position + offset, focal + offset, view_up)" in pan
    assert "zoom_speed = 0.05" in _method("vtk_backend.py", "GLViewWidget", "wheelEvent")
    assert "turn_speed = 0.09" in _method("vtk_backend.py", "GLViewWidget", "mouseMoveEvent")

def test_successful_bake_automatically_activates_wireframe_for_result_inspection():
    load_baked = _method("main.py", "MainWindow", "_load_baked_glb_result")
    assert "self.wireframe_toggle.setEnabled(True)" in load_baked
    assert 'self.scene_view.set_display_mode("both")' in load_baked
    assert "self.wireframe_toggle.setChecked(True)" in load_baked

def test_final_baked_result_automatically_disables_face_overlay():
    load_baked = _method("main.py", "MainWindow", "_load_baked_glb_result")
    install_result = load_baked.index("self.scene_view.set_reconstruction_mesh(")
    disable_overlay = load_baked.index("self.face_overlay_toggle.setChecked(False)")
    assert install_result < disable_overlay
    assert "self.wireframe_toggle.setChecked(True)" in load_baked

def test_viewport_grid_is_a_depth_tested_world_space_ground_plane():
    backend = _source("vtk_backend.py")
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    grid = _method("vtk_backend.py", "GLViewWidget", "_update_viewport_grid_geometry")
    assert "vtk.vtkActor()" in init
    assert "self._renderer.AddActor(self._viewport_grid_actor)" in init
    assert "vtk.vtkActor2D()" not in init
    assert "_viewport_grid_scene_bounds" in grid
    assert "floor_z" in grid
    assert "colors.InsertNextTuple3" in grid
    assert "background_rgb + (line_rgb - background_rgb) * blend" in grid
    assert "fade = 1.0 if orthographic else radial_grid_fade(radius, outer_fade_radius)" in grid
    assert "SetColorModeToDirectScalars" in init
    assert "_refresh_viewport_grid_geometry_if_needed" in init
    assert "self._invert_renderer.AddViewProp(self._navigation_help)" in init

def test_viewport_ground_grid_is_not_rebuilt_per_frame_and_uses_bounded_geometry():
    backend = _source("vtk_backend.py")
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    grid = _method("vtk_backend.py", "GLViewWidget", "_update_viewport_grid_geometry")
    signature = _method("vtk_backend.py", "GLViewWidget", "_viewport_grid_state_signature")
    schedule = _method("vtk_backend.py", "GLViewWidget", "_schedule_viewport_grid_refresh")
    resize = _method("vtk_backend.py", "GLViewWidget", "resizeEvent")
    assert "self._renderer.AddObserver(" not in init
    assert "_viewport_grid_refresh_timer.setSingleShot(True)" in init
    assert "_viewport_grid_refresh_timer.setInterval(90)" in init
    assert "self._viewport_grid_refresh_timer.timeout.connect(" in init
    assert "camera.GetPosition()" not in signature
    assert "camera.GetFocalPoint()" not in signature
    assert "def _schedule_viewport_grid_refresh" in backend
    assert "timer.start()" in schedule
    assert "segment_count = 48" in grid
    assert "outer_fade_radius" in grid
    assert "radial_grid_fade" in grid
    assert "distance *" not in grid
    assert "_update_viewport_grid_geometry" not in resize
    assert "_update_viewport_grid_geometry" not in _method("vtk_backend.py", "GLViewWidget", "_set_camera_orbit")
    assert "_update_viewport_grid_geometry" not in _method("vtk_backend.py", "GLViewWidget", "_pan_camera")
    assert "_schedule_viewport_grid_refresh()" in _method("vtk_backend.py", "GLViewWidget", "addItem")
    assert "_schedule_viewport_grid_refresh()" in _method("vtk_backend.py", "GLViewWidget", "removeItem")
    assert "_schedule_viewport_grid_refresh()" in _method("vtk_backend.py", "_VTKItem", "setVisible")

def test_orthographic_projection_axis_and_grid_are_wired_to_viewport_controls():
    main = _source("main.py")
    backend = _source("vtk_backend.py")
    controls = _method("main.py", "MainWindow", "_on_viewport_projection_changed")
    axis_changed = _method("main.py", "MainWindow", "_on_viewport_projection_axis_changed")
    projection = _method("vtk_backend.py", "GLViewWidget", "set_projection_mode")
    axis = _method("vtk_backend.py", "GLViewWidget", "set_orthographic_axis")
    assert 'CollapsibleSection("VIEWPORT PROJECTION"' in main
    assert 'self.projection_mode_combo.addItem("Orthographic", "orthographic")' in main
    assert 'for axis_name in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):' in main
    assert 'self.projection_axis_combo.addItem(f"{axis_name}-Axis", axis_name)' in main
    assert 'self.scene_view.set_projection_mode(mode)' in controls
    assert 'self.scene_view.set_orthographic_axis(axis)' in axis_changed
    assert 'camera.SetParallelProjection(True)' in projection
    assert 'camera.SetParallelProjection(False)' in projection
    assert 'camera.SetPosition(*(focal + direction * max(distance, 1e-4)))' in axis
    assert 'axis_sign = -1 if normalized.startswith("-") else 1' in axis
    assert '"-X": 180.0' in axis and '"-Y": 270.0' in axis
    assert 'plane_offset = (normal_min - plane_gap) if axis_sign > 0 else (normal_max + plane_gap)' in _method("vtk_backend.py", "GLViewWidget", "_update_viewport_grid_geometry")
    assert 'projection = (self._projection_mode, self._orthographic_axis)' in backend

def test_camera_navigation_stops_on_release_and_ignores_synthetic_cursor_warp_events():
    backend = _source("vtk_backend.py")
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    begin = _method("vtk_backend.py", "GLViewWidget", "_begin_mouse_wrap")
    warp = _method("vtk_backend.py", "GLViewWidget", "_warp_mouse_to_anchor")
    end = _method("vtk_backend.py", "GLViewWidget", "_end_mouse_wrap")
    stop = _method("vtk_backend.py", "GLViewWidget", "_stop_pointer_navigation")
    move = _method("vtk_backend.py", "GLViewWidget", "mouseMoveEvent")
    release = _method("vtk_backend.py", "GLViewWidget", "mouseReleaseEvent")
    smooth = _method("vtk_backend.py", "GLViewWidget", "_step_smooth_look")

    assert "self._mouse_warp_pending = False" in init
    assert "current != self._mouse_wrap_anchor_global" in begin
    assert "self._mouse_warp_pending = True" in warp
    assert "self._mouse_warp_pending = False" in end
    assert "event.buttons()" in move
    assert "QApplication.mouseButtons()" in move
    assert move.index("global_pos == self._mouse_wrap_anchor_global") < move.index("held_buttons = event.buttons()")
    assert "if not (held_buttons & required_button)" in move
    assert "warp_global_pos == self._mouse_wrap_anchor_global" in move
    assert "self._mouse_warp_pending = False" in move
    assert "np.clip(dx, -120.0, 120.0)" in move
    assert "np.clip(dy, -120.0, 120.0)" in move
    assert "self._cancel_smooth_look()" in stop
    assert "self._cancel_camera_motion()" in stop
    assert 'if not self._right_mouse_down:' in smooth
    assert "self._cancel_smooth_look()" in smooth
    assert "self._stop_pointer_navigation(cancel_camera_motion=True)" in release
    assert "easing tail settle after release" not in backend

def test_remote_desktop_camera_navigation_disables_cursor_warp_feedback():
    backend = _source("vtk_backend.py")
    init = _method("vtk_backend.py", "GLViewWidget", "__init__")
    detect = _method("vtk_backend.py", "GLViewWidget", "_detect_remote_input_session")
    refresh = _method("vtk_backend.py", "GLViewWidget", "_refresh_remote_input_session")
    begin = _method("vtk_backend.py", "GLViewWidget", "_begin_mouse_wrap")
    press = _method("vtk_backend.py", "GLViewWidget", "mousePressEvent")
    move = _method("vtk_backend.py", "GLViewWidget", "mouseMoveEvent")
    focus_out = _method("vtk_backend.py", "GLViewWidget", "focusOutEvent")

    assert "self._remote_input_session = self._detect_remote_input_session()" in init
    assert 'GetSystemMetrics(0x1000)' in detect
    assert '"remoting_host.exe"' in detect
    assert '"remoting_desktop.exe"' in detect
    assert 'POLYFITTY_MOUSE_INPUT_MODE' in detect
    assert "if self._remote_input_session:" in begin
    assert "self._mouse_wrap_active = False" in begin
    assert "QtGui.QCursor.setPos" not in begin.split("if self._remote_input_session:", 1)[1].split("current = QtGui.QCursor.pos()", 1)[0]
    assert "if not self._remote_input_session:" in move
    assert "QApplication.mouseButtons()" in move
    assert "self._stop_pointer_navigation(cancel_camera_motion=True)" in focus_out
