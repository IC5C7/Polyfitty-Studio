from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def _class_method(path: str, class_name: str, method_name: str):
    source = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return source, next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )

def _run_bounds(points):
    _source, method = _class_method("app/view3d.py", "SceneView", "_bounds")
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
    exec(compile(module, "view3d_bounds_test", "exec"), namespace)
    return namespace["_bounds"](points)

def test_scene_bounds_returns_min_max_for_mesh_vertices():
    vertices = np.asarray(((-2, 4, 1), (3, -1, 8), (0, 2, 3)), dtype=np.float32)
    bounds = _run_bounds(vertices)
    assert bounds is not None
    np.testing.assert_array_equal(bounds[0], (-2, -1, 1))
    np.testing.assert_array_equal(bounds[1], (3, 4, 8))

def test_scene_bounds_ignores_nonfinite_rows_and_handles_empty_inputs():
    vertices = np.asarray(((1, 2, 3), (np.nan, 0, 0), (np.inf, 1, 2)), dtype=np.float64)
    bounds = _run_bounds(vertices)
    assert bounds is not None
    np.testing.assert_array_equal(bounds[0], (1, 2, 3))
    np.testing.assert_array_equal(bounds[1], (1, 2, 3))
    assert _run_bounds(np.empty((0, 3))) is None
    assert _run_bounds(None) is None
    assert _run_bounds(np.asarray((1, 2))) is None

def test_source_and_baked_result_paths_use_the_restored_bounds_helper():
    view_source = (ROOT / "app/view3d.py").read_text(encoding="utf-8")
    assert "self._source_bounds = self._bounds(points)" in view_source
    assert "self._source_bounds = self._bounds(self._source_vertices)" in view_source
    assert "self._result_bounds = self._bounds(vertices)" in view_source
    assert "def _fit_union_camera(self, *, animate: bool = False):" in view_source

    main_source = (ROOT / "app/main.py").read_text(encoding="utf-8")
    assert "def _load_baked_glb_result" in main_source
    assert "self.scene_view.set_reconstruction_mesh(" in main_source

def test_unneeded_scene_toolbar_labels_removed_and_face_overlay_docked_below_controls():
    main_source = (ROOT / "app/main.py").read_text(encoding="utf-8")
    assert 'QLabel("SOURCE / RECONSTRUCTION")' not in main_source
    assert "self.navigation_hint" not in main_source
    assert "scene_toolbar" not in main_source
    marker = "controls_layout.addWidget(self.vtk_engine_label)"
    overlay = "controls_layout.addWidget(self.face_overlay_toggle, 0, Qt.AlignRight)"
    assert marker in main_source and overlay in main_source
    assert main_source.index(marker) < main_source.index(overlay)

def test_face_overlay_is_enabled_before_automatic_bake_worker_starts():
    main_source = (ROOT / "app/main.py").read_text(encoding="utf-8")
    start = main_source.index("    def _start_automatic_glb_bake(")
    end = main_source.index("    def _on_bake_preview(", start)
    handler = main_source[start:end]
    enable = "self.face_overlay_toggle.setChecked(True)"
    worker_start = "self._export_worker.start()"
    assert enable in handler
    assert worker_start in handler
    assert handler.index(enable) < handler.index(worker_start)
    assert "self.face_overlay_toggle.toggled.connect(self._set_face_overlay_enabled)" in main_source
