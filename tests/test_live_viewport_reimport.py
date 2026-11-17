from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from app.wireframe import build_polygon_wireframe_positions

ROOT = Path(__file__).resolve().parents[1]

def _method_source(file_name: str, class_name: str, method_name: str) -> str:
    source = (ROOT / "app" / file_name).read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""

def test_live_draw_uses_polygon_edges_and_displays_them_by_default():
    source = _method_source("view3d.py", "SceneView", "set_processing_mesh")
    assert "_polygon_wireframe_positions(vertices, triangles, quad_faces)" in source
    assert "edge_item.setVisible(True)" in source
    assert "edge_item.setVisible(False)" not in source
    assert "GLMeshItem(" not in source
    assert "vertexColors" not in source
    assert "width=" not in source
    assert "Draw only original polygon edges" in source

def test_processing_overlay_cleanup_removes_all_transient_items_from_renderer_registry():
    source = _method_source("view3d.py", "SceneView", "clear_processing_mesh")
    assert "if existing is not removed" in source
    assert "if existing is not highlight" in source

def test_reimport_loads_the_current_original_input_without_opening_a_picker():
    source = _method_source("main.py", "MainWindow", "_reimport_source")
    assert "self.input_edit.text().strip()" in source
    assert 'action="reimport"' in source
    assert "QFileDialog.getOpenFileName" not in source
    loader = _method_source("main.py", "MainWindow", "_load_input_source")
    assert "AssetImportWorker(" in loader
    assert "self._show_asset_import_progress(p)" in loader
    assert "self._asset_import_worker.start()" in loader
    source_loader = _method_source("main.py", "MainWindow", "_load_source_preview_asset")
    assert "alpha_mode_override=mode_override" in source_loader
    assert "alpha_cutoff_override=cutoff" in source_loader
    commit = _method_source("main.py", "MainWindow", "_finish_input_source_load")
    assert "self.scene_view.clear_result()" in commit
    assert "self.scene_view.clear_processing_mesh()" in commit
    assert "generated reconstruction and bake previews were cleared" in commit

def test_reimport_button_is_wired_and_disabled_while_source_controls_are_locked():
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert 'QPushButton("↻ REIMPORT")' in source
    assert "self.reimport_btn.clicked.connect(self._reimport_source)" in source
    helper = _method_source("main.py", "MainWindow", "_set_source_file_controls_enabled")
    assert "self.reimport_btn.setEnabled(state and has_file)" in helper

def test_quad_live_draw_shows_only_real_quad_perimeter_not_shading_diagonal():
    vertices = np.asarray(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=np.float32
    )
    quads = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    edges = build_polygon_wireframe_positions(
        vertices, np.empty((0, 3), dtype=np.int64), quads
    )
    assert edges.shape == (8, 3)

    pairs = edges.reshape(-1, 2, 3)
    diagonal = {tuple(vertices[0]), tuple(vertices[2])}
    assert not any({tuple(pair[0]), tuple(pair[1])} == diagonal for pair in pairs)
