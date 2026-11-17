from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _source(name: str) -> str:
    return (ROOT / "app" / name).read_text(encoding="utf-8")

def _method(source: str, class_name: str, method_name: str) -> str:
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""

def _function(source: str, function_name: str) -> str:
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    return ast.get_source_segment(source, function) or ""

def test_worker_progress_protocol_escapes_unicode_for_windows_pipes():
    worker = _source("worker.py")
    emit = _function(worker, "emit")
    assert "ensure_ascii=True" in emit
    assert "ensure_ascii=False" not in emit

    message = "Li & Nan edge collapse 6,253/6,457 · 680 vertices"
    payload = json.dumps({"message": message}, ensure_ascii=True)
    assert payload.isascii()
    assert json.loads(payload)["message"] == message

def test_mesh_dimension_labels_keep_constant_pixel_font_and_transform_hud_safe_inset():
    view = _source("view3d.py")
    init = _method(view, "SceneView", "__init__")
    position = _method(view, "SceneView", "_position_dimension_hud")
    origin_anchor = _method(view, "SceneView", "_update_transform_hud_position")
    project = _method(view, "SceneView", "_update_dimension_edge_label_positions")
    assert 'for axis in "XYZ"' in init
    assert "text_property.SetFontSize(16)" in init
    assert "SetDisplayPosition(x, y)" in project
    assert "self._update_transform_hud_position()" in position
    assert '[:3, 3]' in origin_anchor
    assert "renderer.WorldToDisplay()" in origin_anchor
    assert "projected[1])) - 18" in origin_anchor
    assert "actor.SetDisplayPosition(x, y)" in origin_anchor

def test_bottom_navigation_help_font_is_larger_and_keeps_bottom_inset():
    backend = _source("vtk_backend.py")
    init = _method(backend, "GLViewWidget", "__init__")
    position = _method(backend, "GLViewWidget", "_position_navigation_help")
    assert "help_text_property.SetFontSize(15)" in init
    assert "SetDisplayPosition(max(1, int(width) // 2), 16)" in position
