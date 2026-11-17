from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_ui_exposes_only_maintained_reconstruction_engines():
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "addItem" and len(node.args) >= 2:
                value = node.args[1]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    values.append(value.value)
    assert {"auto", "polyfit", "li_nan"}.issubset(set(values))
    assert not ({"rough_qem", "blender"} & set(values))

def test_pipeline_rejects_retired_mode_keys():
    source = (ROOT / "app" / "pipeline.py").read_text(encoding="utf-8")
    assert 'if requested not in {"auto", "polyfit", "li_nan", "linan", "feature_preserving"}' in source
    assert "rough_qem" not in source.lower()
    assert "blender" not in source.lower()
