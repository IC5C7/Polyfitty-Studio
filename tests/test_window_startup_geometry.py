from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"

def _method_source(class_name: str, method_name: str) -> str:
    source = (APP / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""

def _load_method(method_name: str):
    method = ast.parse(_method_source("MainWindow", method_name)).body[0]
    namespace = {"startup_trace": lambda *_args, **_kwargs: None}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "main.py", "exec"), namespace)
    return namespace[method_name]

class FakeSettings:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.synced = False

    def value(self, key, default=None, type=None):
        value = self.values.get(key, default)
        if type is bool:
            return bool(value)
        return value

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        self.synced = True

class FakeWindow:
    def __init__(self, settings, restore_result=True, maximized=False):
        self.settings = settings
        self.restore_result = restore_result
        self.maximized = maximized
        self.restored_geometry = None
        self.saved_geometry = b"saved-normal-geometry"

    def restoreGeometry(self, geometry):
        self.restored_geometry = geometry
        return self.restore_result

    def saveGeometry(self):
        return self.saved_geometry

    def isMaximized(self):
        return self.maximized

def test_first_runtime_entry_starts_maximized_when_no_previous_window_state_exists():
    restore = _load_method("_restore_startup_window_geometry")
    settings = FakeSettings()
    window = FakeWindow(settings, restore_result=False)
    restore(window)
    assert window._window_geometry_restored is False
    assert window._startup_window_maximized is True

def test_saved_normal_geometry_restores_to_previous_window_size_on_next_launch():
    restore = _load_method("_restore_startup_window_geometry")
    settings = FakeSettings({
        "window/geometry": b"previous-normal-geometry",
        "window/first_startup_maximize_done_v1": True,
        "window/maximized": False,
    })
    window = FakeWindow(settings, restore_result=True)
    restore(window)
    assert window.restored_geometry == b"previous-normal-geometry"
    assert window._window_geometry_restored is True
    assert window._startup_window_maximized is False

def test_saved_maximized_state_is_restored_on_next_launch():
    restore = _load_method("_restore_startup_window_geometry")
    settings = FakeSettings({
        "window/geometry": b"previous-normal-geometry",
        "window/first_startup_maximize_done_v1": True,
        "window/maximized": True,
    })
    window = FakeWindow(settings, restore_result=True)
    restore(window)
    assert window._startup_window_maximized is True

def test_window_close_persists_geometry_and_maximized_state():
    save = _load_method("_save_window_geometry")
    settings = FakeSettings()
    window = FakeWindow(settings, maximized=False)
    save(window)
    assert settings.values["window/geometry"] == b"saved-normal-geometry"
    assert settings.values["window/maximized"] is False
    assert settings.values["window/first_startup_maximize_done_v1"] is True
    assert settings.synced is True

def test_main_window_geometry_is_shared_without_a_portable_specific_implementation():
        main = (APP / "main.py").read_text(encoding="utf-8")
        entry = (APP / "hidden_entry.py").read_text(encoding="utf-8")
        portable_shim = (APP / "portable_entry.py").read_text(encoding="utf-8")
        builder = (ROOT / "installer" / "build_portable.ps1").read_text(encoding="utf-8-sig")
        assert 'if bool(getattr(window, "_startup_window_maximized", False))' in main
        assert "window.showMaximized()" in main
        assert "window.showNormal()" in main
        assert "PORTABLE_VIEWPORT_COORDINATE_VERSION = 18" in main
        assert "from main import main as gui_main" in entry
        assert "from hidden_entry import main" in portable_shim
        assert "gui_main_symbols" not in portable_shim
        assert "PORTABLE_VIEWPORT_COORDINATE_VERSION) < 18" not in portable_shim
        assert "'-File', $portableInstaller, '-PortableBuild'" in builder
        assert "$MainRuntime" not in builder
        assert "window.showMaximized()" not in builder and "window.showNormal()" not in builder

def test_close_event_saves_window_geometry_before_accepting_close():
    source = (APP / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "closeEvent")
    accept_calls = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "accept"
    ]
    assert len(accept_calls) == 2
    for call in accept_calls:
        parent_save = [
            node for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_save_window_geometry"
            and node.lineno < call.lineno
        ]
        assert parent_save, "Every accepted close path must save window geometry first."
