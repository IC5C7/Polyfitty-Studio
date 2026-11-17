from __future__ import annotations

import ast
import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"

def _method_source(file_name: str, class_name: str, method_name: str) -> str:
    source = (APP / file_name).read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""

def test_item_info_starts_disabled_and_existing_settings_are_migrated_once():
    view = (APP / "view3d.py").read_text(encoding="utf-8")
    main = (APP / "main.py").read_text(encoding="utf-8")
    assert "self._item_info_enabled = False" in view
    assert 'self.settings.value("viewport/item_info_enabled", False, type=bool)' in main
    assert '"viewport/item_info_default_off_v1"' in main
    assert 'self.settings.setValue("viewport/item_info_enabled", False)' in main
    assert 'self.settings.setValue("viewport/item_info_default_off_v1", True)' in main

def test_item_info_uses_a_visible_checked_or_unchecked_checkbox_icon():
    main = (APP / "main.py").read_text(encoding="utf-8")
    entry = (APP / "main.py").read_text(encoding="utf-8")
    assert "self.item_info_enabled_check = QCheckBox()" in main
    assert "self.item_info_enabled_check.setAccessibleName(\"ITEM INFO\")" in main
    assert "☑  ITEM INFO" in main and "☐  ITEM INFO" in main
    assert "self.item_info_enabled_check.toggled.connect(self._sync_item_info_checkbox_icon)" in main
    assert "QCheckBox#ViewportItemInfoEnabled::indicator { width: 0px; height: 0px; margin: 0px; }" in main
    assert "_sync_item_info_checkbox_icon" in entry

def test_item_info_checkbox_icon_tracks_checked_state():
    source = (APP / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    method = next(
        node for node in main_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sync_item_info_checkbox_icon"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "main.py", "exec"), namespace)

    class FakeCheckbox:
        text = ""

        def setText(self, value):
            self.text = value

    checkbox = FakeCheckbox()
    instance = SimpleNamespace(item_info_enabled_check=checkbox)
    namespace["_sync_item_info_checkbox_icon"](instance, False)
    assert checkbox.text == "☐  ITEM INFO"
    namespace["_sync_item_info_checkbox_icon"](instance, True)
    assert checkbox.text == "☑  ITEM INFO"

def test_empty_scene_prompt_is_a_centered_non_pickable_vtk_overlay():
    view = (APP / "view3d.py").read_text(encoding="utf-8")
    assert 'self.empty_scene_prompt = vtk.vtkTextActor()' in view
    assert '"DRAG & DROP A 3D FILE HERE\\nOR CLICK TO BROWSE INPUT"' in view
    assert 'self.empty_scene_prompt.SetPickable(False)' in view
    assert 'self._overlay_renderer.AddViewProp(self.empty_scene_prompt)' in view
    assert 'empty_prompt_property.SetFontSize(19)' in view
    position = _method_source("view3d.py", "SceneView", "_position_empty_scene_prompt")
    assert "int(width) // 2" in position and "int(height) // 2" in position
    assert "SetDisplayPosition" in position

def test_empty_viewport_click_opens_the_same_browse_input_dialog():
    view = _method_source("view3d.py", "SceneView", "_handle_viewport_click")
    main = _method_source("main.py", "MainWindow", "_build_ui")
    assert "if self._empty_scene_prompt_visible" in view
    assert "self.emptyViewportClicked.emit()" in view
    assert view.index("self.emptyViewportClicked.emit()") < view.index("self.pick_actor_at(x, y)")
    assert "self.scene_view.emptyViewportClicked.connect(self._browse_input)" in main
    assert "self.browse_btn.clicked.connect(self._browse_input)" in (APP / "main.py").read_text(encoding="utf-8")

def test_import_hides_onboarding_prompt_only_after_a_source_is_loaded():
    load = _method_source("main.py", "MainWindow", "_finish_input_source_load")
    assert "self.scene_view.set_empty_scene_prompt_visible(False)" in load
    assert load.index("self.scene_view.set_empty_scene_prompt_visible(False)") > load.index("self._render_source_asset()")
    worker = (APP / "main.py").read_text(encoding="utf-8")
    assert "class AssetImportWorker(QThread)" in worker
    assert "QProgressDialog" in worker
    assert "progress_callback: Optional[Callable[[int, str], None]] = None" in (APP / "pointio.py").read_text(encoding="utf-8")
    assert "_progress(100" in (APP / "pointio.py").read_text(encoding="utf-8")

def test_portable_builder_requires_the_same_onboarding_and_default_off_behavior():
        main = (APP / "main.py").read_text(encoding="utf-8")
        entry = (APP / "main.py").read_text(encoding="utf-8")
        builder = (ROOT / "installer" / "build_portable.ps1").read_text(encoding="utf-8")
        assert "PORTABLE_VIEWPORT_COORDINATE_VERSION = 18" in main
        assert "emptyViewportClicked" in entry and "_browse_input" in entry
        assert "_empty_scene_prompt_visible" in entry
        assert "viewport/item_info_default_off_v1" in entry
        assert "robocopy" in builder.lower()
        assert "$MainRuntime" not in builder
        assert "Preparing an isolated build image" in builder
        assert "'-File', $portableInstaller, '-PortableBuild'" in builder
        assert "runtime_source_manifest.py" in builder
        assert "'.runtime'" in builder

def test_location_and_dimension_units_default_to_meters_once_and_preserve_later_choices():
    source = (APP / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    method = next(
        node for node in main_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_default_metric_units"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "main.py", "exec"), namespace)

    class Settings:
        def __init__(self, values):
            self.values = dict(values)
            self.synced = False
        def value(self, key, default=None, type=None):
            value = self.values.get(key, default)
            return bool(value) if type is bool else value
        def setValue(self, key, value):
            self.values[key] = value
        def sync(self):
            self.synced = True

    instance = SimpleNamespace(settings=Settings({
        "viewport/location_display_unit": "cm",
        "viewport/dimension_display_unit": "km",
    }))
    namespace["_ensure_default_metric_units"](instance)
    assert instance.settings.values["viewport/location_display_unit"] == "m"
    assert instance.settings.values["viewport/dimension_display_unit"] == "m"
    assert instance.settings.values["viewport/metric_units_default_m_v1"] is True
    assert instance.settings.synced is True

    instance.settings.values["viewport/location_display_unit"] = "cm"
    instance.settings.values["viewport/dimension_display_unit"] = "mm"
    namespace["_ensure_default_metric_units"](instance)
    assert instance.settings.values["viewport/location_display_unit"] == "cm"
    assert instance.settings.values["viewport/dimension_display_unit"] == "mm"

def test_portable_runtime_requires_meter_defaults_for_both_length_readouts():
        main = (APP / "main.py").read_text(encoding="utf-8")
        entry = (APP / "main.py").read_text(encoding="utf-8")
        builder = (ROOT / "installer" / "build_portable.ps1").read_text(encoding="utf-8-sig")
        installer = (ROOT / "installer" / "install.ps1").read_text(encoding="utf-8-sig")
        assert "PORTABLE_VIEWPORT_COORDINATE_VERSION = 18" in main
        assert 'self.settings.setValue("viewport/location_display_unit", "m")' in main
        assert 'self.settings.setValue("viewport/dimension_display_unit", "m")' in main
        assert "viewport/metric_units_default_m_v1" in entry
        assert "runtime_source_manifest.py" in builder
        assert "Preparing an isolated build image" in builder
        assert "python-3.12.10-embed-amd64.zip" in installer
        guide = (ROOT / "README.md").read_text(encoding="utf-8-sig")
        assert "CPython 3.12.10" in guide
        assert "same application source" in guide
