from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
INSTALLER = ROOT / "installer"

def test_main_exposes_a_versioned_pbr_export_dialog_contract():
    main = (APP / "main.py").read_text(encoding="utf-8-sig")
    assert "PORTABLE_PBR_EXPORT_UI_VERSION = 2" in main
    assert 'QGroupBox("PBR TEXTURES TO EXPORT")' in main
    assert "self.pbr_map_checks" in main
    assert '"pbr_maps"' in main
    assert "supported_pbr_maps_for_format" in main

def test_shared_runtime_manifest_tracks_pbr_and_viewport_features_from_the_main_source():
    main = (APP / "main.py").read_text(encoding="utf-8-sig")
    contract = (APP / "runtime_contract.py").read_text(encoding="utf-8-sig")
    verifier = (INSTALLER / "verify_runtime.py").read_text(encoding="utf-8-sig")
    assert "PORTABLE_PBR_EXPORT_UI_VERSION = 2" in main
    assert "PORTABLE_VIEWPORT_COORDINATE_VERSION = 18" in main
    assert "PORTABLE_VIEWPORT_COLOR_MANAGEMENT_VERSION = 2" in main
    assert "PORTABLE_VIEWPORT_MATERIAL_PARITY_VERSION = 3" in main
    assert '"pbr_export_ui": "PORTABLE_PBR_EXPORT_UI_VERSION"' in contract
    assert '"material_preview_parity": "PORTABLE_VIEWPORT_MATERIAL_PARITY_VERSION"' in contract
    assert "verify_runtime_manifest(require_source_match=True)" in verifier

def test_builder_packages_same_app_source_with_an_independent_embedded_runtime():
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        installer = (INSTALLER / "install.ps1").read_text(encoding="utf-8-sig")
        assert "robocopy" in builder.lower()
        assert "$MainRuntime" not in builder
        assert "runtime_source_manifest.py" in builder
        assert "'-File', $portableInstaller, '-PortableBuild'" in builder
        assert "python-3.12.10-embed-amd64.zip" in installer
        assert "requirements.txt" in installer
        assert "verify_runtime.py" in builder
        assert "PolyfittyStudio.Payload.zip" in builder
        assert "GetManifestResourceStream('PolyfittyStudio.Payload.zip')" in builder
        assert "ExtractToDirectory($PayloadVerifyZip" in builder
        assert "The portable output must contain exactly one distributable file" in builder
        assert "PolyfittyStudio.Runtime.exe" not in builder
        assert "PyInstaller" not in builder

def test_main_source_contains_shared_viewport_and_coordinate_frame_features():
    main = (APP / "main.py").read_text(encoding="utf-8-sig")
    view = (APP / "view3d.py").read_text(encoding="utf-8-sig")
    backend = (APP / "vtk_backend.py").read_text(encoding="utf-8-sig")
    coordinate = (APP / "coordinate_frame.py").read_text(encoding="utf-8-sig")
    assert "PORTABLE_VIEWPORT_COORDINATE_VERSION = 18" in main
    assert "def eventFilter" in main and "QEvent.Type.DragEnter" in main and "QEvent.Type.Drop" in main
    assert "def dragEnterEvent" in main and "def dropEvent" in main
    assert "def _export_reconstruction_geometry" in main
    assert "source_frame_transform" in main
    assert "def set_rotation_display_mode" in view
    assert "def set_item_info_enabled" in view and "def set_item_info_mode" in view
    assert "ViewportItemInfoRotationRow" in main
    assert "[ELR]" in main and "[QTN]" in main
    assert "vtkOrientationMarkerWidget()" in backend
    assert "def _initialize_orientation_gizmo" in backend
    assert "source_frame_oriented_axis_edge_candidates" in coordinate
    assert "select_camera_visible_axis_edge" in coordinate
    assert "def source_frame_camera_pose" in coordinate

def test_shared_runtime_source_contains_the_rounded_radial_grid_fade():
    main = (APP / "main.py").read_text(encoding="utf-8-sig")
    backend = (APP / "vtk_backend.py").read_text(encoding="utf-8-sig")
    fade = (APP / "viewport_grid.py").read_text(encoding="utf-8-sig")
    assert "PORTABLE_VIEWPORT_COORDINATE_VERSION = 18" in main
    assert "radial_grid_fade" in backend
    assert "segment_count = 48" in backend
    assert "outer_fade_radius = half_extent * 0.96" in backend
    assert "def radial_grid_fade" in fade
    assert "fade_start_fraction: float = 0.58" in fade

def test_shared_runtime_pbr_color_management_and_orm_are_wired_into_the_main_viewport():
    main = (APP / "main.py").read_text(encoding="utf-8-sig")
    backend = (APP / "vtk_backend.py").read_text(encoding="utf-8-sig")
    geometry = (APP / "vtk_geometry.py").read_text(encoding="utf-8-sig")
    pointio = (APP / "pointio.py").read_text(encoding="utf-8-sig")
    view = (APP / "view3d.py").read_text(encoding="utf-8-sig")
    assert "PORTABLE_VIEWPORT_COLOR_MANAGEMENT_VERSION = 2" in main
    assert "PORTABLE_VIEWPORT_MATERIAL_PARITY_VERSION = 3" in main
    assert "PORTABLE_ASSET_IMPORT_PROGRESS_VERSION = 1" in main
    assert 'checkbox.setChecked(key in {"base_color", "orm"} and key in available)' in main
    assert "Always baked automatically to preserve per-material AO, Roughness and Metallic values." in main
    assert "vertex_colors_for_render_pass" in backend
    assert "configure_studio_environment(self._renderer)" in backend
    assert "def srgb_to_linear_colors(" in geometry
    assert "def create_studio_environment_texture(" in geometry
    assert "def apply_studio_environment_intensity(" in geometry
    assert "def normalize_vertex_normals(" in geometry
    assert "_PREVIEW_NORMALS_METADATA_KEY" in pointio
    assert "def _material_pbr_factors(" in pointio
    assert "item._polyfitty_pbr_roughness" in view
    assert "ormTextureImage=orm_texture_image" in view
    assert "vertexNormals=expanded_normals" in view
    assert "prop.SetORMTexture(texture)" in backend
    assert "SetOcclusionStrength" in backend

def test_source_frame_helper_is_in_its_declared_module_and_not_mischecked_by_builder():
        coordinate = (APP / "coordinate_frame.py").read_text(encoding="utf-8-sig")
        builder = (INSTALLER / "build_portable.ps1").read_text(encoding="utf-8-sig")
        assert "def source_frame_camera_pose" in coordinate
        assert "source_frame_camera_pose" not in builder
        assert "runtime_source_manifest.py" in builder
        assert "Release source file does not match runtime_manifest.json" in builder
