from __future__ import annotations

"""Shared application-source identity contract for installed and portable builds.

Both products execute the same versioned application source tree and feature
contracts. The installed product provisions its private runtime during setup;
the portable product embeds an independent CPython runtime and core packages
inside its distribution. Runtime environments may differ by packaging mode,
but application source and feature definitions must match exactly.
"""

import hashlib
import json
import re
from pathlib import Path

RUNTIME_CONTRACT_VERSION = 1
RUNTIME_MANIFEST_FILENAME = "runtime_manifest.json"

_FEATURE_CONSTANTS = {
    "pbr_export_ui": "PORTABLE_PBR_EXPORT_UI_VERSION",
    "viewport_coordinate": "PORTABLE_VIEWPORT_COORDINATE_VERSION",
    "viewport_color_management": "PORTABLE_VIEWPORT_COLOR_MANAGEMENT_VERSION",
    "material_preview_parity": "PORTABLE_VIEWPORT_MATERIAL_PARITY_VERSION",
    "asset_import_progress": "PORTABLE_ASSET_IMPORT_PROGRESS_VERSION",
    "viewport_mouse_navigation": "PORTABLE_VIEWPORT_MOUSE_NAVIGATION_VERSION",
    "gpu_startup_preflight": "PORTABLE_GPU_STARTUP_PREFLIGHT_VERSION",
    "gpu_first_run_setup": "PORTABLE_GPU_FIRST_RUN_SETUP_VERSION",
}

_ROOT_RUNTIME_FILES = {
    "VERSION.txt",
    "LICENSE",
    "THIRD_PARTY_NOTICES.txt",
    "START.bat",
    "START_HIDDEN.vbs",
    "BUILD_PORTABLE_EXE.bat",
    "BUILD_PORTABLE_EXE.vbs",
    "INSTALL.bat",
    "INSTALL.vbs",
    "UNINSTALL.bat",
    "CREATE_PORTABLE_SHORTCUT.vbs",
}

_SOURCE_TREE_EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

def source_project_root() -> Path:
    """Return the project root when running from the normal Python source tree."""
    return Path(__file__).resolve().parent.parent

def runtime_bundle_root() -> Path:
    """Return the application source root inside the active product package."""
    return source_project_root()

def _relative_runtime_inputs(root: Path) -> list[Path]:
    """List the code, assets, requirements and launch/build files that define a release."""
    candidates: set[Path] = set()

    for folder_name in ("app", "installer", "packaging", "native"):
        folder = root / folder_name
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() == ".pyc":
                continue
            relative_parts = path.relative_to(root).parts
            if any(part in _SOURCE_TREE_EXCLUDED_PARTS for part in relative_parts):
                continue
            candidates.add(path)

    for path in root.glob("requirements*.txt"):
        if path.is_file():
            candidates.add(path)

    for relative in _ROOT_RUNTIME_FILES:
        path = root / relative
        if path.is_file():
            candidates.add(path)

    return sorted(candidates, key=lambda item: item.relative_to(root).as_posix().lower())

def _read_feature_versions(root: Path) -> dict[str, int]:
    main_path = root / "app" / "main.py"
    if not main_path.is_file():
        raise RuntimeError(f"The shared application entry module is missing: {main_path}")
    source = main_path.read_text(encoding="utf-8-sig")
    result: dict[str, int] = {}
    for key, constant in _FEATURE_CONSTANTS.items():
        match = re.search(rf"(?m)^\s*{re.escape(constant)}\s*=\s*(\d+)\s*$", source)
        if match is None:
            raise RuntimeError(f"The application is missing its runtime feature contract: {constant}")
        result[key] = int(match.group(1))
    return result

def _read_app_version(root: Path) -> str:
    version_path = root / "VERSION.txt"
    if not version_path.is_file():
        raise RuntimeError(f"The release version file is missing: {version_path}")
    version = version_path.read_text(encoding="utf-8-sig").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise RuntimeError(f"Invalid Polyfitty Studio version in {version_path}: {version!r}")

    main_source = (root / "app" / "main.py").read_text(encoding="utf-8-sig")
    app_version_match = re.search(r"(?m)^\s*APP_VERSION\s*=\s*['\"]([^'\"]+)['\"]\s*$", main_source)
    if app_version_match is None:
        raise RuntimeError("app/main.py is missing the APP_VERSION release identity.")
    if app_version_match.group(1).strip() != version:
        raise RuntimeError(
            f"Application version mismatch: VERSION.txt is {version}, "
            f"but app/main.py declares {app_version_match.group(1).strip()}."
        )
    return version

def _fingerprint_file_hashes(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()

def compute_runtime_manifest(root: str | Path) -> dict:
    """Calculate a stable SHA-256 identity for all runtime and build inputs."""
    root_path = Path(root).resolve()
    version = _read_app_version(root_path)
    files: dict[str, str] = {}
    for path in _relative_runtime_inputs(root_path):
        relative = path.relative_to(root_path).as_posix()
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "schema_version": 1,
        "app_version": version,
        "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
        "source_fingerprint": _fingerprint_file_hashes(files),
        "source_file_count": len(files),
        "feature_versions": _read_feature_versions(root_path),
        "files": files,
    }

def write_runtime_manifest(root: str | Path) -> Path:
    root_path = Path(root).resolve()
    manifest = compute_runtime_manifest(root_path)
    destination = root_path / RUNTIME_MANIFEST_FILENAME
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination

def verify_source_runtime_manifest(root: str | Path) -> dict:
    """Verify that the extracted source project exactly matches its release manifest."""
    root_path = Path(root).resolve()
    manifest_path = root_path / RUNTIME_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Runtime parity manifest is missing: {manifest_path}. "
            "Extract the complete latest Polyfitty Studio project ZIP into a new folder."
        )
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"Runtime parity manifest could not be read: {manifest_path}: {exc}") from exc
    actual = compute_runtime_manifest(root_path)

    keys = (
        "schema_version", "app_version", "runtime_contract_version",
        "source_fingerprint", "source_file_count", "feature_versions", "files",
    )
    differences = [key for key in keys if expected.get(key) != actual.get(key)]
    if differences:
        expected_files = expected.get("files", {}) if isinstance(expected.get("files"), dict) else {}
        actual_files = actual.get("files", {})
        changed = sorted(
            path for path in set(expected_files) | set(actual_files)
            if expected_files.get(path) != actual_files.get(path)
        )
        details = ", ".join(changed[:12])
        if len(changed) > 12:
            details += f", … and {len(changed) - 12} more"
        raise RuntimeError(
            "This project folder does not match its release runtime manifest "
            f"(different fields: {', '.join(differences)}). "
            + (f"Changed/missing files: {details}. " if details else "")
            + "Do not mix files from different releases; extract one complete project ZIP into a new folder."
        )
    return expected

def _feature_versions_from_runtime() -> dict[str, int]:
    try:
        import main
    except ImportError:
        from . import main

    return {
        "pbr_export_ui": int(main.PORTABLE_PBR_EXPORT_UI_VERSION),
        "viewport_coordinate": int(main.PORTABLE_VIEWPORT_COORDINATE_VERSION),
        "viewport_color_management": int(main.PORTABLE_VIEWPORT_COLOR_MANAGEMENT_VERSION),
        "material_preview_parity": int(main.PORTABLE_VIEWPORT_MATERIAL_PARITY_VERSION),
        "asset_import_progress": int(main.PORTABLE_ASSET_IMPORT_PROGRESS_VERSION),
        "viewport_mouse_navigation": int(main.PORTABLE_VIEWPORT_MOUSE_NAVIGATION_VERSION),
        "gpu_startup_preflight": int(main.PORTABLE_GPU_STARTUP_PREFLIGHT_VERSION),
        "gpu_first_run_setup": int(main.PORTABLE_GPU_FIRST_RUN_SETUP_VERSION),
    }

def verify_runtime_manifest(*, require_source_match: bool | None = None) -> dict:
    """Verify the shared source release identity used by installed and portable launches.

    Both modes execute the same app/hidden_entry.py from the same project
    folder layout, so every launch recomputes the source-file hashes.
    """
    root = runtime_bundle_root()
    manifest_path = root / RUNTIME_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise RuntimeError(f"The runtime bundle is missing {RUNTIME_MANIFEST_FILENAME}: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"Could not read the runtime identity manifest: {exc}") from exc

    if int(manifest.get("schema_version", -1)) != 1:
        raise RuntimeError("This application uses an unsupported runtime manifest schema.")
    if int(manifest.get("runtime_contract_version", -1)) != RUNTIME_CONTRACT_VERSION:
        raise RuntimeError("The application and runtime manifest use different runtime contract versions.")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("The runtime manifest does not contain the shared application/build source file list.")
    if int(manifest.get("source_file_count", -1)) != len(files):
        raise RuntimeError("The runtime manifest source file count is inconsistent.")
    for relative, file_hash in files.items():
        if (
            not isinstance(relative, str)
            or relative.startswith(("/", "\\"))
            or ".." in relative.replace("\\", "/").split("/")
            or not isinstance(file_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", file_hash)
        ):
            raise RuntimeError("The runtime manifest contains an invalid source file entry.")
    expected_fingerprint = _fingerprint_file_hashes(files)
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("source_fingerprint", ""))):
        raise RuntimeError("The runtime manifest has no valid SHA-256 source fingerprint.")
    if expected_fingerprint != manifest.get("source_fingerprint"):
        raise RuntimeError("The runtime manifest file list does not match its source fingerprint.")

    version_path = root / "VERSION.txt"
    if not version_path.is_file():
        raise RuntimeError(f"The runtime bundle is missing VERSION.txt: {version_path}")
    bundled_version = version_path.read_text(encoding="utf-8-sig").strip()
    if bundled_version != str(manifest.get("app_version", "")):
        raise RuntimeError(
            f"Runtime version mismatch: VERSION.txt is {bundled_version}, "
            f"manifest is {manifest.get('app_version')}."
        )

    actual_features = _feature_versions_from_runtime()
    if actual_features != manifest.get("feature_versions"):
        raise RuntimeError(
            "The application modules and runtime manifest have different feature configurations. "
            f"Application={actual_features}; manifest={manifest.get('feature_versions')}"
        )

    try:
        import main
    except ImportError:
        from . import main
    if str(getattr(main, "APP_VERSION", "")) != str(manifest.get("app_version", "")):
        raise RuntimeError(
            f"Application version mismatch: main.py is {getattr(main, 'APP_VERSION', '<missing>')}, "
            f"manifest is {manifest.get('app_version')}."
        )

    if require_source_match is None:
        require_source_match = True
    if require_source_match:
        source_manifest = verify_source_runtime_manifest(root)
        if source_manifest.get("source_fingerprint") != manifest.get("source_fingerprint"):
            raise RuntimeError("The source runtime and its release identity do not match.")

    return manifest
