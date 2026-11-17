from __future__ import annotations

import os
import sys
import time
import math
import json
import subprocess
import threading
import tempfile
import traceback
from pathlib import Path

import numpy as np
import psutil
from PySide6.QtCore import Qt, QEvent, QThread, Signal, QTimer, QSettings, QDir
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar, QTextEdit, QFrame,
    QSplitter, QMessageBox, QSizePolicy, QSpacerItem, QScrollArea,
    QButtonGroup, QSlider, QCheckBox, QToolButton, QColorDialog,
    QDialog, QDialogButtonBox, QLayout, QProgressDialog,
)

from view3d import SceneView
try:
    from .coordinate_frame import as_transform_matrix, source_frame_to_studio, transform_points
    from .startup_diagnostics import startup_trace
except ImportError:
    from coordinate_frame import as_transform_matrix, source_frame_to_studio, transform_points
    from startup_diagnostics import startup_trace
from pointio import load_preview_asset, load_baked_glb_asset, convert_up_axis, mesh_topology_summary
try:
    from .themes import THEMES, stylesheet_for, resolve_theme_name, initial_theme_mode, detect_system_theme
    from .workspace import mesh_work_dir, reconstructed_obj_path, reconstructed_baked_glb_path
except ImportError:
    from themes import THEMES, stylesheet_for, resolve_theme_name, initial_theme_mode, detect_system_theme
    from workspace import mesh_work_dir, reconstructed_obj_path, reconstructed_baked_glb_path
try:
    from .resolution import RESOLUTION_PRESETS, get_output_face_budget
except ImportError:
    from resolution import RESOLUTION_PRESETS, get_output_face_budget
try:
    from .export_runtime import ensure_pip_available, missing_export_modules, export_dependency_message
except ImportError:
    from export_runtime import ensure_pip_available, missing_export_modules, export_dependency_message
try:
    from .pipeline_progress import reconstruction_pipeline_percent, bake_pipeline_percent, RECONSTRUCTION_WEIGHT
except ImportError:
    from pipeline_progress import reconstruction_pipeline_percent, bake_pipeline_percent, RECONSTRUCTION_WEIGHT
try:
    from .fallback_routing import resolve_polyfit_timeout_fallback_source
except ImportError:
    from fallback_routing import resolve_polyfit_timeout_fallback_source
try:
    from .texture_baking import export_textured_glb, BakeCancelledError, PBR_MAP_LABELS
    from .asset_export import export_baked_asset, default_export_path, get_glb_baked_pbr_maps, supported_pbr_maps_for_format, studio_to_output_matrix
    from .gpu_compute import probe_gpu_runtime
    from .gpu_telemetry import query_gpu_telemetry
except ImportError:
    from texture_baking import export_textured_glb, BakeCancelledError, PBR_MAP_LABELS
    from asset_export import export_baked_asset, default_export_path, get_glb_baked_pbr_maps, supported_pbr_maps_for_format, studio_to_output_matrix
    from gpu_compute import probe_gpu_runtime
    from gpu_telemetry import query_gpu_telemetry

APP_VERSION = "0.2.131"

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
POLYFIT_OPTIMIZATION_TIMEOUT_SECONDS = 1

PORTABLE_PBR_EXPORT_UI_VERSION = 2

PORTABLE_VIEWPORT_COORDINATE_VERSION = 18

PORTABLE_VIEWPORT_COLOR_MANAGEMENT_VERSION = 2

PORTABLE_VIEWPORT_MATERIAL_PARITY_VERSION = 3
PORTABLE_ASSET_IMPORT_PROGRESS_VERSION = 1

PORTABLE_VIEWPORT_MOUSE_NAVIGATION_VERSION = 2

PORTABLE_GPU_STARTUP_PREFLIGHT_VERSION = 2
PORTABLE_GPU_FIRST_RUN_SETUP_VERSION = 3

def _gpu_vendor_short(vendor: str) -> str:
    value = str(vendor or "GPU").strip().lower()
    if "advanced micro devices" in value or "amd" in value or "ati" in value:
        return "AMD"
    if "nvidia" in value:
        return "NVIDIA"
    if "intel" in value:
        return "INTEL"
    return str(vendor or "GPU").strip().upper()[:12] or "GPU"

class _WheelSafeMixin:
    """Let the parent scroll area consume wheel input instead of changing a setting."""

    def wheelEvent(self, event):
        event.ignore()

class WheelSafeComboBox(_WheelSafeMixin, QComboBox):
    pass

class WheelSafeSpinBox(_WheelSafeMixin, QSpinBox):
    pass

class WheelSafeDoubleSpinBox(_WheelSafeMixin, QDoubleSpinBox):
    pass

class WheelSafeSlider(_WheelSafeMixin, QSlider):
    pass

class LightingOrbitWidget(QWidget):
    """Small draggable orbit control for rotating the main studio key light."""

    azimuthChanged = Signal(float)

    def __init__(self, azimuth: float = 10.0, parent=None):
        super().__init__(parent)
        self._azimuth = float(azimuth) % 360.0
        self.setMinimumSize(156, 156)
        self.setMaximumHeight(190)
        self.setToolTip("Drag the sun around the orbit to rotate the main studio light.")
        self.setCursor(Qt.PointingHandCursor)

    def azimuth(self) -> float:
        return float(self._azimuth)

    def setAzimuth(self, value: float, emit: bool = False):
        self._azimuth = float(value) % 360.0
        self.update()
        if emit:
            self.azimuthChanged.emit(self._azimuth)

    def _update_from_point(self, x: float, y: float):
        cx, cy = self.width() / 2.0, self.height() / 2.0
        dx, dy = x - cx, y - cy
        if math.hypot(dx, dy) < 14.0:
            return
        self.setAzimuth(math.degrees(math.atan2(dy, dx)) + 90.0, emit=True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._update_from_point(event.position().x(), event.position().y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._update_from_point(event.position().x(), event.position().y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        side = min(self.width(), self.height())
        cx, cy = self.width() / 2.0, self.height() / 2.0
        radius = side * 0.39
        ring_pen = QPen(self.palette().color(self.foregroundRole()), 1.7)
        ring_pen.setCosmetic(True)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.palette().mid().color()))
        core = radius * 0.58
        painter.drawEllipse(int(cx - core), int(cy - core), int(core * 2), int(core * 2))
        angle = math.radians(self._azimuth - 90.0)
        sx = cx + math.cos(angle) * radius
        sy = cy + math.sin(angle) * radius
        sun_color = QColor("#F6C84A")
        painter.setPen(QPen(sun_color, 2.0))
        painter.setBrush(QBrush(self.palette().window().color()))
        painter.drawEllipse(int(sx - 7), int(sy - 7), 14, 14)
        for ray in range(8):
            a = math.radians(ray * 45.0)
            x1, y1 = sx + math.cos(a) * 11, sy + math.sin(a) * 11
            x2, y2 = sx + math.cos(a) * 17, sy + math.sin(a) * 17
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        painter.end()

class CollapsibleSection(QWidget):
    """Compact arrow-headed option section, collapsed by default."""

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        self.header = QToolButton(self)
        self.header.setObjectName("CollapsibleHeader")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(bool(expanded))
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.setMinimumHeight(34)

        self.content = QWidget(self)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(9, 5, 9, 10)
        self.content_layout.setSpacing(7)
        self.content.setVisible(bool(expanded))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.header)
        layout.addWidget(self.content)
        self.header.toggled.connect(self._on_toggled)

    def _on_toggled(self, expanded: bool):
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)
        self.content.updateGeometry()
        self.updateGeometry()
        parent = self.parentWidget()
        if parent is not None:
            parent.updateGeometry()

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

class MetricCard(QFrame):
    def __init__(self, name: str, initial: str = "—"):
        super().__init__()
        self.setObjectName("Metric")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        self.value = QLabel(initial)
        self.value.setObjectName("MetricValue")
        self.name = QLabel(name.upper())
        self.name.setObjectName("MetricName")
        layout.addWidget(self.value)
        layout.addWidget(self.name)

    def set_value(self, text: str):
        self.value.setText(text)

class PipelineWorker(QThread):
    progress = Signal(int, str)
    log = Signal(str)
    preview_points = Signal(str)
    preview_mesh = Signal(str)
    stage_mesh = Signal(str, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.pid: int | None = None
        self.process: subprocess.Popen | None = None
        self._cancel_requested = False
        self._solver_timed_out = False
        self._solver_timer: threading.Timer | None = None
        self._solver_timeout_seconds: float | None = None
        self._solver_timeout_remaining: float | None = None
        self._solver_timer_started_at: float | None = None
        self._control_lock = threading.RLock()
        self._paused = False
        self._pause_requested = False
        self._suspended_processes: list[psutil.Process] = []
        self._progress_value = 0

    @property
    def is_paused(self) -> bool:
        with self._control_lock:
            return bool(self._paused)

    def _cancel_solver_timer_locked(self) -> None:
        timer = self._solver_timer
        self._solver_timer = None
        if timer is not None:
            timer.cancel()
        if self._solver_timer_started_at is not None:
            elapsed = max(0.0, time.monotonic() - self._solver_timer_started_at)
            configured = float(
                self._solver_timeout_remaining
                if self._solver_timeout_remaining is not None
                else self._solver_timeout_seconds or POLYFIT_OPTIMIZATION_TIMEOUT_SECONDS
            )
            self._solver_timeout_remaining = max(0.0, configured - elapsed)
        self._solver_timer_started_at = None

    def _start_solver_timer_locked(self, seconds: float) -> None:
        if self._cancel_requested or self._paused:
            self._solver_timeout_remaining = max(0.01, float(seconds))
            return
        delay = max(0.01, float(seconds))
        self._solver_timeout_remaining = delay
        self._solver_timer_started_at = time.monotonic()
        timer = threading.Timer(delay, self._on_solver_timeout)
        timer.daemon = True
        self._solver_timer = timer
        timer.start()

    def _suspend_engine_process_tree(self) -> bool:
        process = self.process
        if process is None or process.poll() is not None:
            return True
        suspended: list[psutil.Process] = []
        try:
            parent = psutil.Process(process.pid)
            descendants = parent.children(recursive=True)
            for item in list(reversed(descendants)) + [parent]:
                try:
                    if item.is_running() and item.status() != psutil.STATUS_STOPPED:
                        item.suspend()
                        suspended.append(item)
                except psutil.NoSuchProcess:
                    continue
            self._suspended_processes = suspended
            return True
        except (psutil.AccessDenied, psutil.Error):
            for item in reversed(suspended):
                try:
                    item.resume()
                except psutil.Error:
                    pass
            self._suspended_processes = []
            return False

    def _resume_engine_process_tree(self) -> None:
        suspended = list(self._suspended_processes)
        self._suspended_processes = []
        for item in reversed(suspended):
            try:
                if item.is_running():
                    item.resume()
            except psutil.Error:
                pass

    def pause(self) -> bool:
        with self._control_lock:
            if self._cancel_requested:
                return False
            if self._paused:
                return True
            self._paused = True
            self._pause_requested = True
            self._cancel_solver_timer_locked()
            if not self._suspend_engine_process_tree():
                self._paused = False
                self._pause_requested = False
                if self._solver_timeout_remaining is not None:
                    self._start_solver_timer_locked(self._solver_timeout_remaining)
                return False
            return True

    def resume(self) -> bool:
        with self._control_lock:
            if not self._paused:
                return True
            self._paused = False
            self._pause_requested = False
            self._resume_engine_process_tree()
            if self._solver_timeout_remaining is not None and self._solver_timer is None:
                self._start_solver_timer_locked(self._solver_timeout_remaining)
            return True

    def cancel(self):
        with self._control_lock:
            self._cancel_requested = True
            self._paused = False
            self._pause_requested = False
            if self._solver_timer is not None:
                self._solver_timer.cancel()
                self._solver_timer = None
            self._solver_timer_started_at = None
            self._resume_engine_process_tree()
        self.terminate_process_tree(timeout=3.0)

    def terminate_process_tree(self, timeout: float = 3.0):
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            parent = psutil.Process(process.pid)
            descendants = parent.children(recursive=True)
            for child in reversed(descendants):
                try:
                    child.terminate()
                except psutil.Error:
                    pass
            try:
                parent.terminate()
            except psutil.Error:
                pass
            _, alive = psutil.wait_procs(descendants + [parent], timeout=max(0.1, float(timeout)))
            for item in alive:
                try:
                    item.kill()
                except psutil.Error:
                    pass
            if alive:
                psutil.wait_procs(alive, timeout=max(0.1, float(timeout)))
        except psutil.Error:
            try:
                process.kill()
            except Exception:
                pass
        try:
            process.wait(timeout=max(0.1, float(timeout)))
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except Exception:
                pass
            process.wait()

    def stop_and_wait(self, timeout_ms: int = 5000) -> bool:
        self.cancel()
        if self.wait(max(0, int(timeout_ms))):
            return True
        self.terminate_process_tree(timeout=5.0)
        return self.wait(5000)

    def _on_solver_timeout(self):
        with self._control_lock:
            self._solver_timer = None
            self._solver_timer_started_at = None
            if self._paused or self._cancel_requested:
                return
            self._solver_timeout_remaining = None
            process = self.process
            if process is None or process.poll() is not None:
                return
            self._solver_timed_out = True
        self.terminate_process_tree(timeout=5.0)

    def _run_engine_process(self, config: dict, config_filename: str, progress_start: int = 0) -> dict:
        work_dir = Path(config.get("work_dir") or mesh_work_dir(config["input_path"]))
        work_dir.mkdir(parents=True, exist_ok=True)
        config = dict(config)
        config["work_dir"] = str(work_dir)
        self.config = config
        config_path = work_dir / config_filename
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        worker_script = APP_DIR / "worker.py"

        command = [sys.executable, "-u", str(worker_script), str(config_path)]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        with self._control_lock:
            self._solver_timed_out = False
            self._solver_timer = None
            self._solver_timeout_seconds = None
            self._solver_timeout_remaining = None
            self._solver_timer_started_at = None
        with self._control_lock:
            self.process = subprocess.Popen(
                command,
                cwd=tempfile.gettempdir(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            self.pid = self.process.pid
            cancel_on_start = self._cancel_requested
            pause_failed = (
                self._pause_requested and not self._cancel_requested
                and not self._suspend_engine_process_tree()
            )
            if pause_failed:
                self._paused = False
                self._pause_requested = False
        if cancel_on_start:
            self.terminate_process_tree(timeout=3.0)
        elif pause_failed:
            self.log.emit("Pause was not applied because the engine process could not be suspended safely.")
        tail: list[str] = []
        error_message: str | None = None
        completed_result = None
        assert self.process.stdout is not None
        try:
            for raw_line in self.process.stdout:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    tail.append(line)
                    tail = tail[-80:]
                    self.log.emit(line)
                    continue
                if not isinstance(event, dict):
                    self.log.emit(line)
                    continue
                event_type = event.get("type")
                if event_type == "progress":
                    value = int(event.get("value", 0))
                    message = str(event.get("message", ""))
                    display_value = value
                    if progress_start > 0:
                        display_value = progress_start + int(round(value * (100 - progress_start) / 100.0))
                    self._progress_value = max(self._progress_value, int(display_value))
                    self.progress.emit(display_value, message)
                    if progress_start == 0 and value >= 77 and "SCIP optimization" in message and self._solver_timer is None:
                        timeout = int(config.get(
                            "polyfit_optimization_timeout_seconds",
                            POLYFIT_OPTIMIZATION_TIMEOUT_SECONDS,
                        ))
                        with self._control_lock:
                            self._solver_timeout_seconds = float(max(1, timeout))
                            self._start_solver_timer_locked(self._solver_timeout_seconds)
                elif event_type == "log":
                    self.log.emit(str(event.get("message", "")))
                elif event_type == "preview_points":
                    self.preview_points.emit(str(event["path"]))
                elif event_type == "preview_mesh":
                    self.preview_mesh.emit(str(event["path"]))
                elif event_type == "stage_mesh":
                    self.stage_mesh.emit(str(event["path"]), str(event.get("label", "Live processing mesh")))
                elif event_type == "completed":
                    completed_result = event.get("result")
                elif event_type == "error":
                    error_message = str(event.get("message", "Unknown worker error"))
                else:
                    self.log.emit(line)
            return_code = self.process.wait()
        finally:
            with self._control_lock:
                if self._solver_timer is not None:
                    self._solver_timer.cancel()
                    self._solver_timer = None
                self._solver_timeout_seconds = None
                self._solver_timeout_remaining = None
                self._solver_timer_started_at = None
        return {
            "return_code": return_code,
            "timed_out": bool(self._solver_timed_out),
            "cancelled": bool(self._cancel_requested),
            "error_message": error_message,
            "completed_result": completed_result,
            "tail": tail,
            "work_dir": work_dir,
        }

    def _make_timeout_fallback_config(
        self,
        original_config: dict,
        fallback_input_path: Path,
        engine: str,
        source_up_axis: str,
        input_source_kind: str,
        fallback_reason: str = "SCIP face selection exceeded the configured time limit.",
    ) -> dict:
        original_input = str(original_config.get("input_path", ""))
        fallback = dict(original_config)
        fallback.update({
            "input_path": str(fallback_input_path),
            "engine_mode": engine,
            "source_up_axis": str(source_up_axis),
            "requested_engine": "AUTO_FALLBACK" if str(original_config.get("engine_mode", "auto")).lower() == "auto" else "POLYFIT_TIMEOUT_FALLBACK",
            "automatic_fallback_from": "POLYFIT",
            "automatic_fallback_reason": str(fallback_reason),
            "automatic_fallback_input_source": str(input_source_kind),
            "texture_bake_source": original_input,
        })
        return fallback

    def run(self):
        try:
            original_config = dict(self.config)
            requested_engine = str(original_config.get("engine_mode", "auto")).lower()
            work_dir = Path(original_config.get("work_dir") or mesh_work_dir(original_config["input_path"]))
            original_config["work_dir"] = str(work_dir)
            self._progress_value = 0
            if requested_engine in {"auto", "polyfit"}:

                stale_candidate = work_dir / "polyfit_candidate_preview.obj"
                try:
                    stale_candidate.unlink(missing_ok=True)
                except OSError as exc:
                    self.log.emit(f"Could not clear the previous PolyFit candidate preview: {exc}")
            result = self._run_engine_process(original_config, "worker_config.json")

            initial_failed = bool(
                result["timed_out"]
                or result["return_code"] != 0
                or result["error_message"]
                or result["completed_result"] is None
            )

            should_auto_fallback = requested_engine == "auto" and initial_failed

            should_timeout_fallback = (
                requested_engine == "polyfit"
                and result["timed_out"]
                and not result["cancelled"]
            )
            if (should_auto_fallback or should_timeout_fallback) and not result["cancelled"] and not self._cancel_requested:
                candidate_path = work_dir / "polyfit_candidate_preview.obj"
                original_input = Path(original_config["input_path"])
                try:
                    fallback_source, fallback_source_kind = resolve_polyfit_timeout_fallback_source(
                        original_input,
                        candidate_path,
                    )
                except Exception as source_exc:
                    self.log.emit(
                        "Automatic Li & Nan fallback was unavailable because no usable source triangle mesh "
                        f"or PolyFit candidate exists: {source_exc}"
                    )
                else:
                    fallback_axis = (
                        str(original_config.get("source_up_axis", "y"))
                        if fallback_source_kind == "original_input_mesh" else "z"
                    )
                    if result["timed_out"]:
                        failure_reason = "PolyFit SCIP face selection exceeded the configured time limit."
                    elif result["error_message"]:
                        failure_reason = str(result["error_message"]).splitlines()[0]
                    else:
                        failure_reason = f"PolyFit worker exited with code {result['return_code']} without a valid result."
                    if fallback_source_kind == "original_input_mesh":
                        self.log.emit(
                            "AUTO PolyFit did not complete successfully. Automatically switching to Li & Nan 2021 "
                            "using the ORIGINAL INPUT MESH. The source file remains unchanged. "
                            f"Reason: {failure_reason}"
                        )
                    else:
                        self.log.emit(
                            "PolyFit did not complete successfully and no original triangle mesh was supplied; "
                            "switching to Li & Nan 2021 using the generated candidate mesh. "
                            f"Reason: {failure_reason}"
                        )

                    fallback_progress_start = min(90, max(0, int(self._progress_value)))
                    self.progress.emit(
                        fallback_progress_start,
                        "Automatic fallback · Li & Nan 2021 feature-preserving reconstruction",
                    )
                    fallback_config = self._make_timeout_fallback_config(
                        original_config,
                        fallback_source,
                        "li_nan",
                        fallback_axis,
                        fallback_source_kind,
                        fallback_reason=failure_reason,
                    )
                    result = self._run_engine_process(
                        fallback_config,
                        "worker_config_fallback.json",
                        progress_start=fallback_progress_start,
                    )
                    if (
                        result["return_code"] != 0
                        or result["error_message"]
                        or result["completed_result"] is None
                    ) and not result["cancelled"]:
                        fallback_error = result["error_message"] or "Li & Nan fallback did not produce a completed result."
                        self.log.emit(
                            f"Li & Nan automatic fallback could not finish: {fallback_error}. "
                            "The original input remains unchanged."
                        )

            if result["cancelled"] or self._cancel_requested:
                self.failed.emit("PIPELINE_CANCELLED: Reconstruction was cancelled by the user. The output OBJ may not exist if saving had not finished.")
            elif result["return_code"] != 0:
                detail = "\n".join(result["tail"][-30:])
                message = result["error_message"] or f"The engine process exited with code {result['return_code']}."
                if detail:
                    message += "\n\n--- Process output ---\n" + detail
                self.failed.emit(message)
            elif result["error_message"]:
                self.failed.emit(result["error_message"])
            elif result["completed_result"] is None:
                detail = "\n".join(result["tail"][-30:])
                self.failed.emit("The engine process exited without a completion event." + ("\n" + detail if detail else ""))
            else:
                self.completed.emit(result["completed_result"])
        except Exception:
            self.failed.emit("Could not start or monitor the engine process.\n\n" + traceback.format_exc())
        finally:
            if self._solver_timer is not None:
                self._solver_timer.cancel()
                self._solver_timer = None
            if self.process is not None and self.process.poll() is None:
                self.terminate_process_tree()

class ExportOptionsDialog(QDialog):
    """Format, coordinate and portable PBR controls for exporting the baked GLB master."""

    FORMATS = (
        ("GLB · Binary glTF (recommended)", ".glb"),
        ("glTF · JSON + external assets", ".gltf"),
        ("FBX · Autodesk FBX 7.4 + copied textures", ".fbx"),
        ("OBJ · Wavefront + MTL / texture", ".obj"),
        ("PLY · Polygon File Format", ".ply"),
        ("STL · Stereolithography", ".stl"),
        ("3MF · 3D Manufacturing Format", ".3mf"),
        ("OFF · Object File Format", ".off"),
    )

    def __init__(self, parent=None, available_pbr_maps=None):
        super().__init__(parent)
        self.setWindowTitle("Export Options · Polyfitty Studio")
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.format_combo = WheelSafeComboBox()
        for label, suffix in self.FORMATS:
            self.format_combo.addItem(label, suffix)
        self.format_combo.setCurrentIndex(0)
        form.addRow("File format", self.format_combo)

        self.alpha_mode_combo = WheelSafeComboBox()
        self.alpha_mode_combo.addItem("Opaque", "OPAQUE")
        self.alpha_mode_combo.addItem("Alpha Clip", "ALPHA_CLIP")
        self.alpha_mode_combo.addItem("Alpha Hashed", "ALPHA_HASHED")
        self.alpha_mode_combo.addItem("Alpha Blend", "ALPHA_BLEND")
        self.alpha_mode_combo.setCurrentIndex(self.alpha_mode_combo.findData("ALPHA_CLIP"))
        self.alpha_mode_combo.setToolTip(
            "GLB/glTF use the standard OPAQUE, MASK or BLEND field. Alpha Hashed is retained "
            "in Polyfitty material extras and uses MASK as a fallback for other glTF viewers."
        )
        form.addRow("Alpha Blend Mode", self.alpha_mode_combo)

        self.alpha_cutoff_spin = WheelSafeDoubleSpinBox()
        self.alpha_cutoff_spin.setRange(0.0, 1.0)
        self.alpha_cutoff_spin.setDecimals(2)
        self.alpha_cutoff_spin.setSingleStep(0.05)
        self.alpha_cutoff_spin.setValue(0.5)
        self.alpha_cutoff_spin.setToolTip("Alpha threshold from 0 to 1. Used by Alpha Clip / Alpha Hashed.")
        form.addRow("Alpha Cutoff", self.alpha_cutoff_spin)

        self.shadow_mode_combo = WheelSafeComboBox()
        self.shadow_mode_combo.addItem("None", "NONE")
        self.shadow_mode_combo.addItem("Opaque", "OPAQUE")
        self.shadow_mode_combo.addItem("Alpha Clip", "ALPHA_CLIP")
        self.shadow_mode_combo.addItem("Alpha Hashed", "ALPHA_HASHED")
        self.shadow_mode_combo.setCurrentIndex(self.shadow_mode_combo.findData("OPAQUE"))
        self.shadow_mode_combo.setToolTip(
            "Shadow mode is stored in Polyfitty material extras because core glTF has no shadow-mode field."
        )
        form.addRow("Shadow Mode", self.shadow_mode_combo)

        self.axis_combo = WheelSafeComboBox()
        self.axis_combo.addItem("Y-up · glTF standard", "y")
        self.axis_combo.addItem("Z-up · Polyfitty Studio", "z")
        self.axis_combo.addItem("X-up", "x")
        self.axis_combo.setCurrentIndex(self.axis_combo.findData("y"))
        self.axis_combo.setToolTip(
            "Default: Y-up · glTF standard. The Studio Z-up frame is converted back to Y-up "
            "with the inverse X-axis rotation (−90°) during export. Z-up and X-up remain selectable."
        )
        form.addRow("Output up-axis", self.axis_combo)

        self.specular_spin = WheelSafeDoubleSpinBox()
        self.specular_spin.setRange(0.0, 1.0)
        self.specular_spin.setDecimals(2)
        self.specular_spin.setSingleStep(0.05)
        self.specular_spin.setValue(0.25)
        form.addRow("Specular", self.specular_spin)

        self.roughness_spin = WheelSafeDoubleSpinBox()
        self.roughness_spin.setRange(0.0, 1.0)
        self.roughness_spin.setDecimals(2)
        self.roughness_spin.setSingleStep(0.05)
        self.roughness_spin.setValue(1.0)
        form.addRow("Roughness", self.roughness_spin)

        self.metallic_spin = WheelSafeDoubleSpinBox()
        self.metallic_spin.setRange(0.0, 1.0)
        self.metallic_spin.setDecimals(2)
        self.metallic_spin.setSingleStep(0.05)
        self.metallic_spin.setValue(0.0)
        form.addRow("Metallic", self.metallic_spin)
        layout.addLayout(form)

        available = set(PBR_MAP_LABELS if available_pbr_maps is None else available_pbr_maps)
        pbr_group = QGroupBox("PBR TEXTURES TO EXPORT")
        pbr_layout = QGridLayout(pbr_group)
        pbr_layout.setContentsMargins(10, 8, 10, 8)
        pbr_layout.setHorizontalSpacing(12)
        pbr_layout.setVerticalSpacing(6)
        self.pbr_map_checks = {}
        for index, (key, label) in enumerate(PBR_MAP_LABELS.items()):
            checkbox = QCheckBox(label)
            checkbox.setChecked(key in {"base_color", "orm"} and key in available)
            checkbox.setEnabled(key in available)
            checkbox.setToolTip(
                "This map was included in the baked GLB master."
                if key in available else
                "This map was not selected during baking and is unavailable for this export."
            )
            self.pbr_map_checks[key] = checkbox
            pbr_layout.addWidget(checkbox, index // 2, index % 2)
        layout.addWidget(pbr_group)

        self.format_note = QLabel(
            "GLB / glTF retain PBR settings. FBX exports mesh, normals, UVs and Base Color as an external image copied beside the FBX (Copy Path); "
            "FBX material values use a Phong-compatible approximation. OBJ writes MTL + image sidecars; STL / OFF / 3MF preserve geometry only."
        )
        self.format_note.setWordWrap(True)
        self.format_note.setObjectName("Muted")
        layout.addWidget(self.format_note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.available_pbr_maps = available
        self.format_combo.currentIndexChanged.connect(self._refresh_pbr_map_controls)
        self.alpha_mode_combo.currentIndexChanged.connect(self._refresh_alpha_controls)
        self._refresh_pbr_map_controls()
        self._refresh_alpha_controls()

    def _refresh_pbr_map_controls(self, *_args) -> None:
        suffix = str(self.format_combo.currentData() or ".glb").lower()
        supported = set(supported_pbr_maps_for_format(suffix))
        for key, checkbox in self.pbr_map_checks.items():
            baked = key in self.available_pbr_maps
            enabled = baked and key in supported
            checkbox.setEnabled(enabled)
            if not baked:
                tooltip = "This map was not selected during baking and is unavailable for this export."
            elif key not in supported:
                tooltip = f"The {suffix.upper()} format cannot preserve this PBR map."
            elif key == "height":
                tooltip = (
                    "Embedded in Polyfitty asset metadata; glTF has no core standard "
                    "Height / Displacement material slot."
                )
            else:
                tooltip = "This map was included in the baked GLB master and can be exported."
            checkbox.setToolTip(tooltip)

        format_notes = {
            ".glb": (
                "GLB preserves selected PBR textures in one binary file. Alpha Clip maps to glTF MASK, "
                "Alpha Blend to BLEND; Alpha Hashed and Shadow Mode are also stored in Polyfitty material extras. "
                "Height / Displacement is stored in Polyfitty asset metadata because glTF has no core slot."
            ),
            ".gltf": (
                "glTF preserves selected PBR textures as data-URI images. Alpha Clip maps to MASK, "
                "Alpha Blend to BLEND; Alpha Hashed and Shadow Mode are stored in Polyfitty material extras. "
                "Height / Displacement is stored in Polyfitty asset metadata because glTF has no core slot."
            ),
            ".fbx": (
                "FBX 7.4 ASCII export includes geometry, normals, UVs, vertex colors and the selected Base Color image. "
                "Texture Path Mode is fixed to Copy Path: the image is copied beside the FBX and linked by a relative path. "
                "Material shading is approximated through FBX Phong properties; metallic/roughness and shadow mode are not fully portable."
            ),
            ".obj": "OBJ exports Base Color only through MTL + image sidecar; other PBR maps are unavailable.",
            ".ply": "PLY can convert Base Color to vertex colors; other PBR maps are unavailable.",
            ".stl": "STL preserves geometry only and cannot store PBR textures.",
            ".3mf": "3MF export preserves geometry only in this workflow; PBR textures are unavailable.",
            ".off": "OFF preserves geometry only and cannot store PBR textures.",
        }
        self.format_note.setText(format_notes.get(
            suffix, "This format does not preserve PBR textures."
        ))

    def _refresh_alpha_controls(self, *_args) -> None:
        suffix = str(self.format_combo.currentData() or ".glb").lower()
        gltf_supported = suffix in {".glb", ".gltf"}
        fbx_supported = suffix == ".fbx"
        alpha_mode = str(self.alpha_mode_combo.currentData() or "ALPHA_CLIP")
        self.alpha_mode_combo.setEnabled(gltf_supported or fbx_supported)
        self.alpha_cutoff_spin.setEnabled(
            (gltf_supported or fbx_supported) and alpha_mode in {"ALPHA_CLIP", "ALPHA_HASHED"}
        )
        self.shadow_mode_combo.setEnabled(gltf_supported)
        if fbx_supported:
            self.alpha_mode_combo.setToolTip(
                "FBX stores image/material opacity. Alpha Clip / Hashed are baked into the copied Base Color image; "
                "Shadow Mode has no portable FBX material equivalent."
            )
        elif not gltf_supported:
            self.alpha_mode_combo.setToolTip(
                f"{suffix.upper()} cannot preserve selectable alpha blend / shadow modes. "
                "Choose GLB, glTF or FBX to export supported material opacity."
            )
        else:
            self.alpha_mode_combo.setToolTip(
                "Alpha Clip maps to glTF MASK, Alpha Blend to BLEND. Alpha Hashed is retained "
                "in Polyfitty extras and uses MASK as a compatibility fallback."
            )

    def options(self) -> dict:
        return {
            "suffix": str(self.format_combo.currentData() or ".glb"),
            "alpha_mode": str(self.alpha_mode_combo.currentData() or "ALPHA_CLIP"),
            "alpha_cutoff": float(self.alpha_cutoff_spin.value()),
            "shadow_mode": str(self.shadow_mode_combo.currentData() or "OPAQUE"),
            "output_up_axis": str(self.axis_combo.currentData() or "y"),
            "specular": float(self.specular_spin.value()),
            "roughness": float(self.roughness_spin.value()),
            "metallic": float(self.metallic_spin.value()),
            "pbr_maps": [key for key, checkbox in self.pbr_map_checks.items()
                         if checkbox.isEnabled() and checkbox.isChecked()],
        }

def _discover_system_gpu_names() -> list[str]:
    """On Windows, list display adapters independently of CUDA/OpenCL availability."""
    if sys.platform != "win32":
        return []
    powershell_script = "Get-CimInstance -ClassName Win32_VideoController | ForEach-Object { $_.Name }"
    powershell_exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    commands = []
    if powershell_exe.is_file():
        commands.append([str(powershell_exe), "-NoProfile", "-NonInteractive", "-Command", powershell_script])
    commands.extend([
        [
            "powershell.exe", "-NoProfile", "-NonInteractive",
            "-Command",
            powershell_script,
        ],
        ["wmic.exe", "path", "win32_VideoController", "get", "name"],
        ])
    for command in commands:
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=5, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode != 0:
                continue
            names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            names = [name for name in names if name.lower() not in {"name"}]
            if names:
                return list(dict.fromkeys(names))
        except (OSError, subprocess.SubprocessError):
            continue
    return []

class GPUDiscoveryWorker(QThread):
    """Enumerate optional GPU compute devices without blocking the GUI thread."""

    completed = Signal(object)

    def run(self) -> None:
        try:
            payload = probe_gpu_runtime()
            payload["display_adapters"] = _discover_system_gpu_names()
        except Exception as exc:
            payload = {
                "devices": [],
                "errors": [f"GPU scan failed ({type(exc).__name__}: {exc})"],
                "display_adapters": _discover_system_gpu_names(),
            }
        self.completed.emit(payload)

class GPUPerformanceMonitor(QThread):
    """Poll OS/driver GPU telemetry outside the GUI thread while Diagnostics is open."""

    updated = Signal(object)

    def __init__(self, vendor: str, device_name: str, parent=None):
        super().__init__(parent)
        self.vendor = str(vendor or "GPU")
        self.device_name = str(device_name or "Unknown GPU")

    def run(self) -> None:
        while not self.isInterruptionRequested():
            try:
                payload = query_gpu_telemetry(self.vendor, self.device_name, os.getpid())
            except Exception as exc:
                payload = {
                    "utilization_percent": None,
                    "memory_used_bytes": None,
                    "memory_total_bytes": None,
                    "source": "unavailable",
                    "detail": f"GPU telemetry polling failed ({type(exc).__name__}: {exc})",
                }
            self.updated.emit(payload)

            for _ in range(12):
                if self.isInterruptionRequested():
                    return
                self.msleep(200)

class ExportWorker(QThread):
    """Run automatic reconstruction baking or baked-master format conversion off the GUI thread."""

    progress = Signal(int, str)
    gpu_status = Signal(object)
    bake_preview = Signal(object)
    bake_edges = Signal(object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self, *, output_path: str, suffix: str, target_mesh_path: str | None = None,
        bake_enabled: bool = False, resolution: int = 2048,
        source_mesh_path: str = "", source_up_axis: str = "y",
        projection_mode: str = "cage",
        cage_extrusion_percent: float = 5.0, preview_path: str | None = None,
        auto_bake: bool = False, baked_input_path: str | None = None,
        output_up_axis: str = "y", specular: float = 0.25,
        roughness: float = 1.0, metallic: float = 0.0,
        alpha_mode: str = "ALPHA_CLIP", alpha_cutoff: float = 0.5,
        shadow_mode: str = "OPAQUE",
        compute_device: str = "auto",
        pbr_maps=None,
        source_frame_transform=None,
    ) -> None:
        super().__init__()
        self.output_path = str(output_path)
        self.suffix = str(suffix).lower()
        self.source_obj = str(target_mesh_path) if target_mesh_path else ""
        self.bake_enabled = bool(bake_enabled)
        self.resolution = int(resolution)
        self.source_mesh_path = str(source_mesh_path or "")
        self.source_up_axis = str(source_up_axis or "y")
        self.projection_mode = str(projection_mode or "cage")
        self.cage_extrusion_percent = float(cage_extrusion_percent)
        self.preview_path = str(preview_path) if preview_path else None
        self.auto_bake = bool(auto_bake)
        self.baked_input_path = str(baked_input_path) if baked_input_path else None
        self.output_up_axis = str(output_up_axis or "y")
        self.specular = float(specular)
        self.roughness = float(roughness)
        self.metallic = float(metallic)
        self.alpha_mode = str(alpha_mode or "ALPHA_CLIP")
        self.alpha_cutoff = float(alpha_cutoff)
        self.shadow_mode = str(shadow_mode or "OPAQUE")
        self.compute_device = str(compute_device or "auto")
        self.pbr_maps = tuple(
            str(key) for key in (("base_color",) if pbr_maps is None else pbr_maps)
        )
        self.source_frame_transform = as_transform_matrix(source_frame_transform)
        self._pause_gate = threading.Event()
        self._pause_gate.set()

    @property
    def is_paused(self) -> bool:
        return not self._pause_gate.is_set()

    def pause(self) -> bool:
        if self.isInterruptionRequested():
            return False
        self._pause_gate.clear()
        return True

    def resume(self) -> bool:
        self._pause_gate.set()
        return True

    def request_cancel(self) -> None:
        """Request safe cancellation at the next bake processing checkpoint."""
        self._pause_gate.set()
        self.requestInterruption()

    def _bake_control_check(self) -> bool:
        """Block at bake checkpoints while paused and report cancellation when requested."""
        while not self._pause_gate.wait(0.1):
            if self.isInterruptionRequested():
                return True
        return self.isInterruptionRequested()

    def _raise_if_bake_cancelled(self) -> None:
        if self._bake_control_check():
            raise BakeCancelledError("Automatic GLB baking was cancelled by the user.")

    def _report_bake_progress(self, fraction: float | None, message: str) -> None:
        message = str(message)
        status_prefix = "__PFS_GPU_STATUS__"
        if message.startswith(status_prefix):
            try:
                payload = json.loads(message[len(status_prefix):])
                if isinstance(payload, dict):
                    self.gpu_status.emit(payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            return
        if fraction is None:
            self.progress.emit(-1, message)
        else:
            value = max(0, min(100, int(round(float(fraction) * 100.0))))
            self.progress.emit(value, message)

    def _emit_bake_preview(self, payload: dict) -> None:
        """Forward bounded atlas snapshots to the GUI thread without blocking bake."""
        self.bake_preview.emit(payload)

    def _emit_bake_edges(self, payload: dict) -> None:
        """Forward the lightweight rolling 10-face edge trail to the viewport."""
        self.bake_edges.emit(payload)

    def _export_reconstruction_geometry(self, mesh, suffix: str) -> list[Path]:
        """Export source-local reconstruction geometry at the original object transform."""
        import trimesh

        output_frame = studio_to_output_matrix(self.output_up_axis) @ self.source_frame_transform
        destination = Path(self.output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if suffix in {".glb", ".gltf"}:
            scene = trimesh.Scene()
            scene.add_geometry(
                mesh,
                node_name="Polyfitty_Source_Origin",
                geom_name="Polyfitty_Reconstruction",
                transform=output_frame,
            )
            exported = scene.export(file_type=suffix[1:])
            if suffix == ".glb":
                destination.write_bytes(exported if isinstance(exported, bytes) else bytes(exported))
                return [destination]
            if not isinstance(exported, dict):
                raise RuntimeError("The glTF exporter did not return its external asset files.")
            written = []
            for name, payload in exported.items():
                target_name = destination.name if str(name).lower().endswith(".gltf") else Path(name).name
                target = destination.parent / target_name
                target.write_bytes(payload if isinstance(payload, bytes) else str(payload).encode("utf-8"))
                written.append(target)
            return written

        mesh = mesh.copy()
        mesh.apply_transform(output_frame)
        if suffix == ".fbx":
            try:
                from .asset_export import export_fbx_mesh
            except ImportError:
                from asset_export import export_fbx_mesh
            info = export_fbx_mesh(
                mesh, destination, output_up_axis=self.output_up_axis,
                allow_material_texture=True,
            )
            return [Path(item) for item in info.get("written_files", [])]
        exported = mesh.export(str(destination))
        if isinstance(exported, dict):
            written = [destination]
            for name, payload in exported.items():
                target = destination.parent / Path(name).name
                target.write_bytes(payload if isinstance(payload, bytes) else str(payload).encode("utf-8"))
                written.append(target)
            return written
        if isinstance(exported, (bytes, bytearray)) and not destination.exists():
            destination.write_bytes(bytes(exported))
        return [destination]

    def run(self) -> None:
        try:
            if self.auto_bake:
                self._raise_if_bake_cancelled()
            bake_info = None
            export_info = None
            preview_asset = None
            suffix = self.suffix

            if self.baked_input_path:
                self.progress.emit(-1, "Step 0/5 · Checking export dependencies")
                pip_ready, pip_detail = ensure_pip_available(
                    sys.executable, ROOT_DIR / "installer" / "ensure_pip.ps1", ROOT_DIR / ".cache"
                )
                if self.auto_bake:
                    self._raise_if_bake_cancelled()
                if not pip_ready:
                    raise RuntimeError(
                        "The private Python package manager is unavailable.\n\n"
                        f"{pip_detail}\n\nRun INSTALL.bat to repair the application runtime."
                    )
                missing = missing_export_modules(suffix, False, conversion_from_baked=True)
                if missing:
                    raise RuntimeError(export_dependency_message(missing))
                self.progress.emit(8, "Baked GLB master ready · preparing format conversion")
                export_info = export_baked_asset(
                    self.baked_input_path, self.output_path,
                    output_up_axis=self.output_up_axis,
                    specular=self.specular, roughness=self.roughness,
                    metallic=self.metallic, pbr_maps=self.pbr_maps,
                    alpha_mode=self.alpha_mode, alpha_cutoff=self.alpha_cutoff,
                    shadow_mode=self.shadow_mode,
                    progress_callback=self._report_bake_progress,
                )
            else:
                if suffix != ".obj":
                    self.progress.emit(-1, "Step 0/7 · Checking private Python export dependencies")
                    pip_ready, pip_detail = ensure_pip_available(
                        sys.executable, ROOT_DIR / "installer" / "ensure_pip.ps1", ROOT_DIR / ".cache"
                    )
                    if self.auto_bake:
                        self._raise_if_bake_cancelled()
                    if not pip_ready:
                        raise RuntimeError(
                            "The private Python package manager is unavailable.\n\n"
                            f"{pip_detail}\n\nRun INSTALL.bat to repair the application runtime."
                        )
                    missing = missing_export_modules(suffix, self.bake_enabled)
                    if missing:
                        raise RuntimeError(export_dependency_message(missing))
                    self.progress.emit(5, "Export dependencies ready · preparing generated OBJ")

                if suffix == ".glb" and self.bake_enabled:
                    source_obj = Path(self.source_obj)
                    if not source_obj.is_file() or source_obj.suffix.lower() != ".obj":
                        raise ValueError(
                            "The generated reconstruction OBJ is unavailable; cannot bake the GLB target geometry."
                        )
                    bake_info = export_textured_glb(
                        self.output_path, resolution=self.resolution,
                        source_mesh_path=self.source_mesh_path or None,
                        source_up_axis=self.source_up_axis,
                        source_frame_transform=self.source_frame_transform,
                        projection_mode=self.projection_mode,
                        cage_extrusion_percent=self.cage_extrusion_percent,
                        target_mesh_path=source_obj,
                        vertex_color_preview_path=self.preview_path,
                        progress_callback=self._report_bake_progress,
                        compute_device=self.compute_device,
                        preview_callback=self._emit_bake_preview,
                        edge_preview_callback=self._emit_bake_edges,
                        cancel_check=self._bake_control_check,
                        pbr_maps=self.pbr_maps,
                    )
                    self._raise_if_bake_cancelled()
                    self.progress.emit(-1, "GLB validation · Re-open baked file and inspect mesh / UV / embedded texture")
                    preview_asset = load_baked_glb_asset(self.output_path)
                    self._raise_if_bake_cancelled()
                    if not preview_asset.get("has_texture"):
                        self.progress.emit(-1, "GLB validation · Checking baked geometry and material fallback")
                    self.progress.emit(100, "GLB validation complete · preparing final VTK result inspection")
                elif suffix == ".obj":
                    self.progress.emit(-1, "Applying source object origin and writing OBJ mesh")
                    try:
                        from .mesh_io import load_obj_trimesh
                    except ImportError:
                        from mesh_io import load_obj_trimesh
                    mesh = load_obj_trimesh(self.source_obj)
                    self._export_reconstruction_geometry(mesh, suffix)
                    self.progress.emit(100, "OBJ export complete")
                else:
                    self.progress.emit(-1, "Loading generated mesh for export conversion")
                    try:
                        from .mesh_io import load_obj_trimesh
                    except ImportError:
                        from mesh_io import load_obj_trimesh
                    mesh = load_obj_trimesh(self.source_obj)
                    self.progress.emit(-1, f"Applying source object origin and writing {suffix.upper()} mesh data")
                    self._export_reconstruction_geometry(mesh, suffix)
                    self.progress.emit(100, f"{suffix.upper()} export complete")

            self.completed.emit({
                "path": self.output_path,
                "suffix": suffix,
                "bake_info": bake_info,
                "export_info": export_info,
                "preview_asset": preview_asset,
                "auto_bake": self.auto_bake,
            })
        except BakeCancelledError as exc:

            try:
                Path(self.output_path).unlink(missing_ok=True)
            except OSError:
                pass
            self.failed.emit(f"BAKE_CANCELLED: {exc}")
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

class AssetImportWorker(QThread):
    """Decode an imported source asset off the GUI thread and report honest stages."""

    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self, path: str, *, mesh_sample_count: int, merge_distance: float,
        alpha_mode_override: str | None, alpha_cutoff_override: float,
    ) -> None:
        super().__init__()
        self.path = str(path)
        self.mesh_sample_count = int(mesh_sample_count)
        self.merge_distance = float(merge_distance)
        self.alpha_mode_override = alpha_mode_override
        self.alpha_cutoff_override = float(alpha_cutoff_override)

    def run(self) -> None:
        try:
            asset = load_preview_asset(
                self.path,
                mesh_sample_count=self.mesh_sample_count,
                merge_distance=self.merge_distance,
                alpha_mode_override=self.alpha_mode_override,
                alpha_cutoff_override=self.alpha_cutoff_override,
                progress_callback=lambda value, message: self.progress.emit(int(value), str(message)),
            )
            self.completed.emit(asset)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

class MainWindow(QMainWindow):
    def __init__(self, startup_gpu_probe: dict | None = None):
        startup_trace("MainWindow.__init__ entered")
        super().__init__()
        self.setWindowTitle("Polyfitty Studio | Polygonal Surface Reconstruction")
        self.setAcceptDrops(True)

        self.setMinimumSize(1024, 640)
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(1580, max(1024, available.width() - 28)),
                min(940, max(640, available.height() - 52)),
            )
        else:
            self.resize(1580, 940)
        self.worker: PipelineWorker | None = None
        self._export_worker: ExportWorker | None = None
        self.last_result: dict | None = None
        self.last_output: Path | None = None
        self.last_baked_glb: Path | None = None
        self._auto_bake_active = False
        self._task_paused = False
        self._stage_text_before_pause = ""
        self._source_asset_raw: dict | None = None
        self._asset_import_worker: AssetImportWorker | None = None
        self._asset_import_progress: QProgressDialog | None = None
        self._pending_asset_import: tuple[str, str] | None = None
        self._source_frame_transform_raw = np.eye(4, dtype=np.float64)
        self._source_frame_transform_studio = np.eye(4, dtype=np.float64)
        self._mesh_comparison_stats = {"input": None, "reconstruction": None, "output": None}
        self._mesh_comparison_kind = "unknown"
        self._live_mesh_preview_loaded = False
        self._axis_control_allowed = True
        self.started_at: float | None = None
        self.process = psutil.Process(os.getpid())

        legacy_settings = QSettings("PolyFitStudio", "PolyFitStudio")
        self.settings = QSettings("PolyfittyStudio", "PolyfittyStudio")
        if not self.settings.allKeys() and legacy_settings.allKeys():
            for setting_key in legacy_settings.allKeys():
                self.settings.setValue(setting_key, legacy_settings.value(setting_key))
            self.settings.sync()
        self._restore_startup_window_geometry()
        if not bool(self.settings.value("bake_gpu_auto_default_migrated_v1", False, type=bool)):
            legacy_compute = str(self.settings.value("bake_compute_device", "auto") or "auto").strip().lower()
            cpu_override_was_explicit = bool(self.settings.value("bake_compute_cpu_override_explicit", False, type=bool))
            if legacy_compute in {"cpu", "cpu-only", "disabled"} and not cpu_override_was_explicit:
                self.settings.setValue("bake_compute_device", "auto")
            self.settings.setValue("bake_gpu_auto_default_migrated_v1", True)

        self._pending_log_messages: list[str] = []
        self._closing_after_bake_cancel = False
        self._gpu_scan_worker: GPUDiscoveryWorker | None = None
        self._gpu_discovery_records: list[dict] = []
        self._gpu_discovery_errors: list[str] = []
        self._gpu_runtime_state_confirmed = False
        self._startup_gpu_probe = dict(startup_gpu_probe) if isinstance(startup_gpu_probe, dict) else None
        self._gpu_rescan_after_export = False
        self._gpu_rescan_pending = False
        self._gpu_telemetry_target: tuple[str, str] | None = None
        self._gpu_telemetry_worker: GPUPerformanceMonitor | None = None
        self._gpu_vram_total_bytes = 0
        saved_theme_mode = self.settings.value("theme_mode", None)
        legacy_theme = str(self.settings.value("theme", "") or "").lower()
        self._preferred_theme = initial_theme_mode(saved_theme_mode, legacy_theme)
        if saved_theme_mode is None:
            self.settings.setValue("theme_mode", self._preferred_theme)
        startup_trace("MainWindow._build_ui starting")
        self._build_ui()
        startup_trace("MainWindow._build_ui completed")
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_live_metrics)

        self._timer.start(2000)
        self._theme_timer = QTimer(self)
        self._theme_timer.timeout.connect(self._refresh_system_theme)
        self._theme_timer.start(1500)
        self._update_live_metrics()
        startup_trace("MainWindow.__init__ completed")

    def eventFilter(self, watched, event):
        """Handle app-wide file drops and suppress hover tooltips."""
        event_type = event.type()
        if event_type in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            path = self._dropped_input_path(event)
            busy = (self.worker is not None and self.worker.isRunning()) or (
                self._export_worker is not None and self._export_worker.isRunning()
            ) or bool(self._auto_bake_active)
            if path is not None and not busy:
                event.acceptProposedAction()
                return True
        elif event_type == QEvent.Type.Drop:
            path = self._dropped_input_path(event)
            busy = (self.worker is not None and self.worker.isRunning()) or (
                self._export_worker is not None and self._export_worker.isRunning()
            ) or bool(self._auto_bake_active)
            if path is not None and not busy:
                self._load_input_source(QDir.toNativeSeparators(str(path)), action="drop")
                event.acceptProposedAction()
                return True
        if event_type == QEvent.Type.ToolTip:
            return True
        return super().eventFilter(watched, event)

    @staticmethod
    def _dropped_input_path(event):
        """Return the first supported local 3D file carried by a drop event."""
        mime = event.mimeData() if hasattr(event, "mimeData") else None
        if mime is None or not mime.hasUrls():
            return None
        supported = {
            ".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts", ".txt",
            ".csv", ".asc", ".obj", ".stl", ".off", ".glb", ".gltf",
            ".3mf", ".vg", ".bvg",
        }
        for url in mime.urls():
            try:
                if not url.isLocalFile():
                    continue
                path = Path(url.toLocalFile())
                if path.is_file() and path.suffix.lower() in supported:
                    return path
            except (OSError, TypeError, ValueError):
                continue
        return None

    def dragEnterEvent(self, event):
        path = self._dropped_input_path(event)
        busy = (self.worker is not None and self.worker.isRunning()) or (
            self._export_worker is not None and self._export_worker.isRunning()
        ) or bool(self._auto_bake_active)
        if path is not None and not busy:
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        path = self._dropped_input_path(event)
        busy = (self.worker is not None and self.worker.isRunning()) or (
            self._export_worker is not None and self._export_worker.isRunning()
        ) or bool(self._auto_bake_active)
        if path is None or busy:
            event.ignore()
            return
        self._load_input_source(QDir.toNativeSeparators(str(path)), action="drop")
        event.acceptProposedAction()

    def _restore_startup_window_geometry(self):
        """Restore the previous normal size/state, or maximize on the first launch."""
        self._window_geometry_restored = False
        saved_geometry = self.settings.value("window/geometry", None)
        if saved_geometry:
            try:
                self._window_geometry_restored = bool(self.restoreGeometry(saved_geometry))
            except Exception as exc:
                startup_trace("Saved window geometry could not be restored: {0}".format(exc))

        first_maximize_done = bool(self.settings.value(
            "window/first_startup_maximize_done_v1", False, type=bool
        ))
        saved_maximized = bool(self.settings.value("window/maximized", False, type=bool))
        self._startup_window_maximized = bool(saved_maximized or not first_maximize_done)
        startup_trace(
            "Window geometry restored={0}; startup_maximized={1}".format(
                self._window_geometry_restored, self._startup_window_maximized
            )
        )

    def _save_window_geometry(self):
        """Persist maximized state and Qt's normal geometry for the next restore."""
        if not hasattr(self, "settings"):
            return
        try:
            self.settings.setValue("window/geometry", self.saveGeometry())
            self.settings.setValue("window/maximized", bool(self.isMaximized()))
            self.settings.setValue("window/first_startup_maximize_done_v1", True)
            self.settings.sync()
        except Exception as exc:
            startup_trace("Window geometry could not be saved: {0}".format(exc))

    @staticmethod
    def _configure_responsive_forms(root_widget):
        """Wrap form labels and let editors use available width in narrow docks."""
        for layout in root_widget.findChildren(QFormLayout):
            layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
            layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            layout.setFormAlignment(Qt.AlignTop)
            layout.setHorizontalSpacing(8)
            layout.setVerticalSpacing(7)
            for row in range(layout.rowCount()):
                label_item = layout.itemAt(row, QFormLayout.LabelRole)
                if label_item is not None:
                    label_widget = label_item.widget()
                    if isinstance(label_widget, QLabel):
                        label_widget.setWordWrap(True)
                        label_widget.setMinimumWidth(0)
                field_item = layout.itemAt(row, QFormLayout.FieldRole)
                if field_item is not None:
                    field_widget = field_item.widget()
                    if field_widget is not None:
                        field_widget.setMinimumWidth(0)
                        if not isinstance(field_widget, QCheckBox):
                            field_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _build_ui(self):
        central = QWidget()
        central.setAcceptDrops(True)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 7, 8, 7)
        root.setSpacing(6)

        header = QFrame()
        header.setObjectName("Header")
        h = QHBoxLayout(header)
        h.setContentsMargins(13, 6, 13, 6)
        brand_mark = QLabel()
        brand_mark.setObjectName("BrandMark")
        brand_mark.setFixedSize(42, 42)
        brand_pixmap = QPixmap(str(APP_DIR / "assets" / "PolyfittyStudio.png"))
        if not brand_pixmap.isNull():
            brand_mark.setPixmap(brand_pixmap.scaled(42, 42, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            h.addWidget(brand_mark)
        brand_col = QVBoxLayout()
        brand = QLabel("POLYFITTY STUDIO")
        brand.setObjectName("Brand")
        subtitle = QLabel("UNIFIED 3D SCENE  →  SHAPE-AWARE RECONSTRUCTION  →  MESH QUALITY")
        subtitle.setObjectName("SubTitle")
        brand_col.addWidget(brand)
        brand_col.addWidget(subtitle)
        h.addLayout(brand_col)
        h.addStretch(1)
        theme_label = QLabel("THEME")
        theme_label.setObjectName("Muted")
        h.addWidget(theme_label)
        self.theme_combo = WheelSafeComboBox()
        self.theme_combo.addItem("System", "system")
        for key, theme in THEMES.items():
            self.theme_combo.addItem(theme["label"], key)
        theme_index = self.theme_combo.findData(self._preferred_theme)
        if theme_index >= 0:
            self.theme_combo.setCurrentIndex(theme_index)
        self.theme_combo.setFixedWidth(132)
        self.theme_combo.setToolTip("System follows the Windows app appearance (Light or Dark). You can also select a palette manually.")
        self.theme_combo.currentIndexChanged.connect(self._apply_theme)
        h.addWidget(self.theme_combo)
        self.gpu_status_label = QLabel("● GPU · SCANNING")
        self.gpu_status_label.setObjectName("GPUStatus")
        self.gpu_status_label.setAlignment(Qt.AlignCenter)
        self.gpu_status_label.setMinimumWidth(112)
        self.gpu_status_label.setMaximumWidth(190)
        self.gpu_status_label.setToolTip("Scanning CUDA, HIP, oneAPI/SYCL and OpenCL GPU compute adapters.")
        h.addWidget(self.gpu_status_label)
        self.header_status = QLabel("● READY · PYTHON + SCIP")
        self._set_header_status_style("ready")
        h.addWidget(self.header_status)
        root.addWidget(header)

        project_panel = QFrame()
        project_panel.setObjectName("Panel")
        project_row = QHBoxLayout(project_panel)
        project_row.setContentsMargins(9, 4, 9, 4)
        project_row.setSpacing(6)
        input_label = QLabel("INPUT")
        input_label.setObjectName("Section")
        input_label.setMinimumWidth(38)
        project_row.addWidget(input_label)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Point cloud / mesh / VG / BVG input file")
        self.input_edit.setReadOnly(True)
        self.input_edit.setMinimumWidth(180)
        project_row.addWidget(self.input_edit, 2)
        self.browse_btn = QPushButton("Browse Input")
        self.browse_btn.clicked.connect(self._browse_input)
        project_row.addWidget(self.browse_btn)
        self.reimport_btn = QPushButton("↻ REIMPORT")
        self.reimport_btn.setToolTip(
            "Reload the original file from the INPUT path. This clears the current reconstruction preview/result "
            "and rebuilds the source scene and input mesh statistics from the original asset."
        )
        self.reimport_btn.setEnabled(False)
        self.reimport_btn.clicked.connect(self._reimport_source)
        project_row.addWidget(self.reimport_btn)

        output_label = QLabel("OUTPUT")
        output_label.setObjectName("Section")
        output_label.setMinimumWidth(48)
        project_row.addWidget(output_label)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("TEMP workspace / reconstructed baked GLB")
        self.output_edit.setMinimumWidth(180)
        self.output_edit.setReadOnly(True)
        self.output_edit.setToolTip("Shows the TEMP workspace until the reconstructed baked GLB is generated, then shows the GLB path.")
        project_row.addWidget(self.output_edit, 2)
        root.addWidget(project_panel)

        action_panel = QFrame()
        action_panel.setObjectName("Panel")
        action_layout = QHBoxLayout(action_panel)
        action_layout.setContentsMargins(9, 4, 9, 4)
        action_layout.setSpacing(6)
        self.run_btn = QPushButton("▶  RECONSTRUCT")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self._start_pipeline)
        action_layout.addWidget(self.run_btn, 2)
        self.pause_btn = QPushButton("Ⅱ PAUSE")
        self.pause_btn.setToolTip("Pause or resume the active reconstruction or automatic texture bake.")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        action_layout.addWidget(self.pause_btn, 1)
        self.cancel_btn = QPushButton("■ STOP")
        self.cancel_btn.setToolTip("Cancel the active reconstruction or automatic GLB texture bake.")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_pipeline)
        action_layout.addWidget(self.cancel_btn, 1)
        self.export_btn = QPushButton("EXPORT MESH · GLB / OBJ / PLY / STL / 3MF")
        self.export_btn.setObjectName("Export")
        self.export_btn.clicked.connect(self._export_mesh)
        self.export_btn.setEnabled(False)
        action_layout.addWidget(self.export_btn, 2)
        root.addWidget(action_panel)

        body_splitter = QSplitter(Qt.Horizontal)
        body_splitter.setChildrenCollapsible(False)
        body_splitter.setHandleWidth(5)
        root.addWidget(body_splitter, 1)

        left_panel = QFrame()
        left_panel.setObjectName("Panel")
        left_panel.setMinimumWidth(250)
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(9, 8, 9, 8)
        left.setSpacing(7)
        left.setSizeConstraint(QLayout.SetMinimumSize)

        options_title = QLabel("OPTIONS · EXPAND TO CONFIGURE")
        options_title.setObjectName("Section")
        left.addWidget(options_title)

        axis_section = CollapsibleSection("SOURCE COORDINATE AXIS", expanded=False)
        axis_form = QFormLayout()
        self.source_up_axis = WheelSafeComboBox()
        self.source_up_axis.addItem("Y-up → Z-up · Rotate X +90° (default)", "y")
        self.source_up_axis.addItem("Z-up · No conversion", "z")
        self.source_up_axis.addItem("X-up → Z-up · Rotate Y -90°", "x")
        self.source_up_axis.setCurrentIndex(self.source_up_axis.findData("y"))
        self.source_up_axis.setToolTip("Set the source up-axis. Y-up is converted to the Studio Z-up frame by rotating X +90°. The same rotation is applied to the source preview and reconstruction input/output.")
        self.source_up_axis.currentIndexChanged.connect(self._on_axis_changed)
        axis_form.addRow("Source up-axis", self.source_up_axis)
        axis_section.add_layout(axis_form)
        left.addWidget(axis_section)

        cleanup_section = CollapsibleSection("MESH CLEANUP", expanded=False)
        cleanup_form = QFormLayout()
        self.merge_vertices_checkbox = QCheckBox("Merge vertices by distance")
        self.merge_vertices_checkbox.setChecked(True)
        self.merge_vertices_checkbox.setToolTip(
            "Weld mesh vertices closer than the selected Euclidean distance. Choose whether to apply "
            "the cleanup to the original input mesh, the reconstructed output mesh, or both. "
            "The original input file is never modified."
        )
        self.merge_vertices_checkbox.setEnabled(True)
        self.merge_target_combo = WheelSafeComboBox()
        self.merge_target_combo.addItem("INPUT + OUTPUT", "both")
        self.merge_target_combo.addItem("INPUT ONLY", "input")
        self.merge_target_combo.addItem("OUTPUT ONLY", "output")
        self.merge_target_combo.setCurrentIndex(self.merge_target_combo.findData("both"))
        self.merge_target_combo.setEnabled(False)
        self.merge_target_combo.setToolTip("Select which mesh stages receive Merge by Distance.")
        self.merge_distance_spin = WheelSafeDoubleSpinBox()
        self.merge_distance_spin.setDecimals(7)
        self.merge_distance_spin.setRange(0.0000001, 1000000.0)
        self.merge_distance_spin.setSingleStep(0.0001)
        self.merge_distance_spin.setValue(0.0001)
        self.merge_distance_spin.setEnabled(False)
        self.merge_distance_spin.setToolTip("Euclidean merge distance in mesh units. Default: 0.0001.")
        cleanup_form.addRow(self.merge_vertices_checkbox)
        cleanup_form.addRow("Apply to", self.merge_target_combo)
        cleanup_form.addRow("Merge distance", self.merge_distance_spin)
        self.merge_vertices_checkbox.toggled.connect(self._on_mesh_merge_options_changed)
        self.merge_target_combo.currentIndexChanged.connect(self._on_mesh_merge_options_changed)
        self.merge_distance_spin.valueChanged.connect(self._on_mesh_merge_options_changed)
        cleanup_section.add_layout(cleanup_form)
        left.addWidget(cleanup_section)

        overlay_section = CollapsibleSection("VIEWPORT OVERLAYS", expanded=True)
        overlay_toggles = QVBoxLayout()
        overlay_toggles.setContentsMargins(0, 0, 0, 0)
        overlay_toggles.setSpacing(4)
        self.wireframe_toggle = QCheckBox("Wireframe")
        self.wireframe_toggle.setToolTip("Show all source/result polygon edges.")
        self.face_islands_toggle = QCheckBox("Connected faces")
        self.face_islands_toggle.setToolTip("Show edge-connected face groups.")
        for toggle in (self.wireframe_toggle, self.face_islands_toggle):
            toggle.setChecked(False)
            toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            overlay_toggles.addWidget(toggle)
        overlay_toggles_widget = QWidget()
        overlay_toggles_widget.setLayout(overlay_toggles)
        overlay_section.add_widget(overlay_toggles_widget)
        left.addWidget(overlay_section)

        bake_section = CollapsibleSection("TEXTURE BAKING / EXPORT", expanded=False)
        bake_form = QFormLayout()
        self.texture_bake_checkbox = QCheckBox("Bake GLB after reconstruction")
        self.texture_bake_checkbox.setChecked(True)
        self.texture_bake_checkbox.setEnabled(False)
        self.texture_bake_checkbox.setToolTip(
            "Every successful reconstruction is baked to an internal GLB master. The final result viewport "
            "loads this actual GLB, and all later format conversions use the same baked master."
        )
        bake_form.addRow(self.texture_bake_checkbox)
        self.bake_projection_combo = WheelSafeComboBox()
        self.bake_projection_combo.addItem("Cage projection · auto-extruded cage", "cage")
        self.bake_projection_combo.addItem("Direct bake · nearest source surface (no cage)", "direct")
        saved_projection_mode = str(self.settings.value("bake_projection_mode", "direct") or "direct")
        projection_index = self.bake_projection_combo.findData(saved_projection_mode)
        if projection_index >= 0:
            self.bake_projection_combo.setCurrentIndex(projection_index)
        self.bake_projection_combo.setToolTip(
            "Cage projection uses GPU-accelerated ray/triangle intersection when a supported GPU is ready. "
            "Direct bake uses GPU-accelerated nearest-point/triangle queries when available and does not build a cage. "
            "Bake Compute Auto can switch remaining batches to CPU when same-input measurements show CPU is at least 10% faster; GPU Only stops instead of falling back."
        )
        self.bake_projection_combo.currentIndexChanged.connect(self._on_bake_projection_changed)
        bake_form.addRow("Bake projection", self.bake_projection_combo)
        self.bake_compute_combo = WheelSafeComboBox()
        self.bake_compute_combo.addItem("AUTO · GPU first → performance-aware CPU switch", "auto")
        self.bake_compute_combo.addItem("GPU only · strict GPU execution", "gpu")
        self.bake_compute_combo.addItem("CPU only · compatibility mode", "cpu")
        saved_compute_device = str(self.settings.value("bake_compute_device", "auto") or "auto")
        saved_index = self.bake_compute_combo.findData(saved_compute_device)
        if saved_index >= 0:
            self.bake_compute_combo.setCurrentIndex(saved_index)
        self.bake_compute_combo.setToolTip(
            "AUTO: starts with the best available GPU (CUDA, HIP, oneAPI/SYCL, then OpenCL), verifies the first real bake batch, "
            "and compares GPU vs CPU on the exact same inputs. If CPU is at least 10% faster, the verified current "
            "batch is kept and all remaining batches switch to CPU. GPU runtime/parity errors also use CPU. "
            "GPU only: never falls back; the bake stops with an error if GPU execution fails. "
            "CPU only: bypasses GPU. The comparison covers the ray/nearest-triangle compute stage, not the entire bake."
        )
        self.bake_compute_combo.currentIndexChanged.connect(self._on_bake_compute_device_changed)
        bake_form.addRow("Bake compute device", self.bake_compute_combo)
        gpu_status_box = QWidget()
        gpu_status_layout = QHBoxLayout(gpu_status_box)
        gpu_status_layout.setContentsMargins(0, 0, 0, 0)
        gpu_status_layout.setSpacing(6)
        gpu_status_box.setMinimumHeight(30)
        self.gpu_status_detail = QLabel("GPU SCAN")
        self.gpu_status_detail.setObjectName("Muted")
        self.gpu_status_detail.setWordWrap(False)
        self.gpu_status_detail.setMinimumWidth(0)
        self.gpu_status_detail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        gpu_status_layout.addWidget(self.gpu_status_detail, 1)
        self.gpu_status_tooltip = self.gpu_status_detail
        self.gpu_rescan_btn = QPushButton("Rescan")
        self.gpu_rescan_btn.setObjectName("GPURescanButton")
        self.gpu_rescan_btn.setFixedSize(88, 30)

        self.gpu_rescan_btn.setStyleSheet(
            "QPushButton#GPURescanButton { padding: 1px 8px; font-size: 8pt; font-weight: 700; }"
        )
        self.gpu_rescan_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.gpu_rescan_btn.clicked.connect(self._start_gpu_scan)
        gpu_status_layout.addWidget(self.gpu_rescan_btn, 0)
        bake_form.addRow("GPU", gpu_status_box)
        self.texture_resolution_combo = WheelSafeComboBox()
        self.texture_resolution_combo.addItem("1K · 1024²", 1024)
        self.texture_resolution_combo.addItem("2K · 2048²", 2048)
        self.texture_resolution_combo.addItem("4K · 4096²", 4096)
        self.texture_resolution_combo.setCurrentIndex(1)
        bake_form.addRow("Bake resolution", self.texture_resolution_combo)
        saved_bake_maps = self.settings.value("bake_pbr_maps", ["base_color", "orm"])
        if isinstance(saved_bake_maps, str):
            saved_bake_maps = [value.strip() for value in saved_bake_maps.split(",") if value.strip()]
        saved_bake_map_set = set(saved_bake_maps or ["base_color", "orm"])

        saved_bake_map_set.add("orm")
        self.bake_pbr_group = QGroupBox("PBR MAPS TO BAKE")
        bake_pbr_layout = QGridLayout(self.bake_pbr_group)
        bake_pbr_layout.setContentsMargins(8, 6, 8, 6)
        bake_pbr_layout.setHorizontalSpacing(10)
        bake_pbr_layout.setVerticalSpacing(4)
        self.bake_pbr_map_checks = {}
        for index, (key, label) in enumerate(PBR_MAP_LABELS.items()):
            checkbox = QCheckBox(label)
            checkbox.setChecked(key in saved_bake_map_set)
            if key == "orm":
                checkbox.setEnabled(False)
                checkbox.setToolTip(
                    "Always baked automatically to preserve per-material AO, Roughness and Metallic values."
                )
            else:
                checkbox.setToolTip(
                    "Bake this source PBR channel into an independent image. "
                    "Maps not present on the source material use neutral defaults."
                )
            self.bake_pbr_map_checks[key] = checkbox
            bake_pbr_layout.addWidget(checkbox, index // 2, index % 2)
            checkbox.toggled.connect(self._save_bake_pbr_map_settings)
        bake_form.addRow(self.bake_pbr_group)
        self.cage_extrusion_spin = WheelSafeDoubleSpinBox()
        self.cage_extrusion_spin.setDecimals(1)
        self.cage_extrusion_spin.setRange(0.1, 50.0)
        self.cage_extrusion_spin.setSingleStep(1.0)
        self.cage_extrusion_spin.setValue(5.0)
        self.cage_extrusion_spin.setSuffix(" %")
        self.cage_extrusion_spin.setToolTip(
            "Extrudes the automatic bake cage by this percentage of the reconstructed mesh bounding-box diagonal. "
            "Increase it when the original source surface lies farther from the reconstruction; excessively large "
            "values can allow rays to hit unrelated surfaces."
        )
        bake_form.addRow("Cage extrusion", self.cage_extrusion_spin)
        self._on_bake_projection_changed()
        bake_section.add_layout(bake_form)
        left.addWidget(bake_section)

        engine_section = CollapsibleSection("RECONSTRUCTION ENGINE", expanded=False)
        engine_form = QFormLayout()
        self.engine_mode = WheelSafeComboBox()
        self.engine_mode.addItem("AUTO · PolyFit first → Li & Nan fallback", "auto")
        self.engine_mode.addItem("PolyFit · Point clouds / planar structures", "polyfit")
        self.engine_mode.addItem("Li & Nan 2021 · Feature-preserving simplification", "li_nan")
        self.engine_mode.setToolTip(
            "AUTO runs PolyFit first. For an existing triangle mesh, a PolyFit error, timeout, or worker-process crash "
            "automatically triggers Li & Nan 2021 on the original input mesh. Point clouds and VG/BVG inputs use PolyFit. "
            "Li & Nan preserves planar features through bilateral normal filtering, planar-region extraction, and region-constrained QEM."
        )
        engine_form.addRow("Reconstruction method", self.engine_mode)
        self.engine_route_hint = QLabel("AUTO · PolyFit first → Li & Nan fallback for meshes · Point clouds / VG / BVG → PolyFit")
        self.engine_route_hint.setObjectName("Muted")
        self.engine_route_hint.setWordWrap(True)
        engine_form.addRow("AUTO routing", self.engine_route_hint)
        self.linan_options_widget = QWidget()
        linan_form = QFormLayout(self.linan_options_widget)
        linan_form.setContentsMargins(0, 4, 0, 4)
        linan_form.setSpacing(5)
        self.linan_target_vertices = WheelSafeSpinBox()
        self.linan_target_vertices.setRange(0, 100000000)
        self.linan_target_vertices.setValue(0)
        self.linan_target_vertices.setSpecialValueText("AUTO")
        self.linan_target_vertices.setToolTip(
            "Li & Nan AUTO uses about 2 × the detected planar-region count as the target vertex count. "
            "Enter a positive value to override the automatic target."
        )
        linan_form.addRow("Target vertices (0=AUTO)", self.linan_target_vertices)

        self.linan_feature_settings = QWidget()
        feature_form = QFormLayout(self.linan_feature_settings)
        feature_form.setContentsMargins(0, 2, 0, 2)
        feature_form.setSpacing(4)
        self.linan_filter_iterations = WheelSafeSpinBox()
        self.linan_filter_iterations.setRange(1, 100)
        self.linan_filter_iterations.setValue(10)
        self.linan_filter_iterations.setToolTip(
            "Li & Nan report 10–20 iterations as a practical range. The default is 10."
        )
        feature_form.addRow("Filter iterations", self.linan_filter_iterations)
        self.linan_filter_normal_angle = WheelSafeDoubleSpinBox()
        self.linan_filter_normal_angle.setDecimals(1)
        self.linan_filter_normal_angle.setRange(1.0, 89.0)
        self.linan_filter_normal_angle.setSingleStep(1.0)
        self.linan_filter_normal_angle.setValue(20.0)
        self.linan_filter_normal_angle.setSuffix("°")
        self.linan_filter_normal_angle.setToolTip(
            "Normal similarity threshold for bilateral face-normal filtering. Paper default: 20 degrees."
        )
        feature_form.addRow("Filter normal angle", self.linan_filter_normal_angle)
        self.linan_region_normal_angle = WheelSafeDoubleSpinBox()
        self.linan_region_normal_angle.setDecimals(1)
        self.linan_region_normal_angle.setRange(1.0, 89.0)
        self.linan_region_normal_angle.setSingleStep(1.0)
        self.linan_region_normal_angle.setValue(15.0)
        self.linan_region_normal_angle.setSuffix("°")
        self.linan_region_normal_angle.setToolTip(
            "Normal-direction tolerance for edge-connected planar region growing. The paper's experiments use 15 degrees."
        )
        feature_form.addRow("Region normal angle", self.linan_region_normal_angle)
        self.linan_small_region_area = WheelSafeDoubleSpinBox()
        self.linan_small_region_area.setDecimals(6)
        self.linan_small_region_area.setRange(0.0, 1e12)
        self.linan_small_region_area.setValue(0.0)
        self.linan_small_region_area.setSpecialValueText("AUTO")
        self.linan_small_region_area.setToolTip(
            "Small-region area threshold in squared mesh units. The paper does not prescribe a universal value. "
            "AUTO uses four times the median input triangle area."
        )
        feature_form.addRow("Small-region area (0=AUTO)", self.linan_small_region_area)
        self.linan_fitting_threshold = WheelSafeDoubleSpinBox()
        self.linan_fitting_threshold.setDecimals(6)
        self.linan_fitting_threshold.setRange(0.0, 1e12)
        self.linan_fitting_threshold.setValue(0.0)
        self.linan_fitting_threshold.setSpecialValueText("AUTO")
        self.linan_fitting_threshold.setToolTip(
            "Sum-of-vertex-distances threshold in mesh units. The paper leaves this data-dependent. "
            "AUTO uses a scale-aware bounding-box-diagonal heuristic (documented in the implementation notes)."
        )
        feature_form.addRow("Planar fit threshold (0=AUTO)", self.linan_fitting_threshold)
        self.linan_boundary_lock = QCheckBox("Protect open mesh boundaries\n(application safety)")
        self.linan_boundary_lock.setChecked(True)
        self.linan_boundary_lock.setToolTip(
            "Application safety option, not a Li & Nan paper requirement. Boundary-adjacent vertices are kept fixed; "
            "the requested target may not be reached on open surfaces."
        )
        feature_form.addRow(self.linan_boundary_lock)
        self.linan_intersection_guard = QCheckBox("Protect close surfaces during filtering\nand edge collapse (forced ON)")
        self.linan_intersection_guard.setChecked(True)
        self.linan_intersection_guard.setEnabled(False)
        self.linan_intersection_guard.setToolTip(
            "Both application-side guards are forced ON for direct Li & Nan and AUTO mesh inputs: Stage 1 attenuates filter displacement near close overlapping surfaces, and Stage 3 rejects newly introduced near-surface crossings during edge collapse. These safeguards are not part of the paper's equations and can increase processing time."
        )
        feature_form.addRow(self.linan_intersection_guard)
        linan_form.addRow(self.linan_feature_settings)
        engine_form.addRow(self.linan_options_widget)

        self.engine_mode.currentIndexChanged.connect(self._on_engine_mode_changed)
        self._on_engine_mode_changed()
        engine_section.add_layout(engine_form)
        left.addWidget(engine_section)

        resolution_section = CollapsibleSection("OUTPUT RESOLUTION", expanded=False)
        resolution_form = QFormLayout()
        self.resolution_preset = WheelSafeComboBox()
        self.resolution_preset.addItem("Draft · Fast preview", "draft")
        self.resolution_preset.addItem("Standard · Balanced", "standard")
        self.resolution_preset.addItem("High · Detailed", "high")
        self.resolution_preset.addItem("Ultra · Maximum detail", "ultra")
        self.resolution_preset.addItem("Custom · Manual", "custom")
        self.resolution_preset.setCurrentIndex(1)
        self.resolution_preset.setToolTip(
            "Controls the maximum surface sample count. PolyFit polygon count is determined by the native SCIP model-complexity weight."
        )
        resolution_form.addRow("Resolution preset", self.resolution_preset)
        self.resolution_hint = QLabel("180k samples · Balanced")
        self.resolution_hint.setObjectName("Muted")
        self.resolution_hint.setWordWrap(True)
        resolution_form.addRow(self.resolution_hint)
        self.max_points = WheelSafeComboBox()
        for label, value in [("100k points", 100000), ("180k points", 180000), ("300k points", 300000), ("500k points", 500000), ("No cap", 0)]:
            self.max_points.addItem(label, value)
        self.max_points.setCurrentIndex(1)
        self.max_points.setToolTip("Maximum number of surface samples used for reconstruction.")
        resolution_form.addRow("Max samples", self.max_points)
        resolution_section.add_layout(resolution_form)
        left.addWidget(resolution_section)
        self._applying_resolution_preset = False
        self._resolution_preset_values = RESOLUTION_PRESETS
        self.resolution_preset.currentIndexChanged.connect(self._apply_resolution_preset)
        self.max_points.currentIndexChanged.connect(self._on_manual_resolution_changed)

        seg_section = CollapsibleSection("PLANAR SEGMENTATION · POLYFIT", expanded=False)
        seg_form = QFormLayout()
        self.preserve_small_features = QCheckBox("Preserve small features / components")
        self.preserve_small_features.setChecked(True)
        self.preserve_small_features.setToolTip(
            "Preserves small roof planes, eaves, protrusions, and disconnected mesh components. "
            "Applies minimum surface sampling per mesh component, a lower plane-detection threshold, and a smaller normal radius for detail."
        )
        seg_form.addRow(self.preserve_small_features)
        self.small_feature_min_points = WheelSafeSpinBox()
        self.small_feature_min_points.setRange(12, 5000)
        self.small_feature_min_points.setValue(40)
        self.small_feature_min_points.setEnabled(True)
        self.preserve_small_features.toggled.connect(self.small_feature_min_points.setEnabled)
        self.small_feature_min_points.setToolTip("Minimum point count for a small plane. Lower values detect smaller structures but may include noisy planes.")
        seg_form.addRow("Small-plane min points", self.small_feature_min_points)
        self.voxel = WheelSafeDoubleSpinBox()
        self.voxel.setDecimals(5)
        self.voxel.setRange(0, 10000)
        self.voxel.setSingleStep(0.001)
        self.voxel.setValue(0.0)
        self.voxel.setToolTip("0 disables voxel downsampling. Uses input coordinate units.")
        seg_form.addRow("Voxel size (0=OFF)", self.voxel)
        self.threshold = WheelSafeDoubleSpinBox()
        self.threshold.setDecimals(6)
        self.threshold.setRange(0, 100000)
        self.threshold.setSingleStep(0.001)
        self.threshold.setValue(0.0)
        self.threshold.setToolTip("0 automatically uses 0.25% of the bounding-box diagonal.")
        seg_form.addRow("Plane tolerance (0=AUTO)", self.threshold)
        self.min_plane_points = WheelSafeSpinBox()
        self.min_plane_points.setRange(30, 10000000)
        self.min_plane_points.setValue(250)
        seg_form.addRow("Min points per plane", self.min_plane_points)
        self.max_planes = WheelSafeSpinBox()
        self.max_planes.setRange(1, 512)
        self.max_planes.setValue(48)
        self.max_planes.setToolTip(
            "Increase this value for models with many small roof planes, eaves, or auxiliary planes. "
            "A warning is shown if the plane limit is reached while unclassified points remain."
        )
        seg_form.addRow("Max planes", self.max_planes)
        self.iterations = WheelSafeSpinBox()
        self.iterations.setRange(50, 5000)
        self.iterations.setSingleStep(50)
        self.iterations.setValue(350)
        seg_form.addRow("RANSAC iterations", self.iterations)
        seg_section.add_layout(seg_form)
        left.addWidget(seg_section)

        weight_section = CollapsibleSection("ADVANCED · POLYFIT WEIGHTS", expanded=False)
        weight_form = QFormLayout()
        self.w_fit = WheelSafeDoubleSpinBox()
        self.w_fit.setDecimals(3); self.w_fit.setRange(0.0, 1.0); self.w_fit.setSingleStep(0.01); self.w_fit.setValue(0.43)
        self.w_coverage = WheelSafeDoubleSpinBox()
        self.w_coverage.setDecimals(3); self.w_coverage.setRange(0.0, 1.0); self.w_coverage.setSingleStep(0.01); self.w_coverage.setValue(0.27)
        self.w_complexity = WheelSafeDoubleSpinBox()
        self.w_complexity.setDecimals(3); self.w_complexity.setRange(0.0, 1.0); self.w_complexity.setSingleStep(0.01); self.w_complexity.setValue(0.30)
        weight_form.addRow("Data fitting", self.w_fit)
        weight_form.addRow("Model coverage", self.w_coverage)
        weight_form.addRow("Model complexity", self.w_complexity)
        weight_section.add_layout(weight_form)
        left.addWidget(weight_section)

        shortcut_section = CollapsibleSection("VIEWPORT SHORTCUTS", expanded=False)
        shortcut_form = QFormLayout()
        shortcut_form.setContentsMargins(2, 2, 2, 2)
        shortcut_form.setHorizontalSpacing(8)
        shortcut_form.setVerticalSpacing(4)

        shortcut_intro = QLabel("Move: hold RMB + WASD/QE · Shift: Ramp Speed · Orbit: ALT + LMB · Pan: MMB · Zoom: Wheel")
        shortcut_intro.setObjectName("Muted")
        shortcut_intro.setWordWrap(True)
        shortcut_form.addRow(shortcut_intro)

        fixed_shortcuts = (
            ("Fly forward", "W"),
            ("Fly backward", "S"),
            ("Fly left", "A"),
            ("Fly right", "D"),
            ("Fly down", "Q"),
            ("Fly up", "E"),
            ("Focus / restore camera", "F"),
            ("Toggle wireframe", "X"),
            ("Toggle connected faces", "C"),
            ("Toggle view-aligned grid", "G"),
            ("Reimport source", "R"),
            ("Toggle face overlay", "Z"),
        )
        for label_text, key_text in fixed_shortcuts:
            key_label = QLabel(key_text)
            key_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            key_label.setMinimumWidth(38)
            key_label.setMaximumWidth(48)
            key_label.setObjectName("ShortcutKey")
            shortcut_form.addRow(label_text, key_label)
        shortcut_section.add_layout(shortcut_form)
        left.addWidget(shortcut_section)
        left.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setObjectName("OptionsScrollArea")
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setMinimumWidth(258)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        left_scroll.setAlignment(Qt.AlignTop)
        left_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_scroll.setWidget(left_panel)
        self.left_options_scroll = left_scroll
        body_splitter.addWidget(left_scroll)

        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(5)
        body_splitter.addWidget(right_widget)
        body_splitter.setStretchFactor(0, 1)
        body_splitter.setStretchFactor(1, 4)
        initial_left_width = max(310, min(360, int(self.width() * 0.21)))
        body_splitter.setSizes([initial_left_width, max(700, self.width() - initial_left_width - 20)])

        progress_panel = QFrame()
        progress_panel.setObjectName("Panel")
        progress_layout = QVBoxLayout(progress_panel)
        progress_layout.setContentsMargins(9, 4, 9, 4)
        progress_layout.setSpacing(5)
        ptop = QHBoxLayout()
        ptop.setSpacing(9)
        self.stage_label = QLabel("Ready · Select an input file")
        self.stage_label.setObjectName("Section")
        self.stage_label.setMinimumWidth(260)
        self.stage_label.setToolTip("Unified Reconstruction → OBJ validation → Texture Baking → GLB validation pipeline")
        self.percent_label = QLabel("0%")
        self.percent_label.setObjectName("ProgressPercent")
        ptop.addWidget(self.stage_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        ptop.addWidget(self.progress_bar, 1)
        ptop.addWidget(self.percent_label)
        progress_layout.addLayout(ptop)
        progress_panel.setMaximumHeight(38)
        right.addWidget(progress_panel, 0)

        scene_panel = QFrame()
        scene_panel.setObjectName("Panel")
        scene_layout = QVBoxLayout(scene_panel)
        scene_layout.setContentsMargins(7, 5, 7, 5)
        scene_layout.setSpacing(4)

        self.scene_info = QLabel(self)
        self.scene_info.setVisible(False)
        self.face_overlay_toggle = QToolButton()
        self.face_overlay_toggle.setObjectName("ViewportFaceOverlayToggle")
        self.face_overlay_toggle.setText("FACE OVERLAY OFF")
        self.face_overlay_toggle.setCheckable(True)
        self.face_overlay_toggle.setChecked(False)
        self.face_overlay_toggle.setToolTip(
            "Show surface overlays through the mesh. Reconstruction previews remain edge-only."
        )
        self.face_overlay_toggle.setFixedHeight(28)
        self.face_overlay_toggle.setMinimumWidth(132)
        self.face_overlay_toggle.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.face_overlay_toggle.toggled.connect(self._set_face_overlay_enabled)

        startup_trace("SceneView construction starting")
        self.scene_view = SceneView()
        startup_trace("SceneView construction completed")
        self.scene_view.emptyViewportClicked.connect(self._browse_input)
        self._apply_viewport_shortcuts()
        self.scene_view.set_infront_mode("both")
        self.scene_view.set_wireframe_enabled(False)
        self.scene_view.set_sharp_edges_enabled(False)
        self.scene_view.set_face_islands_enabled(False)
        self.scene_view.viewportShortcutTriggered.connect(self._handle_viewport_shortcut)

        self.scene_controls = QFrame()
        self.scene_controls.setObjectName("ViewportControls")
        self.scene_controls.setAttribute(Qt.WA_StyledBackground, True)
        self.scene_controls.setMinimumWidth(0)
        self.scene_controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        controls_layout = QVBoxLayout(self.scene_controls)
        controls_layout.setContentsMargins(11, 8, 11, 9)
        controls_layout.setSpacing(5)
        controls_title = QLabel("VIEWPORT CONTROLS")
        controls_title.setObjectName("Section")
        controls_title.setStyleSheet("font-weight:800; font-size:8pt; letter-spacing:1px;")
        controls_layout.addWidget(controls_title)

        item_info_row = QHBoxLayout()
        item_info_row.setSpacing(6)
        self.item_info_enabled_check = QCheckBox()
        self.item_info_enabled_check.setObjectName("ViewportItemInfoEnabled")
        self.item_info_enabled_check.setAccessibleName("ITEM INFO")
        if not bool(self.settings.value("viewport/item_info_default_off_v1", False, type=bool)):
            self.settings.setValue("viewport/item_info_enabled", False)
            self.settings.setValue("viewport/item_info_default_off_v1", True)
            self.settings.sync()
        saved_item_info_enabled = bool(
            self.settings.value("viewport/item_info_enabled", False, type=bool)
        )
        saved_item_info_mode = str(
            self.settings.value("viewport/item_info_mode", "DIM") or "DIM"
        ).strip().upper()
        if saved_item_info_mode not in {"LOC", "ROT", "SCALE", "DIM"}:
            saved_item_info_mode = "DIM"
        self.item_info_enabled_check.setChecked(saved_item_info_enabled)
        self._sync_item_info_checkbox_icon(saved_item_info_enabled)
        self.item_info_enabled_check.setToolTip(
            "Show one item-information overlay at a time: location, rotation, scale, or dimensions."
        )
        item_info_row.addWidget(self.item_info_enabled_check)
        item_info_row.addStretch(1)
        controls_layout.addLayout(item_info_row)

        item_info_mode_row = QHBoxLayout()
        item_info_mode_row.setSpacing(3)
        self.item_info_mode_group = QButtonGroup(self.scene_controls)
        self.item_info_mode_group.setExclusive(True)
        self.item_info_mode_buttons = {}
        for mode in ("LOC", "ROT", "SCALE", "DIM"):
            button = QToolButton(self.scene_controls)
            button.setObjectName(f"ViewportItemInfo{mode.title()}Button")
            button.setText("[SCL]" if mode == "SCALE" else f"[{mode}]")
            button.setCheckable(True)
            button.setAutoRaise(False)
            button.setFixedHeight(27)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setToolTip({
                "LOC": "Show source-object origin location.",
                "ROT": "Show source-object rotation; switch Euler / Quaternion in the viewport.",
                "SCALE": "Show source-object X / Y / Z scale.",
                "DIM": "Show X / Y / Z dimensions on the nearest camera-visible oriented bounding-box edges.",
            }[mode])
            button.setChecked(mode == saved_item_info_mode)
            self.item_info_mode_group.addButton(button)
            self.item_info_mode_buttons[mode] = button
            item_info_mode_row.addWidget(button, 1)
        controls_layout.addLayout(item_info_mode_row)

        self._ensure_default_metric_units()
        saved_location_unit = str(
            self.settings.value("viewport/location_display_unit", "m") or "m"
        ).strip().lower()
        if saved_location_unit not in {"mm", "cm", "m", "km"}:
            saved_location_unit = "m"
        saved_dimension_unit = str(
            self.settings.value("viewport/dimension_display_unit", "m") or "m"
        ).strip().lower()
        if saved_dimension_unit not in {"mm", "cm", "m", "km"}:
            saved_dimension_unit = "m"
        saved_rotation_display_mode = str(
            self.settings.value("viewport/rotation_display_mode", "euler") or "euler"
        ).strip().lower()
        if saved_rotation_display_mode not in {"euler", "quaternion"}:
            saved_rotation_display_mode = "euler"

        self.item_info_unit_row_widget = QWidget(self.scene_controls)
        self.item_info_unit_row_widget.setObjectName("ViewportItemInfoUnitRow")
        item_info_unit_row = QHBoxLayout(self.item_info_unit_row_widget)
        item_info_unit_row.setContentsMargins(0, 0, 0, 0)
        item_info_unit_row.setSpacing(3)
        self.item_info_unit_group = QButtonGroup(self.item_info_unit_row_widget)
        self.item_info_unit_group.setExclusive(True)
        self.item_info_unit_buttons = {}
        selected_initial_unit = (
            saved_location_unit if saved_item_info_mode == "LOC" else saved_dimension_unit
        )
        unit_button_style = (
            "QToolButton { color:#d8e4f1; background:rgba(27,34,43,150); "
            "border:1px solid rgba(190,205,222,80); border-radius:4px; "
            "font-size:9px; font-weight:800; padding:2px 4px; }"
            "QToolButton:checked { color:#ffffff; background:rgba(59,88,117,230); "
            "border:1px solid rgba(142,190,235,220); }"
            "QToolButton:hover { border-color:rgba(205,225,246,170); }"
        )
        for unit in ("mm", "cm", "m", "km"):
            button = QToolButton(self.item_info_unit_row_widget)
            button.setObjectName(f"ViewportItemInfoUnit{unit.upper()}Button")
            button.setText(f"[{unit.upper()}]")
            button.setCheckable(True)
            button.setAutoRaise(False)
            button.setFixedHeight(24)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setStyleSheet(unit_button_style)
            button.setToolTip(f"Display location and dimensions in {unit}.")
            button.setChecked(unit == selected_initial_unit)
            self.item_info_unit_group.addButton(button)
            self.item_info_unit_buttons[unit] = button
            item_info_unit_row.addWidget(button, 1)
            button.toggled.connect(
                lambda checked, selected_unit=unit: self._on_item_info_unit_changed(
                    selected_unit, checked
                )
            )
        controls_layout.addWidget(self.item_info_unit_row_widget)

        self.item_info_rotation_row_widget = QWidget(self.scene_controls)
        self.item_info_rotation_row_widget.setObjectName("ViewportItemInfoRotationRow")
        item_info_rotation_row = QHBoxLayout(self.item_info_rotation_row_widget)
        item_info_rotation_row.setContentsMargins(0, 0, 0, 0)
        item_info_rotation_row.setSpacing(3)
        self.item_info_rotation_group = QButtonGroup(self.item_info_rotation_row_widget)
        self.item_info_rotation_group.setExclusive(True)
        self.item_info_rotation_buttons = {}
        for rotation_mode, label in (("euler", "[ELR]"), ("quaternion", "[QTN]")):
            button = QToolButton(self.item_info_rotation_row_widget)
            button.setObjectName(
                "ViewportItemInfoRotationEulerButton"
                if rotation_mode == "euler"
                else "ViewportItemInfoRotationQuaternionButton"
            )
            button.setText(label)
            button.setCheckable(True)
            button.setAutoRaise(False)
            button.setFixedHeight(24)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setStyleSheet(unit_button_style)
            button.setToolTip(
                "Display rotation as Euler XYZ angles."
                if rotation_mode == "euler"
                else "Display rotation as Quaternion XYZW."
            )
            button.setChecked(rotation_mode == saved_rotation_display_mode)
            self.item_info_rotation_group.addButton(button)
            self.item_info_rotation_buttons[rotation_mode] = button

            item_info_rotation_row.addWidget(button, 2)
            button.toggled.connect(
                lambda checked, selected_mode=rotation_mode: self._on_item_info_rotation_changed(
                    selected_mode, checked
                )
            )
        controls_layout.addWidget(self.item_info_rotation_row_widget)

        self.item_info_enabled_check.toggled.connect(self._sync_item_info_checkbox_icon)
        self.item_info_enabled_check.toggled.connect(self._on_item_info_enabled_changed)
        for mode, button in self.item_info_mode_buttons.items():
            button.toggled.connect(
                lambda checked, selected_mode=mode: self._on_item_info_mode_changed(selected_mode, checked)
            )
        self.scene_view.set_item_info_enabled(saved_item_info_enabled)
        self.scene_view.set_item_info_mode(saved_item_info_mode)
        self.scene_view.set_location_display_unit(saved_location_unit)
        self.scene_view.set_dimension_display_unit(saved_dimension_unit)
        self.scene_view.set_rotation_display_mode(saved_rotation_display_mode)
        self._update_item_info_controls_visibility()

        surface_label_row = QHBoxLayout()
        self.source_surface_label = QLabel("A · SOURCE 100%")
        self.result_surface_label = QLabel("B · RESULT 0%")
        surface_label_row.addWidget(self.source_surface_label)
        surface_label_row.addStretch(1)
        surface_label_row.addWidget(self.result_surface_label)
        controls_layout.addLayout(surface_label_row)
        self.surface_mix_slider = WheelSafeSlider(Qt.Horizontal)
        self.surface_mix_slider.setRange(0, 100)
        self.surface_mix_slider.setValue(0)
        self.surface_mix_slider.setToolTip(
            "0% = A · Source fully visible / B · Result hidden. "
            "100% = B · Result fully visible / A · Source hidden."
        )
        controls_layout.addWidget(self.surface_mix_slider)

        self.wireframe_options_widget = QWidget()
        wireframe_options_layout = QVBoxLayout(self.wireframe_options_widget)
        wireframe_options_layout.setContentsMargins(0, 0, 0, 0)
        wireframe_options_layout.setSpacing(4)
        wire_mix_labels = QHBoxLayout()
        self.source_wire_label = QLabel("A · SOURCE 0%")
        wire_mix_labels.addWidget(self.source_wire_label)
        wire_mix_labels.addStretch(1)
        self.result_wire_label = QLabel("B · RESULT 100%")
        wire_mix_labels.addWidget(self.result_wire_label)
        wireframe_options_layout.addLayout(wire_mix_labels)
        self.wire_mix_slider = WheelSafeSlider(Qt.Horizontal)
        self.wire_mix_slider.setRange(0, 100)
        self.wire_mix_slider.setValue(100)
        self.wire_mix_slider.setToolTip("Blend source and result wireframes.")
        wireframe_options_layout.addWidget(self.wire_mix_slider)
        wireframe_options_title = QLabel("WIRE FRAME")
        wireframe_options_title.setObjectName("Section")
        wireframe_options_layout.addWidget(wireframe_options_title)
        controls_layout.addWidget(self.wireframe_options_widget)

        self.connected_faces_options_widget = QWidget()
        connected_faces_layout = QVBoxLayout(self.connected_faces_options_widget)
        connected_faces_layout.setContentsMargins(0, 0, 0, 0)
        connected_faces_layout.setSpacing(3)
        connected_faces_row = QHBoxLayout()
        connected_faces_title = QLabel("CONNECTED FACE OPACITY")
        connected_faces_title.setObjectName("Section")
        self.connected_faces_opacity_label = QLabel("100%")
        self.connected_faces_opacity_label.setObjectName("Muted")
        connected_faces_row.addWidget(connected_faces_title)
        connected_faces_row.addStretch(1)
        connected_faces_row.addWidget(self.connected_faces_opacity_label)
        connected_faces_layout.addLayout(connected_faces_row)
        self.connected_faces_opacity_slider = WheelSafeSlider(Qt.Horizontal)
        self.connected_faces_opacity_slider.setRange(10, 100)
        self.connected_faces_opacity_slider.setValue(100)
        self.connected_faces_opacity_slider.valueChanged.connect(self._on_face_islands_opacity_changed)
        connected_faces_layout.addWidget(self.connected_faces_opacity_slider)
        controls_layout.addWidget(self.connected_faces_options_widget)

        self.overlay_mode_widget = QWidget()
        overlay_mode_box = QVBoxLayout(self.overlay_mode_widget)
        overlay_mode_box.setContentsMargins(0, 0, 0, 0)
        overlay_mode_box.setSpacing(3)
        overlay_mode_label = QLabel("OVERLAY COLOR MODE")
        overlay_mode_label.setObjectName("Section")
        overlay_mode_label.setWordWrap(True)
        overlay_mode_box.addWidget(overlay_mode_label)
        self.overlay_color_mode_combo = WheelSafeComboBox()
        self.overlay_color_mode_combo.addItem("INVERT", "invert")
        self.overlay_color_mode_combo.addItem("WHITE", "solid")
        self.overlay_color_mode_combo.addItem("CUSTOM COLORS", "custom")
        saved_overlay_mode = str(self.settings.value("viewport/overlay_color_mode", "invert") or "invert").lower()
        saved_overlay_index = self.overlay_color_mode_combo.findData(saved_overlay_mode)
        self.overlay_color_mode_combo.setCurrentIndex(saved_overlay_index if saved_overlay_index >= 0 else 0)
        self.overlay_color_mode_combo.currentIndexChanged.connect(self._on_viewport_overlay_color_mode_changed)
        self.overlay_color_mode_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        overlay_mode_box.addWidget(self.overlay_color_mode_combo)
        controls_layout.addWidget(self.overlay_mode_widget)

        self._viewport_overlay_color_buttons = {}
        self._overlay_color_widgets = {}
        for overlay_key, overlay_label in (("wireframe", "WIREFRAME COLOR"), ("sharp_edges", "SHARP EDGE COLOR")):
            color_widget = QWidget()
            color_group = QVBoxLayout(color_widget)
            color_group.setContentsMargins(0, 0, 0, 0)
            color_group.setSpacing(3)
            color_title = QLabel(overlay_label)
            color_title.setObjectName("Section")
            color_title.setWordWrap(True)
            color_group.addWidget(color_title)
            color_row = QHBoxLayout()
            color_row.setSpacing(6)
            for layer, short_label in (("source", "A · SOURCE"), ("result", "B · RESULT")):
                setting_key = f"viewport/overlay_colors/{overlay_key}/{layer}"
                saved_color = QColor(str(self.settings.value(setting_key, "#FFFFFF") or "#FFFFFF"))
                if not saved_color.isValid():
                    saved_color = QColor("#FFFFFF")
                button = QPushButton(short_label)
                button.setFixedHeight(23)
                button.setMinimumWidth(72)
                button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                button.setToolTip(f"Choose a custom {overlay_label.lower()} for the {layer} mesh. Selecting a color switches the mode to CUSTOM.")
                self._style_overlay_color_button(button, saved_color)
                button.clicked.connect(
                    lambda _checked=False, kind=overlay_key, mesh_layer=layer, target=button:
                    self._choose_viewport_overlay_color(kind, mesh_layer, target)
                )
                self._viewport_overlay_color_buttons[(overlay_key, layer)] = button
                color_row.addWidget(button)
                rgb = (saved_color.redF(), saved_color.greenF(), saved_color.blueF())
                if overlay_key == "wireframe":
                    self.scene_view.set_wireframe_color(layer, rgb)
                else:
                    self.scene_view.set_sharp_edge_color(layer, rgb)
            color_group.addLayout(color_row)
            self._overlay_color_widgets[overlay_key] = color_widget
            controls_layout.addWidget(color_widget)

        self.scene_view.set_overlay_color_mode(str(self.overlay_color_mode_combo.currentData() or "invert"))
        pass_label = QLabel("VTK RENDER MODE")
        pass_label.setObjectName("Section")
        controls_layout.addWidget(pass_label)
        pass_row = QHBoxLayout()
        pass_row.setSpacing(7)
        self.render_pass_combo = WheelSafeComboBox()
        self.render_pass_combo.addItem("— VTK RENDER MODES —", "")

        vtk_modes = (
            ("PBR", "pbr"),
            ("Diffuse (Albedo)", "diffuse_albedo"),
            ("Solid Surface", "surface"),
        )
        for label, identifier in vtk_modes:
            self.render_pass_combo.addItem(label, f"vtk:{identifier}")
        diffuse_index = self.render_pass_combo.findData("vtk:diffuse_albedo")
        if diffuse_index >= 0:
            self.render_pass_combo.setCurrentIndex(diffuse_index)
        self.render_pass_combo.setToolTip(
            "PBR uses physically based surface shading with the editable studio lights; "
            "Ambient Occlusion is not applied. Diffuse (Albedo) displays only the unmodified "
            "image texture RGB without material tint, solid material colors, vertex colors, "
            "or lighting. Mesh parts without an image texture use neutral white. "
            "Use the left-panel Wireframe toggle for 1 px edges."
        )
        pass_row.addWidget(self.render_pass_combo, 1)
        controls_layout.addLayout(pass_row)

        projection_section = CollapsibleSection("VIEWPORT PROJECTION", expanded=True)
        projection_form = QFormLayout()
        projection_form.setContentsMargins(0, 0, 0, 0)
        projection_form.setHorizontalSpacing(8)
        projection_form.setVerticalSpacing(5)

        self.projection_mode_combo = WheelSafeComboBox()
        self.projection_mode_combo.addItem("Perspective", "perspective")
        self.projection_mode_combo.addItem("Orthographic", "orthographic")
        saved_projection_mode = str(
            self.settings.value("viewport/projection_mode", "perspective") or "perspective"
        ).strip().lower()
        if saved_projection_mode not in {"perspective", "orthographic"}:
            saved_projection_mode = "perspective"
        self.projection_mode_combo.setCurrentIndex(
            max(0, self.projection_mode_combo.findData(saved_projection_mode))
        )
        self.projection_mode_combo.setToolTip(
            "Perspective uses depth-dependent perspective projection. Orthographic removes perspective distortion "
            "and snaps the camera to the selected reference axis."
        )
        projection_form.addRow("Projection", self.projection_mode_combo)

        self.projection_axis_combo = WheelSafeComboBox()
        for axis_name in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
            self.projection_axis_combo.addItem(f"{axis_name}-Axis", axis_name)
        saved_projection_axis = str(
            self.settings.value("viewport/orthographic_axis", "+Z") or "+Z"
        ).strip().upper()
        if saved_projection_axis in {"X", "Y", "Z"}:
            saved_projection_axis = "+" + saved_projection_axis
        if saved_projection_axis not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
            saved_projection_axis = "+Z"
        self.projection_axis_combo.setCurrentIndex(
            max(0, self.projection_axis_combo.findData(saved_projection_axis))
        )
        self.projection_axis_combo.setToolTip(
            "Choose the positive or negative world-axis direction to look along in Orthographic mode. "
            "The grid is drawn on the matching perpendicular plane, behind the model."
        )
        projection_form.addRow("Reference Axis", self.projection_axis_combo)
        projection_section.add_layout(projection_form)
        controls_layout.addWidget(projection_section)
        self.projection_mode_combo.currentIndexChanged.connect(self._on_viewport_projection_changed)
        self.projection_axis_combo.currentIndexChanged.connect(self._on_viewport_projection_axis_changed)
        self.scene_view.set_projection_mode(saved_projection_mode)
        self.scene_view.set_orthographic_axis(saved_projection_axis)
        self.projection_axis_combo.setEnabled(saved_projection_mode == "orthographic")

        source_alpha_section = CollapsibleSection("INPUT MATERIAL ALPHA", expanded=False)
        source_alpha_form = QFormLayout()
        source_alpha_form.setContentsMargins(0, 0, 0, 0)
        source_alpha_form.setHorizontalSpacing(8)
        source_alpha_form.setVerticalSpacing(5)
        self.input_alpha_mode_combo = WheelSafeComboBox()
        self.input_alpha_mode_combo.addItem("Alpha Clip", "ALPHA_CLIP")
        self.input_alpha_mode_combo.addItem("Opaque", "OPAQUE")
        self.input_alpha_mode_combo.addItem("Alpha Hashed", "ALPHA_HASHED")
        self.input_alpha_mode_combo.addItem("Alpha Blend", "ALPHA_BLEND")
        self.input_alpha_mode_combo.addItem("Use File", "USE_FILE")

        saved_input_alpha_mode = str(
            self.settings.value("viewport/input_alpha_mode", "USE_FILE") or "USE_FILE"
        ).strip().upper()
        if saved_input_alpha_mode not in {"ALPHA_CLIP", "OPAQUE", "ALPHA_HASHED", "ALPHA_BLEND", "USE_FILE"}:
            saved_input_alpha_mode = "USE_FILE"
        self.input_alpha_mode_combo.setCurrentIndex(
            max(0, self.input_alpha_mode_combo.findData(saved_input_alpha_mode))
        )
        self.input_alpha_mode_combo.setToolTip(
            "Controls alpha display for the imported source mesh only. Use File reads the original GLB/glTF material mode. "
            "Alpha Hashed uses an Alpha Clip fallback in the VTK viewport."
        )
        source_alpha_form.addRow("Alpha Mode", self.input_alpha_mode_combo)
        self.input_alpha_cutoff_spin = WheelSafeDoubleSpinBox()
        self.input_alpha_cutoff_spin.setRange(0.0, 1.0)
        self.input_alpha_cutoff_spin.setDecimals(2)
        self.input_alpha_cutoff_spin.setSingleStep(0.05)
        self.input_alpha_cutoff_spin.setValue(float(self.settings.value(
            "viewport/input_alpha_cutoff", 0.5, type=float
        )))
        self.input_alpha_cutoff_spin.setToolTip(
            "Alpha threshold from 0 to 1. Used by Alpha Clip / Alpha Hashed on the imported source preview."
        )
        source_alpha_form.addRow("Alpha Cutoff", self.input_alpha_cutoff_spin)
        source_alpha_section.add_layout(source_alpha_form)
        controls_layout.addWidget(source_alpha_section)
        self.input_alpha_mode_combo.currentIndexChanged.connect(self._on_input_material_alpha_changed)
        self.input_alpha_cutoff_spin.editingFinished.connect(self._on_input_material_alpha_changed)
        self._refresh_input_alpha_controls()

        camera_section = CollapsibleSection("VIEWPORT CAMERA", expanded=False)
        camera_form = QFormLayout()
        camera_form.setContentsMargins(0, 0, 0, 0)
        camera_form.setHorizontalSpacing(8)
        camera_form.setVerticalSpacing(5)

        self.camera_focal_length_spin = WheelSafeDoubleSpinBox()
        self.camera_focal_length_spin.setRange(1.0, 500.0)
        self.camera_focal_length_spin.setDecimals(4)
        self.camera_focal_length_spin.setSingleStep(0.1)
        self.camera_focal_length_spin.setSuffix(" mm")
        self.camera_focal_length_spin.setValue(float(self.settings.value(
            "viewport/camera/focal_length_mm", 26.9915, type=float
        )))
        self.camera_focal_length_spin.setToolTip(
            "Viewport lens focal length using a 36 mm horizontal sensor reference. "
            "The active VTK camera updates immediately."
        )
        camera_form.addRow("Focal Length", self.camera_focal_length_spin)

        self.camera_clip_start_spin = WheelSafeDoubleSpinBox()
        self.camera_clip_start_spin.setRange(0.001, 1_000_000.0)
        self.camera_clip_start_spin.setDecimals(3)
        self.camera_clip_start_spin.setSingleStep(0.1)
        self.camera_clip_start_spin.setSuffix(" m")
        self.camera_clip_start_spin.setValue(float(self.settings.value(
            "viewport/camera/clip_start_m", 0.1, type=float
        )))
        self.camera_clip_start_spin.setToolTip(
            "Near clipping plane in metres. Must remain below Clip End."
        )
        camera_form.addRow("Clip Start", self.camera_clip_start_spin)

        self.camera_clip_end_spin = WheelSafeDoubleSpinBox()
        self.camera_clip_end_spin.setRange(0.002, 10_000_000.0)
        self.camera_clip_end_spin.setDecimals(3)
        self.camera_clip_end_spin.setSingleStep(10.0)
        self.camera_clip_end_spin.setSuffix(" m")
        saved_clip_start = self.camera_clip_start_spin.value()
        saved_clip_end = float(self.settings.value(
            "viewport/camera/clip_end_m", 10000.0, type=float
        ))
        self.camera_clip_end_spin.setValue(max(saved_clip_end, saved_clip_start + 0.001))
        self.camera_clip_end_spin.setToolTip(
            "Far clipping plane in metres. Must remain above Clip Start."
        )
        camera_form.addRow("Clip End", self.camera_clip_end_spin)

        camera_note = QLabel("Lens and clipping values are linked to the live viewport camera.")
        camera_note.setObjectName("Muted")
        camera_note.setWordWrap(True)
        camera_section.add_widget(camera_note)
        camera_section.add_layout(camera_form)
        controls_layout.addWidget(camera_section)
        self.camera_focal_length_spin.valueChanged.connect(self._apply_viewport_camera_settings)
        self.camera_clip_start_spin.valueChanged.connect(self._apply_viewport_camera_settings)
        self.camera_clip_end_spin.valueChanged.connect(self._apply_viewport_camera_settings)
        self._apply_viewport_camera_settings()

        studio_lighting_section = CollapsibleSection("ENVIRONMENT & LIGHTING", expanded=False)
        studio_lighting_note = QLabel("PBR / Solid only · LIGHT 1–3 control direct lights; ENVIRONMENT controls IBL intensity. Drag the light control to rotate the key light.")
        studio_lighting_note.setObjectName("Muted")
        studio_lighting_note.setWordWrap(True)
        studio_lighting_section.add_widget(studio_lighting_note)

        self.studio_lights_enabled = QCheckBox("Enable studio lighting")
        self.studio_lights_enabled.setChecked(
            bool(self.settings.value("viewport/studio_lighting/enabled", True, type=bool))
        )
        self.studio_lights_enabled.setToolTip(
            "Master switch for studio lights in PBR and Solid Surface modes. Albedo is always independent of studio lighting."
        )
        studio_lighting_section.add_widget(self.studio_lights_enabled)

        rotation_section = CollapsibleSection("LIGHTING ROTATION", expanded=False)
        self.studio_rotation_widget = LightingOrbitWidget(
            float(self.settings.value("viewport/studio_lighting/key_azimuth", 10.0, type=float))
        )
        rotation_layout = QHBoxLayout()
        rotation_layout.setContentsMargins(0, 0, 0, 0)
        rotation_layout.addStretch(1)
        rotation_layout.addWidget(self.studio_rotation_widget)
        rotation_layout.addStretch(1)
        rotation_section.add_layout(rotation_layout)
        studio_lighting_section.add_widget(rotation_section)

        self.studio_light_controls = {}

        def add_light_control(key, title, default_intensity, expanded=False):
            light_section = CollapsibleSection(title, expanded=expanded)
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setHorizontalSpacing(8)
            form.setVerticalSpacing(5)

            enabled = QCheckBox("Enabled")
            enabled.setChecked(bool(self.settings.value(
                f"viewport/studio_lighting/{key}/enabled", True, type=bool
            )))
            form.addRow(enabled)

            color_row = QHBoxLayout()
            color_mode = WheelSafeComboBox()
            color_mode.addItem("HSV", "hsv")
            hex_edit = QLineEdit(str(self.settings.value(
                f"viewport/studio_lighting/{key}/color", "#FFFFFF"
            )))
            hex_edit.setMaxLength(7)
            hex_edit.setFixedWidth(92)
            color_row.addWidget(color_mode, 1)
            color_row.addWidget(hex_edit, 1)
            color_widget = QWidget()
            color_widget.setLayout(color_row)
            form.addRow("Color", color_widget)

            stored_color = QColor(hex_edit.text())
            if not stored_color.isValid():
                stored_color = QColor("#FFFFFF")
                hex_edit.setText("#FFFFFF")

            def add_color_slider(label, component, minimum, maximum, value):
                row = QHBoxLayout()
                row.setSpacing(6)
                slider = WheelSafeSlider(Qt.Horizontal)
                slider.setRange(minimum, maximum)
                slider.setValue(int(value))
                spin = WheelSafeSpinBox()
                spin.setRange(minimum, maximum)
                spin.setValue(int(value))
                spin.setFixedWidth(58)
                slider.valueChanged.connect(spin.setValue)
                spin.valueChanged.connect(slider.setValue)
                row.addWidget(slider, 1)
                row.addWidget(spin)
                widget = QWidget()
                widget.setLayout(row)
                form.addRow(label, widget)
                return slider, spin

            hue, hue_spin = add_color_slider("Hue", "hue", 0, 359, max(0, stored_color.hue()))
            saturation, saturation_spin = add_color_slider(
                "Saturation", "saturation", 0, 255, stored_color.saturation()
            )
            brightness, brightness_spin = add_color_slider(
                "Value", "value", 0, 255, stored_color.value()
            )
            saved_intensity = int(self.settings.value(
                f"viewport/studio_lighting/{key}/intensity", default_intensity, type=int
            ))
            intensity, intensity_spin = add_color_slider(
                "Intensity", "intensity", 0, 100, max(0, min(100, saved_intensity))
            )

            def color_from_hsv(*_args):
                color = QColor.fromHsv(hue.value(), saturation.value(), brightness.value())
                hex_edit.blockSignals(True)
                hex_edit.setText(color.name(QColor.NameFormat.HexRgb).upper())
                hex_edit.blockSignals(False)
                self._apply_studio_lighting_settings()

            def controls_from_hex():
                color = QColor(hex_edit.text().strip())
                if not color.isValid():
                    return
                control_pairs = (
                    (hue, hue_spin, max(0, color.hue())),
                    (saturation, saturation_spin, color.saturation()),
                    (brightness, brightness_spin, color.value()),
                )
                for slider, spin, value in control_pairs:
                    slider.blockSignals(True)
                    slider.setValue(value)
                    slider.blockSignals(False)
                    spin.blockSignals(True)
                    spin.setValue(value)
                    spin.blockSignals(False)
                self._apply_studio_lighting_settings()

            for widget in (hue, saturation, brightness):
                widget.valueChanged.connect(color_from_hsv)
            hex_edit.editingFinished.connect(controls_from_hex)
            enabled.toggled.connect(self._apply_studio_lighting_settings)
            intensity.valueChanged.connect(self._apply_studio_lighting_settings)

            light_section.add_layout(form)
            studio_lighting_section.add_widget(light_section)
            self.studio_light_controls[key] = {
                "enabled": enabled, "hex": hex_edit, "hue": hue,
                "saturation": saturation, "value": brightness,
                "hue_spin": hue_spin, "saturation_spin": saturation_spin,
                "value_spin": brightness_spin,
                "intensity": intensity, "intensity_spin": intensity_spin,
            }

        add_light_control("key", "LIGHT 1", 100, expanded=False)
        add_light_control("fill", "LIGHT 2", 100, expanded=False)
        add_light_control("back", "LIGHT 3", 100, expanded=False)
        add_light_control("environment", "ENVIRONMENT", 40, expanded=False)
        self.studio_light_controls["environment"]["intensity"].setToolTip(
            "Scales the real linear-light image-based-lighting environment. 40% is the neutral studio baseline."
        )

        studio_lighting_buttons = QHBoxLayout()
        studio_lighting_buttons.addStretch(1)
        self.reset_studio_lighting_btn = QPushButton("Reset defaults")
        self.reset_studio_lighting_btn.setToolTip("Restore the default Windows 3D Viewer-style studio lighting setup.")
        studio_lighting_buttons.addWidget(self.reset_studio_lighting_btn)
        studio_lighting_section.add_layout(studio_lighting_buttons)
        controls_layout.addWidget(studio_lighting_section)

        self.vtk_engine_label = QLabel("VTK · IN-PROCESS")
        self.vtk_engine_label.setObjectName("Muted")
        self.vtk_engine_label.setToolTip("Visualization Toolkit renderer embedded directly in the Polyfitty Studio viewport.")
        controls_layout.addWidget(self.vtk_engine_label)
        controls_layout.addSpacing(2)
        controls_layout.addWidget(self.face_overlay_toggle, 0, Qt.AlignRight)
        self.surface_mix_slider.valueChanged.connect(self._on_surface_mix_changed)
        self.wireframe_toggle.toggled.connect(self.scene_view.set_wireframe_enabled)
        self.face_islands_toggle.toggled.connect(self.scene_view.set_face_islands_enabled)
        self.wireframe_toggle.toggled.connect(self._update_overlay_control_visibility)
        self.face_islands_toggle.toggled.connect(self._update_overlay_control_visibility)
        self.scene_view.selectionChanged.connect(self._update_overlay_control_visibility)
        self._update_overlay_control_visibility()
        self.wire_mix_slider.valueChanged.connect(self._on_wire_mix_changed)
        self.render_pass_combo.currentIndexChanged.connect(self._on_render_pass_changed)
        self.studio_lights_enabled.toggled.connect(self._apply_studio_lighting_settings)
        self.studio_rotation_widget.azimuthChanged.connect(self._apply_studio_lighting_settings)
        self.reset_studio_lighting_btn.clicked.connect(self._reset_studio_lighting_settings)
        self._on_surface_mix_changed(self.surface_mix_slider.value())
        self._on_wire_mix_changed(self.wire_mix_slider.value())
        self._apply_studio_lighting_settings()
        self._on_render_pass_changed(self.render_pass_combo.currentIndex())
        self.viewport_splitter = QSplitter(Qt.Horizontal)
        self.viewport_splitter.setObjectName("ViewportAndControlsSplitter")
        self.viewport_splitter.setChildrenCollapsible(False)
        self.viewport_splitter.setHandleWidth(5)
        self.viewport_splitter.addWidget(self.scene_view)
        self.viewport_controls_scroll = QScrollArea()
        self.viewport_controls_scroll.setObjectName("ViewportControlsDock")
        self.viewport_controls_scroll.setWidgetResizable(True)
        self.viewport_controls_scroll.setFrameShape(QFrame.NoFrame)
        self.viewport_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.viewport_controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.viewport_controls_scroll.setMinimumWidth(238)
        self.viewport_controls_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viewport_controls_scroll.setWidget(self.scene_controls)
        self.viewport_splitter.addWidget(self.viewport_controls_scroll)
        self.viewport_splitter.setStretchFactor(0, 3)
        self.viewport_splitter.setStretchFactor(1, 1)
        initial_controls_width = max(250, min(330, int(self.width() * 0.20)))
        self.viewport_splitter.setSizes([max(340, self.width() - initial_controls_width - 360), initial_controls_width])
        self.viewport_splitter.setToolTip(
            "Viewport Controls is docked to the right and scales with the studio window. Drag the divider to resize the panel; "
            "its width is not capped to a fixed pixel size."
        )
        scene_layout.addWidget(self.viewport_splitter, 1)
        right.addWidget(scene_panel, 1)

        mesh_comparison_panel = QFrame()
        mesh_comparison_panel.setObjectName("Panel")
        mesh_comparison_layout = QVBoxLayout(mesh_comparison_panel)
        mesh_comparison_layout.setContentsMargins(8, 4, 8, 5)
        mesh_comparison_layout.setSpacing(2)
        comparison_title_row = QHBoxLayout()
        comparison_title = QLabel("MESH COMPARISON · INPUT → RECONSTRUCTION → BAKED OUTPUT")
        comparison_title.setObjectName("Section")
        comparison_title_row.addWidget(comparison_title)
        comparison_title_row.addStretch(1)
        self.mesh_comparison_status = QLabel("Load a mesh to compare before and after geometry.")
        self.mesh_comparison_status.setObjectName("Muted")
        self.mesh_comparison_status.setToolTip(
            "Polygon reduction is a geometry-complexity indicator, not proof of better visual fidelity. "
            "Check boundary/non-manifold edges and inspect the shape. Baked GLB file size includes texture images."
        )
        comparison_title_row.addWidget(self.mesh_comparison_status)
        mesh_comparison_layout.addLayout(comparison_title_row)
        comparison_grid = QGridLayout()
        comparison_grid.setContentsMargins(0, 0, 0, 0)
        comparison_grid.setHorizontalSpacing(10)
        comparison_grid.setVerticalSpacing(1)
        comparison_headers = ("METRIC", "INPUT MESH", "RECONSTRUCTION", "BAKED OUTPUT")
        for column, header_text in enumerate(comparison_headers):
            header_label = QLabel(header_text)
            header_label.setObjectName("Muted")
            header_label.setStyleSheet("font-size:7pt;font-weight:800;letter-spacing:0.4px;")
            comparison_grid.addWidget(header_label, 0, column)
        self.mesh_comparison_labels = {}
        comparison_rows = (
            ("vertices", "VERTICES"),
            ("triangles", "TRIANGLES"),
            ("unique_edges", "UNIQUE EDGES"),
            ("topology", "BOUNDARY / NON-MANIFOLD"),
            ("components", "COMPONENTS / WATERTIGHT"),
            ("surface_area", "SURFACE AREA"),
            ("file_size_mb", "FILE SIZE"),
        )
        for row_index, (metric_key, metric_label) in enumerate(comparison_rows, start=1):
            name_label = QLabel(metric_label)
            name_label.setObjectName("Muted")
            name_label.setStyleSheet("font-size:7pt;font-weight:700;")
            comparison_grid.addWidget(name_label, row_index, 0)
            for column, stage_key in enumerate(("input", "reconstruction", "output"), start=1):
                value_label = QLabel("—")
                value_label.setObjectName("MeshComparisonValue")
                value_label.setStyleSheet("font-size:8pt;font-weight:650;")
                comparison_grid.addWidget(value_label, row_index, column)
                self.mesh_comparison_labels[(metric_key, stage_key)] = value_label
        mesh_comparison_layout.addLayout(comparison_grid)
        mesh_comparison_panel.setMinimumWidth(760)
        mesh_comparison_panel.setMinimumHeight(142)
        mesh_comparison_panel.setMaximumHeight(168)
        mesh_comparison_scroll = QScrollArea()
        mesh_comparison_scroll.setObjectName("MeshComparisonScrollArea")
        mesh_comparison_scroll.setWidgetResizable(True)
        mesh_comparison_scroll.setFrameShape(QFrame.NoFrame)
        mesh_comparison_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        mesh_comparison_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        mesh_comparison_scroll.setMinimumHeight(154)
        mesh_comparison_scroll.setMaximumHeight(178)
        mesh_comparison_scroll.setWidget(mesh_comparison_panel)
        self.mesh_comparison_scroll = mesh_comparison_scroll
        right.addWidget(mesh_comparison_scroll, 0)

        self.diagnostics_toggle = QToolButton()
        self.diagnostics_toggle.setObjectName("DiagnosticsToggle")
        self.diagnostics_toggle.setText("PERFORMANCE / MESH INFO / CONSOLE")
        self.diagnostics_toggle.setCheckable(True)
        self.diagnostics_toggle.setChecked(False)
        self.diagnostics_toggle.setArrowType(Qt.RightArrow)
        self.diagnostics_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.diagnostics_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.diagnostics_toggle.setFixedHeight(30)
        right.addWidget(self.diagnostics_toggle, 0)

        lower = QSplitter(Qt.Horizontal)
        lower.setObjectName("DiagnosticsPanel")
        lower.setChildrenCollapsible(False)
        lower.setHandleWidth(4)

        metric_panel = QFrame()
        metric_panel.setObjectName("Panel")
        ml = QVBoxLayout(metric_panel)
        ml.setContentsMargins(7, 5, 7, 5)
        ml.setSpacing(4)
        mtitle = QLabel("MESH QUALITY / PERFORMANCE")
        mtitle.setObjectName("Section")
        ml.addWidget(mtitle)

        metric_row = QHBoxLayout()
        metric_row.setSpacing(5)
        self.card_cpu = MetricCard("CPU", "0%")
        self.card_ram = MetricCard("RAM", "0 MB")
        self.card_elapsed = MetricCard("TIME", "00:00")
        self.card_points = MetricCard("POINTS")
        self.card_planes = MetricCard("PLANES")
        self.card_faces = MetricCard("FACES")
        for card in [self.card_cpu, self.card_ram, self.card_elapsed, self.card_points, self.card_planes, self.card_faces]:
            card.setMaximumHeight(48)
            card_layout = card.layout()
            if card_layout is not None:
                card_layout.setContentsMargins(6, 3, 6, 3)
                card_layout.setSpacing(0)
            metric_row.addWidget(card, 1)
        ml.addLayout(metric_row)

        gpu_metric_row = QHBoxLayout()
        gpu_metric_row.setSpacing(5)
        self.card_gpu_load = MetricCard("GPU LOAD", "N/A")
        self.card_gpu_memory = MetricCard("GPU VRAM", "—")
        self.card_gpu_rate = MetricCard("GPU RATE", "—")
        self.card_gpu_speedup = MetricCard("GPU VS CPU · QUERY", "N/A")
        for card in [self.card_gpu_load, self.card_gpu_memory, self.card_gpu_rate, self.card_gpu_speedup]:
            card.setMaximumHeight(48)
            card_layout = card.layout()
            if card_layout is not None:
                card_layout.setContentsMargins(6, 3, 6, 3)
                card_layout.setSpacing(0)
            gpu_metric_row.addWidget(card, 1)
        ml.addLayout(gpu_metric_row)

        self.card_gpu_load.setToolTip("Live GPU utilization from driver/OS telemetry. N/A means the adapter driver does not expose a supported counter.")
        self.card_gpu_memory.setToolTip("GPU memory usage reported by the driver/OS. Total VRAM may be shown even when live usage is unavailable.")
        self.card_gpu_rate.setToolTip("Measured candidate triangle tests per second during real GPU bake batches (cage ray intersection or direct nearest-surface queries).")
        self.card_gpu_speedup.setToolTip("Measured GPU throughput compared with NumPy CPU on the exact same first production bake batch. This covers the ray-intersection or nearest-point query stage, not the whole bake.")

        performance_gpu_row = QHBoxLayout()
        performance_gpu_row.setContentsMargins(0, 1, 0, 1)
        performance_gpu_row.setSpacing(8)
        self.performance_gpu_status = QLabel("● GPU · SCANNING")
        self.performance_gpu_status.setObjectName("GPUStatus")
        self.performance_gpu_status.setMinimumWidth(142)
        self.performance_gpu_status.setMaximumWidth(196)
        self.performance_gpu_detail = QLabel("Scanning CUDA / HIP / SYCL / OpenCL adapters…")
        self.performance_gpu_detail.setObjectName("Muted")
        self.performance_gpu_detail.setWordWrap(True)
        self.performance_gpu_detail.setMinimumWidth(120)
        performance_gpu_row.addWidget(self.performance_gpu_status, 0)
        performance_gpu_row.addWidget(self.performance_gpu_detail, 1)
        ml.addLayout(performance_gpu_row)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        self.metric_labels = {}
        metric_names = [
            ("vertices", "Vertices"), ("triangles", "Triangles"), ("unique_edges", "Unique edges"),
            ("components", "Components"), ("boundary_edges", "Boundary edges"), ("nonmanifold_edges", "Non-manifold edges"),
            ("watertight", "Watertight"), ("winding_consistent", "Winding"), ("surface_area", "Surface area"),
            ("volume", "Volume"), ("plane_coverage", "Plane coverage"), ("file_size_mb", "Output size"),
            ("reconstruction_method", "Engine"),
        ]
        for index, (key, label) in enumerate(metric_names):
            cell = QHBoxLayout()
            cell.setSpacing(4)
            k = QLabel(label + ":")
            k.setObjectName("Muted")
            v = QLabel("—")
            v.setObjectName("DiagnosticsMetricValue")
            cell.addWidget(k)
            cell.addWidget(v)
            cell.addStretch(1)
            grid.addLayout(cell, index // 4, index % 4)
            self.metric_labels[key] = v
        ml.addLayout(grid)
        metric_panel.setMinimumWidth(560)
        metric_scroll = QScrollArea()
        metric_scroll.setObjectName("DiagnosticsMetricsScrollArea")
        metric_scroll.setWidgetResizable(True)
        metric_scroll.setFrameShape(QFrame.NoFrame)
        metric_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        metric_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        metric_scroll.setMinimumWidth(300)
        metric_scroll.setWidget(metric_panel)
        self.diagnostics_metrics_scroll = metric_scroll
        lower.addWidget(metric_scroll)

        log_panel = QFrame()
        log_panel.setObjectName("Panel")
        ll = QVBoxLayout(log_panel)
        ll.setContentsMargins(7, 5, 7, 5)
        ll.setSpacing(4)
        log_title = QLabel("PIPELINE CONSOLE")
        log_title.setObjectName("Section")
        ll.addWidget(log_title)
        self.log_box = QTextEdit()
        self.log_box.setObjectName("DiagnosticsConsole")
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumWidth(180)
        self.log_box.setMinimumHeight(95)
        self.log_box.document().setMaximumBlockCount(900)
        ll.addWidget(self.log_box, 1)

        for pending_message in self._pending_log_messages:
            self._append_log(pending_message)
        self._pending_log_messages.clear()
        lower.addWidget(log_panel)
        lower.setSizes([760, 470])
        lower.setMaximumHeight(250)
        lower.setMinimumHeight(185)
        lower.setVisible(False)
        self.diagnostics_panel = lower
        self.diagnostics_toggle.toggled.connect(self._set_diagnostics_visible)
        right.addWidget(lower, 0)

        self._configure_responsive_forms(central)
        self.setCentralWidget(central)
        self._apply_theme()
        self._log("Polyfitty Studio is ready. Run INSTALL.bat, then choose an input file.")
        if self._startup_gpu_probe is not None:

            self._on_gpu_scan_completed(self._startup_gpu_probe)
            self._startup_gpu_probe = None
        else:

            self._start_gpu_scan()

    def _start_gpu_telemetry(self):
        if not self._gpu_telemetry_target:
            return
        worker = self._gpu_telemetry_worker
        if worker is not None and worker.isRunning():
            same_target = (
                worker.vendor == self._gpu_telemetry_target[0]
                and worker.device_name == self._gpu_telemetry_target[1]
            )
            if same_target:
                return
            self._stop_gpu_telemetry()
        vendor, device_name = self._gpu_telemetry_target
        worker = GPUPerformanceMonitor(vendor, device_name, self)
        self._gpu_telemetry_worker = worker
        worker.updated.connect(self._on_gpu_telemetry_updated)
        worker.finished.connect(self._on_gpu_telemetry_finished)
        worker.start()

    def _stop_gpu_telemetry(self):
        worker = self._gpu_telemetry_worker
        if worker is None:
            return
        if worker.isRunning():
            worker.requestInterruption()
            if not worker.wait(9000):
                worker.requestInterruption()
                worker.wait()
        self._gpu_telemetry_worker = None
        worker.deleteLater()

    def _on_gpu_telemetry_finished(self):
        worker = self._gpu_telemetry_worker
        if worker is not None and not worker.isRunning():
            self._gpu_telemetry_worker = None
            worker.deleteLater()

    def _on_gpu_telemetry_updated(self, payload: dict):
        utilization = payload.get("utilization_percent")
        self.card_gpu_load.set_value("N/A" if utilization is None else f"{float(utilization):.0f}%")
        memory_used = payload.get("memory_used_bytes")
        memory_total = payload.get("memory_total_bytes") or self._gpu_vram_total_bytes
        if memory_used is not None and memory_total:
            memory_text = f"{memory_used / (1024 ** 3):.1f}/{memory_total / (1024 ** 3):.1f} GB"
        elif memory_used is not None:
            memory_text = f"{memory_used / (1024 ** 3):.1f} GB used"
        elif memory_total:
            memory_text = f"— / {memory_total / (1024 ** 3):.1f} GB"
        else:
            memory_text = "N/A"
        self.card_gpu_memory.set_value(memory_text)
        detail = str(payload.get("detail") or payload.get("source") or "GPU telemetry")
        self.card_gpu_load.setToolTip(f"{detail}\nSource: {payload.get('source', 'unknown')}")
        self.card_gpu_memory.setToolTip(f"{detail}\nLive used-memory telemetry may not be exposed by this driver.")

    def _update_gpu_compute_metrics(self, payload: dict):
        rate = float(payload.get("gpu_tests_per_second", 0.0) or 0.0)
        if rate > 0:
            self.card_gpu_rate.set_value(f"{rate / 1_000_000:.2f}M/s" if rate >= 1_000_000 else f"{rate / 1_000:.1f}K/s")
        else:
            self.card_gpu_rate.set_value("—")
        if payload.get("benchmark_valid"):
            ratio = float(payload.get("speedup_ratio", 0.0) or 0.0)
            gain = float(payload.get("speedup_gain_percent", 0.0) or 0.0)
            sign = "+" if gain >= 0 else ""
            self.card_gpu_speedup.set_value(f"{ratio:.2f}× ({sign}{gain:.0f}%)")
            cpu_ms = float(payload.get("cpu_reference_seconds", 0.0) or 0.0) * 1000.0
            gpu_ms = float(payload.get("gpu_benchmark_seconds", 0.0) or 0.0) * 1000.0
            time_saved = float(payload.get("time_saved_percent", 0.0) or 0.0)
            time_saved_sign = "+" if time_saved >= 0 else ""
            detail = str(payload.get("benchmark_detail") or "Measured same-input CPU/GPU comparison.")
            self.card_gpu_speedup.setToolTip(
                f"CPU reference: {cpu_ms:.2f} ms\nGPU: {gpu_ms:.2f} ms\n"
                f"Throughput gain: {sign}{gain:.1f}%\nTime saved: {time_saved_sign}{time_saved:.1f}%\n{detail}"
            )
        else:
            self.card_gpu_speedup.set_value("Measuring…" if payload.get("state") == "ready" else "N/A")
            self.card_gpu_speedup.setToolTip(
                str(payload.get("benchmark_detail") or "A CPU/GPU comparison is available after the first real GPU bake batch.")
            )

    def _set_gpu_status_ui(self, state: str, title: str, detail: str, tooltip: str = ""):
        self._gpu_current_state = str(state)
        compact = {
            "scanning": "GPU · SCAN",
            "ready": "GPU · READY",
            "active": "GPU · ACTIVE",
            "warning": "CPU · FALLBACK",
            "cpu": "CPU · ONLY",
            "error": "GPU · OFF",
        }.get(str(state).lower(), "GPU · STATUS")
        if hasattr(self, "gpu_status_label"):
            self.gpu_status_label.setText("● " + compact)
            self.gpu_status_label.setProperty("state", str(state))
            style = self.gpu_status_label.style()
            if style is not None:
                style.unpolish(self.gpu_status_label)
                style.polish(self.gpu_status_label)
            self.gpu_status_label.setToolTip(tooltip or detail)
        if hasattr(self, "gpu_status_detail"):
            self.gpu_status_detail.setText(compact.replace(" · ", " "))
            self.gpu_status_detail.setToolTip(tooltip or detail)
        if hasattr(self, "performance_gpu_status"):
            self.performance_gpu_status.setText(str(title))
            self.performance_gpu_status.setProperty("state", str(state))
            style = self.performance_gpu_status.style()
            if style is not None:
                style.unpolish(self.performance_gpu_status)
                style.polish(self.performance_gpu_status)
            self.performance_gpu_status.setToolTip(tooltip or detail)
        if hasattr(self, "performance_gpu_detail"):
            self.performance_gpu_detail.setText(str(detail))
            self.performance_gpu_detail.setToolTip(tooltip or detail)

    def _start_gpu_scan(self):
        export_worker = getattr(self, "_export_worker", None)
        if export_worker is not None and export_worker.isRunning():
            self._gpu_rescan_pending = True
            self._log("GPU rescan queued until the active bake/export job has finished.")
            return
        worker = getattr(self, "_gpu_scan_worker", None)
        if worker is not None and worker.isRunning():
            self._gpu_rescan_pending = True
            return
        self._gpu_rescan_pending = False
        self._gpu_runtime_state_confirmed = False
        self._set_gpu_status_ui(
            "scanning", "● GPU · SCANNING",
            "Scanning…",
        )
        if hasattr(self, "gpu_rescan_btn"):
            self.gpu_rescan_btn.setEnabled(False)
        worker = GPUDiscoveryWorker(self)
        self._gpu_scan_worker = worker
        worker.completed.connect(self._on_gpu_scan_completed)
        worker.finished.connect(self._on_gpu_scan_finished)
        worker.start()

    def _on_gpu_scan_finished(self):
        worker = self._gpu_scan_worker
        self._gpu_scan_worker = None
        if worker is not None:
            worker.deleteLater()
        if hasattr(self, "gpu_rescan_btn"):
            export_worker = getattr(self, "_export_worker", None)
            self.gpu_rescan_btn.setEnabled(not (export_worker is not None and export_worker.isRunning()))
        if getattr(self, "_gpu_rescan_pending", False) and not getattr(self, "_closing_after_bake_cancel", False):
            self._log("A queued GPU rescan is starting after the previous scan finished.")
            QTimer.singleShot(0, self._start_gpu_scan)

    def _on_gpu_scan_completed(self, payload: dict):
        self._gpu_discovery_records = list(payload.get("devices") or [])
        self._gpu_discovery_errors = list(payload.get("errors") or [])
        self._gpu_display_adapters = list(payload.get("display_adapters") or [])
        if self._gpu_runtime_state_confirmed:
            return

        errors = self._gpu_discovery_errors
        devices = self._gpu_discovery_records
        compute_ready = bool(payload.get("compute_ready"))
        if compute_ready and devices:
            first = next((item for item in devices if item.get("active")), devices[0])
            vendor = str(payload.get("active_vendor") or first.get("vendor", "GPU"))
            backend = str(payload.get("active_backend") or first.get("backend", "OpenCL")).upper()
            name = str(payload.get("active_device") or first.get("name", "Unknown GPU"))
            memory = int(payload.get("active_memory_bytes") or first.get("memory_bytes", 0) or 0)
            memory_text = f" · {memory / (1024 ** 3):.1f} GiB" if memory else ""
            self._gpu_telemetry_target = (vendor, name)
            self._gpu_vram_total_bytes = memory
            if self.diagnostics_panel.isVisible():
                self._start_gpu_telemetry()
            lines = [
                f"{'ACTIVE · ' if item.get('active') else ''}{item.get('vendor', 'GPU')} · "
                f"{item.get('name', 'Unknown GPU')} · {item.get('backend', 'UNKNOWN')}"
                + (f" · {int(item.get('memory_bytes', 0)) / (1024 ** 3):.1f} GiB" if item.get('memory_bytes') else "")
                for item in devices
            ]
            detail = (
                f"Compute-ready: {vendor} {name} · {backend}{memory_text}.\n"
                "A real GPU ray/triangle kernel self-test passed. GPU acceleration is used by cage-ray and direct-nearest-triangle texture baking; "
                "Li–Nan filtering, planar-region extraction and decimation remain CPU stages."
            )
            cuda_errors = [item for item in errors if "cuda" in item.lower() or "cupy" in item.lower()]
            if backend != "CUDA" and cuda_errors:
                if any("nvcuda.dll was not found" in item.lower() for item in cuda_errors):
                    detail += (
                        "\nNo NVIDIA CUDA driver was found; this is expected on AMD/Intel-only systems. "
                        f"The selected {backend} GPU backend is active."
                    )
                else:
                    detail += "\nNVIDIA CUDA was not activated; the selected GPU backend is " + backend + ".\n"
                    detail += "CUDA diagnostics: " + " | ".join(cuda_errors)
            tooltip = detail + "\n\nDetected compute adapters:\n" + "\n".join(lines)
            if errors:
                tooltip += "\n\nOther adapter/runtime diagnostics:\n" + "\n".join(errors)
            self._set_gpu_status_ui(
                "ready", f"● GPU · {_gpu_vendor_short(vendor)} READY", detail, tooltip,
            )
        elif devices:
            lines = [
                f"{item.get('vendor', 'GPU')} · {item.get('name', 'Unknown GPU')} · {item.get('backend', 'UNKNOWN')}"
                for item in devices
            ]
            reason = str(payload.get("fallback_reason") or "; ".join(errors) or "GPU kernel self-test failed")
            detail = (
                "A GPU adapter is visible, but no CUDA/HIP/SYCL/OpenCL compute backend passed the kernel self-test. "
                f"CPU baking will be used until the driver/runtime issue is repaired. Reason: {reason}"
            )
            tooltip = detail + "\n\nDetected adapters:\n" + "\n".join(lines)
            if self._gpu_display_adapters:
                tooltip += "\n\nWindows display adapters:\n" + "\n".join(self._gpu_display_adapters)
            self._set_gpu_status_ui("warning", "● CPU · GPU FALLBACK", detail, tooltip)
        else:
            error_text = " | ".join(str(item) for item in errors)
            opencl_missing = any("opencl runtime unavailable" in item.lower() for item in errors)
            pyopencl_not_bundled = "no module named 'pyopencl'" in error_text.lower()
            opencl_icd_missing = any(
                token in error_text.lower()
                for token in ("platform_not_found_khr", "zero platforms", "no vendor icd platform")
            )
            opencl_gpu_not_exposed = "none exposed an available gpu" in error_text.lower()
            display_adapters = self._gpu_display_adapters
            amd_adapter = next(
                (name for name in display_adapters if "amd" in name.lower() or "radeon" in name.lower()),
                None,
            )
            nvidia_adapter = next(
                (name for name in display_adapters if "nvidia" in name.lower() or "geforce" in name.lower()),
                None,
            )
            intel_adapter = next(
                (name for name in display_adapters if "intel" in name.lower() or "arc" in name.lower()),
                None,
            )
            sycl_runtime_missing = any("intel oneapi/sycl runtime unavailable" in item.lower() for item in errors)
            sycl_gpu_not_exposed = any(
                "sycl runtime was loaded, but it exposed no gpu devices" in item.lower()
                or "sycl gpu enumeration failed" in item.lower()
                for item in errors
            )
            if amd_adapter:
                if pyopencl_not_bundled:
                    title = "● GPU · OPENCL PACKAGE MISSING"
                    detail = (
                        f"Windows detected the AMD display adapter: {amd_adapter}, but this product environment is missing PyOpenCL. "
                        "Run the GPU setup for this installation, or restart the portable package so its first-run GPU setup can provision the local OpenCL Python layer. "
                        "The target PC must still have the AMD driver and its OpenCL ICD installed."
                    )
                elif opencl_icd_missing:
                    title = "● GPU · AMD OPENCL DRIVER MISSING"
                    detail = (
                        f"Windows detected the AMD display adapter: {amd_adapter}, and PyOpenCL is present, "
                        "but the AMD driver did not expose an OpenCL ICD platform. Update or repair the AMD graphics driver "
                        "with its OpenCL component, then rescan GPU."
                    )
                elif opencl_gpu_not_exposed:
                    title = "● GPU · AMD GPU NOT EXPOSED BY OPENCL"
                    detail = (
                        f"Windows detected the AMD display adapter: {amd_adapter}, and an OpenCL platform is present, "
                        "but it did not expose an available GPU device. Check the AMD compute driver and whether the adapter "
                        "is enabled for compute, then rescan GPU."
                    )
                else:
                    title = "● GPU · AMD COMPUTE OFF"
                    detail = (
                        f"Windows detected the AMD display adapter: {amd_adapter}, but no usable OpenCL GPU compute device passed discovery. "
                        "Check the AMD driver/OpenCL component and review the GPU diagnostics before rescanning."
                    )
                state = "warning"
            elif intel_adapter and sycl_runtime_missing:
                title = "● GPU · INTEL SYCL RUNTIME MISSING"
                detail = (
                    f"Windows detected the Intel GPU adapter: {intel_adapter}, but the embedded/installed Python environment could not load oneAPI/SYCL. "
                    "Repair the GPU compute packages: use the main Studio setup manager for an installed copy, or relaunch the portable copy to retry its first-run GPU setup. "
                    "The target PC still needs a compatible Intel graphics driver and Level Zero or OpenCL plugin."
                )
                state = "warning"
            elif intel_adapter and sycl_gpu_not_exposed:
                title = "● GPU · INTEL SYCL DEVICE NOT EXPOSED"
                detail = (
                    f"Windows detected the Intel GPU adapter: {intel_adapter}, and oneAPI/SYCL loaded, but no GPU device was exposed. "
                    "Update the Intel graphics driver and verify that the Level Zero or OpenCL compute plugin is installed and enabled. "
                    "OpenCL remains available as a compatibility fallback."
                )
                state = "warning"
            elif nvidia_adapter:
                if pyopencl_not_bundled:
                    title = "● GPU · GPU PACKAGE MISSING"
                    detail = (
                        f"Windows detected the NVIDIA display adapter: {nvidia_adapter}, but the Python environment is missing PyOpenCL "
                        "and may also lack CuPy/CUDA. Repair GPU packages through the main Studio setup manager, or relaunch the portable copy to retry first-run GPU setup."
                    )
                elif opencl_icd_missing:
                    title = "● GPU · NVIDIA OPENCL DRIVER MISSING"
                    detail = (
                        f"Windows detected the NVIDIA display adapter: {nvidia_adapter}, but the installed driver did not expose an OpenCL ICD. "
                        "Update or repair the NVIDIA driver. The application environment can include CUDA and OpenCL together; "
                        "a compatible CUDA-12 NVIDIA driver is still required for the CUDA backend."
                    )
                elif opencl_gpu_not_exposed:
                    title = "● GPU · NVIDIA GPU NOT EXPOSED BY OPENCL"
                    detail = (
                        f"Windows detected the NVIDIA display adapter: {nvidia_adapter}, and an OpenCL platform is present, "
                        "but it did not expose an available GPU device. Update the NVIDIA driver and check the Windows graphics "
                        "preference for Polyfitty Studio. The application environment can include CUDA 12 and OpenCL together; only a compatible NVIDIA GPU uses CUDA."
                    )
                else:
                    title = "● GPU · NVIDIA COMPUTE OFF"
                    detail = (
                        f"Windows detected the NVIDIA display adapter: {nvidia_adapter}, but no CUDA/OpenCL compute device passed discovery. "
                        "Check the NVIDIA driver, OpenCL ICD, bundled CUDA runtime and GPU diagnostics."
                    )
                state = "warning"
            elif pyopencl_not_bundled:
                title = "● GPU · OPENCL PACKAGE MISSING"
                detail = (
                    "This Polyfitty Studio Python environment is missing PyOpenCL, so the OpenCL discovery backend is unavailable. "
                    "Run the installed Studio setup manager, or relaunch the portable copy to retry its first-run GPU setup. "
                    "The target PC must still have a compatible GPU vendor driver and OpenCL ICD."
                )
                state = "warning"
            elif opencl_icd_missing or opencl_missing:
                title = "● GPU · OPENCL DRIVER MISSING"
                detail = (
                    "PyOpenCL is present, but Windows did not expose a usable vendor OpenCL ICD platform. "
                    "Update or repair the NVIDIA/AMD/Intel graphics driver with its OpenCL compute component, "
                    "then rescan GPU. NVIDIA CUDA 12 can be bundled alongside OpenCL; it requires a compatible NVIDIA driver."
                )
                state = "warning"
            elif opencl_gpu_not_exposed:
                title = "● GPU · OPENCL FOUND, NO GPU DEVICE"
                detail = (
                    "The OpenCL loader found a vendor platform, but it did not expose an available GPU device. "
                    "Check the display driver and the Windows graphics preference for Polyfitty Studio."
                )
                state = "warning"
            else:
                title = "● GPU · NOT DETECTED"
                detail = (
                    "No supported CUDA/HIP/SYCL/OpenCL GPU compute device was exposed. Check the vendor graphics driver and compute runtime/ICD. "
                    "CPU baking remains available."
                )
                state = "error"
            if display_adapters:
                detail += "\nWindows display adapters: " + "; ".join(display_adapters)
            if payload.get("fallback_reason"):
                detail += "\nCompute probe: " + str(payload.get("fallback_reason"))
            tooltip = detail + ("\n\nDiagnostics:\n" + "\n".join(errors) if errors else "")
            self._set_gpu_status_ui(state, title, detail, tooltip)
        if hasattr(self, "gpu_rescan_btn"):
            export_worker = getattr(self, "_export_worker", None)
            self.gpu_rescan_btn.setEnabled(not (export_worker is not None and export_worker.isRunning()))

    def _on_gpu_compute_status(self, payload: dict):
        state = str(payload.get("state", "unknown"))
        self._gpu_runtime_state_confirmed = True
        vendor = str(payload.get("vendor", "GPU"))
        device = str(payload.get("device", "Unknown GPU"))
        backend = str(payload.get("backend", "UNKNOWN")).upper()
        precision = str(payload.get("precision", ""))
        batches = int(payload.get("batch_count", 0) or 0)
        ray_tests = int(payload.get("gpu_intersection_tests_total", 0) or 0)
        closest_tests = int(payload.get("gpu_closest_point_tests_total", 0) or 0)
        workload = str(payload.get("gpu_workload", "") or "")
        detail = str(payload.get("detail", ""))
        memory = int(payload.get("memory_bytes", 0) or 0)
        memory_text = f" · {memory / (1024 ** 3):.1f} GiB" if memory else ""
        vendor_badge = _gpu_vendor_short(vendor)
        if vendor.upper() != "CPU" and device not in {"", "CPU", "Unknown GPU"}:
            self._gpu_telemetry_target = (vendor, device)
            if memory:
                self._gpu_vram_total_bytes = memory
            if self.diagnostics_panel.isVisible():
                self._start_gpu_telemetry()
        self._update_gpu_compute_metrics(payload)
        if state == "ready":
            title = f"● GPU · {vendor_badge} READY"
            message = f"{vendor} {device} · {backend} · {precision}{memory_text} · kernel self-test passed. Waiting for bake work."
            self._set_gpu_status_ui("ready", title, message, message)
        elif state == "active":
            title = f"● GPU · {vendor_badge} ACTIVE"
            workload_text = {
                "cage-ray/triangle": "cage ray/triangle",
                "direct-nearest-triangle": "direct nearest-point/triangle",
                "mixed-ray-and-nearest-triangle": "cage ray + direct nearest-point/triangle",
            }.get(workload, "GPU bake kernels")
            message = (
                f"ACTIVE: {vendor} {device} · {backend} · {precision} · "
                f"{batches:,} successful GPU batches · workload={workload_text} · "
                f"ray tests={ray_tests:,} · closest-point tests={closest_tests:,}. {detail}"
            )
            self._set_gpu_status_ui("active", title, message, message)
        elif state == "cpu_selected":
            message = f"CPU-only mode selected. {detail}"
            self._set_gpu_status_ui("cpu", "● GPU · CPU ONLY", message, message)
        elif state == "cpu_performance_fallback":
            message = (
                f"AUTO PERFORMANCE SWITCH · GPU benchmark was slower than CPU. "
                f"The verified first batch was retained; remaining bake batches are running on CPU. {detail}"
            )
            self._set_gpu_status_ui("cpu", "● CPU · AUTO PERFORMANCE SWITCH", message, message)
        elif state == "gpu_required_error":
            message = f"GPU Only was selected and GPU execution could not continue. {detail}"
            self._set_gpu_status_ui("error", "● GPU · REQUIRED BUT UNAVAILABLE", message, message)
        else:
            reason = str(payload.get("fallback_reason") or detail or "GPU unavailable")
            title = "● CPU · GPU FALLBACK"
            message = f"GPU acceleration is not active. CPU bake is running. Reason: {reason}"
            self.card_gpu_rate.set_value("N/A")
            self.card_gpu_speedup.set_value("N/A")
            self.card_gpu_speedup.setToolTip("GPU acceleration is not active for the current bake.")
            display_adapters = getattr(self, "_gpu_display_adapters", [])
            if display_adapters:
                message += " Windows display adapters: " + "; ".join(display_adapters)
            self._set_gpu_status_ui("error", title, message, message)
        self._log(f"GPU status · {self.gpu_status_label.text()} · {self.gpu_status_detail.text()}")

    def _refresh_system_theme(self):
        """Apply Windows app-mode changes while System theme mode is selected."""
        if not hasattr(self, "theme_combo") or self.theme_combo.currentData() != "system":
            return
        detected = detect_system_theme()
        if detected != getattr(self, "_resolved_theme_name", None):
            self._apply_theme()

    def _set_header_status_style(self, state: str = "ready"):
        theme = THEMES.get(getattr(self, "_resolved_theme_name", "graphite"), THEMES["graphite"])
        state_colors = {
            "ready": (theme["success"], theme["success_bg"], theme["success"]),
            "active": (theme["info"], theme["info_bg"], theme["info"]),
            "warning": (theme["warning"], theme["warning_bg"], theme["warning"]),
            "error": (theme["error"], theme["error_bg"], theme["error"]),
        }
        foreground, background, border = state_colors.get(state, state_colors["ready"])
        self.header_status.setStyleSheet(
            "QLabel { "
            f"color: {foreground}; background: {background}; border: 1px solid {border}; "
            "font-weight: 700; padding: 8px 12px; border-radius: 8px; }"
        )

    def _apply_theme(self, *_args):
        theme_mode = str(self.theme_combo.currentData() or "system") if hasattr(self, "theme_combo") else "system"
        theme_name = resolve_theme_name(theme_mode)
        self._resolved_theme_name = theme_name
        theme = THEMES.get(theme_name, THEMES["graphite"])
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet_for(theme_name))
        if hasattr(self, "face_overlay_toggle"):
            accent = theme.get("accent", "#278fc4")
            border = theme.get("border", "#344760")
            text_color = theme.get("text", "#d8e2f1")
            self.face_overlay_toggle.setStyleSheet(
                "QToolButton#ViewportFaceOverlayToggle { "
                f"background: {theme.get('panel', '#111a2a')}; color: {text_color}; "
                f"border: 1px solid {border}; border-radius: 6px; padding: 4px 9px; font-weight: 700; }} "
                "QToolButton#ViewportFaceOverlayToggle:checked { "
                f"background: {accent}; color: #ffffff; border-color: {accent}; }}"
            )
        if hasattr(self, "item_info_mode_buttons"):
            accent = theme.get("accent", "#278fc4")
            border = theme.get("border", "#344760")
            muted = theme.get("muted", "#98a8bc")
            text_color = theme.get("text", "#d8e2f1")
            for button in self.item_info_mode_buttons.values():
                button.setStyleSheet(
                    "QToolButton { "
                    f"background: {theme.get('panel_alt', '#172334')}; color: {text_color}; "
                    f"border: 1px solid {border}; border-radius: 5px; padding: 3px 2px; font-weight: 700; "
                    "} "
                    f"QToolButton:hover {{ border-color: {accent}; color: {text_color}; }} "
                    f"QToolButton:checked {{ background: {accent}; color: #ffffff; border-color: {accent}; }} "
                    f"QToolButton:disabled {{ color: {muted}; }}"
                )
        if hasattr(self, "scene_controls"):
            panel_rgb = tuple(int(theme["panel"][index:index + 2], 16) for index in (1, 3, 5))
            border_rgb = tuple(int(theme["border"][index:index + 2], 16) for index in (1, 3, 5))
            self.scene_controls.setStyleSheet(
                "QFrame#ViewportControls { "
                f"background: rgb({panel_rgb[0]}, {panel_rgb[1]}, {panel_rgb[2]}); "
                f"border: 1px solid rgba({border_rgb[0]}, {border_rgb[1]}, {border_rgb[2]}, 230); "
                "border-radius: 7px; } "
                f"QFrame#ViewportControls QLabel, QFrame#ViewportControls QCheckBox {{ "
                f"background: transparent; color: {theme['text']}; }} "
                f"QFrame#ViewportControls QCheckBox#ViewportItemInfoEnabled {{ "
                f"font-weight: 800; font-size: 9pt; padding: 3px 2px; }} "
                "QFrame#ViewportControls QCheckBox#ViewportItemInfoEnabled::indicator { width: 0px; height: 0px; margin: 0px; } "
                f"QFrame#ViewportControls QCheckBox#ViewportItemInfoEnabled:checked {{ color: {theme['accent']}; }} "
                f"QFrame#ViewportControls QLabel#Muted {{ color: {theme['muted']}; }} "
                f"QFrame#ViewportControls QSlider {{ background: transparent; }}"
            )
        if hasattr(self, "scene_view"):
            self.scene_view.set_theme(theme_name, theme["scene"])
        for widget_name, color_key in (
            ("source_surface_label", "accent"), ("source_wire_label", "accent"),
            ("result_surface_label", "accent_alt"), ("result_wire_label", "accent_alt"),
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setStyleSheet(f"color: {theme[color_key]}; font-weight: 800;")
        if hasattr(self, "header_status"):
            self._set_header_status_style(getattr(self, "_header_status_state", "ready"))
        if hasattr(self, "settings"):

            self.settings.setValue("theme_mode", theme_mode)
            self.settings.setValue("theme", theme_name)

    def _style_overlay_color_button(self, button: QPushButton, color: QColor) -> None:
        foreground = "#FFFFFF" if color.lightnessF() < 0.52 else "#111111"
        theme = THEMES.get(getattr(self, "_resolved_theme_name", "graphite"), THEMES["graphite"])
        button.setStyleSheet(
            "QPushButton { "
            f"background-color: {color.name(QColor.NameFormat.HexRgb)}; "
            f"color: {foreground}; "
            "border: 1px solid rgba(150, 160, 175, 180); "
            "border-radius: 4px; padding: 2px 6px; font-weight: 700; } "
            f"QPushButton:hover {{ border: 1px solid {theme['accent']}; }}"
        )

    def _update_item_info_controls_visibility(self):
        mode = str(getattr(self.scene_view, "_item_info_mode", "DIM") or "DIM").upper()
        enabled = bool(getattr(self.scene_view, "_item_info_enabled", False))
        unit_row = getattr(self, "item_info_unit_row_widget", None)
        rotation_row = getattr(self, "item_info_rotation_row_widget", None)
        if unit_row is not None:
            unit_row.setVisible(enabled and mode in {"LOC", "DIM"})
        if rotation_row is not None:
            rotation_row.setVisible(enabled and mode == "ROT")
        self._sync_item_info_unit_buttons()
        self._sync_item_info_rotation_buttons()

    def _ensure_default_metric_units(self):
        """Apply meter defaults once while preserving later user-selected units."""
        migration_key = "viewport/metric_units_default_m_v1"
        if bool(self.settings.value(migration_key, False, type=bool)):
            return
        self.settings.setValue("viewport/location_display_unit", "m")
        self.settings.setValue("viewport/dimension_display_unit", "m")
        self.settings.setValue(migration_key, True)
        self.settings.sync()

    def _sync_item_info_unit_buttons(self):
        buttons = getattr(self, "item_info_unit_buttons", {})
        if not buttons:
            return
        mode = str(getattr(self.scene_view, "_item_info_mode", "DIM") or "DIM").upper()
        if mode not in {"LOC", "DIM"}:
            return
        key = "location_display_unit" if mode == "LOC" else "dimension_display_unit"
        unit = str(self.settings.value(f"viewport/{key}", "m") or "m").strip().lower()
        if unit not in {"mm", "cm", "m", "km"}:
            unit = "m"
        for button_unit, button in buttons.items():
            button.blockSignals(True)
            button.setChecked(button_unit == unit)
            button.blockSignals(False)

    def _sync_item_info_rotation_buttons(self):
        buttons = getattr(self, "item_info_rotation_buttons", {})
        if not buttons:
            return
        mode = str(self.settings.value("viewport/rotation_display_mode", "euler") or "euler").strip().lower()
        if mode not in {"euler", "quaternion"}:
            mode = "euler"
        for button_mode, button in buttons.items():
            button.blockSignals(True)
            button.setChecked(button_mode == mode)
            button.blockSignals(False)

    def _sync_item_info_checkbox_icon(self, enabled: bool):
        checkbox = getattr(self, "item_info_enabled_check", None)
        if checkbox is not None:
            checkbox.setText("☑  ITEM INFO" if bool(enabled) else "☐  ITEM INFO")

    def _on_item_info_enabled_changed(self, enabled: bool):
        enabled = bool(enabled)
        self.settings.setValue("viewport/item_info_enabled", enabled)
        self.scene_view.set_item_info_enabled(enabled)
        self._update_item_info_controls_visibility()

    def _on_item_info_mode_changed(self, mode: str, checked: bool):
        if not checked:
            return
        normalized = str(mode or "DIM").strip().upper()
        if normalized not in {"LOC", "ROT", "SCALE", "DIM"}:
            normalized = "DIM"
        self.settings.setValue("viewport/item_info_mode", normalized)
        self.scene_view.set_item_info_mode(normalized)
        self._update_item_info_controls_visibility()

    def _on_item_info_unit_changed(self, unit, checked):
        if not checked:
            return
        normalized = str(unit or "m").strip().lower()
        if normalized not in {"mm", "cm", "m", "km"}:
            normalized = "m"
        mode = str(getattr(self.scene_view, "_item_info_mode", "DIM") or "DIM").upper()
        if mode == "LOC":
            self.settings.setValue("viewport/location_display_unit", normalized)
            self.scene_view.set_location_display_unit(normalized)
        elif mode == "DIM":
            self.settings.setValue("viewport/dimension_display_unit", normalized)
            self.scene_view.set_dimension_display_unit(normalized)
        self._sync_item_info_unit_buttons()

    def _on_item_info_rotation_changed(self, mode, checked):
        if not checked:
            return
        normalized = str(mode or "euler").strip().lower()
        if normalized not in {"euler", "quaternion"}:
            normalized = "euler"
        self.settings.setValue("viewport/rotation_display_mode", normalized)
        self.scene_view.set_rotation_display_mode(normalized)
        self._sync_item_info_rotation_buttons()

    def _on_viewport_overlay_color_mode_changed(self, *_args):
        if not hasattr(self, "overlay_color_mode_combo"):
            return
        mode = str(self.overlay_color_mode_combo.currentData() or "invert").lower()
        self.settings.setValue("viewport/overlay_color_mode", mode)
        if hasattr(self, "scene_view"):
            self.scene_view.set_overlay_color_mode(mode)

    def _on_viewport_projection_changed(self, *_args):
        if not hasattr(self, "scene_view") or not hasattr(self, "projection_mode_combo"):
            return
        mode = str(self.projection_mode_combo.currentData() or "perspective").lower()
        self.settings.setValue("viewport/projection_mode", mode)
        self.projection_axis_combo.setEnabled(mode == "orthographic")
        try:
            self.scene_view.set_projection_mode(mode)
            if mode == "orthographic":
                axis = str(self.projection_axis_combo.currentData() or "+Z")
                self.scene_view.set_orthographic_axis(axis)
        except (TypeError, ValueError, RuntimeError) as exc:
            self._log(f"Viewport projection change failed: {exc}")

    def _on_viewport_projection_axis_changed(self, *_args):
        if not hasattr(self, "scene_view") or not hasattr(self, "projection_axis_combo"):
            return
        axis = str(self.projection_axis_combo.currentData() or "+Z").upper()
        self.settings.setValue("viewport/orthographic_axis", axis)
        if str(self.projection_mode_combo.currentData() or "perspective") == "orthographic":
            try:
                self.scene_view.set_orthographic_axis(axis)
            except (TypeError, ValueError, RuntimeError) as exc:
                self._log(f"Orthographic reference-axis change failed: {exc}")

    def _apply_viewport_camera_settings(self, *_args):
        """Persist editable lens/clipping controls and apply them to VTK's camera."""
        if not hasattr(self, "scene_view") or not hasattr(self, "camera_focal_length_spin"):
            return
        focal_length = float(self.camera_focal_length_spin.value())
        clip_start = float(self.camera_clip_start_spin.value())
        clip_end = float(self.camera_clip_end_spin.value())
        sender = self.sender()
        if clip_end <= clip_start:
            if sender is self.camera_clip_start_spin:
                clip_end = min(self.camera_clip_end_spin.maximum(), clip_start + 0.001)
                self.camera_clip_end_spin.blockSignals(True)
                self.camera_clip_end_spin.setValue(clip_end)
                self.camera_clip_end_spin.blockSignals(False)
            else:
                clip_start = max(self.camera_clip_start_spin.minimum(), clip_end - 0.001)
                self.camera_clip_start_spin.blockSignals(True)
                self.camera_clip_start_spin.setValue(clip_start)
                self.camera_clip_start_spin.blockSignals(False)
        try:
            self.scene_view.set_camera_focal_length(focal_length)
            self.scene_view.set_camera_clip_range(clip_start, clip_end)
        except (AttributeError, TypeError, ValueError) as exc:
            self._log(f"Viewport camera settings were not applied: {exc}")
            return
        self.settings.setValue("viewport/camera/focal_length_mm", focal_length)
        self.settings.setValue("viewport/camera/clip_start_m", clip_start)
        self.settings.setValue("viewport/camera/clip_end_m", clip_end)

    def _choose_viewport_overlay_color(self, overlay: str, layer: str, button: QPushButton) -> None:
        """Choose and persist an independent source/result overlay color."""
        setting_key = f"viewport/overlay_colors/{overlay}/{layer}"
        current = QColor(str(self.settings.value(setting_key, "#FFFFFF") or "#FFFFFF"))
        if not current.isValid():
            current = QColor("#FFFFFF")
        title = f"{overlay.replace('_', ' ').title()} · {layer.title()} Color"
        chosen = QColorDialog.getColor(current, self, title)
        if not chosen.isValid():
            return
        custom_index = self.overlay_color_mode_combo.findData("custom")
        if custom_index >= 0:
            self.overlay_color_mode_combo.setCurrentIndex(custom_index)
        rgb = (chosen.redF(), chosen.greenF(), chosen.blueF())
        if overlay == "wireframe":
            self.scene_view.set_wireframe_color(layer, rgb)
        elif overlay == "sharp_edges":
            self.scene_view.set_sharp_edge_color(layer, rgb)
        else:
            return
        self.settings.setValue(setting_key, chosen.name(QColor.NameFormat.HexRgb).upper())
        self._style_overlay_color_button(button, chosen)

    def _on_face_islands_opacity_changed(self, value: int):
        opacity = max(0.0, min(1.0, float(value) / 100.0))
        self.connected_faces_opacity_label.setText(f"{int(value)}%")
        self.scene_view.set_face_islands_opacity(opacity)

    def _apply_viewport_shortcuts(self):
        """Apply the application's fixed viewport bindings."""
        if not hasattr(self, "scene_view"):
            return
        navigation = {
            "forward": "W", "backward": "S", "left": "A",
            "right": "D", "down": "Q", "up": "E",
        }
        actions = {
            "focus_view": "F", "toggle_wireframe": "X",
            "toggle_connected_faces": "C", "toggle_grid": "G",
            "reimport": "R", "toggle_face_overlay": "Z",
        }
        self.scene_view.set_navigation_shortcuts(navigation)
        self.scene_view.set_viewport_shortcuts(actions)

    def _handle_viewport_shortcut(self, action: str):
        if action == "focus_view":
            self.scene_view.focus_view()
        elif action == "toggle_wireframe":
            self.wireframe_toggle.setChecked(not self.wireframe_toggle.isChecked())
        elif action == "toggle_connected_faces":
            self.face_islands_toggle.setChecked(not self.face_islands_toggle.isChecked())
        elif action == "toggle_grid":
            self.scene_view.toggle_viewport_grid()
        elif action == "reimport":
            self._reimport_source()
        elif action == "toggle_face_overlay":
            self.face_overlay_toggle.setChecked(not self.face_overlay_toggle.isChecked())

    def _update_overlay_control_visibility(self, *_args):
        wire = bool(self.wireframe_toggle.isChecked())
        connected = bool(self.face_islands_toggle.isChecked())
        selected_layer = getattr(self.scene_view, "selected_mesh_layer", None)
        sharp = selected_layer in {"source", "result"}
        self.wireframe_options_widget.setVisible(wire)
        self.connected_faces_options_widget.setVisible(connected)
        self.overlay_mode_widget.setVisible(wire or sharp)
        for key, widget in self._overlay_color_widgets.items():
            widget.setVisible(wire if key == "wireframe" else sharp)
        for (key, layer), button in self._viewport_overlay_color_buttons.items():
            if key == "sharp_edges":
                button.setVisible(sharp and layer == selected_layer)

    def _set_face_overlay_enabled(self, enabled: bool):
        """Enable or disable front-overlaid faces in the 3D preview scene."""
        visible = bool(enabled)
        self.scene_view.set_face_overlay_enabled(visible)
        self.face_overlay_toggle.setText(
            "FACE OVERLAY ON" if visible else "FACE OVERLAY OFF"
        )

    def _set_diagnostics_visible(self, expanded: bool):
        """Expand compact diagnostics only on demand; keep the 3D viewport dominant."""
        self.diagnostics_panel.setVisible(bool(expanded))
        if expanded:
            self._start_gpu_telemetry()
        else:
            self._stop_gpu_telemetry()
        self.diagnostics_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.diagnostics_toggle.setText(
            "HIDE DIAGNOSTICS" if expanded else "PERFORMANCE / MESH INFO / CONSOLE"
        )

    @staticmethod
    def _weight_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.01)
        spin.setValue(value)
        return spin

    def _log(self, message: str):
        """Append a console message, buffering it during early UI construction."""
        if not hasattr(self, "log_box"):
            self._pending_log_messages.append(str(message))
            return
        self._append_log(message)

    def _append_log(self, message: str):

        stamp = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{stamp}] {message}")

    def _apply_resolution_preset(self, _index: int):
        key = str(self.resolution_preset.currentData() or "custom")
        if key == "custom":
            self.resolution_hint.setText(self._custom_resolution_hint())
            return
        values = self._resolution_preset_values.get(key)
        if values is None:
            return
        sample_count, hint = values
        self._applying_resolution_preset = True
        try:
            target_index = self.max_points.findData(sample_count)
            if target_index >= 0:
                self.max_points.setCurrentIndex(target_index)
        finally:
            self._applying_resolution_preset = False
        self.resolution_hint.setText(hint)

    def _on_manual_resolution_changed(self, *_args):
        if self._applying_resolution_preset:
            return
        if hasattr(self, "resolution_preset"):
            custom_index = self.resolution_preset.findData("custom")
            if custom_index >= 0 and self.resolution_preset.currentIndex() != custom_index:
                self.resolution_preset.blockSignals(True)
                self.resolution_preset.setCurrentIndex(custom_index)
                self.resolution_preset.blockSignals(False)
            self.resolution_hint.setText(self._custom_resolution_hint())

    def _custom_resolution_hint(self) -> str:
        sample_count = int(self.max_points.currentData() or 0)
        sample_label = "No cap" if sample_count <= 0 else f"{sample_count:,}"
        return f"{sample_label} samples · Custom"

    def _auto_input_is_mesh_candidate(self) -> bool:
        """Return whether an imported triangle mesh can use the Li & Nan fallback."""
        asset = getattr(self, "_source_asset_raw", None)
        return bool(asset and asset.get("kind") == "mesh")

    def _effective_linan_mode(self) -> bool:
        """Return whether the current engine choice resolves to mesh simplification."""
        requested = str(self.engine_mode.currentData() or "auto").lower()
        return requested in {"li_nan", "linan", "feature_preserving"} or (
            requested == "auto" and self._auto_input_is_mesh_candidate()
        )

    def _on_engine_mode_changed(self, *_args):
        """Keep Li & Nan controls and engine-routing guidance in sync."""
        if not hasattr(self, "linan_options_widget"):
            return
        requested = str(self.engine_mode.currentData() or "auto").lower()
        explicit_linan = requested in {"li_nan", "linan", "feature_preserving"}
        auto_mesh_candidate = requested == "auto" and self._auto_input_is_mesh_candidate()
        enabled = explicit_linan or auto_mesh_candidate
        self.linan_options_widget.setEnabled(enabled)
        if hasattr(self, "linan_feature_settings"):
            self.linan_feature_settings.setEnabled(enabled)

        if hasattr(self, "engine_route_hint"):
            if explicit_linan:
                hint = "Li & Nan 2021 · Filter → planar regions → QEM"
            elif auto_mesh_candidate:
                hint = "AUTO · Mesh → PolyFit first → Li & Nan fallback on failure"
            elif requested == "auto":
                hint = "AUTO · PolyFit first · Mesh fallback → Li & Nan 2021"
            else:
                hint = "PolyFit · Point clouds and planar structures"
            self.engine_route_hint.setText(hint)

        if hasattr(self, "linan_target_vertices"):
            self.linan_target_vertices.setToolTip(
                "Li & Nan AUTO uses about 2 × the detected planar-region count. "
                "Enter a positive value to override the automatic target."
            )
        if hasattr(self, "merge_vertices_checkbox"):
            if enabled:
                self.merge_vertices_checkbox.setToolTip(
                    "The selected mesh is simplified directly. Input cleanup is applied before filtering and decimation; "
                    "the original file remains untouched and is used as the high-poly appearance source for texture baking."
                )
            else:
                self.merge_vertices_checkbox.setToolTip(
                    "Checks input mesh vertices using Merge by Distance. When enabled, the merged geometry shown in the "
                    "source-scene preview is used for reconstruction surface sampling. The original input file is not modified."
                )

    def _on_surface_mix_changed(self, value: int):
        mix = max(0, min(100, int(value)))
        self.source_surface_label.setText(f"A · SOURCE {100 - mix}%")
        self.result_surface_label.setText(f"B · RESULT {mix}%")
        self.scene_view.set_surface_mix(mix)

    def _on_wire_mix_changed(self, value: int):
        mix = max(0, min(100, int(value)))
        self.source_wire_label.setText(f"A · SOURCE {100 - mix}%")
        self.result_wire_label.setText(f"B · RESULT {mix}%")
        self.scene_view.set_wireframe_mix(mix)

    def _reset_viewport_to_new_source(self):
        """Show a newly imported input as A-Source 100% with B-Result hidden."""
        self.surface_mix_slider.setValue(0)
        self._on_surface_mix_changed(0)
        self.wire_mix_slider.setValue(0)
        self._on_wire_mix_changed(0)
        self.scene_view.set_display_mode("both")
        self.scene_view.reset_view(animate=False)

    def _set_output_workspace_display(self, input_path: str | Path | None = None):
        source = str(input_path or self.input_edit.text().strip())
        if not source:
            self.output_edit.clear()
            return
        work_dir = mesh_work_dir(source)
        display_path = QDir.toNativeSeparators(str(work_dir))
        self.output_edit.setText(display_path)
        self.output_edit.setToolTip(
            "TEMP workspace. The generated reconstructed baked GLB will appear here: "
            f"{display_path}"
        )

    def _set_output_generated_display(self, baked_path: str | Path):
        path = Path(baked_path)
        if path.is_file():
            display_path = QDir.toNativeSeparators(str(path))
            self.output_edit.setText(display_path)
            self.output_edit.setToolTip(f"Generated reconstructed baked GLB: {display_path}")
            return
        self._set_output_workspace_display()

    def _set_source_file_controls_enabled(self, enabled: bool) -> None:
        """Enable input selection and source re-import together, respecting the active path."""
        state = bool(enabled)
        if hasattr(self, "browse_btn"):
            self.browse_btn.setEnabled(state)
        if hasattr(self, "reimport_btn"):
            current = self.input_edit.text().strip() if hasattr(self, "input_edit") else ""
            try:
                has_file = bool(current) and Path(current).is_file()
            except OSError:
                has_file = False
            self.reimport_btn.setEnabled(state and has_file)

    def _set_asset_import_options_enabled(self, enabled: bool) -> None:
        """Freeze parse-affecting options while their values are snapshotted by the import worker."""
        state = bool(enabled)
        for name in (
            "input_alpha_mode_combo", "input_alpha_cutoff_spin",
            "merge_vertices_checkbox", "merge_target_combo", "merge_distance_spin",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(state)
        if state:
            self._refresh_input_alpha_controls()
            self._set_mesh_merge_controls_enabled()

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select 3D Input", str(Path.home()),
            "3D Data (*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.pts *.txt *.csv *.asc *.obj *.stl *.off *.glb *.gltf *.3mf *.vg *.bvg);;All files (*.*)",
        )
        if not path:
            return
        self._load_input_source(QDir.toNativeSeparators(path), action="select")

    def _reimport_source(self):
        """Reload the current INPUT file, never the generated workspace/result copy."""
        worker = getattr(self, "worker", None)
        export_worker = getattr(self, "_export_worker", None)
        if (worker is not None and worker.isRunning()) or (
            export_worker is not None and export_worker.isRunning()
        ) or bool(getattr(self, "_auto_bake_active", False)):
            self._log("REIMPORT ignored while reconstruction or texture baking is active.")
            return
        source_path = self.input_edit.text().strip()
        if not source_path or not Path(source_path).is_file():
            QMessageBox.warning(
                self, "Reimport source", "The original INPUT file is missing. Select it again with Browse Input."
            )
            self._set_source_file_controls_enabled(True)
            return
        self._load_input_source(source_path, action="reimport")

    def _load_source_preview_asset(
        self, path, *, mesh_sample_count=65000, merge_distance=0.0,
        progress_callback=None, alpha_mode_override=None, alpha_cutoff_override=None,
    ):
        """Synchronous helper for small refreshes; never read widgets off-thread."""
        if alpha_mode_override is None:
            alpha_mode = str(self.input_alpha_mode_combo.currentData() or "ALPHA_CLIP").upper()
            mode_override = None if alpha_mode == "USE_FILE" else alpha_mode
        else:
            mode_override = alpha_mode_override
        cutoff = (
            float(self.input_alpha_cutoff_spin.value())
            if alpha_cutoff_override is None else float(alpha_cutoff_override)
        )
        return load_preview_asset(
            path,
            mesh_sample_count=mesh_sample_count,
            merge_distance=merge_distance,
            alpha_mode_override=mode_override,
            alpha_cutoff_override=cutoff,
            progress_callback=progress_callback,
        )

    def _refresh_input_alpha_controls(self, *_args):
        if not hasattr(self, "input_alpha_mode_combo"):
            return
        mode = str(self.input_alpha_mode_combo.currentData() or "ALPHA_CLIP").upper()
        self.input_alpha_cutoff_spin.setEnabled(mode in {"ALPHA_CLIP", "ALPHA_HASHED"})

    def _on_input_material_alpha_changed(self, *_args):
        """Persist and re-render source alpha without changing reconstruction data."""
        if not hasattr(self, "input_alpha_mode_combo") or not hasattr(self, "input_alpha_cutoff_spin"):
            return
        mode = str(self.input_alpha_mode_combo.currentData() or "ALPHA_CLIP").upper()
        cutoff = float(self.input_alpha_cutoff_spin.value())
        self.settings.setValue("viewport/input_alpha_mode", mode)
        self.settings.setValue("viewport/input_alpha_cutoff", cutoff)
        self._refresh_input_alpha_controls()
        input_path = self.input_edit.text().strip() if hasattr(self, "input_edit") else ""
        if not input_path or not Path(input_path).is_file():
            return
        if not self._source_asset_raw or self._source_asset_raw.get("kind") != "mesh":
            return
        if (self.worker is not None and self.worker.isRunning()) or (
            self._export_worker is not None and self._export_worker.isRunning()
        ) or (self._asset_import_worker is not None and self._asset_import_worker.isRunning()):
            return
        try:
            merge_distance = self.merge_distance_spin.value() if self._mesh_cleanup_includes_input() else 0.0
            asset = self._load_source_preview_asset(
                input_path, mesh_sample_count=65000, merge_distance=merge_distance
            )
            if asset.get("kind") != "mesh":
                return
            self._source_asset_raw = asset
            self._render_source_asset(fit_camera=False)
            self._log(f"Input material preview updated · alpha={mode} · cutoff={cutoff:.2f}")
        except Exception as exc:
            self._log(f"Input material alpha preview update failed: {exc}")

    def _show_asset_import_progress(self, path: Path):
        dialog = QProgressDialog(
            f"Reading {path.name} · decoding geometry and materials…", "", 0, 0, self
        )
        dialog.setWindowTitle("Importing 3D Asset")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumWidth(390)
        dialog.setValue(0)
        dialog.show()
        self._asset_import_progress = dialog

    def _on_asset_import_progress(self, value: int, message: str):
        dialog = self._asset_import_progress
        if dialog is None:
            return
        dialog.setLabelText(str(message))
        if int(value) < 0:

            dialog.setRange(0, 0)
        else:
            if dialog.maximum() == 0:
                dialog.setRange(0, 100)
            dialog.setValue(max(0, min(100, int(value))))

    def _close_asset_import_progress(self):
        dialog = self._asset_import_progress
        self._asset_import_progress = None
        if dialog is not None:
            dialog.setValue(100)
            dialog.close()
            dialog.deleteLater()

    def _on_asset_import_completed(self, asset):
        pending = self._pending_asset_import
        self._close_asset_import_progress()
        if pending is None:
            return
        native_path, action = pending
        self._pending_asset_import = None
        self._finish_input_source_load(native_path, action=action, new_source_asset=asset)

    def _on_asset_import_failed(self, message: str):
        pending = self._pending_asset_import
        self._close_asset_import_progress()
        self._pending_asset_import = None
        path = pending[0] if pending else ""
        action = pending[1] if pending else "select"
        short_message = str(message).split("\n\n", 1)[0]
        self._log(f"Source {'reimport' if action == 'reimport' else 'preview load'} failed: {short_message}")
        self._set_source_file_controls_enabled(True)
        self._set_asset_import_options_enabled(True)
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(bool(self.input_edit.text().strip()))
        QMessageBox.critical(self, "Source import failed", f"{path}\n\n{short_message}")

    def _on_asset_import_thread_finished(self):
        worker = self._asset_import_worker
        self._asset_import_worker = None
        if worker is not None:
            worker.deleteLater()

    def _load_input_source(self, path: str | Path, *, action: str = "select"):
        """Start an asynchronous source import with a GUI progress indicator."""
        native_path = QDir.toNativeSeparators(str(path))
        p = Path(native_path)
        if not p.is_file():
            QMessageBox.warning(self, "Input file", f"The selected source file does not exist:\n{native_path}")
            self._set_source_file_controls_enabled(True)
            return
        current = getattr(self, "_asset_import_worker", None)
        if current is not None and current.isRunning():
            self._log("Asset import ignored because another source is still being parsed.")
            return

        is_bvg = p.suffix.lower() == ".bvg"
        if is_bvg:
            self._finish_input_source_load(native_path, action=action, new_source_asset=None)
            return

        cleanup_scope = str(self.merge_target_combo.currentData() or "both").lower()
        apply_input_cleanup = bool(
            self.merge_vertices_checkbox.isChecked() and cleanup_scope in {"input", "both"}
        )
        merge_distance = self.merge_distance_spin.value() if apply_input_cleanup else 0.0
        alpha_mode = str(self.input_alpha_mode_combo.currentData() or "ALPHA_CLIP").upper()
        mode_override = None if alpha_mode == "USE_FILE" else alpha_mode
        alpha_cutoff = float(self.input_alpha_cutoff_spin.value())

        self._pending_asset_import = (native_path, action)
        self._set_source_file_controls_enabled(False)
        self._set_asset_import_options_enabled(False)
        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self._show_asset_import_progress(p)
        self._on_asset_import_progress(-1, f"Reading {p.name} · decoding geometry, UVs and materials…")

        self._asset_import_worker = AssetImportWorker(
            native_path,
            mesh_sample_count=65000,
            merge_distance=merge_distance,
            alpha_mode_override=mode_override,
            alpha_cutoff_override=alpha_cutoff,
        )
        self._asset_import_worker.progress.connect(self._on_asset_import_progress)
        self._asset_import_worker.completed.connect(self._on_asset_import_completed)
        self._asset_import_worker.failed.connect(self._on_asset_import_failed)
        self._asset_import_worker.finished.connect(self._on_asset_import_thread_finished)
        self._asset_import_worker.start()

    def _finish_input_source_load(
        self, native_path: str, *, action: str, new_source_asset: dict | None,
    ):
        """Commit a parsed source into the scene only after the worker succeeded."""
        p = Path(native_path)
        self.input_edit.setText(native_path)
        self._set_output_workspace_display(p)
        self.export_btn.setEnabled(False)
        self.last_result = None
        self.last_output = None
        self.last_baked_glb = None
        self._source_asset_raw = new_source_asset
        if new_source_asset:
            source_modes = new_source_asset.get("material_render_modes") or []
            if source_modes:
                mode_summary = "; ".join(
                    f"{item.get('name', 'Material')}: alpha={item.get('alpha_mode')} "
                    f"(cutoff={float(item.get('alpha_cutoff', 0.5)):.2f}), shadow={item.get('shadow_mode')}"
                    for item in source_modes
                )
                self._log("Imported glTF material render modes · " + mode_summary)
        self._source_frame_transform_raw = as_transform_matrix(
            new_source_asset.get("source_frame_transform") if new_source_asset else None
        )
        self.scene_view.clear_result()
        self.scene_view.clear_processing_mesh()
        self._reset_viewport_to_new_source()
        self._live_mesh_preview_loaded = False
        self.wireframe_toggle.setChecked(False)
        self.wireframe_toggle.setEnabled(True)
        self._reset_mesh_comparison()
        self.card_points.set_value("—")
        self.card_planes.set_value("—")
        self.card_faces.set_value("—")
        self.card_elapsed.set_value("00:00")
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(0)
        if hasattr(self, "percent_label"):
            self.percent_label.setText("0%")
        if hasattr(self, "header_status"):
            self.header_status.setText("● INPUT READY")

        is_segmented = p.suffix.lower() in {".vg", ".bvg"}
        self._axis_control_allowed = not is_segmented
        self.source_up_axis.setEnabled(not is_segmented)
        if is_segmented and self.source_up_axis.currentData() != "z":
            target_index = self.source_up_axis.findData("z")
            if target_index >= 0:
                self.source_up_axis.setCurrentIndex(target_index)

        if p.suffix.lower() == ".bvg":
            self.scene_view.clear_source()
            self.merge_vertices_checkbox.setEnabled(True)
            self.merge_target_combo.setEnabled(self.merge_vertices_checkbox.isChecked())
            self.merge_distance_spin.setEnabled(self.merge_vertices_checkbox.isChecked())
            self.scene_info.setText("BVG · PolyFit segmented input (source-scene preview is not available)")
            self._set_mesh_comparison_stage(
                "input", {"file_size_mb": p.stat().st_size / (1024.0 * 1024.0)}, "segmented"
            )
            self._log("BVG input: direct PolyFit reconstruction only · source-scene preview skipped")
        else:
            self.merge_vertices_checkbox.setEnabled(True)
            self.merge_target_combo.setEnabled(self.merge_vertices_checkbox.isChecked())
            self.merge_distance_spin.setEnabled(self.merge_vertices_checkbox.isChecked())
            self._render_source_asset()
            self.scene_view.set_display_mode("both")
            source_stats = dict(self._source_asset_raw.get("mesh_stats") or {})
            source_stats["file_size_mb"] = p.stat().st_size / (1024.0 * 1024.0)
            if self._source_asset_raw.get("kind") == "points":
                source_stats["points"] = int(len(self._source_asset_raw.get("points", ())))
            self._set_mesh_comparison_stage(
                "input", source_stats, self._source_asset_raw.get("kind", "unknown")
            )
            if self._source_asset_raw["kind"] == "points":
                self.card_points.set_value(f"{len(self._source_asset_raw['points']):,}")
            else:
                self.card_points.set_value(f"{len(self._source_asset_raw['vertices']):,} verts")
            self._log(f"Source-scene preview loaded: {p.name}")

        self.scene_view.set_empty_scene_prompt_visible(False)
        self._on_engine_mode_changed()
        self._set_source_file_controls_enabled(True)
        self._set_asset_import_options_enabled(True)
        self.run_btn.setEnabled(True)
        action_label = "Original source reimported" if action == "reimport" else "Input selected"
        self.stage_label.setText(f"{action_label} · Source scene displayed")
        self.scene_info.setText(
            "Original source reloaded · ready for reconstruction"
            if action == "reimport" and p.suffix.lower() != ".bvg"
            else self.scene_info.text()
        )
        if action == "reimport":
            self._log(
                f"REIMPORT complete · original source loaded from INPUT: {p.name} · "
                "generated reconstruction and bake previews were cleared."
            )
        self._log(f"Input selected: {p.name}" if action != "reimport" else f"Original INPUT restored: {p.name}")

    def _local_studio_points_to_scene(self, values):
        """Map local reconstruction/bake coordinates back onto the source scene frame."""
        return transform_points(values, self._source_frame_transform_studio)

    def _render_source_asset(self, *, fit_camera: bool = True):
        if not self._source_asset_raw:
            self._source_frame_transform_studio = np.eye(4, dtype=np.float64)
            if hasattr(self, "scene_view"):
                self.scene_view.set_source_frame_transform(self._source_frame_transform_studio)
            return
        axis = str(self.source_up_axis.currentData() or "y")
        self._source_frame_transform_studio = source_frame_to_studio(
            self._source_frame_transform_raw, axis
        )
        self.scene_view.set_source_frame_transform(self._source_frame_transform_studio)
        asset = self._source_asset_raw
        if asset["kind"] == "mesh":
            vertices = convert_up_axis(asset["vertices"], axis)
            material_parts = []
            for part in asset.get("material_parts", ()):
                material_parts.append({
                    **part,
                    "vertices": convert_up_axis(part["vertices"], axis),
                    "normals": (
                        None if part.get("normals") is None
                        else convert_up_axis(part["normals"], axis)
                    ),
                })
            self.scene_view.set_source_mesh(
                vertices, asset["faces"], asset.get("colors"),
                has_texture=bool(asset.get("has_texture", False)),
                material_parts=material_parts,
                fit_camera=fit_camera,
                normals=(
                    None if asset.get("normals") is None
                    else convert_up_axis(asset["normals"], axis)
                ),
            )
            self.wireframe_toggle.setEnabled(True)
        else:
            points = convert_up_axis(asset["points"], axis)
            self.scene_view.set_source_points(points, asset.get("colors"))
            self.wireframe_toggle.setEnabled(True)
        label = asset.get("label", "source asset")
        if axis != "z":
            label += f" · {axis.upper()}-up → Z-up"
        self.scene_info.setText(label)

    def _mesh_cleanup_includes_input(self) -> bool:
        scope = str(self.merge_target_combo.currentData() or "both").lower()
        is_mesh = bool(self._source_asset_raw and self._source_asset_raw.get("kind") == "mesh")
        return bool(self.merge_vertices_checkbox.isChecked() and is_mesh and scope in {"input", "both"})

    def _mesh_cleanup_includes_output(self) -> bool:
        scope = str(self.merge_target_combo.currentData() or "both").lower()
        return bool(self.merge_vertices_checkbox.isChecked() and scope in {"output", "both"})

    def _set_mesh_merge_controls_enabled(self):
        has_input = bool(self.input_edit.text().strip() and Path(self.input_edit.text().strip()).is_file())
        self.merge_vertices_checkbox.setEnabled(has_input)
        self.merge_target_combo.setEnabled(has_input and self.merge_vertices_checkbox.isChecked())
        self.merge_distance_spin.setEnabled(has_input and self.merge_vertices_checkbox.isChecked())

    def _on_mesh_merge_options_changed(self, *_args):
        checked = self.merge_vertices_checkbox.isChecked()
        has_input = bool(self.input_edit.text().strip() and Path(self.input_edit.text().strip()).is_file())
        self.merge_target_combo.setEnabled(checked and has_input)
        self.merge_distance_spin.setEnabled(checked and has_input)
        input_path = self.input_edit.text().strip()
        if not input_path or not Path(input_path).is_file():
            return
        if self.worker is not None and self.worker.isRunning():
            return
        merge_distance = self.merge_distance_spin.value() if self._mesh_cleanup_includes_input() else 0.0
        try:
            if self._source_asset_raw is not None and self._source_asset_raw.get("kind") == "mesh":
                self._source_asset_raw = self._load_source_preview_asset(
                    input_path, mesh_sample_count=65000, merge_distance=merge_distance
                )
                source_stats = dict(self._source_asset_raw.get("mesh_stats") or {})
                source_stats["file_size_mb"] = Path(input_path).stat().st_size / (1024.0 * 1024.0)
                self._set_mesh_comparison_stage("input", source_stats, self._source_asset_raw.get("kind", "unknown"))
                self._on_engine_mode_changed()
                self._render_source_asset()
                self._reset_viewport_to_new_source()
                self.card_points.set_value(f"{len(self._source_asset_raw['vertices']):,} verts")
                if merge_distance > 0.0:
                    self._log(
                        f"Input preview Merge by Distance applied · distance {merge_distance:g} · "
                        f"{self._source_asset_raw.get('label', '')}"
                    )
                else:
                    self._log("Input preview uses the unmerged source geometry.")
            else:
                self._log("Mesh Cleanup target changed. Point-cloud inputs are not vertex-welded; OUTPUT cleanup remains available.")

            if self.last_result is not None or self.last_output is not None:
                self.scene_view.clear_result()
                self.last_result = None
                self.last_output = None
                self.last_baked_glb = None
                self._set_mesh_comparison_stage("reconstruction", None)
                self._set_mesh_comparison_stage("output", None)
                self._set_output_workspace_display()
                self._live_mesh_preview_loaded = False
                self.export_btn.setEnabled(False)
                self.stage_label.setText("Mesh Cleanup settings changed · Reconstruct again")
        except Exception as exc:
            self._log(f"Merge by Distance preview failed: {exc}")

    def _on_bake_compute_device_changed(self, *_args):
        if not hasattr(self, "bake_compute_combo"):
            return
        mode = str(self.bake_compute_combo.currentData() or "auto").lower()
        self.settings.setValue("bake_compute_device", mode)
        self.settings.setValue("bake_compute_cpu_override_explicit", mode == "cpu")

    def _on_bake_projection_changed(self, *_args):
        """Keep cage-only controls unavailable when direct baking is selected."""
        if not hasattr(self, "bake_projection_combo"):
            return
        mode = str(self.bake_projection_combo.currentData() or "cage")
        self.settings.setValue("bake_projection_mode", mode)
        if hasattr(self, "cage_extrusion_spin"):
            self.cage_extrusion_spin.setEnabled(mode == "cage")

    def _on_axis_changed(self, _index: int):
        if self._source_asset_raw:
            self._render_source_asset()

            if self.last_result is not None or self.last_output is not None:
                self.scene_view.clear_result()
                self.last_result = None
                self.last_output = None
                self.last_baked_glb = None
                self._set_mesh_comparison_stage("reconstruction", None)
                self._set_mesh_comparison_stage("output", None)
                self._set_output_workspace_display()
                self._live_mesh_preview_loaded = False
                self.export_btn.setEnabled(False)
                self.stage_label.setText("Axis conversion changed · Reconstruct in the new coordinate frame")
                self._log("The coordinate axis changed, so the previous result was removed from the scene. Run reconstruction again with the new axis.")
            self._apply_scene_mode()

    def _apply_scene_mode(self):

        self.scene_view.set_display_mode("both")

    def _start_pipeline(self):
        if self.worker is not None and self.worker.isRunning():
            return
        input_path = self.input_edit.text().strip()
        if not input_path or not Path(input_path).exists():
            QMessageBox.warning(self, "Input file", "Select a valid input file first.")
            return
        selected_engine = str(self.engine_mode.currentData() or "auto").lower()
        ui_linan_mode = self._effective_linan_mode()
        if selected_engine not in {"li_nan", "linan", "feature_preserving"} and not ui_linan_mode and abs((self.w_fit.value() + self.w_coverage.value() + self.w_complexity.value()) - 1.0) > 0.03:
            QMessageBox.warning(self, "Check weights", "The three weights differ significantly from a sum of 1.0. Consider adjusting them closer to the defaults.")
        if selected_engine in {"li_nan", "linan", "feature_preserving"} and Path(input_path).suffix.lower() in {".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts", ".txt", ".csv", ".asc"}:
            QMessageBox.warning(
                self,
                "Mesh reconstruction requires a mesh",
                "The selected reconstruction mode requires an existing mesh. Select an OBJ, STL, OFF, GLB, glTF, 3MF, or mesh-based PLY input instead of a point cloud.",
            )
            return
        work_dir = mesh_work_dir(input_path)
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Temporary Workspace", f"Could not create the temporary workspace:\n{work_dir}\n\n{exc}")
            return
        output_path = str(reconstructed_obj_path(input_path))
        if Path(input_path).resolve() == Path(output_path).resolve():
            QMessageBox.warning(self, "Path error", "Input and temporary output paths must be different.")
            return
        self._set_output_workspace_display(input_path)
        config = {
            "input_path": input_path,
            "output_path": output_path,
            "work_dir": str(work_dir),
            "source_up_axis": str(self.source_up_axis.currentData() or "y"),
            "source_frame_transform": self._source_frame_transform_raw.tolist(),
            "merge_mesh_vertices": bool(self._mesh_cleanup_includes_input()),
            "merge_mesh_input": bool(self._mesh_cleanup_includes_input()),
            "merge_mesh_output": bool(self._mesh_cleanup_includes_output()),
            "merge_cleanup_scope": str(self.merge_target_combo.currentData() or "both"),
            "merge_distance": float(self.merge_distance_spin.value()),
            "resolution_preset": str(self.resolution_preset.currentData() or "custom"),
            "output_face_budget": get_output_face_budget(str(self.resolution_preset.currentData() or "custom")),
            "max_points": int(self.max_points.currentData()),
            "preserve_small_features": bool(self.preserve_small_features.isChecked()),
            "small_feature_min_points": int(self.small_feature_min_points.value()),
            "voxel_size": self.voxel.value(),
            "distance_threshold": self.threshold.value(),
            "min_plane_points": self.min_plane_points.value(),
            "max_planes": self.max_planes.value(),
            "ransac_iterations": self.iterations.value(),
            "engine_mode": str(self.engine_mode.currentData()),
            "linan_target_vertices": int(self.linan_target_vertices.value()),
            "linan_filter_iterations": int(self.linan_filter_iterations.value()),
            "linan_filter_normal_angle": float(self.linan_filter_normal_angle.value()),
            "linan_region_normal_angle": float(self.linan_region_normal_angle.value()),
            "linan_small_region_area_threshold": float(self.linan_small_region_area.value()),
            "linan_fitting_threshold": float(self.linan_fitting_threshold.value()),
            "linan_boundary_lock": bool(self.linan_boundary_lock.isChecked()),
            "linan_intersection_guard": bool(
                selected_engine in {"li_nan", "linan", "feature_preserving"}
                or (selected_engine == "auto" and self._auto_input_is_mesh_candidate())
            ),
            "fitting": self.w_fit.value(),
            "coverage": self.w_coverage.value(),
            "complexity": self.w_complexity.value(),
            "polyfit_optimization_timeout_seconds": POLYFIT_OPTIMIZATION_TIMEOUT_SECONDS,
        }
        self.scene_view.reset_view()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.percent_label.setText("0%")
        self.stage_label.setText("Pipeline 1/2 · Reconstruction · Starting")
        self.header_status.setText("● RUNNING · PIPELINE")
        self._header_status_state = "active"
        self._set_header_status_style("active")
        self.run_btn.setEnabled(False)
        self._set_task_paused_state(False)
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self._set_source_file_controls_enabled(False)
        self.output_edit.setEnabled(False)
        self.source_up_axis.setEnabled(False)
        self.merge_vertices_checkbox.setEnabled(False)
        self.merge_target_combo.setEnabled(False)
        self.merge_distance_spin.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.last_result = None
        self.last_output = None
        self.last_baked_glb = None
        self._set_mesh_comparison_stage("reconstruction", None)
        self._set_mesh_comparison_stage("output", None)
        self._live_mesh_preview_loaded = False
        self.started_at = time.perf_counter()
        self.card_points.set_value("—")
        self.card_planes.set_value("—")
        self.card_faces.set_value("—")
        self.card_elapsed.set_value("00:00")
        self._log("=" * 46)
        self._log(f"Reconstruction started · source up-axis={config['source_up_axis'].upper()} → Studio Z-up")
        self._log(
            f"Output resolution: {config['resolution_preset']} · "
            f"Max samples {config['max_points'] or 'No cap'}"
        )
        active_engine = str(config.get("engine_mode", "auto")).lower()
        if active_engine == "auto" and ui_linan_mode:
            self._log(
                "AUTO detected a triangulated mesh candidate. PolyFit will run first; if it errors, times out, or its isolated worker crashes, Li & Nan 2021 will retry using the original input mesh."
            )
        if active_engine == "polyfit":
            self._log(
                "PolyFit mode: save the official native polygon result directly · "
                "Output face count is controlled by the SCIP model-complexity weight (no post-process face cap)"
            )
        elif active_engine == "li_nan":
            self._log(
                "Li & Nan feature-preserving route: filter → planar regions/contours → hierarchical QEM · "
                f"target {'AUTO (about 2× planar-region count)' if not config['linan_target_vertices'] else f"{config['linan_target_vertices']:,}"} vertices"
            )
        else:
            self._log("AUTO reconstruction: PolyFit first · Li & Nan 2021 is the automatic fallback when a usable mesh is available")
        if config["merge_mesh_input"] or config["merge_mesh_output"]:
            targets = " + ".join(
                label for enabled, label in ((config["merge_mesh_input"], "INPUT"), (config["merge_mesh_output"], "OUTPUT"))
                if enabled
            )
            self._log(
                f"Mesh Cleanup · {targets} · Merge by Distance {config['merge_distance']:g} · "
                "the original input file remains untouched."
            )
        self.scene_view.clear_processing_mesh()
        self.worker = PipelineWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.preview_points.connect(self._on_preview_points)
        self.worker.preview_mesh.connect(self._on_preview_mesh)
        self.worker.stage_mesh.connect(self._on_stage_mesh)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _set_task_paused_state(self, paused: bool) -> None:
        self._task_paused = bool(paused)
        if hasattr(self, "pause_btn"):
            self.pause_btn.setText("▶ RESUME" if self._task_paused else "Ⅱ PAUSE")
            self.pause_btn.setToolTip(
                "Resume the paused task." if self._task_paused
                else "Pause or resume the active reconstruction or automatic texture bake."
            )
        if self._task_paused:
            self._stage_text_before_pause = self.stage_label.text()
            self.stage_label.setText(f"Paused · {self._stage_text_before_pause}")
        elif self._stage_text_before_pause:
            if self.stage_label.text().startswith("Paused · "):
                self.stage_label.setText(self._stage_text_before_pause)
            self._stage_text_before_pause = ""

    def _set_task_stage_text(self, value: str) -> None:
        message = str(value)
        if self._task_paused:
            self._stage_text_before_pause = message
            self.stage_label.setText(f"Paused · {message}")
        else:
            self.stage_label.setText(message)

    def _active_pause_worker(self):
        export_worker = getattr(self, "_export_worker", None)
        if self._auto_bake_active and export_worker is not None and export_worker.isRunning():
            return export_worker
        worker = getattr(self, "worker", None)
        if worker is not None and worker.isRunning():
            return worker
        return None

    def _toggle_pause(self):
        worker = self._active_pause_worker()
        if worker is None:
            self._set_task_paused_state(False)
            self.pause_btn.setEnabled(False)
            return
        if self._task_paused:
            if worker.resume():
                self._set_task_paused_state(False)
                self._log("Task resumed.")
            return
        if not worker.pause():
            self._log("Pause could not be applied safely. The task continues running.")
            return
        self._set_task_paused_state(True)
        self._log("Task paused at a safe checkpoint. Press RESUME to continue or STOP to cancel.")

    def _cancel_pipeline(self):
        export_worker = getattr(self, "_export_worker", None)
        if self._auto_bake_active and export_worker is not None and export_worker.isRunning():
            self.cancel_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.stage_label.setText("Cancel requested · Stopping automatic texture bake")
            self._log("Cancellation requested. The bake will stop at its next safe processing checkpoint.")
            export_worker.request_cancel()
            return
        if self.worker is None or not self.worker.isRunning():
            return
        self.cancel_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.stage_label.setText("Cancel requested · Stopping engine process")
        self._log("Cancellation requested. Terminating the isolated reconstruction engine process.")
        self.worker.cancel()

    def _on_progress(self, value: int, message: str):
        """Render reconstruction progress as the first segment of one continuous pipeline."""
        self.progress_bar.setRange(0, 100)
        overall = reconstruction_pipeline_percent(value)
        self.progress_bar.setValue(overall)
        self.percent_label.setText(f"{overall}%")
        self._set_task_stage_text(f"Pipeline 1/2 · Reconstruction · {message}")

    def _on_preview_points(self, preview_path: str):
        try:
            with np.load(preview_path) as preview:
                points = np.asarray(preview["points"])
                colors = np.asarray(preview["colors"])

            if self._source_asset_raw and self._source_asset_raw.get("kind") == "points":
                points = self._local_studio_points_to_scene(points)
                self.scene_view.set_source_points(points, colors)
                self._apply_scene_mode()
            self.card_points.set_value(f"{len(points):,}")
        except Exception as exc:
            self._log(f"Source 3D preview failed: {exc}")

    def _on_stage_mesh(self, preview_path: str, label: str):
        """Display real intermediate processing geometry as a fluorescent overlay."""
        try:
            with np.load(preview_path) as preview:
                vertices = np.asarray(preview["vertices"], dtype=np.float32)
                faces = np.asarray(preview["faces"], dtype=np.uint32)
                quads = np.asarray(preview["quads"], dtype=np.uint32) if "quads" in preview.files else None
                highlight_segments = np.asarray(preview["highlight_segments"], dtype=np.float32) if "highlight_segments" in preview.files else None
            if len(vertices) == 0 or (len(faces) == 0 and (quads is None or len(quads) == 0)):
                return
            vertices = self._local_studio_points_to_scene(vertices)
            if highlight_segments is not None:
                highlight_segments = self._local_studio_points_to_scene(highlight_segments.reshape(-1, 3)).reshape(-1, 2, 3)
            self.scene_view.set_processing_mesh(
                vertices, faces, label=label, quads=quads,
                highlight_segments=highlight_segments,
            )
            self.scene_info.setText(f"LIVE BUILD · {label}")
        except Exception as exc:
            self._log(f"Live processing-mesh overlay failed: {exc}")

    def _on_preview_mesh(self, preview_path: str):
        """Show the completed surface as soon as geometry is ready, before metrics finish."""
        try:
            with np.load(preview_path) as preview:
                vertices = np.asarray(preview["vertices"])
                faces = np.asarray(preview["faces"])
                colors = np.asarray(preview["colors"]) if "colors" in preview.files else None
            vertices = self._local_studio_points_to_scene(vertices)
            self.scene_view.clear_processing_mesh()
            self.scene_view.set_reconstruction_mesh(
                vertices, faces, colors,
                has_texture=bool(self._source_asset_raw and self._source_asset_raw.get("has_texture", False)),
            )
            self._live_mesh_preview_loaded = True
            self.wireframe_toggle.setEnabled(True)
            self.scene_view.set_display_mode("both")
            self.scene_info.setText(
                f"LIVE RECONSTRUCTION · {len(vertices):,} vertices / {len(faces):,} faces · final GLB bake pending"
            )
            self._log(
                f"Provisional geometry preview displayed: {len(vertices):,} vertices / {len(faces):,} faces · final inspection will load the baked GLB"
            )
        except Exception as exc:
            self._log(f"Live result-mesh preview failed: {exc}")

    def _on_completed(self, result: dict):
        self.last_result = result
        self.last_output = Path(result["output_path"])
        self.last_baked_glb = None
        self._set_output_workspace_display()
        self.scene_view.clear_processing_mesh()
        metrics = result["metrics"]
        engine_name = metrics.get("reconstruction_method", "PolyFit")
        self.scene_info.setText(
            f"{engine_name} reconstruction complete · generating baked GLB for final inspection"
        )
        self.card_faces.set_value(f"{metrics['triangles']:,}")
        self.card_planes.set_value(str(metrics.get("plane_count", "—")) if metrics.get("plane_count", 0) else "—")
        self._update_result_metrics(metrics)
        reconstruction_stats = dict(metrics)
        try:
            reconstruction_stats["file_size_mb"] = Path(result["output_path"]).stat().st_size / (1024.0 * 1024.0)
        except OSError:
            pass
        self._set_mesh_comparison_stage("reconstruction", reconstruction_stats)
        elapsed = int(metrics.get("elapsed_seconds", 0))
        self.card_elapsed.set_value(f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(RECONSTRUCTION_WEIGHT)
        self.percent_label.setText(f"{RECONSTRUCTION_WEIGHT}%")
        self.stage_label.setText("Pipeline 2/2 · Texture Baking · Starting automatic GLB bake")
        self.header_status.setText("● BAKING · GLB MASTER")
        self._header_status_state = "active"
        self._set_header_status_style("active")
        self.export_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self._set_task_paused_state(False)
        self._set_source_file_controls_enabled(False)
        self.output_edit.setEnabled(False)
        self.source_up_axis.setEnabled(False)
        self._set_mesh_merge_controls_enabled()
        self._log(
            f"Reconstruction completed: {engine_name} · {metrics['vertices']:,} vertices / "
            f"{metrics['triangles']:,} triangles · {metrics['elapsed_seconds']:.1f}s"
        )
        if metrics["boundary_edges"] > 0 or metrics["nonmanifold_edges"] > 0:
            self._log(f"Quality warning: boundary edges={metrics['boundary_edges']:,}, non-manifold edges={metrics['nonmanifold_edges']:,}")
        if not metrics["watertight"]:
            if str(engine_name).startswith("Li & Nan"):
                self._log(
                    "Input mesh is not watertight. Li & Nan preserves open boundaries by default but does not fill holes "
                    "or guarantee a watertight reconstruction."
                )
            else:
                self._log("Warning: the imported mesh is open or fragmented. The selected reconstruction path may leave boundaries open.")
        self._start_automatic_glb_bake(result)

    def _save_bake_pbr_map_settings(self, _checked=None):
        """Persist the explicit PBR map selection for the next bake."""
        checks = getattr(self, "bake_pbr_map_checks", {})
        if checks:
            selected = [key for key, checkbox in checks.items() if checkbox.isChecked()]
            self.settings.setValue("bake_pbr_maps", selected)

    def _start_automatic_glb_bake(self, result: dict):
        """Bake the generated OBJ to the canonical GLB master after every reconstruction."""
        if self._export_worker is not None and self._export_worker.isRunning():
            self._log("Automatic GLB bake was not started because another export worker is still active.")
            return
        source_obj = Path(result["output_path"])
        if not source_obj.is_file():
            self._on_export_failed(f"Generated reconstruction OBJ is missing: {source_obj}")
            return

        self.scene_view.reset_view()

        self.face_overlay_toggle.setChecked(True)
        self.surface_mix_slider.setValue(100)
        self.wire_mix_slider.setValue(100)
        self._auto_bake_active = True
        self._gpu_runtime_state_confirmed = False
        if hasattr(self, "gpu_rescan_btn"):
            self.gpu_rescan_btn.setEnabled(False)
        baked_path = reconstructed_baked_glb_path(self.input_edit.text().strip())
        baked_path.unlink(missing_ok=True)
        self._set_output_workspace_display()

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(RECONSTRUCTION_WEIGHT)
        self.percent_label.setText(f"{RECONSTRUCTION_WEIGHT}%")
        self.stage_label.setText("Pipeline 2/2 · Texture Baking · Preparing generated OBJ")
        self.header_status.setText("● PIPELINE · TEXTURE BAKING")
        self.run_btn.setEnabled(False)
        self._set_task_paused_state(False)
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self._set_source_file_controls_enabled(False)
        self.output_edit.setEnabled(False)
        self.source_up_axis.setEnabled(False)
        self.bake_compute_combo.setEnabled(False)
        self.bake_projection_combo.setEnabled(False)
        self.cage_extrusion_spin.setEnabled(False)
        self.bake_pbr_group.setEnabled(False)
        self.export_btn.setEnabled(False)
        compute_mode = str(self.bake_compute_combo.currentData() or "auto").lower()
        metrics = result.get("metrics") or {}
        engine_name = str(metrics.get("reconstruction_method", ""))
        is_linan_result = engine_name.lower().startswith("li & nan")
        if is_linan_result and compute_mode == "auto":
            self._log(
                "Li & Nan 2021 final bake · performance-aware Auto is enabled "
                "(GPU first; compare the first production batch with CPU, then switch remaining batches when CPU is ≥10% faster)."
            )
        elif is_linan_result and compute_mode == "gpu":
            self._log("Li & Nan 2021 final bake · strict GPU Only selected; GPU failure will stop the bake.")
        elif is_linan_result and compute_mode == "cpu":
            self._log("Li & Nan 2021 final bake · CPU-only compatibility override was explicitly selected.")
        self._log(
            f"Automatic GLB bake started as pipeline continuation · target OBJ: {source_obj.name} · "
            f"compute={compute_mode.upper()}"
        )
        source_path = self.input_edit.text().strip()
        preview_path = result.get("mesh_preview_path")
        projection_mode = str(self.bake_projection_combo.currentData() or "cage")
        pbr_maps = [key for key, checkbox in self.bake_pbr_map_checks.items() if checkbox.isChecked()]
        if not pbr_maps:
            pbr_maps = []
        self.settings.setValue("bake_pbr_maps", pbr_maps)
        self._log(
            "Bake geometry chain · original source mesh (high-poly appearance) → "
            f"{source_obj.name} (Reconstruction low-poly target) → "
            f"{'generated cage' if projection_mode == 'cage' else 'direct sampling · no cage'} · "
            f"PBR maps={', '.join(pbr_maps) if pbr_maps else 'none'}"
        )
        self._export_worker = ExportWorker(
            output_path=str(baked_path), suffix=".glb", target_mesh_path=str(source_obj),
            bake_enabled=True,
            resolution=int(self.texture_resolution_combo.currentData() or 2048),
            source_mesh_path=source_path,
            source_up_axis=str(self.source_up_axis.currentData() or "y"),
            projection_mode=projection_mode,
            cage_extrusion_percent=float(self.cage_extrusion_spin.value()),
            preview_path=str(preview_path) if preview_path else None,
            auto_bake=True,
            compute_device=compute_mode,
            pbr_maps=pbr_maps,
            source_frame_transform=self._source_frame_transform_studio,
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.gpu_status.connect(self._on_gpu_compute_status)
        self._export_worker.bake_preview.connect(self._on_bake_preview)
        self._export_worker.bake_edges.connect(self._on_bake_edges)
        self._export_worker.completed.connect(self._on_export_completed)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.finished.connect(self._on_export_thread_finished)
        self._export_worker.start()

    def _on_bake_preview(self, payload: dict):
        """Display the live GLB atlas, projection cage and active chart in the 3D scene."""
        try:
            texture_image = np.asarray(payload.get("texture_image"), dtype=np.uint8)
            if texture_image.ndim != 3 or texture_image.shape[2] not in (3, 4):
                return
            vertices = payload.get("vertices")
            if vertices is not None:
                vertices = self._local_studio_points_to_scene(np.asarray(vertices, dtype=np.float32))
                cage_outer = payload.get("cage_outer_vertices")
                cage_inner = payload.get("cage_inner_vertices")
                if cage_outer is not None:
                    cage_outer = self._local_studio_points_to_scene(cage_outer)
                if cage_inner is not None:
                    cage_inner = self._local_studio_points_to_scene(cage_inner)
                self.scene_view.set_bake_texture_preview(
                    vertices,
                    np.asarray(payload.get("faces"), dtype=np.uint32),
                    np.asarray(payload.get("uvs"), dtype=np.float32),
                    texture_image,
                    diffuse_texture_image=payload.get("diffuse_texture_image"),
                )
            else:
                self.scene_view.update_bake_texture_preview(
                    texture_image,
                    diffuse_texture_image=payload.get("diffuse_texture_image"),
                )

            if vertices is None:
                cage_outer = payload.get("cage_outer_vertices")
                cage_inner = payload.get("cage_inner_vertices")
                if cage_outer is not None:
                    cage_outer = self._local_studio_points_to_scene(cage_outer)
                if cage_inner is not None:
                    cage_inner = self._local_studio_points_to_scene(cage_inner)
            self.scene_view.set_bake_process_visualization(
                cage_outer_vertices=cage_outer,
                cage_inner_vertices=cage_inner,
                cage_faces=payload.get("cage_faces"),
            )
        except Exception as exc:

            self._log(f"Live bake viewport update skipped: {exc}")

    def _on_bake_edges(self, payload: dict):
        """Draw the current and preceding nine baked faces as a live onion trail."""
        try:
            segments = payload.get("recent_edge_segments")
            if segments is not None:
                segments = self._local_studio_points_to_scene(
                    np.asarray(segments).reshape(-1, 3)
                ).reshape(-1, 2, 3)
            self.scene_view.set_bake_recent_edges(segments)
        except Exception as exc:
            self._log(f"Live bake edge overlay update skipped: {exc}")

    def _load_baked_glb_result(self, path: Path, asset: dict | None = None):
        """Display the already-validated baked GLB mesh, UVs and image in VTK."""
        if asset is None:
            asset = load_baked_glb_asset(path)
        self.scene_view.set_reconstruction_mesh(
            asset["vertices"], asset["faces"], asset.get("colors"),
            has_texture=bool(asset.get("has_texture")),
            uvs=asset.get("uvs"), texture_image=asset.get("texture_image"),
            diffuse_texture_image=asset.get("diffuse_texture_image"),
            orm_texture_image=asset.get("orm_texture_image"),
            pbr_factors=asset.get("pbr_factors"),
            occlusion_strength=asset.get("occlusion_strength", 1.0),
        )

        self.face_overlay_toggle.setChecked(False)
        output_stats = dict(asset.get("mesh_stats") or mesh_topology_summary(asset["vertices"], asset["faces"]))
        if "file_size_mb" not in output_stats:
            try:
                output_stats["file_size_mb"] = Path(path).stat().st_size / (1024.0 * 1024.0)
            except OSError:
                pass
        self._set_mesh_comparison_stage("output", output_stats)
        self._live_mesh_preview_loaded = True
        self.wireframe_toggle.setEnabled(True)
        self.scene_view.set_display_mode("both")

        self.wireframe_toggle.setChecked(True)
        self.scene_info.setText(
            f"BAKED GLB RESULT · {len(asset['vertices']):,} vertices / {len(asset['faces']):,} faces · "
            f"{'embedded UV texture' if asset.get('has_texture') else 'GLB material / vertex colors'}"
        )
        self._log(
            f"Final result inspection loaded from GLB: {path.name} · "
            f"{len(asset['vertices']):,} vertices / {len(asset['faces']):,} faces · "
            f"texture={'yes' if asset.get('has_texture') else 'no'}"
        )

    def _on_failed(self, error: str):
        self.scene_view.clear_processing_mesh()
        if str(error).startswith("PIPELINE_CANCELLED:"):
            self._set_task_paused_state(False)
            self.pause_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
            self.run_btn.setEnabled(True)
            self._set_source_file_controls_enabled(True)
            self.output_edit.setEnabled(True)
            self.source_up_axis.setEnabled(self._axis_control_allowed)
            self._set_mesh_merge_controls_enabled()
            self.header_status.setText("● RECONSTRUCTION CANCELLED")
            self.stage_label.setText("Reconstruction cancelled · no automatic bake was started")
            self._log(str(error).partition(":")[2].strip())
            return
        self._set_task_paused_state(False)
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        if self.started_at is not None:
            elapsed = int(time.perf_counter() - self.started_at)
            self.card_elapsed.set_value(f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        self.header_status.setText("● FAILED · REVIEW CONSOLE")
        self._header_status_state = "error"
        self._set_header_status_style("error")
        self.stage_label.setText("Failed · Check the console for details")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._set_source_file_controls_enabled(True)
        self.output_edit.setEnabled(True)
        self.source_up_axis.setEnabled(self._axis_control_allowed)
        self._set_mesh_merge_controls_enabled()
        self._log("Failed: " + error.replace("\n", "<br>"))
        QMessageBox.critical(self, "Reconstruction failed", error[-3500:])

    def _reset_mesh_comparison(self):
        self._mesh_comparison_stats = {"input": None, "reconstruction": None, "output": None}
        self._mesh_comparison_kind = "unknown"
        self._refresh_mesh_comparison()

    def _set_mesh_comparison_stage(self, stage: str, stats: dict | None, kind: str | None = None):
        if stage not in self._mesh_comparison_stats:
            return
        self._mesh_comparison_stats[stage] = dict(stats) if stats else None
        if stage == "input" and kind:
            self._mesh_comparison_kind = str(kind)
        self._refresh_mesh_comparison()

    @staticmethod
    def _mesh_delta_text(value, baseline, suffix=""):
        if value is None:
            return "—"
        numeric_value = float(value)
        numeric_baseline = float(baseline) if baseline is not None else None
        if suffix:
            text = f"{numeric_value:,.2f}{suffix}"
        elif numeric_value.is_integer():
            text = f"{int(numeric_value):,}"
        else:
            text = f"{numeric_value:,.4g}"
        if numeric_baseline in (None, 0.0):
            return text
        numeric_delta = numeric_value - numeric_baseline
        change = numeric_delta / abs(numeric_baseline) * 100.0
        if abs(change) < 0.05:
            return f"{text}  ·  0.0%"
        arrow = "↓" if change < 0 else "↑"
        if suffix:
            delta_text = f"{abs(numeric_delta):,.2f}{suffix}"
        elif numeric_value.is_integer() and numeric_baseline.is_integer():
            delta_text = f"{abs(int(round(numeric_delta))):,}"
        else:
            delta_text = f"{abs(numeric_delta):,.4g}"
        return f"{text}  ·  {arrow} {delta_text} ({abs(change):.1f}%)"

    @staticmethod
    def _mesh_delta_color(value, baseline, lower_is_better=True):
        if value is None or baseline is None:
            return "font-weight:650;"
        delta = float(value) - float(baseline)
        if abs(delta) < max(1e-9, abs(float(baseline)) * 0.0005):
            return "font-weight:650;"
        return "font-weight:750;"

    def _refresh_mesh_comparison(self):
        if not hasattr(self, "mesh_comparison_labels"):
            return
        source = self._mesh_comparison_stats.get("input") or {}
        reconstruction = self._mesh_comparison_stats.get("reconstruction") or {}
        output = self._mesh_comparison_stats.get("output") or {}
        stages = {"input": source, "reconstruction": reconstruction, "output": output}
        baseline_vertices = source.get("points") if self._mesh_comparison_kind == "points" else source.get("vertices")
        baseline_triangles = source.get("triangles")
        baseline_edges = source.get("unique_edges")
        baseline_boundary = source.get("boundary_edges")
        baseline_nonmanifold = source.get("nonmanifold_edges")
        baseline_components = source.get("components")
        baseline_area = source.get("surface_area")
        baseline_size = source.get("file_size_mb")
        for stage, stats in stages.items():
            is_input = stage == "input"
            if self._mesh_comparison_kind == "points" and is_input:
                vertex_value = stats.get("points")
                vertex_text = f"{int(vertex_value):,} pts" if vertex_value is not None else "—"
            else:
                vertex_value = stats.get("vertices")
                vertex_text = f"{int(vertex_value):,}" if vertex_value is not None else "—"
            if (
                not is_input and self._mesh_comparison_kind != "points"
                and vertex_value is not None and baseline_vertices not in (None, 0)
            ):
                vertex_text = self._mesh_delta_text(int(vertex_value), baseline_vertices)
                self.mesh_comparison_labels[("vertices", stage)].setStyleSheet(
                    "font-size:8pt;font-weight:650;" + self._mesh_delta_color(vertex_value, baseline_vertices)
                )
            else:
                self.mesh_comparison_labels[("vertices", stage)].setStyleSheet("font-size:8pt;font-weight:650;")
            self.mesh_comparison_labels[("vertices", stage)].setText(vertex_text)

            for key, base in (("triangles", baseline_triangles), ("unique_edges", baseline_edges)):
                value = stats.get(key)
                text = f"{int(value):,}" if value is not None else "—"
                if not is_input and value is not None and base not in (None, 0):
                    text = self._mesh_delta_text(int(value), base)
                    style = "font-size:8pt;font-weight:650;" + self._mesh_delta_color(value, base)
                else:
                    style = "font-size:8pt;font-weight:650;"
                self.mesh_comparison_labels[(key, stage)].setText(text)
                self.mesh_comparison_labels[(key, stage)].setStyleSheet(style)

            boundary = stats.get("boundary_edges")
            nonmanifold = stats.get("nonmanifold_edges")
            if boundary is None and nonmanifold is None:
                topology_text = "—"
            else:
                topology_text = f"{int(boundary or 0):,} / {int(nonmanifold or 0):,}"
                if not is_input and boundary is not None and nonmanifold is not None:
                    b_change = "=" if baseline_boundary is None else (
                        f"↓{baseline_boundary - boundary:,}" if boundary < baseline_boundary else
                        (f"↑{boundary - baseline_boundary:,}" if boundary > baseline_boundary else "=")
                    )
                    n_change = "=" if baseline_nonmanifold is None else (
                        f"↓{baseline_nonmanifold - nonmanifold:,}" if nonmanifold < baseline_nonmanifold else
                        (f"↑{nonmanifold - baseline_nonmanifold:,}" if nonmanifold > baseline_nonmanifold else "=")
                    )
                    topology_text += f"  ·  B {b_change} / NM {n_change}"
            self.mesh_comparison_labels[("topology", stage)].setText(topology_text)
            self.mesh_comparison_labels[("topology", stage)].setStyleSheet("font-size:8pt;font-weight:650;")

            components = stats.get("components")
            watertight = stats.get("watertight")
            if components is None:
                component_text = "—"
            else:
                component_text = f"{int(components):,} / {'CLOSED' if watertight else 'OPEN'}"
            self.mesh_comparison_labels[("components", stage)].setText(component_text)
            self.mesh_comparison_labels[("components", stage)].setStyleSheet("font-size:8pt;font-weight:650;")

            area = stats.get("surface_area")
            area_text = f"{float(area):,.4g}" if area is not None else "—"
            if not is_input and area is not None and baseline_area not in (None, 0):
                area_text = self._mesh_delta_text(float(area), baseline_area)
            self.mesh_comparison_labels[("surface_area", stage)].setText(area_text)
            self.mesh_comparison_labels[("surface_area", stage)].setStyleSheet("font-size:8pt;font-weight:650;")

            size = stats.get("file_size_mb")
            size_text = f"{float(size):,.2f} MB" if size is not None else "—"
            if not is_input and size is not None and baseline_size not in (None, 0):
                size_text = self._mesh_delta_text(float(size), baseline_size, " MB")
                style = "font-size:8pt;font-weight:650;" + self._mesh_delta_color(size, baseline_size)
            else:
                style = "font-size:8pt;font-weight:650;"
            self.mesh_comparison_labels[("file_size_mb", stage)].setText(size_text)
            self.mesh_comparison_labels[("file_size_mb", stage)].setStyleSheet(style)

        if not source:
            self.mesh_comparison_status.setText("Load a mesh to compare before and after geometry.")
            self.mesh_comparison_status.setStyleSheet("font-size:7pt;")
            return
        if not reconstruction:
            self.mesh_comparison_status.setText("INPUT READY · waiting for reconstruction")
            self.mesh_comparison_status.setStyleSheet("font-size:7pt;font-weight:700;")
            return
        if self._mesh_comparison_kind == "points" or baseline_triangles in (None, 0):
            self.mesh_comparison_status.setText("POINT-CLOUD / SEGMENTED INPUT · POLYGON DELTA N/A")
            self.mesh_comparison_status.setStyleSheet("font-size:7pt;font-weight:700;")
            return
        result_triangles = reconstruction.get("triangles")
        if result_triangles is None:
            self.mesh_comparison_status.setText("RECONSTRUCTION METRICS UNAVAILABLE")
            self.mesh_comparison_status.setStyleSheet("font-size:7pt;")
            return
        delta = (float(result_triangles) - float(baseline_triangles)) / float(baseline_triangles) * 100.0
        boundary_worse = (
            reconstruction.get("boundary_edges") is not None and baseline_boundary is not None
            and reconstruction["boundary_edges"] > baseline_boundary
        )
        nonmanifold_worse = (
            reconstruction.get("nonmanifold_edges") is not None and baseline_nonmanifold is not None
            and reconstruction["nonmanifold_edges"] > baseline_nonmanifold
        )
        components_worse = (
            reconstruction.get("components") is not None and baseline_components is not None
            and baseline_components > 0 and reconstruction["components"] > baseline_components
        )
        if delta < -0.05:
            if boundary_worse or nonmanifold_worse or components_worse:
                message = f"POLYGONS ↓ {abs(delta):.1f}% · TOPOLOGY WARNING"
                style = "font-size:7pt;font-weight:800;"
            else:
                message = f"POLYGONS ↓ {abs(delta):.1f}% · TOPOLOGY STABLE"
                style = "font-size:7pt;font-weight:800;"
        elif delta > 0.05:
            message = f"POLYGONS ↑ {delta:.1f}% · DENSER GEOMETRY"
            style = "font-size:7pt;font-weight:800;"
        else:
            message = "POLYGONS · NO MATERIAL CHANGE"
            style = "font-size:7pt;font-weight:800;"
        if output:
            message += "  ·  BAKED GLB READY"
        self.mesh_comparison_status.setText(message)
        self.mesh_comparison_status.setStyleSheet(style)

    def _update_result_metrics(self, m: dict):
        values = {
            "vertices": f"{m['vertices']:,}",
            "triangles": f"{m['triangles']:,}",
            "unique_edges": f"{m['unique_edges']:,}",
            "components": str(m["components"]),
            "boundary_edges": f"{m['boundary_edges']:,}",
            "nonmanifold_edges": f"{m['nonmanifold_edges']:,}",
            "watertight": "YES" if m["watertight"] else "NO",
            "winding_consistent": "YES" if m["winding_consistent"] else "NO",
            "surface_area": f"{m['surface_area']:.4g}",
            "volume": f"{m['volume']:.4g}" if m["volume"] is not None else "N/A",
            "plane_coverage": (
                "OFF" if m.get("planar_region_extraction_enabled") is False
                else f"{m['plane_coverage'] * 100:.1f}%" if m.get("plane_coverage") is not None
                else "pre-segmented"
            ),
            "file_size_mb": f"{m['file_size_mb']:.2f}",
            "reconstruction_method": str(m.get("reconstruction_method", "PolyFit")),
        }
        for key, value in values.items():
            self.metric_labels[key].setText(value)

    def _update_live_metrics(self):

        if hasattr(self, "diagnostics_panel") and not self.diagnostics_panel.isVisible():
            return
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram_bytes = self.process.memory_info().rss
            if self.worker is not None and self.worker.isRunning() and self.worker.pid:
                try:
                    ram_bytes += psutil.Process(self.worker.pid).memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            ram = ram_bytes / (1024 * 1024)
            self.card_cpu.set_value(f"{cpu:.0f}%")
            self.card_ram.set_value(f"{ram:.0f} MB")
        except Exception:
            pass
        if self.started_at is not None and self.worker is not None and self.worker.isRunning():
            elapsed = int(time.perf_counter() - self.started_at)
            self.card_elapsed.set_value(f"{elapsed // 60:02d}:{elapsed % 60:02d}")

    def _export_mesh(self):
        if self._export_worker is not None and self._export_worker.isRunning():
            return
        if self.last_baked_glb is None or not self.last_baked_glb.is_file():
            QMessageBox.warning(
                self, "Export unavailable",
                "The automatic baked GLB master is not ready yet. Complete reconstruction and GLB baking first.",
            )
            return

        available_pbr_maps = get_glb_baked_pbr_maps(self.last_baked_glb)
        options_dialog = ExportOptionsDialog(
            self, available_pbr_maps=available_pbr_maps
        )
        if options_dialog.exec() != QDialog.Accepted:
            return
        options = options_dialog.options()
        suffix = options["suffix"]
        filters = {
            ".glb": "GL Transmission Format Binary (*.glb)",
            ".gltf": "GL Transmission Format JSON (*.gltf)",
            ".fbx": "Autodesk FBX 7.4 (*.fbx)",
            ".obj": "Wavefront OBJ (*.obj)",
            ".ply": "Polygon File Format (*.ply)",
            ".stl": "STereoLithography (*.stl)",
            ".3mf": "3D Manufacturing Format (*.3mf)",
            ".off": "Object File Format (*.off)",
        }
        initial = str(default_export_path(self.last_baked_glb, suffix))
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export baked result", initial, filters[suffix]
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() != suffix:
            output_path = output_path.with_suffix(suffix)
        if output_path.resolve() == self.last_baked_glb.resolve():
            QMessageBox.warning(
                self, "Export destination is reserved",
                "The automatically baked GLB is the immutable master for result inspection and future exports. "
                "Choose another output filename.",
            )
            return

        self._auto_bake_active = False
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.percent_label.setText("0%")
        self.stage_label.setText(f"Export · Preparing {suffix.upper()} conversion from baked GLB")
        self.header_status.setText(f"● EXPORTING · {suffix.upper()}")
        self.export_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self._set_source_file_controls_enabled(False)
        self._log(
            f"Export started from baked GLB master · {output_path.name} · "
            f"axis={options['output_up_axis'].upper()} · alpha={options['alpha_mode']} "
            f"({options['alpha_cutoff']:.2f}) · shadow={options['shadow_mode']} · "
            f"specular={options['specular']:.2f} · roughness={options['roughness']:.2f} · "
            f"metallic={options['metallic']:.2f} · "
            f"PBR maps={', '.join(options['pbr_maps']) if options['pbr_maps'] else 'none'}"
        )
        self._export_worker = ExportWorker(
            output_path=str(output_path), suffix=suffix,
            baked_input_path=str(self.last_baked_glb),
            output_up_axis=options["output_up_axis"],
            specular=options["specular"], roughness=options["roughness"],
            metallic=options["metallic"],
            alpha_mode=options["alpha_mode"],
            alpha_cutoff=options["alpha_cutoff"],
            shadow_mode=options["shadow_mode"],
            pbr_maps=options["pbr_maps"],
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.completed.connect(self._on_export_completed)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.finished.connect(self._on_export_thread_finished)
        self._export_worker.start()

    def _on_export_progress(self, value: int, message: str):
        """Update the main pipeline bar; no separate closable progress dialog is used."""
        is_bake_stage = bool(self._auto_bake_active)
        task_label = (
            f"Pipeline 2/2 · Texture Baking · {message}"
            if is_bake_stage else f"Export · {message}"
        )
        self._set_task_stage_text(task_label)
        if int(value) < 0:

            self.progress_bar.setRange(0, 0)
            self.percent_label.setText("2/2" if is_bake_stage else "…")
            return

        self.progress_bar.setRange(0, 100)
        if is_bake_stage:
            overall = bake_pipeline_percent(value)
        else:
            overall = max(0, min(100, int(value)))
        self.progress_bar.setValue(overall)
        self.percent_label.setText(f"{overall}%")

    def _restore_export_controls(self):
        """Restore controls after the worker ends; progress remains in the main window."""
        self.progress_bar.setRange(0, 100)
        self.export_btn.setEnabled(True)
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(True)
        if hasattr(self, "pause_btn"):
            self._set_task_paused_state(False)
            self.pause_btn.setEnabled(False)
        if hasattr(self, "cancel_btn"):
            self.cancel_btn.setEnabled(False)
        if hasattr(self, "bake_compute_combo"):
            self.bake_compute_combo.setEnabled(True)
        if hasattr(self, "bake_projection_combo"):
            self.bake_projection_combo.setEnabled(True)
            self._on_bake_projection_changed()
        if hasattr(self, "bake_pbr_group"):
            self.bake_pbr_group.setEnabled(True)
        if hasattr(self, "gpu_rescan_btn"):
            scan_worker = getattr(self, "_gpu_scan_worker", None)
            export_worker = getattr(self, "_export_worker", None)
            scan_active = scan_worker is not None and scan_worker.isRunning()
            export_active = export_worker is not None and export_worker.isRunning()
            self.gpu_rescan_btn.setEnabled(not (scan_active or export_active))

    def _on_export_thread_finished(self):
        worker = self._export_worker
        self._export_worker = None
        if worker is not None:
            worker.deleteLater()
        if hasattr(self, "gpu_rescan_btn"):
            scan_worker = getattr(self, "_gpu_scan_worker", None)
            self.gpu_rescan_btn.setEnabled(not (scan_worker is not None and scan_worker.isRunning()))
        should_rescan = bool(
            getattr(self, "_gpu_rescan_after_export", False)
            or getattr(self, "_gpu_rescan_pending", False)
        )
        self._gpu_rescan_after_export = False
        if should_rescan and not getattr(self, "_closing_after_bake_cancel", False):
            self._gpu_rescan_pending = True
            self._log("Export completed. Automatically rescanning CUDA/HIP/SYCL/OpenCL and running the GPU kernel self-test.")
            QTimer.singleShot(0, self._start_gpu_scan)

    def _on_export_completed(self, result: dict):
        self._gpu_rescan_after_export = True
        self._restore_export_controls()
        path = Path(str(result.get("path", "")))
        if result.get("auto_bake"):
            self._auto_bake_active = False
            self.last_baked_glb = path
            self._set_output_generated_display(path)

            self.progress_bar.setRange(0, 0)
            self.percent_label.setText("VTK")
            self.stage_label.setText("Pipeline 2/2 · Result inspection · Installing validated GLB in VTK viewport")
            try:
                self._load_baked_glb_result(path, result.get("preview_asset"))
            except Exception as exc:
                self.last_baked_glb = None
                self._log(f"Baked GLB preview load failed: {exc}")
                self.header_status.setText("● GLB BAKE OK · PREVIEW FAILED")
                self._header_status_state = "warning"
                self._set_header_status_style("warning")
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(100)
                self.percent_label.setText("100%")
                self.stage_label.setText("Pipeline 2/2 · GLB created · Result preview failed")
                self.export_btn.setEnabled(False)
                self.run_btn.setEnabled(True)
                self._set_source_file_controls_enabled(True)
                self.output_edit.setEnabled(True)
                self.source_up_axis.setEnabled(self._axis_control_allowed)
                QMessageBox.warning(self, "GLB preview failed", str(exc))
                return
            bake_info = result.get("bake_info") or {}
            if bake_info:
                if bake_info.get("gpu_active"):
                    self._on_gpu_compute_status({
                        "state": "active", "vendor": bake_info.get("compute_vendor", "GPU"),
                        "device": bake_info.get("compute_device", "GPU"),
                        "backend": bake_info.get("compute_backend", "GPU"),
                        "precision": bake_info.get("compute_precision", ""),
                        "batch_count": bake_info.get("gpu_batch_count", 0),
                        "memory_bytes": bake_info.get("compute_memory_bytes", 0),
                        "cpu_reference_seconds": bake_info.get("cpu_reference_seconds", 0.0),
                        "gpu_benchmark_seconds": bake_info.get("gpu_benchmark_seconds", 0.0),
                        "speedup_ratio": bake_info.get("speedup_ratio", 0.0),
                        "speedup_gain_percent": bake_info.get("speedup_gain_percent", 0.0),
                        "time_saved_percent": bake_info.get("time_saved_percent", 0.0),
                        "gpu_compute_seconds_total": bake_info.get("gpu_compute_seconds_total", 0.0),
                        "gpu_intersection_tests_total": bake_info.get("gpu_intersection_tests_total", 0),
                        "gpu_closest_point_tests_total": bake_info.get("gpu_closest_point_tests_total", 0),
                        "gpu_workload": bake_info.get("gpu_workload", ""),
                        "gpu_tests_per_second": bake_info.get("gpu_tests_per_second", 0.0),
                        "benchmark_valid": bake_info.get("benchmark_valid", False),
                        "benchmark_detail": bake_info.get("benchmark_detail", ""),
                        "detail": "Bake completed with GPU compute.",
                    })
                elif bake_info.get("gpu_fallback_reason"):
                    self._on_gpu_compute_status({
                        "state": "cpu_fallback", "fallback_reason": bake_info.get("gpu_fallback_reason"),
                        "detail": "Bake completed on CPU.",
                    })
            self.header_status.setText("● COMPLETE · BAKED GLB READY")
            self._header_status_state = "ready"
            self._set_header_status_style("ready")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.percent_label.setText("100%")
            self.stage_label.setText("Pipeline complete · baked GLB loaded for inspection and export")
            self.export_btn.setEnabled(True)
            self.run_btn.setEnabled(True)
            self._set_source_file_controls_enabled(True)
            self.output_edit.setEnabled(True)
            self.source_up_axis.setEnabled(self._axis_control_allowed)
            if bake_info:
                uv_layout_label = (
                    "xatlas connected charts"
                    if bake_info.get("uv_layout_method") == "xatlas_connected_charts"
                    else "per-triangle fallback charts"
                )
                self._log(
                    f"Automatic GLB bake completed · {bake_info.get('resolution', 0)}² · "
                    f"{bake_info.get('triangles', 0):,} triangles · "
                    f"{bake_info.get('uv_chart_count', 'N/A')} {uv_layout_label} · "
                    f"{bake_info.get('target_face_components_before_uv_unwrap', 'N/A')} target topology components · "
                    f"{bake_info.get('file_bytes', 0) / (1024 * 1024):.2f} MB"
                )
                projection_label = {
                    "cage_raycast_both_sides": "cage ray projection",
                    "direct_nearest_surface": "direct nearest-surface bake · no cage",
                    "vertex_color_fallback": "transferred vertex-color fallback",
                }.get(str(bake_info.get("projection_method", "")), "unknown projection")
                self._log(
                    f"Bake projection · {projection_label} · "
                    f"original source preserved={bool(bake_info.get('source_mesh_preserved_for_bake'))}"
                )
                repair_keys = (
                    "bake_welded_duplicate_vertices",
                    "bake_removed_degenerate_triangles",
                    "bake_removed_duplicate_triangles",
                    "bake_winding_flipped_triangles",
                    "bake_nonmanifold_edges",
                )
                repair_values = [
                    f"{key.removeprefix('bake_').replace('_', ' ')}={bake_info[key]:,}"
                    for key in repair_keys if key in bake_info
                ]
                if repair_values:
                    self._log("Bake target topology check · " + " · ".join(repair_values))
                self._log(
                    f"Bake compute backend · {bake_info.get('compute_backend', 'CPU')} · "
                    f"{bake_info.get('compute_device', 'CPU')} · "
                    f"{bake_info.get('gpu_batch_count', 0)} GPU batches · "
                    f"workload={bake_info.get('gpu_workload', 'not-started')} · "
                    f"ray tests={bake_info.get('gpu_intersection_tests_total', 0):,} · "
                    f"closest-point tests={bake_info.get('gpu_closest_point_tests_total', 0):,} · "
                    f"{bake_info.get('compute_precision', 'float64')}"
                )
                if bake_info.get("gpu_fallback_reason"):
                    self._log(f"GPU/CPU fallback detail · {bake_info['gpu_fallback_reason']}")
            self._log("Reconstruction pipeline fully complete · final inspection and format export now use the baked GLB master")
            return

        export_info = result.get("export_info") or {}
        self._log(f"Export completed: {path}")
        if export_info:
            self._log(
                f"Export settings · format={export_info.get('format', '').upper()} · "
                f"axis={export_info.get('output_up_axis', '').upper()} · "
                f"alpha={export_info.get('alpha_mode', 'n/a')} ({export_info.get('alpha_cutoff', 0.5):.2f}) · "
                f"shadow={export_info.get('shadow_mode', 'n/a')} · "
                f"specular={export_info.get('specular', 0):.2f} · "
                f"roughness={export_info.get('roughness', 0):.2f} · "
                f"metallic={export_info.get('metallic', 0):.2f}"
            )
            details = f"Saved successfully.\n{path}\n\nFormat: {export_info.get('format', '').upper()}\nOutput up-axis: {export_info.get('output_up_axis', '').upper()}"
            if export_info.get("render_modes_preserved"):
                details += (
                    f"\nAlpha Blend Mode: {export_info.get('alpha_mode', 'n/a')}"
                    f"\nAlpha Cutoff: {export_info.get('alpha_cutoff', 0.5):.2f}"
                    f"\nShadow Mode: {export_info.get('shadow_mode', 'n/a')}"
                )
            elif export_info.get("render_mode_limitations"):
                details += "\n\nNote: " + export_info["render_mode_limitations"]
            if export_info.get("material_limitations"):
                details += "\n\nNote: " + export_info["material_limitations"]
        else:
            details = f"Saved successfully.\n{path}"
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.percent_label.setText("100%")
        self.stage_label.setText(f"Export complete · {path.suffix.upper().lstrip('.') or path.name}")
        self.header_status.setText("● EXPORT COMPLETE")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._set_source_file_controls_enabled(True)
        QMessageBox.information(self, "Export", details)

    def _on_export_failed(self, error: str):
        self.scene_view.clear_bake_texture_preview()
        if str(error).startswith("BAKE_CANCELLED:"):
            was_auto_bake = bool(self._auto_bake_active)
            self._auto_bake_active = False
            if was_auto_bake:
                self.last_baked_glb = None
                self._set_output_workspace_display()
            self._log("Automatic GLB bake cancelled safely. The reconstructed OBJ remains in the temporary workspace.")
            self._set_task_paused_state(False)
            self.pause_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
            if self._closing_after_bake_cancel:
                return
            self.bake_compute_combo.setEnabled(True)
            self.bake_projection_combo.setEnabled(True)
            self.bake_pbr_group.setEnabled(True)
            self._on_bake_projection_changed()
            self.stage_label.setText("Automatic bake cancelled · reconstruction OBJ retained in temporary workspace")
            self.header_status.setText("● RECONSTRUCTION READY · BAKE CANCELLED")
            self.run_btn.setEnabled(True)
            self._set_source_file_controls_enabled(True)
            self.output_edit.setEnabled(True)
            self.source_up_axis.setEnabled(self._axis_control_allowed)
            return
        was_auto_bake = bool(self._auto_bake_active)
        self._auto_bake_active = False
        if was_auto_bake:
            self.last_baked_glb = None
            self._set_output_workspace_display()
        self._restore_export_controls()
        self._log("Export failed: " + str(error).replace("\n", "<br>"))
        if was_auto_bake:
            self.header_status.setText("● RECONSTRUCTION OK · GLB BAKE FAILED")
            self._header_status_state = "warning"
            self._set_header_status_style("warning")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(RECONSTRUCTION_WEIGHT)
            self.percent_label.setText(f"{RECONSTRUCTION_WEIGHT}%")
            self.stage_label.setText("Pipeline 2/2 · Automatic GLB bake failed · See console")
            self.export_btn.setEnabled(False)
            self.run_btn.setEnabled(True)
            self._set_source_file_controls_enabled(True)
            self.output_edit.setEnabled(True)
            QMessageBox.critical(self, "Automatic GLB bake failed", str(error)[-6000:])
            return
        self.progress_bar.setRange(0, 100)
        self.stage_label.setText("Export failed · See console for details")
        self.header_status.setText("● EXPORT FAILED")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._set_source_file_controls_enabled(True)
        QMessageBox.critical(self, "Export failed", str(error)[-6000:])

    def _on_render_pass_changed(self, _index: int = -1):
        """Switch the embedded VTK renderer between the supported appearance modes."""
        value = str(self.render_pass_combo.currentData() or "")
        if not value or ":" not in value:
            return
        namespace, identifier = value.split(":", 1)
        if namespace == "vtk":
            self.scene_view.set_render_pass(identifier)
            label = self.render_pass_combo.currentText()
            self.scene_info.setText(f"VTK · {label}")
            self._log(f"VTK render mode: {label}")

    def _apply_studio_lighting_settings(self, *_args):
        """Persist Windows 3D Viewer-style light colors, strengths and direction."""
        controls = getattr(self, "studio_light_controls", None)
        if not controls or not hasattr(self, "scene_view"):
            return
        enabled = bool(self.studio_lights_enabled.isChecked())
        light_values = {}
        for key, widgets in controls.items():
            color = QColor(widgets["hex"].text().strip())
            if not color.isValid():
                color = QColor("#FFFFFF")
            light_values[key] = {
                "enabled": bool(widgets["enabled"].isChecked()),
                "color": (color.redF(), color.greenF(), color.blueF()),
                "intensity": int(widgets["intensity"].value()),
            }
            self.settings.setValue(f"viewport/studio_lighting/{key}/enabled", light_values[key]["enabled"])
            self.settings.setValue(f"viewport/studio_lighting/{key}/color", color.name(QColor.NameFormat.HexRgb).upper())
            self.settings.setValue(f"viewport/studio_lighting/{key}/intensity", light_values[key]["intensity"])
        azimuth = self.studio_rotation_widget.azimuth()
        self.scene_view.set_studio_lighting_enabled(enabled)
        self.scene_view.set_studio_lighting(
            key_azimuth=azimuth,
            key_color=light_values["key"]["color"],
            key_intensity=light_values["key"]["intensity"],
            key_enabled=light_values["key"]["enabled"],
            fill_color=light_values["fill"]["color"],
            fill_intensity=light_values["fill"]["intensity"],
            fill_enabled=light_values["fill"]["enabled"],
            back_color=light_values["back"]["color"],
            back_intensity=light_values["back"]["intensity"],
            back_enabled=light_values["back"]["enabled"],
            environment_color=light_values["environment"]["color"],
            environment_intensity=light_values["environment"]["intensity"],
            environment_enabled=light_values["environment"]["enabled"],
        )
        self.settings.setValue("viewport/studio_lighting/enabled", enabled)
        self.settings.setValue("viewport/studio_lighting/key_azimuth", azimuth)

    def _reset_studio_lighting_settings(self):
        """Restore the default three-light plus environment setup."""
        defaults = {"key": 100, "fill": 100, "back": 100, "environment": 40}
        self.studio_lights_enabled.setChecked(True)
        for key, widgets in self.studio_light_controls.items():
            widgets["enabled"].setChecked(True)
            widgets["hex"].setText("#FFFFFF")
            color = QColor("#FFFFFF")
            for name, spin_name, value in (
                ("hue", "hue_spin", 0),
                ("saturation", "saturation_spin", 0),
                ("value", "value_spin", 255),
            ):
                widget = widgets[name]
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
                spin = widgets[spin_name]
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            intensity = widgets["intensity"]
            intensity.blockSignals(True)
            intensity.setValue(defaults[key])
            intensity.blockSignals(False)
            intensity_spin = widgets.get("intensity_spin")
            if intensity_spin is not None:
                intensity_spin.blockSignals(True)
                intensity_spin.setValue(defaults[key])
                intensity_spin.blockSignals(False)
        self.studio_rotation_widget.setAzimuth(10.0)
        self._apply_studio_lighting_settings()

    def closeEvent(self, event):

        scan_worker = getattr(self, "_gpu_scan_worker", None)
        if scan_worker is not None and scan_worker.isRunning():
            scan_worker.wait()
        import_worker = getattr(self, "_asset_import_worker", None)
        if import_worker is not None and import_worker.isRunning():
            answer = QMessageBox.question(
                self, "Import in progress",
                "A 3D asset is still being parsed. Wait for the import to finish before closing Polyfitty Studio?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if answer == QMessageBox.No:
                event.ignore()
                return
            import_worker.wait()
            self._close_asset_import_progress()
        if self._auto_bake_active:

            self._closing_after_bake_cancel = True
            worker = self._export_worker
            if worker is not None and worker.isRunning():
                self._log("Application closing without format Export. Requesting safe cancellation of the internal GLB bake.")
                worker.request_cancel()
                worker.wait()
            self._auto_bake_active = False
            self._stop_gpu_telemetry()
            self._save_window_geometry()
            event.accept()
            return
        if self._export_worker is not None and self._export_worker.isRunning():
            answer = QMessageBox.question(
                self, "Export in progress",
                "A format export is still running. Wait for it to finish before closing Polyfitty Studio?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if answer == QMessageBox.No:
                event.ignore()
                return
            self._export_worker.wait()
        if self.worker is not None and self.worker.isRunning():
            answer = QMessageBox.question(
                self, "Work in progress",
                "The reconstruction engine is running in a separate process. Closing the app will safely stop the current job. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer == QMessageBox.No:
                event.ignore()
                return
            if not self.worker.stop_and_wait(5000):
                process = self.worker.process
                if process is not None and process.poll() is None:
                    self.worker.terminate_process_tree(timeout=5.0)
                self.worker.wait()
        self._stop_gpu_telemetry()
        self._save_window_geometry()
        event.accept()

def _set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PolyfittyStudio.Desktop")
    except Exception:
        pass

def main(app=None, startup_splash=None, startup_gpu_probe=None):
    startup_trace("app.main entry")
    _set_windows_app_user_model_id()
    app = app or QApplication(sys.argv)
    icon_path = APP_DIR / "assets" / "PolyfittyStudio.ico"
    app_icon = QIcon(str(icon_path)) if icon_path.is_file() else QIcon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    app.setOrganizationName("PolyfittyStudio")
    app.setApplicationName("Polyfitty Studio")
    app.setApplicationDisplayName("Polyfitty Studio")
    startup_trace("QApplication created")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 9))
    app.setStyleSheet(stylesheet_for("graphite"))
    if startup_splash is not None:
        if isinstance(startup_gpu_probe, dict):
            gpu_summary = str(startup_gpu_probe.get("startup_splash_summary") or "GPU status checked")
            startup_splash.update(gpu_summary + " · Preparing the 3D workspace...")
        else:
            startup_splash.update("Loading the viewport and GPU runtime adapters...")
    startup_trace("MainWindow construction starting")
    window = MainWindow(startup_gpu_probe=startup_gpu_probe)
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    startup_trace("MainWindow construction returned")
    if startup_splash is not None:
        startup_splash.update("Finalizing the Polyfitty Studio workspace...")
    if bool(getattr(window, "_startup_window_maximized", False)):
        window.showMaximized()
    else:
        window.showNormal()
    window.raise_()
    window.activateWindow()
    app.processEvents()
    window.settings.setValue("window/first_startup_maximize_done_v1", True)
    window.settings.setValue("window/maximized", bool(window.isMaximized()))
    window.settings.sync()
    if startup_splash is not None:
        startup_splash.finish(window)
    startup_trace(
        "MAIN_WINDOW_SHOW_RETURNED visible={0} minimized={1}".format(
            window.isVisible(), window.isMinimized()
        )
    )

    QTimer.singleShot(750, lambda: startup_trace(
        "Qt event loop active; main_window_visible={0}".format(window.isVisible())
    ))
    startup_trace("Entering QApplication.exec")
    return int(app.exec())

if __name__ == "__main__":
    raise SystemExit(main())
