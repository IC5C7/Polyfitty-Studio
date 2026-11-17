from __future__ import annotations

"""VTK rendering backend with a small PyQtGraph-compatible scene API.

The rest of Polyfitty Studio can keep using MeshData / GLMeshItem-style helpers,
while all geometry, color, depth, camera and viewport rendering is handled by
VTK. QVTKRenderWindowInteractor embeds the VTK render window directly in Qt.
"""

from dataclasses import dataclass
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from PySide6 import QtCore, QtGui, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

try:
    from .gl_invert import GLInvertPass
except ImportError:
    from gl_invert import GLInvertPass

try:
    from .startup_diagnostics import startup_trace
except ImportError:
    from startup_diagnostics import startup_trace

try:
    from .viewport_grid import radial_grid_fade
except ImportError:
    from viewport_grid import radial_grid_fade

_DEPTH_TEST = 0x0B71

try:
    from .vtk_geometry import (
        _color_rgba, _vtk_rgba, _vtk_points, _vtk_cells, set_actor_color, set_actor_texture,
        mesh_to_polydata, lines_to_polydata, texture_image_to_vtk_image, set_texture_srgb_mode,
        vertex_colors_for_render_pass, configure_studio_environment,
        apply_studio_environment_intensity, normalize_vertex_normals,
    )
except ImportError:
    from vtk_geometry import (
        _color_rgba, _vtk_rgba, _vtk_points, _vtk_cells, set_actor_color, set_actor_texture,
        mesh_to_polydata, lines_to_polydata, texture_image_to_vtk_image, set_texture_srgb_mode,
        vertex_colors_for_render_pass, configure_studio_environment,
        apply_studio_environment_intensity, normalize_vertex_normals,
    )

@dataclass
class MeshData:
    vertexes: np.ndarray | None = None
    faces: np.ndarray | None = None
    vertexColors: np.ndarray | None = None
    vertexUVs: np.ndarray | None = None
    vertexNormals: np.ndarray | None = None
    textureImage: object | None = None
    ormTextureImage: object | None = None

class _VTKItem:
    def __init__(self):
        self._scene_view: GLViewWidget | None = None
        self._overlay = False
        self._depth_value = 0
        self._invert_overlay = False

    @property
    def actor(self) -> vtk.vtkActor:
        raise NotImplementedError

    def setVisible(self, visible: bool):
        visible = bool(visible)
        changed = bool(self.actor.GetVisibility()) != visible
        self.actor.SetVisibility(visible)
        if self._scene_view is not None:
            if changed:
                self._scene_view._schedule_viewport_grid_refresh()
            self._scene_view.update()

    def isVisible(self) -> bool:
        return bool(self.actor.GetVisibility())

    def setInvertOverlay(self, enabled: bool):
        """Route this item through the dedicated hardware-invert renderer."""
        enabled = bool(enabled)
        if enabled == self._invert_overlay:
            return
        self._invert_overlay = enabled
        if self._scene_view is not None:
            self._scene_view._sync_item_renderer(self)

    def setDepthValue(self, value: float):
        self._depth_value = float(value)

    def setGLOptions(self, options):

        overlay = isinstance(options, dict) and options.get(_DEPTH_TEST, True) is False
        if overlay != self._overlay:
            self._overlay = overlay
            if self._scene_view is not None:
                self._scene_view._sync_item_renderer(self)

class GLMeshItem(_VTKItem):
    def __init__(
        self,
        meshdata: MeshData | None = None,
        smooth: bool = True,
        drawEdges: bool = False,
        drawFaces: bool = True,
        color=(1.0, 1.0, 1.0, 1.0),
        shader: str | None = None,
        glOptions=None,
        textureImage=None,
        ormTextureImage=None,
        **_kwargs,
    ):
        super().__init__()
        self.meshdata = meshdata or MeshData()
        if textureImage is not None:
            self.meshdata.textureImage = textureImage
        if ormTextureImage is not None:
            self.meshdata.ormTextureImage = ormTextureImage
        self._vtk_texture = None
        self._vtk_orm_texture = None
        self._occlusion_strength = 1.0
        self._pbr_enabled = False
        self.smooth = bool(smooth)
        self.draw_edges = bool(drawEdges)
        self.draw_faces = bool(drawFaces)
        self._color = _color_rgba(color)
        self._polydata = vtk.vtkPolyData()
        self._normals = vtk.vtkPolyDataNormals()
        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetInputConnection(self._normals.GetOutputPort())
        self._mapper.ScalarVisibilityOn()
        self._mapper.SetColorModeToDirectScalars()
        self._mapper.SetScalarModeToUsePointData()
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        prop = self._actor.GetProperty()
        prop.SetColor(*self._color[:3])
        prop.SetOpacity(self._color[3])
        prop.SetInterpolationToGouraud() if self.smooth else prop.SetInterpolationToFlat()
        prop.SetEdgeVisibility(self.draw_edges)
        prop.SetEdgeColor(0.055, 0.075, 0.10)
        prop.SetLineWidth(1.0)

        prop.SetBackfaceCulling(True)
        if not self.draw_faces:
            prop.SetRepresentationToWireframe()
        if glOptions is not None:
            self.setGLOptions(glOptions)
        self._rebuild()

    @property
    def actor(self):
        return self._actor

    def _rebuild(self):
        vertexes = self.meshdata.vertexes
        faces = self.meshdata.faces
        if vertexes is None or faces is None:
            self._polydata = vtk.vtkPolyData()
        else:
            vertex_colors = self.meshdata.vertexColors
            if vertex_colors is not None and self._pbr_enabled:
                vertex_colors = vertex_colors_for_render_pass(vertex_colors, self._pbr_enabled)
            self._polydata = mesh_to_polydata(
                vertexes, faces, vertex_colors, self.meshdata.vertexUVs,
                self.meshdata.vertexNormals,
            )
        self._apply_texture_image(self.meshdata.textureImage)
        self._apply_orm_texture_image(self.meshdata.ormTextureImage)
        source_normals = normalize_vertex_normals(
            self.meshdata.vertexNormals, len(vertexes)
        ) if vertexes is not None and self.meshdata.vertexNormals is not None else None
        if source_normals is not None:

            self._mapper.SetInputData(self._polydata)
        else:
            self._normals.SetInputData(self._polydata)
            self._normals.SetConsistency(True)
            self._normals.SetAutoOrientNormals(False)
            self._normals.SplittingOff()
            self._normals.SetComputePointNormals(self.smooth)
            self._normals.SetComputeCellNormals(not self.smooth)
            self._normals.Update()
            self._mapper.SetInputConnection(self._normals.GetOutputPort())
        has_texture = self._vtk_texture is not None
        self._mapper.SetScalarVisibility(
            self._polydata.GetPointData().GetScalars() is not None and not has_texture
        )
        self._mapper.Modified()
        if self._scene_view is not None:
            self._scene_view._schedule_viewport_grid_refresh()
            self._scene_view.update()

    def _apply_texture_image(self, image):
        """Attach a real UV image texture to the VTK actor when the GLB provides one."""
        set_actor_texture(self._actor, None)
        self._vtk_texture = None
        if image is None:
            set_actor_color(self._actor, self._color, preserve_texture_rgb=False)
            return
        try:
            vtk_image = texture_image_to_vtk_image(image)
            texture = vtk.vtkTexture()
            texture.SetInputData(vtk_image)

            texture.SetColorModeToDirectScalars()
            texture.SetPremultipliedAlpha(False)
            texture.InterpolateOn()
            texture.RepeatOff()
            self._vtk_texture = texture
            self._sync_texture_render_binding()
            set_actor_color(self._actor, self._color, preserve_texture_rgb=True)
        except Exception:
            self._vtk_texture = None
            set_actor_texture(self._actor, None)
            set_actor_color(self._actor, self._color, preserve_texture_rgb=False)

    def _apply_orm_texture_image(self, image):
        """Attach a linear glTF ORM (AO/Roughness/Metallic) texture for PBR."""
        prop = self._actor.GetProperty()
        try:
            prop.SetORMTexture(None)
        except Exception:
            pass
        self._vtk_orm_texture = None
        if image is None:
            self._sync_orm_render_binding()
            return
        try:
            vtk_image = texture_image_to_vtk_image(image)
            texture = vtk.vtkTexture()
            texture.SetInputData(vtk_image)
            texture.SetColorModeToDirectScalars()
            texture.SetPremultipliedAlpha(False)
            texture.InterpolateOn()
            texture.RepeatOff()

            set_texture_srgb_mode(texture, False)
            self._vtk_orm_texture = texture
            self._sync_orm_render_binding()
        except Exception:
            self._vtk_orm_texture = None
            self._sync_orm_render_binding()

    def _sync_orm_render_binding(self):
        """Bind ORM only for PBR; keep AO/R/M data out of the unlit Albedo pass."""
        prop = self._actor.GetProperty()
        texture = self._vtk_orm_texture if self._pbr_enabled else None
        try:
            prop.SetORMTexture(texture)
            prop.SetOcclusionStrength(float(np.clip(self._occlusion_strength, 0.0, 1.0)))
        except Exception:

            pass

    def setMeshData(self, vertexes=None, faces=None, vertexColors=None, vertexUVs=None, vertexNormals=None, textureImage=None, ormTextureImage=None, **_kwargs):
        if vertexes is not None:
            self.meshdata.vertexes = np.asarray(vertexes, dtype=np.float32)
        if faces is not None:
            self.meshdata.faces = np.asarray(faces, dtype=np.uint32)
        if vertexColors is not None:
            self.meshdata.vertexColors = np.asarray(vertexColors, dtype=np.float32)
        if vertexUVs is not None:
            self.meshdata.vertexUVs = np.asarray(vertexUVs, dtype=np.float32)
        if vertexNormals is not None:
            self.meshdata.vertexNormals = np.asarray(vertexNormals, dtype=np.float32)
        if textureImage is not None:
            self.meshdata.textureImage = textureImage
        if ormTextureImage is not None:
            self.meshdata.ormTextureImage = ormTextureImage
        self._rebuild()

    def _sync_texture_render_binding(self):
        """Keep VTK's legacy texture and PBR albedo slot consistent with mode."""
        texture = self._vtk_texture
        if texture is None:
            set_actor_texture(self._actor, None)
        elif self._pbr_enabled:
            set_texture_srgb_mode(texture, True)
            set_actor_texture(self._actor, texture, bind_albedo=True)
        else:
            set_actor_texture(self._actor, texture, bind_albedo=False)
            set_texture_srgb_mode(texture, False)
        self._sync_orm_render_binding()

    def _source_appearance_has_alpha(self) -> bool:
        """Return whether the active source image or vertex colors contain transparency."""
        image = self.meshdata.textureImage
        if image is not None:
            try:
                rgba = np.asarray(image.convert("RGBA") if hasattr(image, "convert") else image)
                if rgba.ndim == 3 and rgba.shape[2] >= 4 and rgba.size:
                    alpha = rgba[..., 3]
                    return bool(np.nanmin(alpha) < (0.999 if np.issubdtype(alpha.dtype, np.floating) else 255))
            except (TypeError, ValueError, AttributeError):
                pass
        colors = self.meshdata.vertexColors
        if colors is not None:
            try:
                values = np.asarray(colors, dtype=np.float64)
                if values.ndim == 2 and values.shape[1] >= 4 and values.size:
                    alpha = values[:, 3]
                    threshold = 0.999 if float(np.nanmax(alpha)) <= 1.5 else 254.5
                    return bool(np.nanmin(alpha) < threshold)
            except (TypeError, ValueError):
                pass
        return False

    def setAlphaBlendMode(self, mode: str):
        """Select the correct opaque/translucent VTK pass for source alpha."""
        normalized = str(mode or "NATIVE").strip().upper()
        if normalized in {"ALPHA_CLIP", "ALPHA_HASHED", "ALPHA_BLEND"}:
            translucent = True
        elif normalized == "OPAQUE":
            translucent = False
        else:

            translucent = self._source_appearance_has_alpha()
        if translucent:
            self._actor.ForceOpaqueOff()
            self._actor.ForceTranslucentOn()
        else:
            self._actor.ForceTranslucentOff()
            self._actor.ForceOpaqueOn()
        if self._scene_view is not None:
            self._scene_view.update()
        return normalized

    def setTextureImage(self, image):
        """Replace only the actor texture, retaining the current geometry/UV buffers."""
        self.meshdata.textureImage = image
        self._apply_texture_image(image)
        if self._scene_view is not None:
            self._scene_view.update()

    def setColor(self, color):
        self._color = _color_rgba(color)
        set_actor_color(
            self._actor, self._color, preserve_texture_rgb=self._vtk_texture is not None
        )
        if self._scene_view is not None:
            self._scene_view.update()

    def setPBR(
        self, enabled: bool, roughness: float = 1.0, metallic: float = 0.0,
        specular: float = 0.25, occlusion_strength: float = 1.0,
    ):
        pbr_enabled = bool(enabled)
        changed = pbr_enabled != self._pbr_enabled
        self._pbr_enabled = pbr_enabled
        try:
            strength = float(occlusion_strength)
        except (TypeError, ValueError):
            strength = 1.0
        self._occlusion_strength = float(np.clip(strength, 0.0, 1.0)) if np.isfinite(strength) else 1.0
        if changed:
            self._rebuild()
        elif self._vtk_texture is not None:
            self._sync_texture_render_binding()
        prop = self._actor.GetProperty()
        if enabled:
            prop.SetInterpolationToPBR()
            prop.SetSpecular(float(np.clip(specular, 0.0, 1.0)))
            prop.SetRoughness(float(np.clip(roughness, 0.0, 1.0)))
            prop.SetMetallic(float(np.clip(metallic, 0.0, 1.0)))
        else:
            prop.SetInterpolationToGouraud() if self.smooth else prop.SetInterpolationToFlat()
            prop.SetSpecular(0.0)
            prop.SetMetallic(0.0)
            prop.SetRoughness(1.0)
        self._sync_orm_render_binding()
        if self._scene_view is not None:
            self._scene_view.update()

    def setLighting(self, enabled: bool):
        """Enable/disable VTK's per-fragment lighting for this actor only."""
        prop = self._actor.GetProperty()
        if bool(enabled):
            prop.LightingOn()
        else:
            prop.LightingOff()
        if self._scene_view is not None:
            self._scene_view.update()

class GLLinePlotItem(_VTKItem):
    def __init__(
        self,
        pos: np.ndarray | None = None,
        color=(1.0, 1.0, 1.0, 1.0),
        antialias: bool = True,
        mode: str = "lines",
        glOptions=None,
        **_kwargs,
    ):
        super().__init__()
        self._positions = np.asarray(pos if pos is not None else [], dtype=np.float32).reshape(-1, 3)
        self._color = _color_rgba(color)
        self._mode = str(mode)
        self._polydata = vtk.vtkPolyData()
        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-1.0, -1.0)
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        self._actor.GetProperty().LightingOff()
        self._actor.GetProperty().SetLineWidth(1.0)
        self._apply_color()
        if glOptions is not None:
            self.setGLOptions(glOptions)
        self._rebuild()

    @property
    def actor(self):
        return self._actor

    def _apply_color(self):
        prop = self._actor.GetProperty()
        if self._invert_overlay:
            prop.SetColor(1.0, 1.0, 1.0)
            prop.SetOpacity(1.0)
        else:
            prop.SetColor(*self._color[:3])
            prop.SetOpacity(self._color[3])

    def setInvertOverlay(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self._invert_overlay:
            return
        super().setInvertOverlay(enabled)
        self._apply_color()

    def _rebuild(self):
        self._polydata = lines_to_polydata(self._positions)
        self._mapper.SetInputData(self._polydata)
        self._mapper.ScalarVisibilityOff()
        self._mapper.SetScalarModeToUseCellData()
        self._mapper.SetColorModeToDirectScalars()
        self._actor.GetProperty().SetLineWidth(1.0)
        if self._scene_view is not None:
            self._scene_view.update()

    def setData(self, pos=None, color=None, cellColors=None, **_kwargs):
        if pos is not None:
            self._positions = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
            self._rebuild()
        if color is not None:
            self._color = _color_rgba(color)
            self._apply_color()
        if cellColors is not None:
            self.setSegmentColors(cellColors)
        if self._scene_view is not None:
            self._scene_view.update()

    def setSegmentColors(self, colors):
        values = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if len(values) != self._polydata.GetNumberOfLines():
            raise ValueError("Line overlay segment colors must match the line-cell count.")
        scalars = _vtk_rgba(values)
        scalars.SetName("OverlayRGBA")
        self._polydata.GetCellData().SetScalars(scalars)
        self._mapper.SetScalarModeToUseCellData()
        self._mapper.SetColorModeToDirectScalars()
        self._mapper.ScalarVisibilityOn()
        self._mapper.Modified()
        if self._scene_view is not None:
            self._scene_view.update()

    def clearSegmentColors(self):
        self._polydata.GetCellData().SetScalars(None)
        self._mapper.ScalarVisibilityOff()
        self._mapper.Modified()
        self._apply_color()
        if self._scene_view is not None:
            self._scene_view.update()

class GLScatterPlotItem(_VTKItem):
    def __init__(self, pos=None, color=None, size: float = 2.0, pxMode: bool = True, **_kwargs):
        super().__init__()
        self._positions = np.asarray(pos if pos is not None else [], dtype=np.float32).reshape(-1, 3)
        self._colors = None if color is None else np.asarray(color, dtype=np.float32)
        self._size = float(size)
        self._polydata = vtk.vtkPolyData()
        self._vertex_filter = vtk.vtkVertexGlyphFilter()
        self._vertex_filter.SetInputData(self._polydata)
        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetInputConnection(self._vertex_filter.GetOutputPort())
        self._mapper.SetColorModeToDirectScalars()
        self._mapper.SetScalarModeToUsePointData()
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        self._actor.GetProperty().SetRepresentationToPoints()
        self._actor.GetProperty().SetPointSize(self._size)
        self._actor.GetProperty().LightingOff()
        self._rebuild()

    @property
    def actor(self):
        return self._actor

    def _rebuild(self):
        self._polydata = vtk.vtkPolyData()
        self._polydata.SetPoints(_vtk_points(self._positions))
        if len(self._positions):
            ids = np.arange(len(self._positions), dtype=np.int64).reshape(-1, 1)
            verts = vtk.vtkCellArray()
            verts.SetData(
                numpy_to_vtkIdTypeArray(np.arange(len(ids) + 1, dtype=np.int64), deep=True),
                numpy_to_vtkIdTypeArray(ids.reshape(-1), deep=True),
            )
            self._polydata.SetVerts(verts)
        if self._colors is not None and len(self._colors) == len(self._positions):
            self._polydata.GetPointData().SetScalars(_vtk_rgba(self._colors))
            self._mapper.ScalarVisibilityOn()
        else:
            self._mapper.ScalarVisibilityOff()
        self._vertex_filter.SetInputData(self._polydata)
        self._vertex_filter.Update()
        if self._scene_view is not None:
            self._scene_view.update()

    def setData(self, pos=None, color=None, size=None, **_kwargs):
        if pos is not None:
            self._positions = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
        if color is not None:
            self._colors = np.asarray(color, dtype=np.float32)
        if size is not None:
            self._size = float(size)
            self._actor.GetProperty().SetPointSize(self._size)
        self._rebuild()

class GLViewWidget(QVTKRenderWindowInteractor):
    """QVTK widget exposing the subset of GLViewWidget used by SceneView."""

    viewportClicked = QtCore.Signal(float, float)
    viewportShortcutTriggered = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.opts = {"distance": 8.0, "elevation": 22.0, "azimuth": 35.0}
        self._projection_mode = "perspective"
        self._orthographic_axis = "Z"
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetLayer(0)
        self._renderer.SetBackground(0.098, 0.110, 0.125)
        self._renderer.SetBackgroundAlpha(1.0)
        self._renderer.SetUseDepthPeeling(True)
        self._renderer.SetMaximumNumberOfPeels(80)

        self._renderer.SetOcclusionRatio(0.08)
        self._material_environment_texture = configure_studio_environment(self._renderer)
        environment_scalars = self._material_environment_texture.GetInput().GetPointData().GetScalars()
        self._material_environment_base_radiance = vtk_to_numpy(environment_scalars).copy()

        self._overlay_renderer = vtk.vtkRenderer()
        self._overlay_renderer.SetLayer(1)
        self._overlay_renderer.SetInteractive(False)
        self._overlay_renderer.SetBackground(0.0, 0.0, 0.0)
        self._overlay_renderer.SetBackgroundAlpha(0.0)
        self._overlay_renderer.SetPreserveColorBuffer(True)
        self._overlay_renderer.SetPreserveDepthBuffer(True)
        self._overlay_renderer.SetActiveCamera(self._renderer.GetActiveCamera())

        window = self.GetRenderWindow()
        window.SetAlphaBitPlanes(1)
        window.SetMultiSamples(4)
        self._invert_renderer = vtk.vtkRenderer()
        self._invert_renderer.SetLayer(2)
        self._invert_renderer.SetInteractive(False)
        self._invert_renderer.SetBackground(0.0, 0.0, 0.0)
        self._invert_renderer.SetBackgroundAlpha(0.0)
        self._invert_renderer.SetPreserveColorBuffer(True)
        self._invert_renderer.SetPreserveDepthBuffer(True)
        self._invert_renderer.SetActiveCamera(self._renderer.GetActiveCamera())

        window.SetNumberOfLayers(4)
        window.AddRenderer(self._renderer)
        window.AddRenderer(self._overlay_renderer)
        window.AddRenderer(self._invert_renderer)
        self._invert_pass = GLInvertPass()
        self._invert_renderer.AddObserver(vtk.vtkCommand.StartEvent, self._begin_invert_pass)
        self._invert_renderer.AddObserver(vtk.vtkCommand.EndEvent, self._end_invert_pass)
        self._light_kit = vtk.vtkLightKit()
        self._light_kit.SetKeyLightIntensity(0.95)
        self._light_kit.SetKeyToFillRatio(3.0)
        self._light_kit.SetKeyLightWarmth(0.58)
        self._light_kit.SetFillLightWarmth(0.42)
        self._light_kit.SetBackLightWarmth(0.62)
        self._light_kit.AddLightsToRenderer(self._renderer)
        self._studio_lighting_enabled = True
        self._studio_key_azimuth = 10.0
        self._studio_light_values = {
            "key": {"color": (1.0, 1.0, 1.0), "intensity": 100.0, "enabled": True},
            "fill": {"color": (1.0, 1.0, 1.0), "intensity": 100.0, "enabled": True},
            "back": {"color": (1.0, 1.0, 1.0), "intensity": 100.0, "enabled": True},
            "environment": {"color": (1.0, 1.0, 1.0), "intensity": 40.0, "enabled": True},
        }
        self._capture_studio_light_refs()
        self._apply_studio_light_values()
        self._items: list[_VTKItem] = []
        self._interactor_initialized = False
        self._interactor_init_pending = False
        self._interactor_init_failed = False
        self._orientation_marker_widget = None
        self._orientation_marker_axes = None
        self._navigation_mode = None
        self._last_navigation_pos = None
        self._left_press_pos = None
        self._left_orbit_active = False
        self._right_mouse_down = False
        self._fly_keys: set[str] = set()
        self._fly_keys_by_physical: dict[int, str] = {}
        self._fly_velocity = np.zeros(3, dtype=np.float64)
        self._fly_speed_multiplier = 1.0
        self._fly_shift_acceleration_per_second = 3.0
        self._fly_shift_release_easing_per_second = 5.0
        self._fly_last_step_time = None
        self._shift_down = False
        self._mouse_wrap_active = False
        self._mouse_wrap_anchor_global = None

        self._remote_input_session = self._detect_remote_input_session()

        self._mouse_warp_pending = False
        startup_trace(
            "Viewport mouse input mode: {0}".format(
                "remote-safe (cursor recentering disabled)"
                if self._remote_input_session else "local (cursor recentering enabled)"
            )
        )

        self._navigation_shortcuts = {
            "forward": "W", "backward": "S", "left": "A",
            "right": "D", "down": "Q", "up": "E",
        }
        self._viewport_shortcuts = {
            "focus_view": "F", "toggle_wireframe": "X",
            "toggle_connected_faces": "C", "toggle_grid": "G",
            "reimport": "R", "toggle_face_overlay": "Z",
        }

        self._viewport_grid_enabled = False
        self._viewport_grid_base_spacing = 1.0
        self._viewport_grid_polydata = vtk.vtkPolyData()
        self._viewport_grid_mapper = vtk.vtkPolyDataMapper()
        self._viewport_grid_mapper.SetInputData(self._viewport_grid_polydata)
        self._viewport_grid_mapper.SetScalarModeToUseCellData()
        self._viewport_grid_mapper.SetColorModeToDirectScalars()
        self._viewport_grid_mapper.ScalarVisibilityOn()
        self._viewport_grid_actor = vtk.vtkActor()
        self._viewport_grid_actor.SetMapper(self._viewport_grid_mapper)
        self._viewport_grid_actor.SetPickable(False)
        self._viewport_grid_actor.SetDragable(False)
        grid_property = self._viewport_grid_actor.GetProperty()
        grid_property.LightingOff()
        grid_property.SetLineWidth(1.0)
        grid_property.SetOpacity(1.0)
        self._viewport_grid_actor.SetVisibility(False)
        self._renderer.AddActor(self._viewport_grid_actor)
        self._viewport_grid_signature = None

        self._viewport_grid_refresh_timer = QtCore.QTimer(self)
        self._viewport_grid_refresh_timer.setSingleShot(True)
        self._viewport_grid_refresh_timer.setInterval(90)
        self._viewport_grid_refresh_timer.timeout.connect(
            self._refresh_viewport_grid_geometry_if_needed
        )
        self._fly_timer = QtCore.QTimer(self)
        self._fly_timer.setInterval(16)
        self._fly_timer.timeout.connect(self._step_keyboard_fly)

        self._camera_motion_timer = QtCore.QTimer(self)
        self._camera_motion_timer.setInterval(16)
        self._camera_motion_timer.timeout.connect(self._step_camera_motion)
        self._camera_motion_target_position = None
        self._camera_motion_target_focal = None
        self._camera_motion_target_view_up = None
        self._camera_motion_easing = 0.30

        self._look_timer = QtCore.QTimer(self)
        self._look_timer.setInterval(16)
        self._look_timer.timeout.connect(self._step_smooth_look)
        self._look_target_elevation: float | None = None
        self._look_target_azimuth: float | None = None
        self._look_sensitivity = 0.045
        self._look_easing = 0.28
        self._camera_focal_length_mm = 26.9915
        self._camera_clip_start = 0.1
        self._camera_clip_end = 10000.0
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self._navigation_help = vtk.vtkTextActor()
        self._navigation_help.SetInput("")
        self._navigation_help.SetPickable(False)
        self._navigation_help.SetVisibility(True)
        help_text_property = self._navigation_help.GetTextProperty()

        help_text_property.SetColor(1.0, 1.0, 1.0)
        help_text_property.SetOpacity(1.0)
        help_text_property.SetFontFamilyToArial()
        help_text_property.SetFontSize(15)
        help_text_property.SetBold(False)
        help_text_property.SetShadow(False)
        help_text_property.SetJustificationToCentered()
        help_text_property.SetVerticalJustificationToBottom()

        help_text_property.SetBackgroundOpacity(0.0)
        help_text_property.SetFrame(False)
        help_text_property.SetFrameWidth(0)

        self._invert_renderer.AddViewProp(self._navigation_help)
        self._refresh_navigation_help()
        self._position_navigation_help()
        self.set_camera_focal_length(self._camera_focal_length_mm)
        self.set_camera_clip_range(self._camera_clip_start, self._camera_clip_end)
        self.setCameraPosition(distance=8.0, elevation=22.0, azimuth=35.0)

    @staticmethod
    def _portable_key_sequence(value) -> str:
        """Normalize a user shortcut to Qt's portable key-sequence text."""
        try:
            sequence = value if isinstance(value, QtGui.QKeySequence) else QtGui.QKeySequence(str(value or ""))
            if sequence.count() != 1:
                return ""
            return sequence.toString(QtGui.QKeySequence.SequenceFormat.PortableText).strip()
        except Exception:
            return ""

    def set_navigation_shortcuts(self, shortcuts: dict | None):
        defaults = {
            "forward": "W", "backward": "S", "left": "A",
            "right": "D", "down": "Q", "up": "E",
        }
        values = dict(defaults)
        for action, value in (shortcuts or {}).items():
            if action in values:
                normalized = self._portable_key_sequence(value)
                if normalized:
                    values[action] = normalized
        self._navigation_shortcuts = values
        self._refresh_navigation_help()

    def set_viewport_shortcuts(self, shortcuts: dict | None):
        defaults = {
            "focus_view": "F", "toggle_wireframe": "X",
            "toggle_connected_faces": "C", "toggle_grid": "G",
            "reimport": "R", "toggle_face_overlay": "Z",
        }
        values = dict(defaults)
        for action, value in (shortcuts or {}).items():
            if action in values:
                normalized = self._portable_key_sequence(value)
                if normalized:
                    values[action] = normalized
        self._viewport_shortcuts = values
        self._refresh_navigation_help()

    @staticmethod
    def _event_key_sequence(event) -> str:
        try:
            combined = int(event.modifiers()) | int(event.key())
            return QtGui.QKeySequence(combined).toString(
                QtGui.QKeySequence.SequenceFormat.PortableText
            ).strip()
        except Exception:
            return ""

    def _navigation_action_for_event(self, event) -> str | None:
        """Resolve WASD/QE from the physical Qt key, independent of modifiers.

        Shift is deliberately not part of the movement chord. It is sampled as
        a separate speed modifier, so RMB+WASD/QE works whether Shift is pressed
        before, during, or after the mouse button is held.
        """
        key = int(event.key())
        key_map = QtCore.Qt.Key
        key_actions = {
            int(key_map.Key_W): "forward",
            int(key_map.Key_S): "backward",
            int(key_map.Key_A): "left",
            int(key_map.Key_D): "right",
            int(key_map.Key_Q): "down",
            int(key_map.Key_E): "up",
        }
        return key_actions.get(key)

    def _viewport_action_for_event(self, event) -> str | None:
        """Resolve app commands directly; never depend on QKeySequence modifiers."""
        key = int(event.key())
        key_map = QtCore.Qt.Key
        key_actions = {
            int(key_map.Key_F): "focus_view",
            int(key_map.Key_X): "toggle_wireframe",
            int(key_map.Key_C): "toggle_connected_faces",
            int(key_map.Key_G): "toggle_grid",
            int(key_map.Key_R): "reimport",
            int(key_map.Key_Z): "toggle_face_overlay",
        }
        return key_actions.get(key)

    def _refresh_navigation_help(self):
        if not hasattr(self, "_navigation_help"):
            return
        nav = self._navigation_shortcuts
        actions = self._viewport_shortcuts
        movement = "/".join(nav[key] for key in ("forward", "left", "backward", "right", "down", "up") if nav.get(key))
        action_labels = [
            f"{actions['focus_view']} Focus / Restore" if actions.get("focus_view") else "",
            f"{actions['toggle_wireframe']} Wireframe" if actions.get("toggle_wireframe") else "",
            f"{actions['toggle_connected_faces']} Connected Faces" if actions.get("toggle_connected_faces") else "",
            f"{actions['toggle_grid']} Grid" if actions.get("toggle_grid") else "",
            f"{actions['reimport']} Reimport" if actions.get("reimport") else "",
            f"{actions['toggle_face_overlay']} Face Overlay" if actions.get("toggle_face_overlay") else "",
        ]
        action_text = " · ".join(item for item in action_labels if item)
        self._navigation_help.SetInput(
            "LMB Select · ALT + LMB Orbit · RMB Camera Rotate / Fly · MMB Pan · WHEEL Zoom · SHIFT Ramp Speed"
            f"\nRMB + {movement} Move · {action_text}"
        )
        self._position_navigation_help()
        self.update()

    @staticmethod
    def _detect_remote_input_session() -> bool:
        """Detect Windows remote-control sessions where cursor warping is unsafe.

        Chrome Remote Desktop injects host-side mouse input through its Windows
        remoting processes. Pointer recentering from inside the application can
        therefore feed synthetic movement back into the remote control stream.
        RDP is detected through SM_REMOTESESSION; Chrome Remote Desktop is
        detected by its active host/desktop process names.

        POLYFITTY_MOUSE_INPUT_MODE=remote-safe|local is available for diagnosis
        and for remote tools whose process names are not exposed to the user.
        """
        override = os.environ.get("POLYFITTY_MOUSE_INPUT_MODE", "").strip().lower()
        if override in {"remote", "remote-safe", "safe"}:
            return True
        if override in {"local", "warp"}:
            return False
        if sys.platform != "win32":
            return False
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.GetSystemMetrics.argtypes = [ctypes.c_int]
            user32.GetSystemMetrics.restype = ctypes.c_int
            if user32.GetSystemMetrics(0x1000):
                return True

            class PROCESSENTRY32W(ctypes.Structure):
                _fields_ = [
                    ("dwSize", wintypes.DWORD),
                    ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
            kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
            entry_pointer = ctypes.POINTER(PROCESSENTRY32W)
            kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, entry_pointer]
            kernel32.Process32FirstW.restype = wintypes.BOOL
            kernel32.Process32NextW.argtypes = [wintypes.HANDLE, entry_pointer]
            kernel32.Process32NextW.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL

            snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
            invalid_handle = ctypes.c_void_p(-1).value
            if snapshot == invalid_handle:
                return False
            remote_processes = {
                "remoting_host.exe",
                "remoting_desktop.exe",
                "remoting_me2me_host.exe",
                "remoting_native_messaging_host.exe",
            }
            try:
                entry = PROCESSENTRY32W()
                entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
                if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                    while True:
                        if str(entry.szExeFile).lower() in remote_processes:
                            return True
                        if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                            break
            finally:
                kernel32.CloseHandle(snapshot)
        except Exception as exc:
            try:
                startup_trace(
                    "Remote input session detection unavailable: {0}: {1}".format(
                        type(exc).__name__, exc
                    )
                )
            except Exception:
                pass
        return False

    def _refresh_remote_input_session(self):
        """Refresh remote-session detection when a new navigation gesture begins."""
        detected = self._detect_remote_input_session()
        if detected == self._remote_input_session:
            return
        self._remote_input_session = detected
        self._mouse_wrap_active = False
        self._mouse_wrap_anchor_global = None
        self._mouse_warp_pending = False
        startup_trace(
            "Viewport mouse input mode changed to {0}".format(
                "remote-safe (cursor recentering disabled)"
                if detected else "local (cursor recentering enabled)"
            )
        )

    def _begin_mouse_wrap(self):
        """Center the pointer for local input; avoid warping in remote sessions."""
        if self._remote_input_session:
            self._mouse_wrap_active = False
            self._mouse_wrap_anchor_global = None
            self._mouse_warp_pending = False

            self.unsetCursor()
            return
        try:
            anchor = self.mapToGlobal(self.rect().center())
            self._mouse_wrap_anchor_global = QtCore.QPoint(anchor)
            self._mouse_wrap_active = True
            current = QtGui.QCursor.pos()
            self._mouse_warp_pending = current != self._mouse_wrap_anchor_global
            self.setCursor(QtCore.Qt.CursorShape.BlankCursor)
            if self._mouse_warp_pending:
                QtGui.QCursor.setPos(self._mouse_wrap_anchor_global)
        except Exception:
            self._mouse_wrap_active = False
            self._mouse_wrap_anchor_global = None
            self._mouse_warp_pending = False

    def _warp_mouse_to_anchor(self):
        if not self._mouse_wrap_active or self._mouse_wrap_anchor_global is None:
            return
        try:
            if QtGui.QCursor.pos() != self._mouse_wrap_anchor_global:

                self._mouse_warp_pending = True
                QtGui.QCursor.setPos(self._mouse_wrap_anchor_global)
        except Exception:
            self._mouse_warp_pending = False

    def _end_mouse_wrap(self):
        self._mouse_wrap_active = False
        self._mouse_wrap_anchor_global = None
        self._mouse_warp_pending = False
        self.unsetCursor()

    def _stop_pointer_navigation(self, *, cancel_camera_motion=False):
        """Stop drag navigation immediately if its initiating button is no longer held."""
        was_orbit = self._navigation_mode == "orbit" or self._left_orbit_active
        was_look = self._navigation_mode == "look" or self._right_mouse_down
        self._left_orbit_active = False
        self._right_mouse_down = False
        self._navigation_mode = None
        self._last_navigation_pos = None
        if was_look:
            self._cancel_smooth_look()
            self._fly_keys.clear()
            self._fly_keys_by_physical.clear()
            self._fly_velocity.fill(0.0)
            self._fly_speed_multiplier = 1.0
            self._fly_last_step_time = None
            self._fly_timer.stop()
        if cancel_camera_motion and was_orbit:
            self._cancel_camera_motion()
        self._end_mouse_wrap()

    def _capture_studio_light_refs(self):
        """Keep handles to LightKit's head, key, fill and paired back lights."""
        lights = self._renderer.GetLights()
        lights.InitTraversal()
        values = []
        for _ in range(lights.GetNumberOfItems()):
            light = lights.GetNextItem()
            if light is not None:
                values.append(light)
        self._studio_light_refs = {

            "camera_fill": values[0] if len(values) > 0 else None,
            "key": values[1] if len(values) > 1 else None,
            "fill": values[2] if len(values) > 2 else None,
            "back": values[3:5],
        }

    def _apply_studio_light_values(self):
        self._capture_studio_light_refs()
        baseline = {"key": 0.95, "fill": 0.3166666667, "back": 0.2714285714}
        camera_fill = self._studio_light_refs.get("camera_fill")
        if camera_fill is not None:
            camera_fill.SetColor(1.0, 1.0, 1.0)
            camera_fill.SetIntensity(0.3166666667)
            camera_fill.SetSwitch(1 if self._studio_lighting_enabled else 0)
        for key in ("key", "fill", "back"):
            value = self._studio_light_values[key]
            refs = self._studio_light_refs.get(key)
            refs = refs if isinstance(refs, list) else [refs]
            for light in refs:
                if light is None:
                    continue
                color = tuple(float(np.clip(channel, 0.0, 1.0)) for channel in value["color"])
                light.SetColor(*color)
                light.SetIntensity(float(np.clip(value["intensity"], 0.0, 100.0)) / 100.0 * baseline[key])
                light.SetSwitch(1 if value["enabled"] and self._studio_lighting_enabled else 0)
        environment = self._studio_light_values["environment"]
        apply_studio_environment_intensity(
            self._material_environment_texture,
            self._material_environment_base_radiance,
            environment["intensity"],
            environment["color"],
            enabled=bool(environment["enabled"] and self._studio_lighting_enabled),
        )

    def set_studio_lighting_enabled(self, enabled: bool):
        """Toggle all studio lights used by lit viewport modes."""
        enabled = bool(enabled)
        if enabled == self._studio_lighting_enabled:
            if enabled:
                self._apply_studio_light_values()
            return
        if enabled:
            self._light_kit.AddLightsToRenderer(self._renderer)
        else:
            self._light_kit.RemoveLightsFromRenderer(self._renderer)
        self._studio_lighting_enabled = enabled
        self._apply_studio_light_values()
        self.update()

    def set_studio_lighting(
        self,
        *,
        key_azimuth=None,
        key_color=None, key_intensity=None, key_enabled=None,
        fill_color=None, fill_intensity=None, fill_enabled=None,
        back_color=None, back_intensity=None, back_enabled=None,
        environment_color=None, environment_intensity=None, environment_enabled=None,
    ):
        """Apply the editable Windows 3D Viewer-style light controls."""
        if key_azimuth is not None:
            azimuth = float(key_azimuth) % 360.0
            delta = azimuth - float(self._light_kit.GetKeyLightAzimuth())
            self._light_kit.SetKeyLightAzimuth(azimuth)
            self._light_kit.SetFillLightAzimuth(float(self._light_kit.GetFillLightAzimuth()) + delta)
            self._light_kit.SetBackLightAzimuth(float(self._light_kit.GetBackLightAzimuth()) + delta)
            self._studio_key_azimuth = azimuth
            self._light_kit.Update()

        updates = {
            "key": (key_color, key_intensity, key_enabled),
            "fill": (fill_color, fill_intensity, fill_enabled),
            "back": (back_color, back_intensity, back_enabled),
            "environment": (environment_color, environment_intensity, environment_enabled),
        }
        for key, (color, intensity, enabled) in updates.items():
            current = self._studio_light_values[key]
            if color is not None:
                rgb = tuple(float(v) for v in color)
                if len(rgb) == 3 and np.isfinite(rgb).all():
                    current["color"] = tuple(float(np.clip(v, 0.0, 1.0)) for v in rgb)
            if intensity is not None:
                current["intensity"] = float(np.clip(intensity, 0.0, 100.0))
            if enabled is not None:
                current["enabled"] = bool(enabled)
        if self._studio_lighting_enabled:
            self._apply_studio_light_values()
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._interactor_initialized and not self._interactor_init_pending:

            self._interactor_init_pending = True
            startup_trace("VTK child showEvent; interactor initialization deferred")
            QtCore.QTimer.singleShot(180, self._initialize_interactor_after_show)
        self.update()

    def paintEvent(self, event):

        if not self._interactor_initialized:
            return
        super().paintEvent(event)

    def _initialize_interactor_after_show(self):
        self._interactor_init_pending = False
        if self._interactor_initialized or not self.isVisible():
            return
        startup_trace("Deferred VTK interactor initialization starting")
        try:
            self.Initialize()
            style = vtk.vtkInteractorStyleUser()
            self.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
            self._initialize_orientation_gizmo()
            self._interactor_initialized = True
            startup_trace("Deferred VTK interactor initialization completed")
            self.update()
        except Exception as exc:
            self._interactor_init_failed = True
            startup_trace(
                "Deferred VTK interactor initialization failed: {0}: {1}".format(
                    type(exc).__name__, exc
                )
            )

    def _initialize_orientation_gizmo(self):
        """Create a camera-synchronized XYZ axis widget in the viewport corner."""
        if self._orientation_marker_widget is not None:
            return
        axes = vtk.vtkAxesActor()
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.035)
        axes.SetConeRadius(0.18)
        axes.SetSphereRadius(0.10)
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")
        for caption, color in (
            (axes.GetXAxisCaptionActor2D(), (0.95, 0.30, 0.28)),
            (axes.GetYAxisCaptionActor2D(), (0.34, 0.86, 0.48)),
            (axes.GetZAxisCaptionActor2D(), (0.35, 0.62, 1.00)),
        ):
            prop = caption.GetCaptionTextProperty()
            prop.SetColor(*color)
            prop.SetFontSize(15)
            prop.BoldOn()
            prop.ShadowOff()

        marker = vtk.vtkOrientationMarkerWidget()
        marker.SetOrientationMarker(axes)
        marker.SetInteractor(self.GetRenderWindow().GetInteractor())
        marker.SetViewport(0.82, 0.70, 0.97, 0.88)
        marker.SetEnabled(1)
        marker.InteractiveOff()
        marker_renderer = marker.GetRenderer()
        if marker_renderer is not None:
            marker_renderer.SetLayer(3)
            marker_renderer.SetInteractive(False)
            marker_renderer.SetPreserveColorBuffer(True)
        try:
            marker.SetOutlineColor(0.58, 0.68, 0.80)
        except (AttributeError, TypeError):
            pass
        self._orientation_marker_axes = axes
        self._orientation_marker_widget = marker

    @staticmethod
    def _vector3(value):
        return np.asarray(value, dtype=np.float64).reshape(3)

    def _camera_state(self):
        camera = self._renderer.GetActiveCamera()
        position = self._vector3(camera.GetPosition())
        focal = self._vector3(camera.GetFocalPoint())
        offset = position - focal
        distance = max(float(np.linalg.norm(offset)), 1e-4)
        direction = offset / distance
        elevation = math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0))))
        azimuth = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
        return camera, position, focal, distance, elevation, azimuth

    def _camera_motion_state(self):
        """Return the pending target pose when an eased movement is active."""
        if (
            self._camera_motion_target_position is None
            or self._camera_motion_target_focal is None
        ):
            return self._camera_state()
        camera = self._renderer.GetActiveCamera()
        position = np.asarray(self._camera_motion_target_position, dtype=np.float64)
        focal = np.asarray(self._camera_motion_target_focal, dtype=np.float64)
        offset = position - focal
        distance = max(float(np.linalg.norm(offset)), 1e-4)
        direction = offset / distance
        elevation = math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0))))
        azimuth = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
        return camera, position, focal, distance, elevation, azimuth

    def _cancel_camera_motion(self):
        timer = getattr(self, "_camera_motion_timer", None)
        if timer is not None:
            timer.stop()
        self._camera_motion_target_position = None
        self._camera_motion_target_focal = None
        self._camera_motion_target_view_up = None

    def _queue_camera_motion(self, position, focal, view_up=None):
        """Ease the camera toward a world-space position and focal point."""
        position = np.asarray(position, dtype=np.float64).reshape(3)
        focal = np.asarray(focal, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(focal)):
            return False
        self._camera_motion_target_position = position.copy()
        self._camera_motion_target_focal = focal.copy()
        if view_up is not None:
            candidate_up = np.asarray(view_up, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(candidate_up)) and np.linalg.norm(candidate_up) > 1e-9:
                self._camera_motion_target_view_up = candidate_up / np.linalg.norm(candidate_up)
        elif self._camera_motion_target_view_up is None:
            camera = self._renderer.GetActiveCamera()
            current_up = np.asarray(camera.GetViewUp(), dtype=np.float64)
            self._camera_motion_target_view_up = current_up / max(float(np.linalg.norm(current_up)), 1e-9)
        if not self._camera_motion_timer.isActive():
            self._camera_motion_timer.start()
        return True

    def _step_camera_motion(self):
        """Apply a responsive ease-out transition for orbit, pan, dolly and focus."""
        target_position = self._camera_motion_target_position
        target_focal = self._camera_motion_target_focal
        if target_position is None or target_focal is None:
            self._camera_motion_timer.stop()
            return
        camera = self._renderer.GetActiveCamera()
        current_position = np.asarray(camera.GetPosition(), dtype=np.float64)
        current_focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        position_error = target_position - current_position
        focal_error = target_focal - current_focal
        target_distance = max(float(np.linalg.norm(target_position - target_focal)), 1.0)
        tolerance = max(target_distance * 1.0e-5, 1.0e-5)
        finished = max(
            float(np.linalg.norm(position_error)),
            float(np.linalg.norm(focal_error)),
        ) <= tolerance
        if finished:
            next_position = target_position
            next_focal = target_focal
        else:
            easing = float(np.clip(self._camera_motion_easing, 0.05, 0.8))
            next_position = current_position + position_error * easing
            next_focal = current_focal + focal_error * easing
        previous_distance = max(float(np.linalg.norm(current_position - current_focal)), 1e-5)
        next_distance = max(float(np.linalg.norm(next_position - next_focal)), 1e-5)
        camera.SetPosition(*next_position)
        camera.SetFocalPoint(*next_focal)
        if self._projection_mode == "orthographic" and abs(next_distance - previous_distance) > 1e-9:
            camera.SetParallelScale(float(np.clip(
                camera.GetParallelScale() * next_distance / previous_distance, 1e-5, 1e12
            )))
        if self._camera_motion_target_view_up is not None:
            camera.SetViewUp(*self._camera_motion_target_view_up)
            camera.OrthogonalizeViewUp()
        self._apply_camera_clip_range()
        _camera, _position, _focal, distance, elevation, azimuth = self._camera_state()
        self.opts.update(distance=distance, elevation=elevation, azimuth=azimuth)
        self.update()
        if finished:
            self._camera_motion_timer.stop()
            self._camera_motion_target_position = None
            self._camera_motion_target_focal = None
            self._camera_motion_target_view_up = None

    def _apply_camera_clip_range(self):
        camera = self._renderer.GetActiveCamera()
        camera.SetClippingRange(float(self._camera_clip_start), float(self._camera_clip_end))

    def set_camera_focal_length(self, focal_length_mm: float):
        """Set a 36 mm-wide sensor equivalent focal length on the viewport camera."""
        focal_length = float(focal_length_mm)
        if not math.isfinite(focal_length) or focal_length <= 0.0:
            raise ValueError("Camera focal length must be a finite positive number of millimetres.")
        self._camera_focal_length_mm = float(np.clip(focal_length, 1.0, 500.0))
        camera = self._renderer.GetActiveCamera()
        camera.SetUseHorizontalViewAngle(True)
        horizontal_fov = math.degrees(
            2.0 * math.atan(36.0 / (2.0 * self._camera_focal_length_mm))
        )
        camera.SetViewAngle(horizontal_fov)
        if self._projection_mode == "orthographic":
            _camera, _position, _focal, distance, _elevation, _azimuth = self._camera_state()
            camera.SetParallelScale(self._parallel_scale_for_distance(distance))
        self.update()
        return self._camera_focal_length_mm

    def set_camera_clip_range(self, clip_start: float, clip_end: float):
        """Set editable near/far clipping planes, expressed in metres."""
        near = float(clip_start)
        far = float(clip_end)
        if (
            not math.isfinite(near) or not math.isfinite(far)
            or near <= 0.0 or far <= near
        ):
            raise ValueError("Camera clip range requires 0 < Clip Start < Clip End.")
        self._camera_clip_start = near
        self._camera_clip_end = far
        self._apply_camera_clip_range()
        self.update()
        return self._camera_clip_start, self._camera_clip_end

    @property
    def projection_mode(self) -> str:
        return str(getattr(self, "_projection_mode", "perspective"))

    @property
    def orthographic_axis(self) -> str:
        return str(getattr(self, "_orthographic_axis", "Z"))

    def _parallel_scale_for_distance(self, distance: float) -> float:
        """Match the perspective camera's vertical field of view at distance."""
        camera = self._renderer.GetActiveCamera()
        width, height = self.GetRenderWindow().GetSize()
        if width <= 0 or height <= 0:
            width, height = max(int(self.width()), 1), max(int(self.height()), 1)
        aspect = max(float(width) / max(float(height), 1.0), 1e-4)
        horizontal_fov = math.radians(float(camera.GetViewAngle()))
        vertical_fov = 2.0 * math.atan(math.tan(horizontal_fov * 0.5) / aspect)
        return max(float(distance) * math.tan(vertical_fov * 0.5), 1e-5)

    def set_projection_mode(self, mode: str):
        """Switch between perspective and orthographic camera projection."""
        normalized = str(mode or "perspective").strip().lower()
        if normalized not in {"perspective", "orthographic"}:
            raise ValueError("Projection mode must be 'perspective' or 'orthographic'.")
        camera = self._renderer.GetActiveCamera()
        if normalized == self._projection_mode:
            return normalized
        self._cancel_camera_motion()
        self._cancel_smooth_look()
        _camera, _position, _focal, distance, _elevation, _azimuth = self._camera_state()
        if normalized == "orthographic":
            camera.SetParallelProjection(True)
            camera.SetParallelScale(self._parallel_scale_for_distance(distance))
            self._projection_mode = normalized
            self.set_orthographic_axis(self._orthographic_axis)
        else:
            camera.SetParallelProjection(False)
            self._projection_mode = normalized
        self._viewport_grid_signature = None
        if self._viewport_grid_enabled:
            self._update_viewport_grid_geometry()
        self.update()
        return normalized

    def set_orthographic_axis(self, axis: str):
        """Snap the orthographic camera to either direction of a world axis."""
        normalized = str(axis or "+Z").strip().upper().replace(" ", "")
        if normalized in {"X", "Y", "Z"}:
            normalized = "+" + normalized
        if normalized not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
            raise ValueError("Orthographic axis must be +X, -X, +Y, -Y, +Z or -Z.")
        self._orthographic_axis = normalized
        if self._projection_mode != "orthographic":
            self._viewport_grid_signature = None
            return normalized
        self._cancel_camera_motion()
        self._cancel_smooth_look()
        camera, position, focal, distance, _elevation, _azimuth = self._camera_state()
        preserve_parallel_scale = float(camera.GetParallelScale()) if self._projection_mode == "orthographic" else None
        directions = {
            "+X": (np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0))),
            "-X": (np.asarray((-1.0, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0))),
            "+Y": (np.asarray((0.0, 1.0, 0.0)), np.asarray((0.0, 0.0, 1.0))),
            "-Y": (np.asarray((0.0, -1.0, 0.0)), np.asarray((0.0, 0.0, 1.0))),
            "+Z": (np.asarray((0.0, 0.0, 1.0)), np.asarray((0.0, 1.0, 0.0))),
            "-Z": (np.asarray((0.0, 0.0, -1.0)), np.asarray((0.0, 1.0, 0.0))),
        }
        direction, view_up = directions[normalized]
        camera.SetFocalPoint(*focal)
        camera.SetPosition(*(focal + direction * max(distance, 1e-4)))
        camera.SetViewUp(*view_up)
        camera.OrthogonalizeViewUp()
        camera.SetParallelProjection(True)
        camera.SetParallelScale(
            preserve_parallel_scale
            if preserve_parallel_scale is not None
            else self._parallel_scale_for_distance(distance)
        )
        axis_letter = normalized[-1]
        axis_sign = -1 if normalized.startswith("-") else 1
        elevation = float(axis_sign * 90.0) if axis_letter == "Z" else 0.0
        azimuth = {
            "+X": 0.0, "-X": 180.0,
            "+Y": 90.0, "-Y": 270.0,
            "+Z": 0.0, "-Z": 0.0,
        }[normalized]
        self.opts.update(distance=distance, elevation=elevation, azimuth=azimuth)
        self._apply_camera_clip_range()
        self._viewport_grid_signature = None
        if self._viewport_grid_enabled:
            self._update_viewport_grid_geometry()
        self.update()
        return normalized

    def _set_camera_orbit(self, distance, elevation, azimuth, focal=None):
        _camera, _position, current_focal, _old_distance, _old_elevation, _old_azimuth = self._camera_motion_state()
        focus = current_focal if focal is None else self._vector3(focal)
        elevation = float(np.clip(elevation, -89.0, 89.0))
        azimuth = float(azimuth) % 360.0
        distance = float(np.clip(distance, 1e-4, 1e12))
        el = math.radians(elevation)
        az = math.radians(azimuth)
        direction = np.asarray((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
        position = focus + direction * distance
        self._queue_camera_motion(position, focus, (0.0, 0.0, 1.0))
        self.opts.update(distance=distance, elevation=elevation, azimuth=azimuth)

    def _set_camera_look(self, distance, elevation, azimuth):
        """Rotate the view direction in place, like Unity's RMB fly-look."""
        camera, position, _focal, _old_distance, _old_elevation, _old_azimuth = self._camera_state()
        elevation = float(np.clip(elevation, -89.0, 89.0))
        azimuth = float(azimuth) % 360.0
        distance = float(np.clip(distance, 1e-4, 1e12))
        el = math.radians(elevation)
        az = math.radians(azimuth)
        position_to_focal = np.asarray((
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ))
        focal = position - position_to_focal * distance
        camera.SetFocalPoint(*focal)
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.OrthogonalizeViewUp()
        if self._projection_mode == "orthographic":
            camera.SetParallelProjection(True)
            camera.SetParallelScale(self._parallel_scale_for_distance(distance))
        else:
            camera.SetParallelProjection(False)
        self._apply_camera_clip_range()
        self.opts.update(distance=distance, elevation=elevation, azimuth=azimuth)
        self.update()

    def _cancel_smooth_look(self):
        self._look_timer.stop()
        self._look_target_elevation = None
        self._look_target_azimuth = None

    def _queue_smooth_look(self, dx: float, dy: float):
        """Accumulate a low-sensitivity RMB look target for eased camera motion."""
        _camera, _position, _focal, _distance, elevation, azimuth = self._camera_state()
        if self._look_target_elevation is None or self._look_target_azimuth is None:
            self._look_target_elevation = elevation
            self._look_target_azimuth = azimuth
        turn_speed = self._look_sensitivity * (2.0 if self._shift_down else 1.0)
        self._look_target_elevation = float(np.clip(
            self._look_target_elevation + float(dy) * turn_speed, -89.0, 89.0
        ))
        self._look_target_azimuth = (
            float(self._look_target_azimuth) - float(dx) * turn_speed
        ) % 360.0
        if not self._look_timer.isActive():
            self._look_timer.start()

    def _step_smooth_look(self):
        """Ease RMB camera rotation only while the right mouse button remains held."""
        if not self._right_mouse_down:
            self._cancel_smooth_look()
            return
        if self._look_target_elevation is None or self._look_target_azimuth is None:
            self._look_timer.stop()
            return
        _camera, _position, _focal, distance, elevation, azimuth = self._camera_state()
        elevation_error = self._look_target_elevation - elevation
        azimuth_error = (
            (self._look_target_azimuth - azimuth + 180.0) % 360.0
        ) - 180.0
        finished = max(abs(elevation_error), abs(azimuth_error)) < 0.025
        if finished:
            next_elevation = self._look_target_elevation
            next_azimuth = self._look_target_azimuth
            self._look_timer.stop()
        else:
            next_elevation = elevation + elevation_error * self._look_easing
            next_azimuth = azimuth + azimuth_error * self._look_easing
        self._set_camera_look(distance, next_elevation, next_azimuth)
        if finished:
            self._look_target_elevation = None
            self._look_target_azimuth = None

    @property
    def viewport_grid_enabled(self) -> bool:
        return bool(self._viewport_grid_enabled)

    def set_viewport_grid_enabled(self, enabled: bool):
        """Toggle the perspective world-space ground grid below scene assets."""
        enabled = bool(enabled)
        self._viewport_grid_enabled = enabled
        self._viewport_grid_actor.SetVisibility(enabled)
        if enabled:
            self._viewport_grid_signature = None
            self._update_viewport_grid_geometry()
        else:
            self._viewport_grid_refresh_timer.stop()
        self.update()

    def toggle_viewport_grid(self) -> bool:
        enabled = not self._viewport_grid_enabled
        self.set_viewport_grid_enabled(enabled)
        return enabled

    def _viewport_grid_scene_bounds(self):
        """Return the union of visible, ordinary scene item bounds."""
        bounds = []
        for item in getattr(self, "_items", ()):

            if item is None or getattr(item, "_invert_overlay", False):
                continue
            try:
                actor = item.actor
                if actor is None or not actor.GetVisibility():
                    continue
                current = tuple(float(v) for v in actor.GetBounds())
                if len(current) == 6 and np.isfinite(current).all() and current[0] <= current[1] and current[2] <= current[3] and current[4] <= current[5]:
                    bounds.append(current)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
        if not bounds:
            return None
        values = np.asarray(bounds, dtype=np.float64)
        return (
            float(values[:, 0].min()), float(values[:, 1].max()),
            float(values[:, 2].min()), float(values[:, 3].max()),
            float(values[:, 4].min()), float(values[:, 5].max()),
        )

    @staticmethod
    def _nice_viewport_grid_step(target: float) -> float:
        """Choose a readable 1-2-5 world-space grid interval."""
        target = max(float(target), 0.05)
        exponent = math.floor(math.log10(target))
        scale = 10.0 ** exponent
        fraction = target / scale
        multiplier = 1.0 if fraction < 1.5 else 2.0 if fraction < 3.5 else 5.0 if fraction < 7.5 else 10.0
        return max(0.05, multiplier * scale)

    def _viewport_grid_state_signature(self):
        """Signature of only the state that changes world-space grid geometry."""
        try:
            bounds = self._viewport_grid_scene_bounds()
            bounds_values = tuple(round(float(v), 4) for v in bounds) if bounds is not None else ()
            background = tuple(round(float(v), 4) for v in self._renderer.GetBackground())
            projection = (self._projection_mode, self._orthographic_axis)
            return bounds_values + background + projection
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def _schedule_viewport_grid_refresh(self):
        """Coalesce scene changes; never rebuild the grid in the render loop."""
        if not getattr(self, "_viewport_grid_enabled", False):
            return
        timer = getattr(self, "_viewport_grid_refresh_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()

    def _refresh_viewport_grid_geometry_if_needed(self):
        """Refresh after scene changes only, not on camera movement or frames."""
        if not getattr(self, "_viewport_grid_enabled", False):
            return
        signature = self._viewport_grid_state_signature()
        if signature is not None and signature != self._viewport_grid_signature:
            self._update_viewport_grid_geometry()
            self.update()

    def _update_viewport_grid_geometry(self):
        """Build a ground grid in perspective or an axis-facing plane in orthographic mode."""
        try:
            bounds = self._viewport_grid_scene_bounds()
            if bounds is not None:
                xmin, xmax, ymin, ymax, zmin, zmax = bounds
                center = np.asarray(((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5), dtype=np.float64)
                extents = np.asarray((xmax - xmin, ymax - ymin, zmax - zmin), dtype=np.float64)
                scene_span = max(float(extents.max()), 1.0e-3)
            else:
                center = np.zeros(3, dtype=np.float64)
                extents = np.ones(3, dtype=np.float64)
                scene_span = 1.0

            orthographic = self._projection_mode == "orthographic"
            if orthographic:
                axis_label = str(self._orthographic_axis).upper()
                if axis_label in {"X", "Y", "Z"}:
                    axis_label = "+" + axis_label
                axis_index = {"X": 0, "Y": 1, "Z": 2}[axis_label[-1]]
                axis_sign = -1 if axis_label.startswith("-") else 1
                if axis_index == 0:
                    plane_axes = (1, 2)
                elif axis_index == 1:
                    plane_axes = (0, 2)
                else:
                    plane_axes = (0, 1)
                normal_min = (xmin, ymin, zmin)[axis_index] if bounds is not None else 0.0
                normal_max = (xmax, ymax, zmax)[axis_index] if bounds is not None else 0.0
                plane_gap = max(scene_span * 0.0015, 1.0e-4)

                plane_offset = (normal_min - plane_gap) if axis_sign > 0 else (normal_max + plane_gap)
                center_u = center[plane_axes[0]]
                center_v = center[plane_axes[1]]
                half_extent = min(max(max(float(extents[list(plane_axes)].max()) * 1.5, 12.0), 12.0), 5000.0)
                floor_z = plane_offset
                plane_label_axis = self._orthographic_axis
            else:
                plane_axes = (0, 1)
                center_u, center_v = float(center[0]), float(center[1])
                floor_z = (zmin - max(scene_span * 0.0015, 1.0e-4)) if bounds is not None else 0.0
                half_extent = min(max(scene_span * 2.5, 12.0), 5000.0)
                plane_label_axis = "Z"

            spacing = self._nice_viewport_grid_step(
                max(self._viewport_grid_base_spacing, half_extent / 24.0)
            )
            major_interval = spacing * 10.0
            u_min, u_max = center_u - half_extent, center_u + half_extent
            v_min, v_max = center_v - half_extent, center_v + half_extent
            u_lines = np.arange(math.ceil(u_min / spacing), math.floor(u_max / spacing) + 1, dtype=np.float64) * spacing
            v_lines = np.arange(math.ceil(v_min / spacing), math.floor(v_max / spacing) + 1, dtype=np.float64) * spacing

            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            colors = vtk.vtkUnsignedCharArray()
            colors.SetName("ViewportGroundGridRGB")
            colors.SetNumberOfComponents(3)
            background_rgb = np.clip(np.asarray(self._renderer.GetBackground(), dtype=np.float64) * 255.0, 0.0, 255.0)
            segment_count = 48
            outer_fade_radius = half_extent * 0.96
            u_segments = np.linspace(u_min, u_max, segment_count + 1)
            v_segments = np.linspace(v_min, v_max, segment_count + 1)
            minor_rgb = (139, 153, 168)
            major_rgb = (165, 178, 191)
            axis_rgb = (194, 112, 99)
            second_axis_rgb = (94, 151, 196)

            def world_point(u_value, v_value):
                value = [0.0, 0.0, 0.0]
                value[plane_axes[0]] = float(u_value)
                value[plane_axes[1]] = float(v_value)
                value[axis_index if orthographic else 2] = float(plane_offset if orthographic else floor_z)
                return value

            def add_segment(first, second, rgb, strength):
                radius = math.hypot((first[0] + second[0]) * 0.5 - center_u,
                                    (first[1] + second[1]) * 0.5 - center_v)
                fade = 1.0 if orthographic else radial_grid_fade(radius, outer_fade_radius)
                blend = float(np.clip(strength * fade, 0.0, 1.0))
                if blend <= 0.012:
                    return
                line_rgb = np.asarray(rgb, dtype=np.float64)
                display_rgb = np.clip(background_rgb + (line_rgb - background_rgb) * blend, 0.0, 255.0).astype(np.uint8)
                first_id = points.InsertNextPoint(*world_point(first[0], first[1]))
                second_id = points.InsertNextPoint(*world_point(second[0], second[1]))
                lines.InsertNextCell(2)
                lines.InsertCellPoint(first_id)
                lines.InsertCellPoint(second_id)
                colors.InsertNextTuple3(int(display_rgb[0]), int(display_rgb[1]), int(display_rgb[2]))

            def line_style(value, main_rgb):
                if abs(value) <= spacing * 1.0e-5:
                    return main_rgb, 0.78
                major_index = value / major_interval
                if abs(major_index - round(major_index)) <= 1.0e-5:
                    return major_rgb, 0.58
                return minor_rgb, 0.40

            for v in v_lines:
                rgb, strength = line_style(float(v), axis_rgb)
                for index in range(segment_count):
                    add_segment((u_segments[index], v), (u_segments[index + 1], v), rgb, strength)
            for u in u_lines:
                rgb, strength = line_style(float(u), second_axis_rgb)
                for index in range(segment_count):
                    add_segment((u, v_segments[index]), (u, v_segments[index + 1]), rgb, strength)

            self._viewport_grid_polydata.SetPoints(points)
            self._viewport_grid_polydata.SetLines(lines)
            self._viewport_grid_polydata.GetCellData().SetScalars(colors)
            self._viewport_grid_polydata.Modified()
            self._viewport_grid_mapper.SetInputData(self._viewport_grid_polydata)
            self._viewport_grid_mapper.Modified()
            self._viewport_grid_signature = self._viewport_grid_state_signature()
        except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
            return

    def _position_navigation_help(self):
        """Center the glyph-only VTK text overlay along the viewport bottom."""
        if not hasattr(self, "_navigation_help"):
            return
        try:
            render_window = self.GetRenderWindow()
            width, height = render_window.GetSize()
            if width <= 0 or height <= 0:
                width, height = int(self.width()), int(self.height())
            self._navigation_help.SetDisplayPosition(max(1, int(width) // 2), 16)
            self._navigation_help.Modified()
        except (AttributeError, RuntimeError, TypeError, ValueError):

            return

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_navigation_help()

    def mousePressEvent(self, event):
        button = event.button()
        if button == QtCore.Qt.MouseButton.LeftButton:
            if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                self._refresh_remote_input_session()
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:

                self._cancel_camera_motion()
                self._cancel_smooth_look()
                self._left_orbit_active = True
                self._navigation_mode = "orbit"
                self._last_navigation_pos = event.position()
                self._begin_mouse_wrap()
            else:
                self._left_press_pos = event.position()
            event.accept()
            return
        if button == QtCore.Qt.MouseButton.RightButton:
            self._refresh_remote_input_session()
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            self._cancel_camera_motion()
            self._cancel_smooth_look()
            _camera, _position, _focal, _distance, elevation, azimuth = self._camera_state()
            self._look_target_elevation = elevation
            self._look_target_azimuth = azimuth
            self._right_mouse_down = True
            self._navigation_mode = "look"
            self._last_navigation_pos = event.position()
            self._fly_keys.clear()
            self._fly_keys_by_physical.clear()
            self._fly_velocity.fill(0.0)
            self._fly_speed_multiplier = 1.0
            self._fly_last_step_time = time.monotonic()
            self._begin_mouse_wrap()
            event.accept()
            return
        if button == QtCore.Qt.MouseButton.MiddleButton:
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            self._cancel_camera_motion()
            self._cancel_smooth_look()
            self._navigation_mode = "pan"
            self._last_navigation_pos = event.position()
            self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        current = event.position()
        previous = self._last_navigation_pos
        if self._navigation_mode in {"orbit", "look"}:

            if self._mouse_wrap_active and self._mouse_warp_pending:
                warp_global_pos = event.globalPosition().toPoint()
                if warp_global_pos == self._mouse_wrap_anchor_global:
                    self._mouse_warp_pending = False
                    event.accept()
                    return
                self._mouse_warp_pending = False

            required_button = (
                QtCore.Qt.MouseButton.LeftButton
                if self._navigation_mode == "orbit"
                else QtCore.Qt.MouseButton.RightButton
            )
            held_buttons = event.buttons()

            if not self._remote_input_session:
                try:
                    held_buttons |= QtWidgets.QApplication.mouseButtons()
                except (AttributeError, RuntimeError):
                    pass
            if not (held_buttons & required_button):
                self._stop_pointer_navigation(cancel_camera_motion=True)
                event.accept()
                return

        if self._navigation_mode in {"orbit", "look"} and self._mouse_wrap_active:
            global_pos = event.globalPosition().toPoint()
            anchor = self._mouse_wrap_anchor_global
            if anchor is None:
                self._stop_pointer_navigation(cancel_camera_motion=True)
                event.accept()
                return
            dx = float(global_pos.x() - anchor.x())
            dy = float(global_pos.y() - anchor.y())

            dx = float(np.clip(dx, -120.0, 120.0))
            dy = float(np.clip(dy, -120.0, 120.0))
            if abs(dx) > 0.0 or abs(dy) > 0.0:
                if self._navigation_mode == "orbit" and self._left_orbit_active:
                    _camera, _position, _focal, distance, elevation, azimuth = self._camera_motion_state()
                    turn_speed = 0.09 * (2.0 if self._shift_down else 1.0)
                    self._set_camera_orbit(
                        distance, elevation + dy * turn_speed, azimuth - dx * turn_speed
                    )
                elif self._navigation_mode == "look" and self._right_mouse_down:
                    self._queue_smooth_look(dx, dy)
            self._warp_mouse_to_anchor()
            event.accept()
            return
        if previous is not None and self._navigation_mode in {"orbit", "look"}:
            dx = float(current.x() - previous.x())
            dy = float(current.y() - previous.y())
            if self._navigation_mode == "orbit" and self._left_orbit_active:
                _camera, _position, _focal, distance, elevation, azimuth = self._camera_motion_state()
                turn_speed = 0.09 * (2.0 if self._shift_down else 1.0)
                self._set_camera_orbit(
                    distance, elevation + dy * turn_speed, azimuth - dx * turn_speed
                )
            elif self._navigation_mode == "look" and self._right_mouse_down:
                self._queue_smooth_look(dx, dy)
            else:
                super().mouseMoveEvent(event)
                return
            self._last_navigation_pos = current
            event.accept()
            return
        if previous is not None and self._navigation_mode == "pan":
            dx = float(current.x() - previous.x())
            dy = float(current.y() - previous.y())
            self._pan_camera(dx, dy)
            self._last_navigation_pos = current
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        button = event.button()
        if button == QtCore.Qt.MouseButton.LeftButton:
            if self._left_orbit_active:
                self._stop_pointer_navigation(cancel_camera_motion=True)
            else:
                start = self._left_press_pos
                self._left_press_pos = None
                if start is not None:
                    end = event.position()
                    click_distance = abs(float(end.x() - start.x())) + abs(float(end.y() - start.y()))
                    if click_distance <= 5.0:
                        self.viewportClicked.emit(float(end.x()), float(end.y()))
            event.accept()
            return
        if button == QtCore.Qt.MouseButton.MiddleButton and self._navigation_mode == "pan":
            self._navigation_mode = None
            self._last_navigation_pos = None
            self.unsetCursor()
            event.accept()
            return
        if button == QtCore.Qt.MouseButton.RightButton:

            self._stop_pointer_navigation(cancel_camera_motion=True)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Unity-style dolly zoom toward/away from the orbit target."""
        delta = float(event.angleDelta().y()) / 120.0
        if abs(delta) < 1e-6:
            delta = float(event.pixelDelta().y()) / 120.0
        if abs(delta) < 1e-6:
            event.accept()
            return
        _camera, _position, _focal, distance, elevation, azimuth = self._camera_motion_state()
        zoom_speed = 0.05 * (2.0 if self._shift_down else 1.0)
        if self._projection_mode == "orthographic":
            camera = self._renderer.GetActiveCamera()
            camera.SetParallelScale(float(np.clip(
                camera.GetParallelScale() * math.exp(-delta * zoom_speed), 1e-5, 1e12
            )))
            self.update()
        else:
            new_distance = distance * math.exp(-delta * zoom_speed)
            self._set_camera_orbit(new_distance, elevation, azimuth)
        event.accept()

    def _pan_camera(self, dx: float, dy: float):
        camera, position, focal, distance, _elevation, _azimuth = self._camera_motion_state()
        forward = focal - position
        forward /= max(float(np.linalg.norm(forward)), 1e-9)
        view_up = self._vector3(camera.GetViewUp())
        right = np.cross(forward, view_up)
        right /= max(float(np.linalg.norm(right)), 1e-9)
        up = np.cross(right, forward)
        up /= max(float(np.linalg.norm(up)), 1e-9)
        reference_scale = camera.GetParallelScale() if self._projection_mode == "orthographic" else distance
        scale = max(float(reference_scale), 1e-3) * 0.0032 * (3.0 if self._shift_down else 1.0)
        offset = (-right * float(dx) + up * float(dy)) * scale
        self._queue_camera_motion(position + offset, focal + offset, view_up)

    def keyPressEvent(self, event):
        key = int(event.key())
        key_map = QtCore.Qt.Key
        shift_modifier = QtCore.Qt.KeyboardModifier.ShiftModifier

        self._shift_down = bool(event.modifiers() & shift_modifier)
        if key == int(key_map.Key_Shift):
            self._shift_down = True
            event.accept()
            return

        if self._right_mouse_down:
            action = self._navigation_action_for_event(event)
            if action is not None:
                if not event.isAutoRepeat():
                    self._fly_keys.add(action)
                    self._fly_keys_by_physical[key] = action
                if not self._fly_timer.isActive():
                    self._fly_timer.start()
                event.accept()
                return

        action = self._viewport_action_for_event(event)
        if action is not None:
            if not event.isAutoRepeat():
                self.viewportShortcutTriggered.emit(action)
            event.accept()
            return

        event.accept()

    def keyReleaseEvent(self, event):

        if event.isAutoRepeat():
            event.accept()
            return
        key = int(event.key())
        if key == int(QtCore.Qt.Key.Key_Shift):
            self._shift_down = False
            event.accept()
            return
        released_action = self._fly_keys_by_physical.pop(key, None)
        if released_action is not None:
            self._fly_keys.discard(released_action)

        self._shift_down = bool(
            QtWidgets.QApplication.keyboardModifiers()
            & QtCore.Qt.KeyboardModifier.ShiftModifier
        )

        event.accept()

    def _step_keyboard_fly(self):
        if not self._right_mouse_down:
            self._fly_timer.stop()
            self._fly_keys.clear()
            self._fly_keys_by_physical.clear()
            self._fly_velocity.fill(0.0)
            self._fly_speed_multiplier = 1.0
            self._fly_last_step_time = None
            return
        now = time.monotonic()
        last_step = self._fly_last_step_time
        dt = 0.016 if last_step is None else float(np.clip(now - last_step, 0.001, 0.05))
        self._fly_last_step_time = now

        try:
            self._shift_down = bool(
                QtWidgets.QApplication.keyboardModifiers()
                & QtCore.Qt.KeyboardModifier.ShiftModifier
            )
        except RuntimeError:
            pass
        camera, position, focal, distance, _elevation, _azimuth = self._camera_state()
        forward = focal - position
        forward /= max(float(np.linalg.norm(forward)), 1e-9)
        view_up = self._vector3(camera.GetViewUp())
        right = np.cross(forward, view_up)
        right /= max(float(np.linalg.norm(right)), 1e-9)
        up = np.cross(right, forward)
        up /= max(float(np.linalg.norm(up)), 1e-9)
        forward_sign = int("forward" in self._fly_keys) - int("backward" in self._fly_keys)
        right_sign = int("right" in self._fly_keys) - int("left" in self._fly_keys)
        up_sign = int("up" in self._fly_keys) - int("down" in self._fly_keys)
        direction = forward * forward_sign + right * right_sign + up * up_sign
        length = float(np.linalg.norm(direction))
        if length > 1e-9:
            direction /= length

        if self._shift_down and length > 1e-9:
            self._fly_speed_multiplier += self._fly_shift_acceleration_per_second * dt
        else:
            release_blend = 1.0 - math.exp(-self._fly_shift_release_easing_per_second * dt)
            self._fly_speed_multiplier += (1.0 - self._fly_speed_multiplier) * release_blend

        base_speed = max(float(distance), 1.0) * 0.0035
        speed = base_speed * self._fly_speed_multiplier
        target_velocity = direction * speed
        velocity_blend = 1.0 - math.exp(-16.0 * dt)
        self._fly_velocity += (target_velocity - self._fly_velocity) * velocity_blend
        velocity_length = float(np.linalg.norm(self._fly_velocity))
        if velocity_length <= max(base_speed * 0.012, 1e-7):
            self._fly_velocity.fill(0.0)
            if not self._fly_keys:
                self._fly_timer.stop()
                self._fly_speed_multiplier = 1.0
                self._fly_last_step_time = None
            return
        offset = self._fly_velocity
        camera.SetPosition(*(position + offset))
        camera.SetFocalPoint(*(focal + offset))
        self._apply_camera_clip_range()
        self.update()

    def pick_actor_at(self, x: float, y: float):
        """Return the polygon actor under a Qt-widget coordinate, or None."""
        try:
            window_width, window_height = self.GetRenderWindow().GetSize()
            scale_x = float(window_width) / max(float(self.width()), 1.0)
            scale_y = float(window_height) / max(float(self.height()), 1.0)
            pick_x = int(round(float(x) * scale_x))
            pick_y = int(window_height - 1 - round(float(y) * scale_y))
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.004)
            if not picker.Pick(pick_x, pick_y, 0.0, self._renderer):
                return None
            return picker.GetActor()
        except Exception:
            return None

    def focusOutEvent(self, event):

        self._stop_pointer_navigation(cancel_camera_motion=True)
        self._right_mouse_down = False
        self._navigation_mode = None
        self._last_navigation_pos = None
        self._left_press_pos = None
        self._left_orbit_active = False
        self._fly_keys.clear()
        self._fly_keys_by_physical.clear()
        self._fly_velocity.fill(0.0)
        self._fly_speed_multiplier = 1.0
        self._fly_last_step_time = None
        self._fly_timer.stop()
        self._cancel_camera_motion()
        self._end_mouse_wrap()
        self._look_timer.stop()
        self._look_target_elevation = None
        self._look_target_azimuth = None
        self._shift_down = False
        self.unsetCursor()
        super().focusOutEvent(event)

    def setBackgroundColor(self, color):
        rgba = _color_rgba(color)
        self._renderer.SetBackground(*rgba[:3])
        self._renderer.SetBackgroundAlpha(rgba[3])
        self._schedule_viewport_grid_refresh()
        self.update()

    def addItem(self, item: _VTKItem):
        if item is None:
            return
        if item not in self._items:
            self._items.append(item)
        item._scene_view = self
        self._sync_item_renderer(item)
        self._schedule_viewport_grid_refresh()

    def removeItem(self, item: _VTKItem):
        if item is None:
            return
        try:
            self._renderer.RemoveActor(item.actor)
            self._overlay_renderer.RemoveActor(item.actor)
            self._invert_renderer.RemoveActor(item.actor)
        finally:
            if item in self._items:
                self._items.remove(item)
            item._scene_view = None
        self._schedule_viewport_grid_refresh()
        self.update()

    def _sync_item_renderer(self, item: _VTKItem):
        self._renderer.RemoveActor(item.actor)
        self._overlay_renderer.RemoveActor(item.actor)
        self._invert_renderer.RemoveActor(item.actor)
        if item._invert_overlay:
            target = self._invert_renderer
        else:
            target = self._overlay_renderer if item._overlay else self._renderer
        target.AddActor(item.actor)
        self._schedule_viewport_grid_refresh()
        self.update()

    def setInvertOverlayDepthTest(self, enabled: bool):
        """Use the base scene depth buffer for invert lines when requested.

        True means hidden lines are depth-occluded by the mesh. False clears
        the invert renderer's depth buffer and intentionally places all invert
        lines in front (the explicit FACE OVERLAY behavior).
        """
        preserve_depth = bool(enabled)
        self._invert_renderer.SetPreserveDepthBuffer(preserve_depth)
        self.update()

    def _begin_invert_pass(self, renderer, _event):

        props = renderer.GetViewProps()
        if props is None or props.GetNumberOfItems() == 0:
            return
        self._invert_pass.begin(self.GetRenderWindow().GetState())

    def _end_invert_pass(self, _renderer, _event):
        self._invert_pass.end()

    def setCameraPosition(self, pos=None, distance=None, elevation=None, azimuth=None):
        self._cancel_camera_motion()
        camera = self._renderer.GetActiveCamera()
        if pos is None:
            focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        elif all(hasattr(pos, member) for member in ("x", "y", "z")):
            focal = np.asarray((pos.x(), pos.y(), pos.z()), dtype=np.float64)
        else:
            focal = np.asarray(pos, dtype=np.float64).reshape(3)
        distance = float(distance if distance is not None else self.opts.get("distance", 8.0))
        elevation = float(elevation if elevation is not None else self.opts.get("elevation", 22.0))
        azimuth = float(azimuth if azimuth is not None else self.opts.get("azimuth", 35.0))
        el = np.radians(elevation)
        az = np.radians(azimuth)
        direction = np.asarray((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)))
        camera.SetFocalPoint(*focal)
        camera.SetPosition(*(focal + direction * max(distance, 1e-3)))
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.OrthogonalizeViewUp()
        self._apply_camera_clip_range()
        self.opts.update(distance=distance, elevation=elevation, azimuth=azimuth)
        self.update()

gl = SimpleNamespace(
    GLViewWidget=GLViewWidget,
    MeshData=MeshData,
    GLMeshItem=GLMeshItem,
    GLLinePlotItem=GLLinePlotItem,
    GLScatterPlotItem=GLScatterPlotItem,
)
