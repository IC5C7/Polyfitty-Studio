from __future__ import annotations

import math

import numpy as np
import vtk
from PySide6 import QtCore, QtWidgets

try:
    from .wireframe import build_wireframe_positions, build_polygon_wireframe_positions, build_sharp_edge_positions, blend_surface_opacity, blend_wireframe_alpha, viewport_render_visibility, infront_applies, infront_gl_options, face_overlay_applies
    from .mesh_visualization import connected_component_mesh_data, apply_layer_alpha
    from .coordinate_frame import source_frame_oriented_bounds, source_frame_oriented_axis_edge_candidates, source_frame_opposite_camera_z_side_axis_edges, select_camera_visible_axis_edge, source_frame_camera_pose, transform_points
    from .vtk_backend import gl
except ImportError:
    from wireframe import build_wireframe_positions, build_polygon_wireframe_positions, build_sharp_edge_positions, blend_surface_opacity, blend_wireframe_alpha, viewport_render_visibility, infront_applies, infront_gl_options, face_overlay_applies
    from mesh_visualization import connected_component_mesh_data, apply_layer_alpha
    from coordinate_frame import source_frame_oriented_bounds, source_frame_oriented_axis_edge_candidates, source_frame_opposite_camera_z_side_axis_edges, select_camera_visible_axis_edge, source_frame_camera_pose, transform_points
    from vtk_backend import gl

class SceneView(gl.GLViewWidget):
    """Single interactive 3D scene with independently visible source/result layers."""

    selectionChanged = QtCore.Signal(object)
    emptyViewportClicked = QtCore.Signal()

    SOURCE_COLOR = (0.18, 0.76, 0.98, 0.92)
    RESULT_COLOR = (1.00, 0.86, 0.10, 0.58)

    DEFAULT_OVERLAY_COLOR = (1.0, 1.0, 1.0)
    SOLID_OVERLAY_COLOR = (1.0, 1.0, 1.0)
    MAX_WIREFRAME_EDGES = 600_000
    MAX_SHARP_EDGES = 250_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene_background_rgba = (25, 28, 32, 255)
        self.setBackgroundColor(self._scene_background_rgba)
        self.opts["distance"] = 8
        self.opts["elevation"] = 22
        self.opts["azimuth"] = 35
        self._source_items = []
        self._result_items = []
        self._source_surface_item = None
        self._source_scatter_item = None
        self._source_point_colors = None
        self._result_surface_item = None
        self._source_bounds = None
        self._source_vertices = None
        self._source_dimension_points = None
        self._source_faces = None
        self._source_vertex_colors = None
        self._source_vertex_normals = None
        self._source_has_texture = False
        self._source_material_parts = []
        self._result_has_texture = False
        self._result_vertex_colors = None
        self._result_uvs = None
        self._result_texture_image = None
        self._result_diffuse_texture_image = None
        self._result_orm_texture_image = None
        self._result_roughness = 0.72
        self._result_metallic = 0.0
        self._result_occlusion_strength = 1.0
        self._material_items = []
        self._material_layers = []
        self._material_mesh_buffers = []
        self._result_bounds = None
        self._show_source = True
        self._show_result = True
        self._source_opacity = 1.0
        self._result_opacity = 0.0
        self._surface_mix = 0.0
        self._result_vertices = None
        self._result_faces = None
        self._source_wire_item = None
        self._result_wire_item = None
        self._source_sharp_edge_item = None
        self._result_sharp_edge_item = None
        self._source_sharp_positions = None
        self._result_sharp_positions = None
        self._sharp_edges_enabled = False
        self._selected_mesh_layer: str | None = None
        self._face_islands_enabled = False
        self._face_island_items = []
        self._face_island_layers = []
        self._face_island_mesh_buffers = []
        self._processing_edge_item = None
        self._processing_highlight_item = None
        self._processing_label = ""
        self._bake_preview_item = None
        self._bake_preview_pbr_texture_image = None
        self._bake_preview_diffuse_texture_image = None
        self._bake_cage_outer_item = None
        self._bake_cage_inner_item = None
        self._bake_active_item = None
        self._bake_active_outline_item = None
        self._bake_recent_edge_item = None
        self._bake_recent_edge_visible = False
        self._source_wire_positions = None
        self._result_wire_positions = None
        self._wireframe_enabled = False
        self._wireframe_mix = 0.5
        self._wireframe_colors = {
            "source": self.DEFAULT_OVERLAY_COLOR,
            "result": self.DEFAULT_OVERLAY_COLOR,
        }
        self._sharp_edge_colors = {
            "source": self.DEFAULT_OVERLAY_COLOR,
            "result": self.DEFAULT_OVERLAY_COLOR,
        }
        self._overlay_color_mode = "invert"
        self._invert_visibility_states = {}

        self._infront_mode = "both"
        self._face_islands_opacity = 1.0

        self._face_overlay_enabled = False
        self._focus_restore_camera = None
        self._focused_mesh_layer: str | None = None
        self.setInvertOverlayDepthTest(True)
        self._infront_items = []
        self._render_pass = "diffuse_albedo"
        self._theme_name = "graphite"
        self._source_frame_transform = np.eye(4, dtype=np.float64)
        self._rotation_display_mode = "euler"
        self._item_info_enabled = False
        self._item_info_mode = "DIM"
        self._location_display_unit = "m"
        self._dimension_display_unit = "m"
        self.viewportClicked.connect(self._handle_viewport_click)
        self.view_reset_button = QtWidgets.QToolButton(self)
        self.view_reset_button.setObjectName("ViewportResetViewButton")
        self.view_reset_button.setText("↺")
        self.view_reset_button.setToolTip("Reset the viewport camera to frame the visible source and result meshes")
        self.view_reset_button.setFixedSize(28, 28)
        self.view_reset_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.view_reset_button.setStyleSheet(
            "QToolButton#ViewportResetViewButton {"
            "color:#e4edf7; background:rgba(27,34,43,150); "
            "border:1px solid rgba(190,205,222,85); border-radius:5px; "
            "font-size:18px; font-weight:600; padding:0px; }"
            "QToolButton#ViewportResetViewButton:hover {"
            "background:rgba(55,72,91,210); border-color:rgba(205,225,246,160); }"
        )
        self.view_reset_button.clicked.connect(self.reset_view)
        self._position_view_reset_button()
        self.view_reset_button.raise_()

        self.dimension_axis_huds = {}
        for axis in "XYZ":
            actor = vtk.vtkTextActor()
            actor.SetInput("")
            actor.SetPickable(False)
            actor.SetVisibility(False)
            text_property = actor.GetTextProperty()
            text_property.SetColor(1.0, 1.0, 1.0)
            text_property.SetOpacity(1.0)
            text_property.SetFontFamilyToArial()
            text_property.SetFontSize(16)
            text_property.SetBold(False)
            text_property.SetShadow(False)
            text_property.SetJustificationToCentered()
            text_property.SetVerticalJustificationToCentered()
            text_property.SetBackgroundOpacity(0.0)
            text_property.SetFrame(False)
            text_property.SetFrameWidth(0)
            self._invert_renderer.AddViewProp(actor)
            self.dimension_axis_huds[axis] = actor

        self.dimension_hud = self.dimension_axis_huds["X"]
        self._dimension_box_source = None
        self._dimension_box_result = None
        self._dimension_label_projected_positions = {}
        self._update_dimension_hud_color()
        self._renderer.AddObserver(
            vtk.vtkCommand.EndEvent, self._update_dimension_edge_label_positions
        )

        self.transform_hud = vtk.vtkTextActor()
        self.transform_hud.SetInput("")
        self.transform_hud.SetPickable(False)
        self.transform_hud.SetVisibility(False)
        transform_text_property = self.transform_hud.GetTextProperty()
        transform_text_property.SetColor(1.0, 1.0, 1.0)
        transform_text_property.SetOpacity(1.0)
        transform_text_property.SetFontFamilyToArial()
        transform_text_property.SetFontSize(14)
        transform_text_property.SetBold(False)
        transform_text_property.SetShadow(False)
        transform_text_property.SetJustificationToCentered()
        transform_text_property.SetVerticalJustificationToTop()
        transform_text_property.SetBackgroundOpacity(0.0)
        transform_text_property.SetFrame(False)
        transform_text_property.SetFrameWidth(0)
        self._invert_renderer.AddViewProp(self.transform_hud)

        self._empty_scene_prompt_visible = True
        self.empty_scene_prompt = vtk.vtkTextActor()
        self.empty_scene_prompt.SetInput(
            "DRAG & DROP A 3D FILE HERE\nOR CLICK TO BROWSE INPUT"
        )
        self.empty_scene_prompt.SetPickable(False)
        self.empty_scene_prompt.SetVisibility(True)
        empty_prompt_property = self.empty_scene_prompt.GetTextProperty()
        empty_prompt_property.SetColor(0.88, 0.92, 0.97)
        empty_prompt_property.SetOpacity(0.96)
        empty_prompt_property.SetFontFamilyToArial()
        empty_prompt_property.SetFontSize(19)
        empty_prompt_property.SetBold(True)
        empty_prompt_property.SetShadow(True)
        empty_prompt_property.SetJustificationToCentered()
        empty_prompt_property.SetVerticalJustificationToCentered()
        empty_prompt_property.SetBackgroundColor(0.025, 0.035, 0.05)
        empty_prompt_property.SetBackgroundOpacity(0.48)
        empty_prompt_property.SetFrame(False)
        empty_prompt_property.SetFrameWidth(0)
        self._overlay_renderer.AddViewProp(self.empty_scene_prompt)

        self._update_dimension_hud_color()
        self._position_empty_scene_prompt()
        self._position_dimension_hud()
        self._update_transform_hud()

    def _update_dimension_hud_color(self):
        """Match the bottom hint: white glyph masks rendered through GL_INVERT."""
        actors = getattr(self, "dimension_axis_huds", None) or {}
        transform_actor = getattr(self, "transform_hud", None)
        if transform_actor is not None:
            actors = {**actors, "__transform__": transform_actor}
        for actor in actors.values():
            try:
                text_property = actor.GetTextProperty()

                text_property.SetColor(1.0, 1.0, 1.0)
                text_property.SetOpacity(1.0)
                text_property.SetBackgroundOpacity(0.0)
                text_property.SetFrame(False)
                text_property.SetFrameWidth(0)
                actor.Modified()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue

    def _position_dimension_hud(self):
        """Refresh the selected item label anchored below the source object origin."""
        self._update_transform_hud_position()

    def _update_transform_hud_position(self):
        actor = getattr(self, "transform_hud", None)
        if actor is None or not self._item_info_enabled or self._item_info_mode not in {"LOC", "ROT", "SCALE"}:
            return
        renderer = getattr(self, "_renderer", None)
        if renderer is None:
            return
        try:
            origin = np.asarray(self._source_frame_transform, dtype=np.float64)[:3, 3]
            renderer.SetWorldPoint(float(origin[0]), float(origin[1]), float(origin[2]), 1.0)
            renderer.WorldToDisplay()
            projected = np.asarray(renderer.GetDisplayPoint(), dtype=np.float64)
            if projected.size < 3 or not np.isfinite(projected[:3]).all() or projected[2] < 0.0 or projected[2] > 1.0:
                actor.SetVisibility(False)
                return
            render_window = self.GetRenderWindow()
            width, height = render_window.GetSize()
            if width <= 0 or height <= 0:
                width, height = int(self.width()), int(self.height())
            x = int(np.clip(round(float(projected[0])), 24, max(24, int(width) - 24)))

            y = int(np.clip(round(float(projected[1])) - 18, 20, max(20, int(height) - 20)))
            actor.SetDisplayPosition(x, y)
            actor.SetVisibility(True)
            actor.Modified()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            actor.SetVisibility(False)

    def _rotation_components(self):
        matrix = np.asarray(self._source_frame_transform, dtype=np.float64)[:3, :3]
        try:
            left, _singular, right = np.linalg.svd(matrix)
            rotation = left @ right
            if np.linalg.det(rotation) < 0.0:
                left[:, -1] *= -1.0
                rotation = left @ right
            from scipy.spatial.transform import Rotation
            value = Rotation.from_matrix(rotation)
            if self._rotation_display_mode == "quaternion":
                return value.as_quat()
            return value.as_euler("xyz", degrees=True)
        except Exception:
            return np.zeros(4 if self._rotation_display_mode == "quaternion" else 3, dtype=np.float64)

    @staticmethod
    def _scale_components(matrix):
        """Return signed axis scales from the source object's affine basis."""
        value = np.asarray(matrix, dtype=np.float64).reshape(4, 4)[:3, :3]
        scales = np.linalg.norm(value, axis=0)
        if np.linalg.det(value) < 0.0:
            scales[0] *= -1.0
        if not np.isfinite(scales).all():
            return np.ones(3, dtype=np.float64)
        return scales

    def _transform_hud_text(self):
        matrix = np.asarray(self._source_frame_transform, dtype=np.float64).reshape(4, 4)
        if self._item_info_mode == "LOC":
            unit = str(getattr(self, "_location_display_unit", "m") or "m").strip().lower()
            if unit not in {"mm", "cm", "m", "km"}:
                unit = "m"
            multiplier, precision = self._dimension_unit_profile(unit)
            location = matrix[:3, 3] * multiplier
            return (
                f"X {location[0]:,.{precision}f}{unit}  "
                f"Y {location[1]:,.{precision}f}{unit}  "
                f"Z {location[2]:,.{precision}f}{unit}"
            )
        if self._item_info_mode == "SCALE":
            scale = self._scale_components(matrix)
            return f"X {scale[0]:,.4f}  Y {scale[1]:,.4f}  Z {scale[2]:,.4f}"
        if self._item_info_mode == "ROT":
            rotation = self._rotation_components()
            if self._rotation_display_mode == "quaternion":
                return (
                    f"W {rotation[3]:.4f}  X {rotation[0]:.4f}  "
                    f"Y {rotation[1]:.4f}  Z {rotation[2]:.4f}"
                )
            return f"X {rotation[0]:.2f}°  Y {rotation[1]:.2f}°  Z {rotation[2]:.2f}°"
        return ""

    def _update_transform_hud(self):
        actor = getattr(self, "transform_hud", None)
        if actor is None:
            return
        visible = self._item_info_enabled and self._item_info_mode in {"LOC", "ROT", "SCALE"}
        text = self._transform_hud_text() if visible else ""
        actor.SetInput(text)
        actor.SetVisibility(bool(text) and visible)
        actor.Modified()
        if visible:
            self._update_transform_hud_position()
        self.update()

    def set_item_info_enabled(self, enabled):
        self._item_info_enabled = bool(enabled)
        self._update_transform_hud()
        self._update_dimension_hud()

    def set_item_info_mode(self, mode):
        normalized = str(mode or "DIM").strip().upper()
        if normalized not in {"LOC", "ROT", "SCALE", "DIM"}:
            normalized = "DIM"
        self._item_info_mode = normalized
        self._update_transform_hud()
        self._update_dimension_hud()

    def set_location_display_unit(self, unit):
        normalized = str(unit or "m").strip().lower()
        if normalized not in {"mm", "cm", "m", "km"}:
            normalized = "m"
        self._location_display_unit = normalized
        self._update_transform_hud()

    def set_dimension_display_unit(self, unit):
        normalized = str(unit or "m").strip().lower()
        if normalized not in {"mm", "cm", "m", "km"}:
            normalized = "m"
        self._dimension_display_unit = normalized
        self._update_dimension_hud()

    def set_source_frame_transform(self, matrix):
        """Update the shared source origin/rotation and rebuild its oriented bounds."""
        try:
            value = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
            if not np.isfinite(value).all():
                raise ValueError("Source frame contains non-finite values.")
            if abs(float(np.linalg.det(value[:3, :3]))) <= 1e-12:
                raise ValueError("Source frame is not invertible.")
            self._source_frame_transform = value.copy()
        except (TypeError, ValueError):
            self._source_frame_transform = np.eye(4, dtype=np.float64)
        self._refresh_dimension_box_caches()
        self._update_transform_hud()
        self._update_dimension_hud()

    def set_rotation_display_mode(self, mode):
        normalized = str(mode or "euler").strip().lower()
        if normalized not in {"euler", "quaternion"}:
            normalized = "euler"
        self._rotation_display_mode = normalized
        self._update_transform_hud()

    def _toggle_rotation_display_mode(self):
        """Retain a small compatibility helper for internal callers and tests."""
        next_mode = "quaternion" if self._rotation_display_mode == "euler" else "euler"
        self.set_rotation_display_mode(next_mode)

    def _current_dimension_layer(self):
        """Return the visible source/result layer currently represented by labels."""
        source_visible = (
            self._show_source and self._source_bounds is not None
            and (1.0 - float(self._surface_mix)) > 0.001
        )
        result_visible = (
            self._show_result and self._result_bounds is not None
            and float(self._surface_mix) > 0.001
        )
        if source_visible and result_visible:
            if self._selected_mesh_layer == "source":
                return "source"
            if self._selected_mesh_layer == "result":
                return "result"
            return "result" if float(self._surface_mix) >= 0.5 else "source"
        if result_visible:
            return "result"
        if source_visible:
            return "source"
        return None

    def _current_dimension_bounds(self):
        """Return world AABB for compatibility with existing viewport framing."""
        layer = self._current_dimension_layer()
        return self._source_bounds if layer == "source" else self._result_bounds if layer == "result" else None

    def _refresh_dimension_box_caches(self):
        """Measure source and result boxes in the same source-object coordinate frame."""
        frame = self._source_frame_transform
        source_points = (
            self._source_vertices if self._source_vertices is not None
            else self._source_dimension_points
        )
        try:
            self._dimension_box_source = (
                source_frame_oriented_bounds(source_points, frame)
                if source_points is not None else None
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            self._dimension_box_source = None
        try:
            self._dimension_box_result = (
                source_frame_oriented_bounds(self._result_vertices, frame)
                if self._result_vertices is not None else None
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            self._dimension_box_result = None

    def _current_dimension_box(self):
        layer = self._current_dimension_layer()
        if layer == "source":
            return self._dimension_box_source
        if layer == "result":
            return self._dimension_box_result
        return None

    @staticmethod
    def _dimension_unit_profile(unit):
        """Return the display multiplier and precision for meter-based geometry."""
        profiles = {
            "mm": (1000.0, 2),
            "cm": (100.0, 2),
            "m": (1.0, 3),
            "km": (0.001, 6),
        }
        return profiles.get(str(unit or "m").strip().lower(), profiles["m"])

    @staticmethod
    def _dimension_hud_text(bounds, unit="m"):
        if bounds is None:
            return ""
        try:
            lower = np.asarray(bounds[0], dtype=np.float64).reshape(3)
            upper = np.asarray(bounds[1], dtype=np.float64).reshape(3)
            dimensions = upper - lower
            if not np.all(np.isfinite(dimensions)):
                return ""
            dimensions = np.maximum(dimensions, 0.0)
        except (TypeError, ValueError, IndexError):
            return ""
        normalized_unit = str(unit or "m").strip().lower()
        if normalized_unit not in {"mm", "cm", "m", "km"}:
            normalized_unit = "m"
        multiplier, precision = SceneView._dimension_unit_profile(normalized_unit)
        dimensions *= multiplier
        return (
            f"X {dimensions[0]:,.{precision}f} {normalized_unit}    "
            f"Y {dimensions[1]:,.{precision}f} {normalized_unit}    "
            f"Z {dimensions[2]:,.{precision}f} {normalized_unit}"
        )

    @staticmethod
    def _dimension_axis_texts(dimension_box, unit="m"):
        if not dimension_box:
            return {"X": "", "Y": "", "Z": ""}
        values = np.asarray(dimension_box.get("dimensions"), dtype=np.float64).reshape(3)
        if not np.isfinite(values).all():
            return {"X": "", "Y": "", "Z": ""}
        normalized_unit = str(unit or "m").strip().lower()
        if normalized_unit not in {"mm", "cm", "m", "km"}:
            normalized_unit = "m"
        multiplier, precision = SceneView._dimension_unit_profile(normalized_unit)
        values = np.maximum(values, 0.0) * multiplier
        return {
            axis: f"{axis} {float(values[index]):,.{precision}f} {normalized_unit}"
            for index, axis in enumerate("XYZ")
        }

    def _update_dimension_hud(self):
        actors = getattr(self, "dimension_axis_huds", None)
        if not actors:
            return
        show_dimensions = self._item_info_enabled and self._item_info_mode == "DIM"
        dimension_box = self._current_dimension_box() if show_dimensions else None
        texts = self._dimension_axis_texts(dimension_box, self._dimension_display_unit)
        for axis, actor in actors.items():
            text = texts.get(axis, "")
            if actor.GetInput() != text:
                actor.SetInput(text)
            actor.SetVisibility(bool(text) and show_dimensions)
            actor.Modified()
        self._update_dimension_edge_label_positions()
        self.update()

    def _update_dimension_edge_label_positions(self, *_args):
        """Place each DIM label on the nearest bbox edge not occluded by mesh."""
        self._update_transform_hud_position()
        actors = getattr(self, "dimension_axis_huds", None)
        if not self._item_info_enabled or self._item_info_mode != "DIM":
            for actor in (actors or {}).values():
                actor.SetVisibility(False)
            self._dimension_label_projected_positions = {}
            return
        dimension_box = self._current_dimension_box()
        renderer = getattr(self, "_renderer", None)
        if not actors or dimension_box is None or renderer is None:
            for actor in (actors or {}).values():
                actor.SetVisibility(False)
            self._dimension_label_projected_positions = {}
            return

        try:
            for actor in actors.values():
                actor.SetVisibility(False)
            edge_candidates = source_frame_oriented_axis_edge_candidates(
                dimension_box["local_bounds"], self._source_frame_transform
            )
            camera = renderer.GetActiveCamera()
            camera_position = np.asarray(camera.GetPosition(), dtype=np.float64).reshape(3)

            def project(point):
                renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
                renderer.WorldToDisplay()
                value = np.asarray(renderer.GetDisplayPoint(), dtype=np.float64)
                if value.size < 3 or not np.isfinite(value[:3]).all():
                    return None
                return value[:3]

            def depth_at(x, y):

                return renderer.GetZ(int(x), int(y))

            render_window = self.GetRenderWindow()
            width, height = render_window.GetSize()
            if width <= 0 or height <= 0:
                width, height = int(self.width()), int(self.height())
            if width <= 0 or height <= 0:
                self._dimension_label_projected_positions = {}
                return

            center = np.asarray(dimension_box["world_corners"], dtype=np.float64).reshape(-1, 3).mean(axis=0)
            center_display = project(center)
            positions = {}
            for axis_index, axis in enumerate("XYZ"):
                actor = actors[axis]
                axis_candidates = edge_candidates[axis_index]
                if axis in {"X", "Y"}:

                    axis_candidates = source_frame_opposite_camera_z_side_axis_edges(
                        axis_candidates,
                        camera_position,
                        self._source_frame_transform,
                        dimension_box["local_bounds"],
                        axis_index,
                    )
                    if len(axis_candidates) == 0:
                        continue
                    selected = select_camera_visible_axis_edge(
                        axis_candidates,
                        camera_position,
                        project,
                        depth_at,
                        (width, height),
                        sample_count=13,
                        depth_tolerance=0.0015,
                        selection_rank=0,
                    )
                    if selected is None:

                        selected = select_camera_visible_axis_edge(
                            axis_candidates,
                            camera_position,
                            project,
                            None,
                            (width, height),
                            sample_count=13,
                            selection_rank=0,
                        )
                else:
                    selected = select_camera_visible_axis_edge(
                        axis_candidates,
                        camera_position,
                        project,
                        depth_at,
                        (width, height),
                        sample_count=13,
                        depth_tolerance=0.0015,
                        selection_rank=1,
                    )
                if selected is None:
                    continue

                anchor_display = np.asarray(selected["display_point"], dtype=np.float64).reshape(3)
                if center_display is not None:
                    outward = anchor_display[:2] - center_display[:2]
                else:
                    outward = np.zeros(2, dtype=np.float64)
                outward_length = float(np.linalg.norm(outward))
                if outward_length > 1e-6:
                    outward /= outward_length
                else:
                    outward = {
                        "X": np.asarray((0.0, -1.0), dtype=np.float64),
                        "Y": np.asarray((-1.0, 0.0), dtype=np.float64),
                        "Z": np.asarray((0.0, 1.0), dtype=np.float64),
                    }[axis]

                label_position = anchor_display[:2] + outward * 17.0
                x = int(round(float(label_position[0])))
                y = int(round(float(label_position[1])))
                if x < 18 or x > int(width) - 18 or y < 18 or y > int(height) - 18:
                    continue
                actor.SetDisplayPosition(x, y)
                actor.SetVisibility(True)
                actor.Modified()
                positions[axis] = (x, y)
            self._dimension_label_projected_positions = positions
        except (AttributeError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
            for actor in actors.values():
                actor.SetVisibility(False)
            self._dimension_label_projected_positions = {}

    def _position_empty_scene_prompt(self):
        actor = getattr(self, "empty_scene_prompt", None)
        if actor is None:
            return
        try:
            render_window = self.GetRenderWindow()
            width, height = render_window.GetSize()
            if width <= 0 or height <= 0:
                width, height = int(self.width()), int(self.height())
            actor.SetDisplayPosition(max(1, int(width) // 2), max(1, int(height) // 2))
            actor.Modified()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return

    def set_empty_scene_prompt_visible(self, visible: bool):
        """Show import guidance only while the studio has no selected input."""
        self._empty_scene_prompt_visible = bool(visible)
        actor = getattr(self, "empty_scene_prompt", None)
        if actor is not None:
            actor.SetVisibility(self._empty_scene_prompt_visible)
            actor.Modified()
            if self._empty_scene_prompt_visible:
                self._position_empty_scene_prompt()
        self.update()

    def _position_view_reset_button(self):
        button = getattr(self, "view_reset_button", None)
        if button is None:
            return
        button.move(max(8, self.width() - button.width() - 12), 12)
        button.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_empty_scene_prompt()
        self._position_dimension_hud()
        self._position_view_reset_button()

    def reset_view(self, animate=None):
        """Frame the current source/result bounds, or restore the default camera."""
        self._focus_restore_camera = None
        self._focused_mesh_layer = None
        self._cancel_camera_motion()
        if hasattr(self, "_cancel_smooth_look"):
            self._cancel_smooth_look()
        self._fly_keys.clear()
        self._fly_keys_by_physical.clear()
        self._fly_velocity.fill(0.0)
        self._fly_speed_multiplier = 1.0
        self._fly_last_step_time = None
        self._fly_timer.stop()
        if self._source_bounds is not None or self._result_bounds is not None:
            if animate is None:
                self._fit_union_camera(animate=True)
            else:
                self._fit_union_camera(animate=bool(animate))
        else:
            elevation = math.radians(22.0)
            azimuth = math.radians(35.0)
            direction = np.asarray((
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ))
            self._queue_camera_motion(direction * 8.0, np.zeros(3), (0.0, 0.0, 1.0))

    def focus_view(self):
        """Toggle between framing the active mesh and the previous camera pose."""
        self._cancel_camera_motion()
        if hasattr(self, "_cancel_smooth_look"):
            self._cancel_smooth_look()
        self._fly_keys.clear()
        self._fly_keys_by_physical.clear()
        self._fly_velocity.fill(0.0)
        self._fly_speed_multiplier = 1.0
        self._fly_last_step_time = None
        self._fly_timer.stop()

        selected = self._selected_mesh_layer
        if selected not in {"source", "result"}:
            if self._source_bounds is not None:
                selected = "source"
            elif self._result_bounds is not None:
                selected = "result"
        if selected in {"source", "result"} and self._selected_mesh_layer != selected:

            self._select_mesh_layer(selected)

        if (
            self._focus_restore_camera is not None
            and selected == self._focused_mesh_layer
        ):
            restore = self._focus_restore_camera
            self._focus_restore_camera = None
            self._focused_mesh_layer = None
            self._queue_camera_motion(
                restore["position"], restore["focal"], restore["view_up"]
            )
            return True

        target_bounds = None
        if selected == "source":
            target_bounds = self._source_bounds
        elif selected == "result":
            target_bounds = self._result_bounds

        if target_bounds is None:
            visible_bounds = []
            source_alpha = (1.0 - float(self._surface_mix)) * float(self._source_opacity)
            result_alpha = float(self._surface_mix) * float(self._result_opacity)
            if self._show_source and source_alpha > 0.001 and self._source_bounds is not None:
                visible_bounds.append(self._source_bounds)
            if self._show_result and result_alpha > 0.001 and self._result_bounds is not None:
                visible_bounds.append(self._result_bounds)
            if not visible_bounds:

                visible_bounds = [b for b in (self._source_bounds, self._result_bounds) if b is not None]
            if not visible_bounds:
                return False
            lo = np.min(np.vstack([bounds[0] for bounds in visible_bounds]), axis=0)
            hi = np.max(np.vstack([bounds[1] for bounds in visible_bounds]), axis=0)
        else:
            lo = np.asarray(target_bounds[0], dtype=np.float64)
            hi = np.asarray(target_bounds[1], dtype=np.float64)

        center = (lo + hi) * 0.5
        half_diagonal = max(float(np.linalg.norm(hi - lo)) * 0.5, 1e-3)
        camera = self._renderer.GetActiveCamera()
        position = np.asarray(camera.GetPosition(), dtype=np.float64)
        focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        direction = position - focal
        current_distance = max(float(np.linalg.norm(direction)), 1e-4)
        direction /= current_distance

        render_width, render_height = self.GetRenderWindow().GetSize()
        if render_width <= 0 or render_height <= 0:
            render_width, render_height = max(1, self.width()), max(1, self.height())
        aspect = max(float(render_width) / max(float(render_height), 1.0), 1e-3)
        view_half_angle = math.radians(float(camera.GetViewAngle()) * 0.5)
        if camera.GetUseHorizontalViewAngle():
            horizontal_half_fov = view_half_angle
            vertical_half_fov = math.atan(math.tan(horizontal_half_fov) / aspect)
        else:
            vertical_half_fov = view_half_angle
            horizontal_half_fov = math.atan(math.tan(vertical_half_fov) * aspect)
        limiting_half_fov = max(min(vertical_half_fov, horizontal_half_fov), math.radians(1.0))
        distance = max(half_diagonal / math.sin(limiting_half_fov) * 1.18, 0.1)

        if self._focus_restore_camera is None:
            self._focus_restore_camera = {
                "position": position.copy(),
                "focal": focal.copy(),
                "view_up": np.asarray(camera.GetViewUp(), dtype=np.float64).copy(),
            }
        self._focused_mesh_layer = selected
        self._queue_camera_motion(center + direction * distance, center, camera.GetViewUp())
        return True

    @property
    def selected_mesh_layer(self) -> str | None:
        return self._selected_mesh_layer

    def _mesh_layer_for_actor(self, actor) -> str | None:
        if actor is None:
            return None

        candidates = [
            (self._source_surface_item, "source"),
            (self._result_surface_item, "result"),
            (self._bake_preview_item, "result"),
        ]
        candidates.extend(zip(self._material_items, self._material_layers))

        candidates.extend(zip(self._face_island_items, self._face_island_layers))
        for item, layer in candidates:
            if item is None:
                continue
            candidate = item.actor
            try:
                same_actor = (
                    actor == candidate
                    or actor.GetAddressAsString("") == candidate.GetAddressAsString("")
                )
            except Exception:
                same_actor = actor == candidate
            if not same_actor:
                continue

            layer = str(layer)
            if layer == "source":
                if not self._show_source or (1.0 - self._surface_mix) <= 0.001:
                    return None
            elif layer == "result":
                if not self._show_result or self._surface_mix <= 0.001:
                    return None
            return layer
        return None

    def _handle_viewport_click(self, x: float, y: float):
        if self._empty_scene_prompt_visible:
            self.emptyViewportClicked.emit()
            return
        actor = self.pick_actor_at(x, y)
        layer = self._mesh_layer_for_actor(actor)

        if layer is None and self._selected_mesh_layer is None:
            return
        if layer == self._selected_mesh_layer:
            layer = None

        self._selected_mesh_layer = layer
        self._sharp_edges_enabled = layer is not None
        self._apply_visibility()
        self.selectionChanged.emit(layer)
        self.update()

    def _clear_mesh_selection(self, layer: str):
        if self._selected_mesh_layer == layer:
            self._selected_mesh_layer = None
            self._sharp_edges_enabled = False
            self.selectionChanged.emit(None)

    def _select_mesh_layer(self, layer: str | None):
        layer = layer if layer in {"source", "result"} else None
        if layer == self._selected_mesh_layer:
            return
        self._selected_mesh_layer = layer
        self._sharp_edges_enabled = layer is not None
        self._apply_visibility()
        self.selectionChanged.emit(layer)

    def _remove_items(self, items):
        removed_ids = {id(item) for item in items}
        for item in items:
            self._invert_visibility_states.pop(id(item), None)
            try:
                self.removeItem(item)
            except Exception:
                pass
        items.clear()
        self._infront_items = [
            (item, kind) for item, kind in self._infront_items if id(item) not in removed_ids
        ]

    def clear_source(self):
        self._focus_restore_camera = None
        self._focused_mesh_layer = None
        self._clear_mesh_selection("source")
        self._remove_face_island_items("source")
        self._remove_items(self._source_items)
        self._remove_items(self._material_items)
        self._material_layers.clear()
        self._material_mesh_buffers.clear()
        self._source_vertices = None
        self._source_faces = None
        self._source_vertex_colors = None
        self._source_vertex_normals = None
        self._source_has_texture = False
        self._source_material_parts = []
        self._source_surface_item = None
        self._source_scatter_item = None
        self._source_point_colors = None
        self._source_wire_item = None
        self._source_wire_positions = None
        self._source_sharp_edge_item = None
        self._source_sharp_positions = None
        self._source_bounds = None
        self._source_dimension_points = None
        self._dimension_box_source = None
        self._update_dimension_hud()
        self._rebuild_material_items()
        self._fit_union_camera()

    def clear_result(self):
        if self._focused_mesh_layer == "result":
            self._focus_restore_camera = None
            self._focused_mesh_layer = None
        self._clear_mesh_selection("result")
        self.clear_bake_texture_preview()
        self._remove_face_island_items("result")
        self._remove_items(self._result_items)
        self._remove_items(self._material_items)
        self._material_layers.clear()
        self._material_mesh_buffers.clear()
        self._result_surface_item = None
        self._result_bounds = None
        self._dimension_box_result = None
        self._result_vertices = None
        self._result_faces = None
        self._result_vertex_colors = None
        self._result_uvs = None
        self._result_texture_image = None
        self._result_diffuse_texture_image = None
        self._result_orm_texture_image = None
        self._result_roughness = 0.72
        self._result_metallic = 0.0
        self._result_occlusion_strength = 1.0
        self._result_has_texture = False
        self._result_wire_item = None
        self._result_wire_positions = None
        self._result_sharp_edge_item = None
        self._result_sharp_positions = None
        self._update_dimension_hud()
        self._rebuild_material_items()
        self._fit_union_camera()

    def clear_bake_texture_preview(self):
        """Remove the transient textured preview and its geometric bake overlays."""
        self.clear_bake_process_visualization()
        item = self._bake_preview_item
        if item is not None:
            try:
                self.removeItem(item)
            except Exception:
                pass
            self._infront_items = [
                (existing, kind) for existing, kind in self._infront_items
                if existing is not item
            ]
        self._bake_preview_item = None
        self._bake_preview_pbr_texture_image = None
        self._bake_preview_diffuse_texture_image = None
        self.update()

    def set_bake_texture_preview(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        uvs: np.ndarray,
        texture_image,
        diffuse_texture_image=None,
    ) -> bool:
        """Create the live bake surface once, then update only its atlas texture.

        The incoming geometry is a bounded viewport-only sample of the final
        UV-seam-split GLB target. Faces in one UV chart share indices in the
        preview; only UV seams duplicate vertices. The preview remains separate
        from the authoritative export mesh.
        """
        if self._bake_preview_item is not None:
            self.update_bake_texture_preview(
                texture_image, diffuse_texture_image=diffuse_texture_image
            )
            return True
        vertex_array = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        face_array = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
        uv_array = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
        if (
            len(vertex_array) < 3
            or len(face_array) == 0
            or len(uv_array) != len(vertex_array)
            or int(face_array.max(initial=0)) >= len(vertex_array)
        ):
            return False
        self._bake_preview_pbr_texture_image = texture_image
        if diffuse_texture_image is not None:
            self._bake_preview_diffuse_texture_image = diffuse_texture_image
        active_texture = self._active_bake_preview_texture()
        mesh_data = gl.MeshData(
            vertexes=vertex_array,
            faces=face_array,
            vertexUVs=uv_array,
            textureImage=active_texture,
        )
        item = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawEdges=False,
            drawFaces=True,
            color=(1.0, 1.0, 1.0, self._result_opacity),
            shader="shaded",
            glOptions="translucent",
            textureImage=active_texture,
        )
        self._configure_material_item(item)
        self._apply_infront_to_item(item, "surface")
        self.addItem(item)
        self._bake_preview_item = item
        self._apply_visibility()
        self.update()
        return True

    def _active_bake_preview_texture(self):
        if self._render_pass == "diffuse_albedo" and self._bake_preview_diffuse_texture_image is not None:
            return self._bake_preview_diffuse_texture_image
        return self._bake_preview_pbr_texture_image

    def update_bake_texture_preview(self, texture_image, diffuse_texture_image=None):
        """Refresh the live atlas on the existing viewport actor without rebuilding geometry."""
        if texture_image is not None:
            self._bake_preview_pbr_texture_image = texture_image
        if diffuse_texture_image is not None:
            self._bake_preview_diffuse_texture_image = diffuse_texture_image
        if self._bake_preview_item is None or self._bake_preview_pbr_texture_image is None:
            return
        active_texture = self._active_bake_preview_texture()
        if active_texture is not None:
            self._bake_preview_item.setTextureImage(active_texture)
        self.update()

    def set_bake_recent_edges(self, recent_segments=None) -> None:
        """Show the last ten baked faces as a single true-color-inverted edge trail."""
        recent = (
            np.empty((0, 2, 3), dtype=np.float32)
            if recent_segments is None
            else np.asarray(recent_segments, dtype=np.float32).reshape(-1, 2, 3)
        )
        recent = recent[np.isfinite(recent).all(axis=(1, 2))]
        item = self._bake_recent_edge_item
        self._bake_recent_edge_visible = bool(len(recent))
        if len(recent):
            positions = np.ascontiguousarray(recent.reshape(-1, 3), dtype=np.float32)
            if item is None:
                item = gl.GLLinePlotItem(
                    pos=positions,
                    color=(1.0, 1.0, 1.0, 1.0),
                    antialias=True,
                    mode="lines",
                    glOptions="translucent",
                )
                item._semantic_color = (1.0, 1.0, 1.0, 1.0)
                self._apply_forced_infront_to_item(item)
                self._bake_recent_edge_item = item
            else:
                item.setData(pos=positions, color=(1.0, 1.0, 1.0, 1.0))
                item.setGLOptions(infront_gl_options())
                item.setDepthValue(200)
                item.setVisible(True)
            self._sync_invert_overlay_visibility(refresh_desired=True)
        elif item is not None:
            item.setVisible(False)
            self._sync_invert_overlay_visibility(refresh_desired=True)
        self.update()

    def _apply_forced_infront_to_item(self, item):
        """Put a transient processing actor in the dedicated depth-free overlay renderer."""
        if item is None:
            return
        item.setGLOptions(infront_gl_options())
        item.setDepthValue(200)
        self.addItem(item)

    def _update_bake_cage_line(self, attr_name: str, positions, color):
        positions = (
            np.empty((0, 3), dtype=np.float32)
            if positions is None else np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        )
        item = getattr(self, attr_name)
        if len(positions) == 0:
            if item is not None:
                self._invert_visibility_states.pop(id(item), None)
                try:
                    self.removeItem(item)
                except Exception:
                    pass
                setattr(self, attr_name, None)
            return
        if item is None:
            item = gl.GLLinePlotItem(
                pos=positions, color=color, antialias=True,
                mode="lines", glOptions="translucent",
            )
            item._semantic_color = tuple(float(value) for value in color)
            self._apply_forced_infront_to_item(item)
            setattr(self, attr_name, item)
        else:
            item._semantic_color = tuple(float(value) for value in color)
            display_color = self._auxiliary_line_display_color(item)
            item.setData(pos=positions, color=display_color)
            item.setGLOptions(infront_gl_options())
            item.setDepthValue(200)
        if item is not None and self._overlay_color_mode == "solid":
            item.setData(color=self._auxiliary_line_display_color(item))
        item.setVisible(True)
        self._sync_invert_overlay_visibility(refresh_desired=True)

    def set_bake_process_visualization(
        self,
        *,
        cage_outer_vertices=None,
        cage_inner_vertices=None,
        cage_faces=None,
        active_vertices=None,
        active_faces=None,
    ) -> None:
        """Draw the actual bake cage and the currently processed chart region in 3D.

        Cage wires and the active chart patch are independent actors rendered in
        the depth-free overlay renderer. They are intentionally not viewport text
        and remain visible above source/result geometry in every render mode.
        """
        if cage_faces is not None:
            faces = np.asarray(cage_faces, dtype=np.uint32).reshape(-1, 3)
            outer = np.asarray(cage_outer_vertices, dtype=np.float32).reshape(-1, 3) if cage_outer_vertices is not None else np.empty((0, 3), dtype=np.float32)
            inner = np.asarray(cage_inner_vertices, dtype=np.float32).reshape(-1, 3) if cage_inner_vertices is not None else np.empty((0, 3), dtype=np.float32)
            outer_lines = self._wireframe_positions(outer, faces) if len(outer) and len(faces) else None
            inner_lines = self._wireframe_positions(inner, faces) if len(inner) and len(faces) else None
            self._update_bake_cage_line(
                "_bake_cage_outer_item", outer_lines, (0.10, 0.92, 1.00, 0.98)
            )
            self._update_bake_cage_line(
                "_bake_cage_inner_item", inner_lines, (1.00, 0.48, 0.10, 0.78)
            )

        vertices = (
            np.asarray(active_vertices, dtype=np.float32).reshape(-1, 3)
            if active_vertices is not None else np.empty((0, 3), dtype=np.float32)
        )
        faces = (
            np.asarray(active_faces, dtype=np.uint32).reshape(-1, 3)
            if active_faces is not None else np.empty((0, 3), dtype=np.uint32)
        )
        valid = (
            len(vertices) >= 3 and len(faces) > 0
            and int(faces.max(initial=0)) < len(vertices)
        )
        if not valid:
            for attr_name in ("_bake_active_item", "_bake_active_outline_item"):
                item = getattr(self, attr_name)
                if item is not None:
                    item.setVisible(False)
            self.update()
            return

        if self._bake_active_item is None:
            item = gl.GLMeshItem(
                meshdata=gl.MeshData(vertexes=vertices, faces=faces),
                smooth=False, drawEdges=False, drawFaces=True,
                color=(1.00, 0.36, 0.035, 0.68),
                shader="shaded", glOptions="translucent",
            )
            self._apply_infront_to_item(item, "surface")
            self.addItem(item)
            self._bake_active_item = item
        else:
            self._bake_active_item.setMeshData(vertexes=vertices, faces=faces)
            self._bake_active_item.setVisible(True)
            self._apply_infront_to_item(self._bake_active_item, "surface")

        outline_positions = build_wireframe_positions(vertices, faces, max_edges=18_000)
        self._update_bake_cage_line(
            "_bake_active_outline_item", outline_positions, (1.00, 0.94, 0.20, 1.00)
        )
        self.update()

    def clear_bake_process_visualization(self):
        """Remove cage shells and current-chart highlights without touching bake data."""
        for attr_name in (
            "_bake_cage_outer_item", "_bake_cage_inner_item",
            "_bake_active_item", "_bake_active_outline_item",
            "_bake_recent_edge_item",
        ):
            item = getattr(self, attr_name, None)
            if item is not None:
                self._invert_visibility_states.pop(id(item), None)
                try:
                    self.removeItem(item)
                except Exception:
                    pass
                self._infront_items = [
                    (existing, kind) for existing, kind in self._infront_items
                    if existing is not item
                ]
                setattr(self, attr_name, None)
        self._bake_recent_edge_visible = False
        self.update()

    def clear_scene(self):
        self.clear_source()
        self.clear_result()
        self.clear_processing_mesh()

    def clear_processing_mesh(self):
        """Remove transient edge-only reconstruction overlays."""
        removed = self._processing_edge_item
        if removed is not None:
            self._invert_visibility_states.pop(id(removed), None)
            try:
                self.removeItem(removed)
            except Exception:
                pass
            self._infront_items = [
                (existing, kind) for existing, kind in self._infront_items
                if existing is not removed
            ]
        self._processing_edge_item = None
        highlight = self._processing_highlight_item
        if highlight is not None:
            self._invert_visibility_states.pop(id(highlight), None)
            try:
                self.removeItem(highlight)
            except Exception:
                pass
            self._infront_items = [
                (existing, kind) for existing, kind in self._infront_items
                if existing is not highlight
            ]
        self._processing_highlight_item = None
        self._processing_label = ""
        self.update()

    def set_processing_mesh(
        self, vertices: np.ndarray, faces: np.ndarray, label: str = "", quads: np.ndarray | None = None,
        highlight_segments: np.ndarray | None = None,
    ):
        """Draw only original polygon edges during reconstruction; never color-fill faces."""
        self.clear_processing_mesh()
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        quad_faces = np.asarray(quads if quads is not None else [], dtype=np.int64).reshape(-1, 4)
        if len(vertices) < 3 or (len(triangles) == 0 and len(quad_faces) == 0):
            return
        valid_tri = (triangles >= 0).all(axis=1) & (triangles < len(vertices)).all(axis=1)
        valid_quad = (quad_faces >= 0).all(axis=1) & (quad_faces < len(vertices)).all(axis=1)
        triangles = triangles[valid_tri]
        quad_faces = quad_faces[valid_quad]
        polygon_count = len(triangles) + len(quad_faces)
        if polygon_count > 180_000:
            tri_budget = int(round(180_000 * len(triangles) / polygon_count))
            quad_budget = 180_000 - tri_budget
            if len(triangles) > tri_budget:
                triangles = triangles[np.linspace(0, len(triangles) - 1, tri_budget, dtype=np.int64)] if tri_budget else np.empty((0, 3), dtype=np.int64)
            if len(quad_faces) > quad_budget:
                quad_faces = quad_faces[np.linspace(0, len(quad_faces) - 1, quad_budget, dtype=np.int64)] if quad_budget else np.empty((0, 4), dtype=np.int64)
        if len(triangles) == 0 and len(quad_faces) == 0:
            return

        self._processing_label = str(label)
        processing_edges = self._polygon_wireframe_positions(vertices, triangles, quad_faces)
        if processing_edges is not None and len(processing_edges):
            edge_item = gl.GLLinePlotItem(
                pos=processing_edges,
                color=(0.82, 0.88, 0.96, 1.0),
                antialias=True,
                mode="lines",
                glOptions="translucent",
            )
            self._apply_infront_to_item(edge_item, "wireframe")
            edge_item._semantic_color = (0.82, 0.88, 0.96, 1.0)
            edge_item.setVisible(True)
            self.addItem(edge_item)
            self._processing_edge_item = edge_item

        if highlight_segments is not None:
            segments = np.asarray(highlight_segments, dtype=np.float32).reshape(-1, 2, 3)
            if len(segments) > 12000:
                keep = np.linspace(0, len(segments) - 1, 12000, dtype=np.int64)
                segments = segments[keep]
            segments = segments[np.isfinite(segments).all(axis=(1, 2))]
            if len(segments):
                highlight_item = gl.GLLinePlotItem(
                    pos=np.ascontiguousarray(segments.reshape(-1, 3), dtype=np.float32),
                    color=(1.0, 0.28, 0.035, 1.0),
                    antialias=True,
                    mode="lines",
                    glOptions="translucent",
                )
                self._apply_forced_infront_to_item(highlight_item)
                highlight_item._semantic_color = (1.0, 0.28, 0.035, 1.0)
                highlight_item.setVisible(True)
                self._processing_highlight_item = highlight_item
        self._sync_invert_overlay_visibility(refresh_desired=True)
        self.update()

    def _wireframe_positions(cls, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray | None:
        return build_wireframe_positions(vertices, faces, cls.MAX_WIREFRAME_EDGES)

    @classmethod
    def _polygon_wireframe_positions(
        cls, vertices: np.ndarray, triangles: np.ndarray, quads: np.ndarray
    ) -> np.ndarray | None:
        return build_polygon_wireframe_positions(
            vertices, triangles, quads, max_edges=cls.MAX_WIREFRAME_EDGES
        )

    @classmethod
    def _sharp_edge_positions(cls, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray | None:
        return build_sharp_edge_positions(vertices, faces, max_edges=cls.MAX_SHARP_EDGES)

    @staticmethod
    def _normalize_overlay_color(color) -> tuple[float, float, float]:
        values = np.asarray(color, dtype=np.float64).reshape(-1)
        if values.size < 3:
            raise ValueError("Overlay color must contain at least three RGB values.")
        values = values[:3]
        if not np.all(np.isfinite(values)):
            raise ValueError("Overlay color must contain finite RGB values.")
        if float(np.max(values)) > 1.5:
            values = values / 255.0
        return tuple(float(value) for value in np.clip(values, 0.0, 1.0))

    def _overlay_rgb(self, overlay: str, layer: str) -> tuple[float, float, float]:
        if self._overlay_color_mode in {"invert", "solid"}:
            return self.SOLID_OVERLAY_COLOR
        colors = self._wireframe_colors if overlay == "wireframe" else self._sharp_edge_colors
        return colors[layer]

    def _sharp_edge_color(self, layer: str) -> tuple[float, float, float, float]:
        alpha = self._source_opacity if layer == "source" else self._result_opacity
        return (*self._overlay_rgb("sharp_edges", layer), float(alpha))

    def _create_sharp_edge_item(self, layer: str):
        positions = self._source_sharp_positions if layer == "source" else self._result_sharp_positions
        if positions is None or len(positions) == 0:
            return None
        item = gl.GLLinePlotItem(
            pos=positions, color=self._sharp_edge_color(layer), antialias=True, mode="lines", glOptions="translucent",
        )
        self._apply_infront_to_item(item, "wireframe")
        self.addItem(item)
        if layer == "source":
            self._source_items.append(item)
            self._source_sharp_edge_item = item
        else:
            self._result_items.append(item)
            self._result_sharp_edge_item = item
        return item

    def _apply_infront_to_item(self, item, kind: str):
        """Apply the selected always-on-top mode to a surface or wireframe item."""
        if item is None:
            return
        self._infront_items = [
            (existing, existing_kind)
            for existing, existing_kind in self._infront_items
            if existing is not item
        ]
        self._infront_items.append((item, kind))
        enabled = infront_applies(self._infront_mode, kind)
        if kind == "surface":
            enabled = face_overlay_applies(self._face_overlay_enabled, self._infront_mode)
        if enabled:
            item.setGLOptions(infront_gl_options())
            item.setDepthValue(100)
        else:
            item.setGLOptions("translucent")
            item.setDepthValue(0)

    def set_infront_mode(self, mode: str):
        """Set the legacy In Front mode; the current UI fixes this to Both."""
        mode = str(mode).strip().lower()
        if mode not in {"surface", "wireframe", "both", "off"}:
            mode = "surface"
        self._infront_mode = mode
        for item, kind in list(self._infront_items):
            self._apply_infront_to_item(item, kind)
        self.update()

    def set_infront_enabled(self, enabled: bool):
        """Backward-compatible setter; True targets both mesh layers."""
        self.set_infront_mode("both" if enabled else "off")

    def set_face_overlay_enabled(self, enabled: bool):
        """Toggle depth-bypassing face overlays for asset and processing surfaces.

        OFF restores normal depth testing for face actors and makes the GPU
        invert line pass respect the mesh depth buffer. ON restores the
        alpha-blended In Front treatment for faces and deliberately draws invert
        lines above the surface. Solid/custom wireframes keep their own policy.
        """
        enabled = bool(enabled)
        self._face_overlay_enabled = enabled
        self.setInvertOverlayDepthTest(not enabled)
        for item, kind in list(self._infront_items):
            if kind == "surface":
                self._apply_infront_to_item(item, kind)
        self.update()

    def _create_wire_item(self, layer: str):
        if layer == "source":
            positions = self._source_wire_positions
        else:
            positions = self._result_wire_positions
        if positions is None or len(positions) == 0:
            return None
        item = gl.GLLinePlotItem(
            pos=positions,
            color=self._wire_color(layer),
            antialias=True,
            mode="lines",
            glOptions="translucent",
        )
        self._apply_infront_to_item(item, "wireframe")
        self.addItem(item)
        if layer == "source":
            self._source_items.append(item)
            self._source_wire_item = item
        else:
            self._result_items.append(item)
            self._result_wire_item = item
        return item

    @staticmethod
    def _bounds(points: np.ndarray | None):
        """Return finite axis-aligned bounds for a point or mesh vertex array.

        Both source-scene loading and baked-GLB result loading depend on this
        helper before the camera can be framed. Invalid/empty coordinates are
        ignored so a malformed vertex cannot poison the entire preview.
        """
        if points is None:
            return None
        try:
            vertices = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        except (TypeError, ValueError):
            return None
        if vertices.size == 0:
            return None
        finite = np.all(np.isfinite(vertices), axis=1)
        vertices = vertices[finite]
        if len(vertices) == 0:
            return None
        lower = np.min(vertices, axis=0)
        upper = np.max(vertices, axis=0)
        return lower, upper

    def _fit_union_camera(self, *, animate: bool = False):
        """Frame the visible geometry using the source origin and its orientation."""
        dimension_boxes = [
            box for box in (
                getattr(self, "_dimension_box_source", None),
                getattr(self, "_dimension_box_result", None),
            ) if box is not None
        ]
        frame = np.asarray(self._source_frame_transform, dtype=np.float64).reshape(4, 4)

        if dimension_boxes:
            local_lower = np.min(
                np.vstack([np.asarray(box["local_bounds"][0], dtype=np.float64) for box in dimension_boxes]),
                axis=0,
            )
            local_upper = np.max(
                np.vstack([np.asarray(box["local_bounds"][1], dtype=np.float64) for box in dimension_boxes]),
                axis=0,
            )
        else:
            bounds = [b for b in (self._source_bounds, self._result_bounds) if b is not None]
            if not bounds:
                return
            world_lower = np.min(np.vstack([b[0] for b in bounds]), axis=0)
            world_upper = np.max(np.vstack([b[1] for b in bounds]), axis=0)
            world_corners = np.asarray(
                [
                    (x, y, z)
                    for x in (world_lower[0], world_upper[0])
                    for y in (world_lower[1], world_upper[1])
                    for z in (world_lower[2], world_upper[2])
                ], dtype=np.float64,
            )
            fallback_box = source_frame_oriented_bounds(world_corners, frame)
            if fallback_box is None:
                return
            local_lower, local_upper = fallback_box["local_bounds"]

        render_window = self.GetRenderWindow()
        width, height = render_window.GetSize() if render_window is not None else (0, 0)
        if width <= 0 or height <= 0:
            width, height = max(1, int(self.width())), max(1, int(self.height()))
        camera = self._renderer.GetActiveCamera()
        try:
            horizontal_fov = float(camera.GetViewAngle())
        except (AttributeError, TypeError, ValueError):
            horizontal_fov = 67.0
        pose = source_frame_camera_pose(
            (local_lower, local_upper),
            frame,
            (width, height),
            horizontal_fov,
            margin=1.28,
        )
        position = np.asarray(pose["position"], dtype=np.float64)
        focal = np.asarray(pose["focal"], dtype=np.float64)
        view_up = np.asarray(pose["view_up"], dtype=np.float64)
        distance = float(pose["distance"])
        direction = np.asarray(pose["direction"], dtype=np.float64)
        elevation = math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0))))
        azimuth = math.degrees(math.atan2(float(direction[1]), float(direction[0])))

        if animate:
            self._queue_camera_motion(position, focal, view_up)
        else:
            self._cancel_camera_motion()
            active_camera = self._renderer.GetActiveCamera()
            active_camera.SetFocalPoint(*focal)
            active_camera.SetPosition(*position)
            active_camera.SetViewUp(*view_up)
            active_camera.OrthogonalizeViewUp()
            self._apply_camera_clip_range()
            self.opts.update(distance=distance, elevation=elevation, azimuth=azimuth)
            self.update()

    def set_source_points(self, points: np.ndarray, colors: np.ndarray | None = None):
        self.clear_source()
        points = np.asarray(points, dtype=np.float32)
        if len(points) == 0:
            return
        self._source_bounds = self._bounds(points)
        self._source_dimension_points = np.asarray(points, dtype=np.float32).copy()
        self._refresh_dimension_box_caches()
        if len(points) > 120000:
            stride = max(1, len(points) // 120000)
            points = points[::stride]
            if colors is not None:
                colors = np.asarray(colors)[::stride]
        if colors is None or len(colors) != len(points):
            colors = np.tile(np.array([self.SOURCE_COLOR], dtype=np.float32), (len(points), 1))
        else:
            colors = np.asarray(colors, dtype=np.float32)
            if colors.ndim == 2 and colors.shape[1] == 3:
                colors = np.column_stack([colors, np.full(len(colors), 0.92, dtype=np.float32)])
            elif colors.ndim == 2 and colors.shape[1] >= 4:
                colors = colors[:, :4].copy()
        colors = np.asarray(colors, dtype=np.float32).copy()
        colors[:, 3] = np.minimum(colors[:, 3], 0.96)
        self._source_point_colors = colors
        item = gl.GLScatterPlotItem(
            pos=points, color=self._source_point_display_colors(), size=2.6, pxMode=True
        )
        self._apply_infront_to_item(item, "other")
        self.addItem(item)
        self._source_items.append(item)
        self._source_scatter_item = item
        self._update_dimension_hud()
        self._apply_visibility()
        self._fit_union_camera()

    def set_source_mesh(
        self, vertices: np.ndarray, faces: np.ndarray,
        colors: np.ndarray | None = None, has_texture: bool = False,
        material_parts: list[dict] | None = None,
        fit_camera: bool = True,
        normals: np.ndarray | None = None,
    ):
        self.clear_source()
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)
        if len(vertices) == 0 or len(faces) == 0:
            return
        self._source_vertices = np.asarray(vertices, dtype=np.float32).copy()
        self._source_faces = np.asarray(faces, dtype=np.uint32).copy()
        self._source_bounds = self._bounds(self._source_vertices)
        self._source_dimension_points = None
        self._refresh_dimension_box_caches()
        self._source_vertex_colors = self._normalize_vertex_colors(colors, len(vertices))
        self._source_vertex_normals = self._normalize_vertex_normals(normals, len(vertices))
        self._source_has_texture = bool(has_texture)
        self._source_material_parts = []
        for part in material_parts or ():
            try:
                part_vertices = np.asarray(part.get("vertices"), dtype=np.float32).reshape(-1, 3)
                part_faces = np.asarray(part.get("faces"), dtype=np.uint32).reshape(-1, 3)
                part_colors = self._normalize_vertex_colors(part.get("colors"), len(part_vertices))
                part_uvs = part.get("uvs")
                part_uvs = None if part_uvs is None else np.asarray(part_uvs, dtype=np.float32)
                if part_uvs is not None and part_uvs.shape != (len(part_vertices), 2):
                    part_uvs = None
                image = part.get("texture_image")
                diffuse_image = part.get("diffuse_texture_image", image)
                orm_image = part.get("orm_texture_image")
                if part_uvs is None or (image is None and orm_image is None):
                    image = None
                    diffuse_image = None
                    orm_image = None
                    part_uvs = None
                elif image is None:
                    diffuse_image = None
                elif diffuse_image is None:
                    diffuse_image = image
                if len(part_vertices) >= 3 and len(part_faces) and int(part_faces.max(initial=0)) < len(part_vertices):
                    part_normals = self._normalize_vertex_normals(part.get("normals"), len(part_vertices))
                    self._source_material_parts.append({
                        "vertices": part_vertices.copy(),
                        "faces": part_faces.copy(),
                        "colors": part_colors,
                        "normals": part_normals,
                        "roughness": float(np.clip(part.get("roughness", 0.72), 0.0, 1.0)),
                        "metallic": float(np.clip(part.get("metallic", 0.0), 0.0, 1.0)),
                        "uvs": None if part_uvs is None else part_uvs.copy(),
                        "texture_image": image,
                        "diffuse_texture_image": diffuse_image,
                        "orm_texture_image": orm_image,
                        "occlusion_strength": float(np.clip(part.get("occlusion_strength", 1.0), 0.0, 1.0)),
                        "alpha_mode": str(part.get("alpha_mode", "NATIVE")).upper(),
                        "alpha_cutoff": float(np.clip(part.get("alpha_cutoff", 0.5), 0.0, 1.0)),
                        "shadow_mode": str(part.get("shadow_mode", "OPAQUE")).upper(),
                    })
            except Exception:
                continue
        self._source_sharp_positions = self._sharp_edge_positions(self._source_vertices, self._source_faces)
        if len(faces) > 240000:
            rng = np.random.default_rng(7)
            keep = np.sort(rng.choice(len(faces), 240000, replace=False))
            faces = faces[keep]
        used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
        vertices = vertices[used]
        faces = inverse.reshape(-1, 3).astype(np.uint32, copy=False)
        source_normals = self._source_vertex_normals
        expanded_normals = None if source_normals is None else source_normals[used]
        mesh_data = gl.MeshData(vertexes=vertices, faces=faces, vertexNormals=expanded_normals)
        item = gl.GLMeshItem(
            meshdata=mesh_data, smooth=True, drawEdges=False, drawFaces=True,
            color=(0.16, 0.66, 0.90, self._source_opacity), shader="shaded", glOptions="translucent",
        )
        self._apply_infront_to_item(item, "surface")
        self.addItem(item)
        self._source_items.append(item)
        self._source_surface_item = item
        self._source_wire_positions = self._wireframe_positions(vertices, faces)
        self._create_wire_item("source")
        self._create_sharp_edge_item("source")
        self._rebuild_face_island_items("source")
        self._rebuild_material_items()
        self._apply_visibility()
        if fit_camera:
            self._fit_union_camera()

    def set_reconstruction_mesh(
        self, vertices: np.ndarray, faces: np.ndarray,
        colors: np.ndarray | None = None, has_texture: bool = False,
        uvs: np.ndarray | None = None, texture_image=None, diffuse_texture_image=None,
        orm_texture_image=None, pbr_factors: dict | None = None, occlusion_strength: float = 1.0,
    ):
        self.clear_result()
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)
        if len(vertices) == 0 or len(faces) == 0:
            return
        self._result_vertices = vertices
        self._result_faces = faces
        self._result_vertex_colors = self._normalize_vertex_colors(colors, len(vertices))
        candidate_uvs = None if uvs is None else np.asarray(uvs, dtype=np.float32)
        self._result_uvs = candidate_uvs if candidate_uvs is not None and candidate_uvs.shape == (len(vertices), 2) else None
        self._result_texture_image = texture_image
        self._result_diffuse_texture_image = (
            texture_image if diffuse_texture_image is None else diffuse_texture_image
        )
        self._result_orm_texture_image = orm_texture_image
        factors = pbr_factors if isinstance(pbr_factors, dict) else {}
        try:
            self._result_roughness = float(np.clip(factors.get("roughness", 0.72), 0.0, 1.0))
        except (TypeError, ValueError):
            self._result_roughness = 0.72
        try:
            self._result_metallic = float(np.clip(factors.get("metallic", 0.0), 0.0, 1.0))
        except (TypeError, ValueError):
            self._result_metallic = 0.0
        try:
            self._result_occlusion_strength = float(np.clip(occlusion_strength, 0.0, 1.0))
        except (TypeError, ValueError):
            self._result_occlusion_strength = 1.0
        self._result_has_texture = bool(
            has_texture or texture_image is not None or diffuse_texture_image is not None
            or orm_texture_image is not None
        )
        self._result_bounds = self._bounds(vertices)
        self._refresh_dimension_box_caches()
        self._create_result_item()
        self._rebuild_material_items()
        self._apply_visibility()
        self._fit_union_camera()

    def _create_result_item(self):
        if self._result_vertices is None or self._result_faces is None:
            return
        vertices = self._result_vertices
        faces = self._result_faces
        self._result_sharp_positions = self._sharp_edge_positions(vertices, faces)
        if len(faces) > 300000:
            stride = max(1, len(faces) // 300000)
            faces = faces[::stride]
        used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
        vertices = vertices[used]
        faces = inverse.reshape(-1, 3).astype(np.uint32, copy=False)
        self._result_wire_positions = self._wireframe_positions(vertices, faces)

        mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
        item = gl.GLMeshItem(
            meshdata=mesh_data, smooth=False, drawEdges=False, drawFaces=True,
            color=(*self.RESULT_COLOR[:3], self._result_opacity),
            shader="shaded", glOptions="translucent",
        )
        self._apply_infront_to_item(item, "surface")
        self.addItem(item)
        self._result_items.append(item)
        self._result_surface_item = item
        self._create_wire_item("result")
        self._create_sharp_edge_item("result")
        self._rebuild_face_island_items("result")

    @staticmethod
    def _normalize_vertex_normals(normals: np.ndarray | None, count: int) -> np.ndarray | None:
        if normals is None:
            return None
        values = np.asarray(normals, dtype=np.float64)
        if values.shape != (int(count), 3) or not np.isfinite(values).all():
            return None
        lengths = np.linalg.norm(values, axis=1)
        if np.any(lengths <= 1e-12):
            return None
        return np.ascontiguousarray(values / lengths[:, None], dtype=np.float32)

    @staticmethod
    def _normalize_vertex_colors(colors: np.ndarray | None, count: int) -> np.ndarray | None:
        if colors is None:
            return None
        values = np.asarray(colors, dtype=np.float32)
        if values.ndim != 2 or len(values) != int(count) or values.shape[1] < 3:
            return None
        if float(np.nanmax(values)) > 1.5:
            values = values / 255.0
        if values.shape[1] == 3:
            values = np.column_stack((values, np.ones(len(values), dtype=np.float32)))
        return np.ascontiguousarray(np.clip(np.nan_to_num(values[:, :4], nan=0.66), 0.0, 1.0))

    def _rebuild_material_items(self):
        """Build PBR material layers or image-only, unlit Diffuse layers.

        PBR and Diffuse/Albedo both preserve the source Base Color image and
        material factor. Texture-less parts retain material and vertex colors.
        Reconstruction results use the corresponding baked GLB image directly.
        """
        self._remove_items(self._material_items)
        self._material_layers.clear()
        self._material_mesh_buffers.clear()
        if self._render_pass not in {"pbr", "diffuse_albedo"}:
            return

        layer_data = []
        diffuse_only = self._render_pass == "diffuse_albedo"
        if self._source_material_parts:
            textured_parts = []
            plain_groups = {}
            for part in self._source_material_parts:
                part_texture = (
                    part.get("diffuse_texture_image") if diffuse_only
                    else part.get("texture_image")
                )
                part_uvs = part.get("uvs")
                alpha_mode = str(part.get("alpha_mode", "NATIVE")).upper()
                part_orm_texture = part.get("orm_texture_image")
                if (part_texture is not None or part_orm_texture is not None) and part_uvs is not None:
                    textured_parts.append((part, part_texture))
                    continue
                part_vertices = np.asarray(part["vertices"], dtype=np.float32)
                part_faces = np.asarray(part["faces"], dtype=np.uint32)

                part_colors = self._normalize_vertex_colors(part.get("colors"), len(part_vertices))
                if part_colors is None:
                    part_colors = np.ones((len(part_vertices), 4), dtype=np.float32)
                if diffuse_only and alpha_mode != "NATIVE":
                    source_colors = self._normalize_vertex_colors(part.get("colors"), len(part_vertices))
                    if source_colors is not None and source_colors.shape[1] >= 4:
                        part_colors[:, 3] = source_colors[:, 3]
                part_normals = self._normalize_vertex_normals(part.get("normals"), len(part_vertices))
                roughness = float(np.clip(part.get("roughness", 0.72), 0.0, 1.0))
                metallic = float(np.clip(part.get("metallic", 0.0), 0.0, 1.0))
                group_key = (
                    alpha_mode, float(part.get("alpha_cutoff", 0.5)),
                    round(roughness, 6), round(metallic, 6),
                )
                group = plain_groups.setdefault(group_key, {
                    "vertices": [], "faces": [], "colors": [], "normals": [], "offset": 0,
                })
                group["vertices"].append(part_vertices)
                group["faces"].append(part_faces.astype(np.int64) + group["offset"])
                group["colors"].append(part_colors)
                if part_normals is None:
                    group["normals"].append(None)
                else:
                    group["normals"].append(part_normals)
                group["offset"] += len(part_vertices)
            for (alpha_mode, alpha_cutoff, roughness, metallic), group in plain_groups.items():
                group_normals = None
                if all(value is not None for value in group["normals"]):
                    group_normals = np.concatenate(group["normals"], axis=0)
                layer_data.append((
                    np.concatenate(group["vertices"], axis=0),
                    np.concatenate(group["faces"], axis=0).astype(np.uint32),
                    np.concatenate(group["colors"], axis=0), None, None, "source",
                    alpha_mode, alpha_cutoff, group_normals, roughness, metallic,
                    None, 1.0,
                ))
            for part, part_texture in textured_parts:
                layer_data.append((
                    part["vertices"], part["faces"], part.get("colors"),
                    part.get("uvs"), part_texture, "source",
                    str(part.get("alpha_mode", "NATIVE")).upper(),
                    float(part.get("alpha_cutoff", 0.5)),
                    part.get("normals"),
                    float(np.clip(part.get("roughness", 0.72), 0.0, 1.0)),
                    float(np.clip(part.get("metallic", 0.0), 0.0, 1.0)),
                    part.get("orm_texture_image"),
                    float(np.clip(part.get("occlusion_strength", 1.0), 0.0, 1.0)),
                ))
        elif self._source_vertices is not None and self._source_faces is not None:
            source_colors = (
                np.ones((len(self._source_vertices), 4), dtype=np.float32)
                if diffuse_only else self._source_vertex_colors
            )
            layer_data.append((
                self._source_vertices, self._source_faces,
                source_colors, None, None, "source", "NATIVE", 0.5,
                self._source_vertex_normals, 0.72, 0.0, None, 1.0,
            ))

        if self._result_vertices is not None and self._result_faces is not None:
            result_texture = self._result_texture_image
            if diffuse_only:
                result_texture = self._result_diffuse_texture_image
            result_orm_texture = self._result_orm_texture_image
            result_uvs = self._result_uvs if (result_texture is not None or result_orm_texture is not None) else None
            if result_uvs is None:
                result_texture = None
                result_orm_texture = None
            result_colors = self._result_vertex_colors
            if result_colors is None:
                result_colors = np.ones((len(self._result_vertices), 4), dtype=np.float32)
            layer_data.append((
                self._result_vertices, self._result_faces,
                result_colors, result_uvs, result_texture, "result", "NATIVE", 0.5,
                None, self._result_roughness, self._result_metallic,
                result_orm_texture, self._result_occlusion_strength,
            ))

        layer_limit = {"source": 240_000, "result": 240_000}
        layer_totals = {
            layer: sum(len(entry[1]) for entry in layer_data if entry[5] == layer)
            for layer in layer_limit
        }
        remaining_faces = dict(layer_limit)
        for vertices, faces, colors, uvs, texture_image, layer, alpha_mode, alpha_cutoff, normals, roughness, metallic, orm_texture_image, occlusion_strength in layer_data:
            layer_remaining = remaining_faces.get(layer, 240_000)
            if vertices is None or faces is None or len(faces) == 0 or layer_remaining <= 0:
                continue
            tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
            valid = (tri >= 0).all(axis=1) & (tri < len(vertices)).all(axis=1)
            tri = tri[valid]
            layer_total = max(1, int(layer_totals.get(layer, len(tri))))
            if layer_total > layer_limit.get(layer, 240_000):
                entry_budget = max(1, int(round(
                    layer_limit.get(layer, 240_000) * len(tri) / layer_total
                )))
            else:
                entry_budget = len(tri)
            entry_budget = min(entry_budget, layer_remaining)
            if len(tri) > entry_budget:
                take = np.linspace(0, len(tri) - 1, entry_budget, dtype=np.int64)
                tri = tri[take]
            if len(tri) == 0:
                continue
            remaining_faces[layer] = layer_remaining - len(tri)
            used, inverse = np.unique(tri.reshape(-1), return_inverse=True)
            expanded_vertices = np.ascontiguousarray(
                np.asarray(vertices, dtype=np.float32)[used], dtype=np.float32
            )
            expanded_faces = inverse.reshape(-1, 3).astype(np.uint32, copy=False)
            color_values = self._normalize_vertex_colors(colors, len(vertices))
            if color_values is None:
                color_values = np.ones((len(vertices), 4), dtype=np.float32)
            expanded_colors = np.ascontiguousarray(color_values[used], dtype=np.float32)
            normal_values = self._normalize_vertex_normals(normals, len(vertices))
            expanded_normals = None if normal_values is None else np.ascontiguousarray(normal_values[used], dtype=np.float32)
            expanded_uvs = None
            if (texture_image is not None or orm_texture_image is not None) and uvs is not None:
                candidate_uvs = np.asarray(uvs, dtype=np.float32)
                if candidate_uvs.shape == (len(vertices), 2) and np.isfinite(candidate_uvs[used]).all():
                    expanded_uvs = np.ascontiguousarray(candidate_uvs[used], dtype=np.float32)
                else:
                    texture_image = None
                    orm_texture_image = None
            else:
                texture_image = None
                orm_texture_image = None

            mesh_data = gl.MeshData(
                vertexes=expanded_vertices,
                faces=expanded_faces,
                vertexColors=None if texture_image is not None else expanded_colors,
                vertexUVs=expanded_uvs,
                vertexNormals=expanded_normals,
                textureImage=texture_image,
                ormTextureImage=orm_texture_image,
            )
            item = gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=False,
                drawEdges=False,
                drawFaces=True,
                color=(1.0, 1.0, 1.0, 1.0),
                shader="shaded",
                glOptions="translucent",
                textureImage=texture_image,
                ormTextureImage=orm_texture_image,
            )
            item._polyfitty_pbr_roughness = float(roughness)
            item._polyfitty_pbr_metallic = float(metallic)
            item._polyfitty_pbr_occlusion_strength = float(occlusion_strength)
            self._configure_material_item(item)
            if hasattr(item, "setAlphaBlendMode"):
                item.setAlphaBlendMode(alpha_mode)
            self._apply_infront_to_item(item, "surface")
            self.addItem(item)
            self._material_items.append(item)
            self._material_layers.append(layer)
            self._material_mesh_buffers.append({
                "item": item,
                "layer": layer,
                "vertices": expanded_vertices,
                "faces": expanded_faces,
                "colors": expanded_colors,
                "textured": texture_image is not None,
                "orm_textured": orm_texture_image is not None,
            })

    def _configure_material_item(self, item):
        """Apply imported PBR factors, or keep Diffuse (Albedo) completely unlit."""
        is_pbr = self._render_pass == "pbr"
        roughness = float(getattr(item, "_polyfitty_pbr_roughness", 0.72))
        metallic = float(getattr(item, "_polyfitty_pbr_metallic", 0.0))
        if hasattr(item, "setPBR"):
            item.setPBR(
                is_pbr,
                roughness=roughness if is_pbr else 1.0,
                metallic=metallic if is_pbr else 0.0,
                specular=0.25 if is_pbr else 0.0,
                occlusion_strength=float(getattr(item, "_polyfitty_pbr_occlusion_strength", 1.0)),
            )
        if hasattr(item, "setLighting"):
            item.setLighting(is_pbr)

    def _configure_active_material_items(self):
        for item in self._material_items:
            self._configure_material_item(item)
        if self._bake_preview_item is not None:
            self._configure_material_item(self._bake_preview_item)
            active_texture = self._active_bake_preview_texture()
            if active_texture is not None:
                self._bake_preview_item.setTextureImage(active_texture)

    def set_theme(self, theme_name: str, scene_color=None):
        self._theme_name = str(theme_name or "graphite").lower()
        if scene_color is not None:
            self._scene_background_rgba = tuple(scene_color)
            self.setBackgroundColor(scene_color)
        self._update_dimension_hud_color()
        self.update()

    def set_render_pass(self, render_pass: str):
        """Select PBR, unlit Diffuse (Albedo), Solid Surface, or Wireframe."""
        normalized = str(render_pass or "diffuse_albedo").strip().lower().replace(" ", "_")
        if normalized == "original_color":
            normalized = "diffuse_albedo"
        if normalized not in {"pbr", "diffuse_albedo", "surface", "wireframe"}:
            normalized = "diffuse_albedo"
        if normalized == self._render_pass:
            self._configure_active_material_items()
            self._apply_visibility()
            return
        self._render_pass = normalized
        self._rebuild_material_items()
        self._configure_active_material_items()
        self._apply_visibility()
        self.update()

    def set_point_cloud(self, points: np.ndarray, colors: np.ndarray | None = None):
        self.set_source_points(points, colors)

    def set_mesh(self, vertices: np.ndarray, faces: np.ndarray):
        self.set_reconstruction_mesh(vertices, faces)

    def set_display_mode(self, mode: str):
        mode = str(mode).lower()
        self._show_source = mode in {"source", "both"}
        self._show_result = mode in {"result", "both"}
        self._apply_visibility()

    def set_surface_mix(self, value: float):
        """Blend source/result surface opacity with one A↔B slider (0..100).

        The VTK-backed mesh wrapper exposes setColor() for surface actors while
        line actors use setData(). Updating both layers redraws the blend at once.
        """
        self._surface_mix = float(np.clip(value, 0.0, 100.0)) / 100.0
        self._source_opacity, self._result_opacity = blend_surface_opacity(value)
        if self._source_surface_item is not None:
            self._source_surface_item.setColor((0.16, 0.66, 0.90, self._source_opacity))
        if self._source_scatter_item is not None:
            self._source_scatter_item.setData(color=self._source_point_display_colors())
        if self._result_surface_item is not None:
            self._result_surface_item.setColor((*self.RESULT_COLOR[:3], self._result_opacity))
        if self._bake_preview_item is not None:
            self._bake_preview_item.setColor((1.0, 1.0, 1.0, self._result_opacity))
        for item, layer in ((self._source_sharp_edge_item, "source"), (self._result_sharp_edge_item, "result")):
            if item is not None:
                item.setData(color=self._sharp_edge_color(layer))
        self._apply_visibility()
        self.update()

    def set_reconstruction_opacity(self, value: float):
        """Backward-compatible setter; normal UI uses set_surface_mix()."""
        self._result_opacity = float(np.clip(value, 0.10, 1.0))
        if self._result_surface_item is not None:
            self._result_surface_item.setColor((*self.RESULT_COLOR[:3], self._result_opacity))
        self.update()

    def _source_point_display_colors(self) -> np.ndarray | None:
        if self._source_point_colors is None:
            return None
        colors = self._source_point_colors.copy()
        colors[:, 3] *= self._source_opacity
        return colors

    def _wire_color(self, layer: str) -> tuple[float, float, float, float]:
        source_alpha, result_alpha = blend_wireframe_alpha(self._wireframe_mix * 100.0)
        alpha = source_alpha if layer == "source" else result_alpha
        return (*self._overlay_rgb("wireframe", layer), alpha)

    def set_overlay_color_mode(self, mode: str):
        normalized = str(mode or "invert").strip().lower()
        if normalized not in {"invert", "solid", "custom"}:
            normalized = "invert"
        if normalized == self._overlay_color_mode:
            return
        self._overlay_color_mode = normalized
        for item, layer in ((self._source_wire_item, "source"), (self._result_wire_item, "result")):
            if item is not None:
                item.clearSegmentColors()
                item.setData(color=self._wire_color(layer))
        for item, layer in ((self._source_sharp_edge_item, "source"), (self._result_sharp_edge_item, "result")):
            if item is not None:
                item.clearSegmentColors()
                item.setData(color=self._sharp_edge_color(layer))
        for item in (
            self._bake_cage_outer_item, self._bake_cage_inner_item,
            self._bake_active_outline_item, self._bake_recent_edge_item,
            self._processing_edge_item, self._processing_highlight_item,
        ):
            if item is not None:
                item.setData(color=self._auxiliary_line_display_color(item))
        self._sync_invert_overlay_visibility(refresh_desired=False)
        self.update()

    def _auxiliary_line_display_color(self, item):
        base = tuple(getattr(item, "_semantic_color", getattr(item, "_color", (1.0, 1.0, 1.0, 1.0))))

        if item is self._processing_highlight_item:
            return base
        if self._overlay_color_mode == "solid":
            return (1.0, 1.0, 1.0, float(base[3] if len(base) > 3 else 1.0))
        return base

    def _invertible_line_items(self):
        return tuple(
            item for item in (
                self._source_wire_item,
                self._result_wire_item,
                self._source_sharp_edge_item,
                self._result_sharp_edge_item,
                self._bake_cage_outer_item,
                self._bake_cage_inner_item,
                self._bake_active_outline_item,
                self._bake_recent_edge_item,
                self._processing_edge_item,
                self._processing_highlight_item,
            ) if item is not None
        )

    def _sync_invert_overlay_visibility(self, refresh_desired: bool = False):
        """Route line overlays through framebuffer inversion when that mode is active."""
        invert = self._overlay_color_mode == "invert"
        for item in self._invertible_line_items():
            key = id(item)
            if refresh_desired or key not in self._invert_visibility_states:
                desired = bool(item.isVisible())
            else:
                desired = self._invert_visibility_states[key]

            if invert:
                self._invert_visibility_states[key] = bool(desired)
                item.setInvertOverlay(True)
                color = getattr(item, "_color", (1.0, 1.0, 1.0, 1.0))
                opacity = float(color[3]) if len(color) > 3 else 1.0
                item.actor.SetVisibility(bool(desired and opacity > 1e-4))
            else:
                item.setInvertOverlay(False)
                restore = self._invert_visibility_states.pop(key, desired)
                item.actor.SetVisibility(bool(desired if refresh_desired else restore))

    def set_wireframe_color(self, layer: str, color) -> None:
        layer = str(layer).strip().lower()
        if layer not in self._wireframe_colors:
            raise ValueError("Wireframe layer must be 'source' or 'result'.")
        self._wireframe_colors[layer] = self._normalize_overlay_color(color)
        item = self._source_wire_item if layer == "source" else self._result_wire_item
        if item is not None:
            item.setData(color=self._wire_color(layer))
        self._sync_invert_overlay_visibility(refresh_desired=False)
        self.update()

    def set_sharp_edge_color(self, layer: str, color) -> None:
        layer = str(layer).strip().lower()
        if layer not in self._sharp_edge_colors:
            raise ValueError("Sharp-edge layer must be 'source' or 'result'.")
        self._sharp_edge_colors[layer] = self._normalize_overlay_color(color)
        item = self._source_sharp_edge_item if layer == "source" else self._result_sharp_edge_item
        if item is not None:
            item.setData(color=self._sharp_edge_color(layer))
        self._sync_invert_overlay_visibility(refresh_desired=False)
        self.update()

    def set_sharp_edges_enabled(self, enabled: bool):
        """Compatibility switch; sharp edges are only shown for the selected mesh."""
        self._sharp_edges_enabled = bool(enabled) and self._selected_mesh_layer is not None
        self._apply_visibility()

    def set_wireframe_enabled(self, enabled: bool):
        self._wireframe_enabled = bool(enabled)
        self._apply_visibility()

    def set_wireframe_mix(self, value: float):
        self._wireframe_mix = float(np.clip(value, 0.0, 100.0)) / 100.0
        for item, layer in ((self._source_wire_item, "source"), (self._result_wire_item, "result")):
            if item is not None:
                item.setData(color=self._wire_color(layer))
        self._apply_visibility()

    def _remove_face_island_items(self, layer: str | None = None):
        """Remove cached connected-component overlays for one or both mesh layers."""
        keep_items = []
        keep_layers = []
        keep_buffers = []
        for item, item_layer, buffer in zip(
            self._face_island_items, self._face_island_layers, self._face_island_mesh_buffers
        ):
            if layer is None or item_layer == layer:
                try:
                    self.removeItem(item)
                except Exception:
                    pass
                self._infront_items = [
                    (existing, kind) for existing, kind in self._infront_items if existing is not item
                ]
            else:
                keep_items.append(item)
                keep_layers.append(item_layer)
                keep_buffers.append(buffer)
        self._face_island_items = keep_items
        self._face_island_layers = keep_layers
        self._face_island_mesh_buffers = keep_buffers

    @staticmethod
    def _configure_face_island_fill(item):
        """Force connected-face overlays to render as visibly filled, unlit faces."""
        prop = item.actor.GetProperty()
        prop.SetRepresentationToSurface()
        prop.SetEdgeVisibility(False)
        prop.LightingOff()
        prop.SetOpacity(1.0)
        mapper = getattr(item, "_mapper", None)
        if mapper is not None:
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()

    def _rebuild_face_island_items(self, layer: str | None = None):
        """Build per-component, randomly colored filled surfaces for each mesh layer."""
        requested_layers = (layer,) if layer in {"source", "result"} else ("source", "result")
        for item_layer in requested_layers:
            self._remove_face_island_items(item_layer)
        if not self._face_islands_enabled:
            return

        for item_layer in requested_layers:
            if item_layer == "source":
                vertices, faces = self._source_vertices, self._source_faces
                seed = 1729
            else:
                vertices, faces = self._result_vertices, self._result_faces
                seed = 7919
            if vertices is None or faces is None or len(faces) == 0:
                continue
            expanded_v, expanded_f, expanded_c = connected_component_mesh_data(
                vertices, faces, seed=seed, max_faces=250_000
            )
            if len(expanded_f) == 0:
                continue
            mesh_data = gl.MeshData(
                vertexes=expanded_v, faces=expanded_f, vertexColors=expanded_c
            )
            item = gl.GLMeshItem(
                meshdata=mesh_data, smooth=False, drawEdges=False, drawFaces=True,
                color=(1.0, 1.0, 1.0, 1.0), shader="shaded", glOptions="translucent",
            )

            self._configure_face_island_fill(item)
            self._apply_infront_to_item(item, "surface")
            self.addItem(item)
            self._face_island_items.append(item)
            self._face_island_layers.append(item_layer)
            self._face_island_mesh_buffers.append({
                "item": item, "layer": item_layer,
                "vertices": np.asarray(expanded_v, dtype=np.float32),
                "faces": np.asarray(expanded_f, dtype=np.uint32),
                "colors": np.asarray(expanded_c, dtype=np.float32),
            })

    def set_face_islands_opacity(self, value: float):
        self._face_islands_opacity = float(np.clip(value, 0.0, 1.0))
        self._apply_visibility()
        self.update()

    def set_face_islands_enabled(self, enabled: bool):
        """Toggle random, repeatable colors for edge-connected face islands."""
        enabled = bool(enabled)
        if enabled == self._face_islands_enabled:
            return
        self._face_islands_enabled = enabled
        if enabled:
            self._rebuild_face_island_items()
        self._apply_visibility()
        self.update()

    def _apply_visibility(self):
        source_wire_alpha, result_wire_alpha = blend_wireframe_alpha(self._wireframe_mix * 100.0)
        pass_mode = self._render_pass
        material_pass = pass_mode in {"pbr", "diffuse_albedo"}
        layer_visibility_state = viewport_render_visibility(
            pass_mode, self._wireframe_enabled, self._sharp_edges_enabled,
            self._face_islands_enabled,
        )
        show_face_islands = layer_visibility_state["connected_face_islands"]
        show_base_surfaces = layer_visibility_state["base_surface"]
        show_material_surfaces = layer_visibility_state["material_surface"]
        show_wires = layer_visibility_state["wireframe"]
        show_sharp_edges = layer_visibility_state["sharp_edges"]

        for item in self._source_items:
            is_wire = item is self._source_wire_item
            is_sharp_edge = item is self._source_sharp_edge_item
            if is_wire:
                visible = self._show_source and show_wires and source_wire_alpha > 0.001
            elif is_sharp_edge:
                visible = (
                    self._show_source and show_sharp_edges
                    and self._selected_mesh_layer == "source"
                )
            elif item is self._source_scatter_item and material_pass:
                visible = self._show_source
            else:
                visible = self._show_source and show_base_surfaces and not show_face_islands
            item.setVisible(visible)

        for item in self._result_items:
            is_wire = item is self._result_wire_item
            is_sharp_edge = item is self._result_sharp_edge_item
            if is_wire:
                visible = self._show_result and show_wires and result_wire_alpha > 0.001
            elif is_sharp_edge:
                visible = (
                    self._show_result and show_sharp_edges
                    and self._selected_mesh_layer == "result"
                )
            else:
                visible = self._show_result and show_base_surfaces and not show_face_islands
            item.setVisible(visible)

        layer_visibility = {"source": self._show_source, "result": self._show_result}

        for buffer in self._face_island_mesh_buffers:
            item, layer = buffer["item"], buffer["layer"]
            alpha = (1.0 - self._surface_mix) if layer == "source" else self._surface_mix
            colors = buffer["colors"].copy()

            colors[:, 3] = 1.0
            overlay_opacity = float(np.clip(alpha * self._face_islands_opacity, 0.0, 1.0))
            try:
                item.setMeshData(
                    vertexes=buffer["vertices"], faces=buffer["faces"],
                    vertexColors=np.ascontiguousarray(colors, dtype=np.float32),
                )
                self._configure_face_island_fill(item)
                item.actor.GetProperty().SetOpacity(overlay_opacity)
            except Exception:
                overlay_opacity = 0.0
            item.setVisible(
                show_face_islands and layer_visibility.get(layer, True) and overlay_opacity > 0.001
            )
        for buffer in self._material_mesh_buffers:
            item, layer = buffer["item"], buffer["layer"]
            alpha = (1.0 - self._surface_mix) if layer == "source" else self._surface_mix
            if buffer.get("textured"):

                item.setColor((1.0, 1.0, 1.0, float(alpha)))
            else:
                colors = apply_layer_alpha(buffer["colors"], self._surface_mix, layer)
                try:
                    item.setMeshData(
                        vertexes=buffer["vertices"], faces=buffer["faces"],
                        vertexColors=colors,
                    )
                except Exception:
                    pass
            visible = (
                show_material_surfaces
                and layer_visibility.get(layer, True) and alpha > 0.001
            )

            if layer == "result" and self._bake_preview_item is not None:
                visible = False
            item.setVisible(visible)
        if self._bake_preview_item is not None:
            self._bake_preview_item.setColor((1.0, 1.0, 1.0, self._result_opacity))
            self._bake_preview_item.setVisible(
                show_material_surfaces
                and self._show_result and self._result_opacity > 0.001
            )
        for item in (
            self._bake_cage_outer_item, self._bake_cage_inner_item,
            self._bake_active_item, self._bake_active_outline_item,
        ):
            if item is not None:
                item.setVisible(True)
        if self._bake_recent_edge_item is not None:
            self._bake_recent_edge_item.setVisible(self._bake_recent_edge_visible)
        self._sync_invert_overlay_visibility(refresh_desired=True)
        self._update_dimension_hud()
