from __future__ import annotations

"""Qt-independent VTK data conversion helpers, covered by headless tests."""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

def _color_rgba(value) -> tuple[float, float, float, float]:
    """Normalize RGB/RGBA values supplied either as 0..1 or 0..255."""
    if value is None:
        return (1.0, 1.0, 1.0, 1.0)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 3:
        array = np.r_[array, 1.0 if np.nanmax(array) <= 1.5 else 255.0]
    if array.size < 4:
        array = np.r_[array, np.ones(4 - array.size)]
    array = np.nan_to_num(array[:4], nan=1.0, posinf=1.0, neginf=0.0)
    if float(np.max(array)) > 1.5:
        array /= 255.0
    return tuple(float(v) for v in np.clip(array, 0.0, 1.0))

def set_actor_color(actor: vtk.vtkActor, color, *, preserve_texture_rgb: bool = False) -> None:
    """Apply actor opacity while avoiding RGB tinting of a real texture map."""
    rgba = _color_rgba(color)
    prop = actor.GetProperty()
    prop.SetColor(*(1.0, 1.0, 1.0) if preserve_texture_rgb else rgba[:3])
    prop.SetOpacity(float(rgba[3]))

def set_actor_texture(
    actor: vtk.vtkActor,
    texture: vtk.vtkTexture | None,
    *,
    bind_albedo: bool = True,
) -> None:
    """Bind an image for classic shading and, optionally, the VTK PBR albedo slot.

    Diffuse (Albedo) is a literal image-color view.  It uses the classic actor
    texture only, because VTK's sRGB decode is correct for PBR lighting but
    makes an unlit image appear much darker when no output gamma transform is
    applied.  PBR additionally binds the same texture to ``albedoTex``.
    """
    actor.SetTexture(texture)
    prop = actor.GetProperty()
    set_base_color_texture = getattr(prop, "SetBaseColorTexture", None)
    if callable(set_base_color_texture):
        set_base_color_texture(texture if bind_albedo else None)

def set_texture_srgb_mode(texture: vtk.vtkTexture | None, pbr_enabled: bool | None = True) -> None:
    """Use sRGB decoding only while the texture is bound to the PBR albedo slot.

    Diffuse mode intentionally samples the stored RGB byte values without
    linearization so its unlit output matches the source image pixels.  The
    albedoTex slot is detached in that mode; this avoids VTK 9.6's validation
    error for a non-sRGB texture attached to vtkProperty.SetBaseColorTexture().
    """
    if texture is not None:
        texture.SetUseSRGBColorSpace(True if pbr_enabled is None else bool(pbr_enabled))

def srgb_to_linear_colors(colors: np.ndarray) -> np.ndarray:
    """Decode display-encoded sRGB vertex colors for physically-based shading.

    Diffuse/Albedo uses sRGB bytes directly so its unlit output matches the
    source image. PBR albedo values must be linear-light values; decode only
    when rebuilding the mesh for the PBR render pass.
    """
    values = np.asarray(colors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] not in (3, 4):
        raise ValueError("Vertex colors must have shape (N, 3) or (N, 4).")
    values = values.copy()
    if values.size and float(np.nanmax(values)) > 1.5:
        values /= 255.0
    values = np.clip(np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    rgb = values[:, :3]
    values[:, :3] = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )
    return np.ascontiguousarray(values, dtype=np.float32)

def vertex_colors_for_render_pass(colors: np.ndarray, pbr_enabled: bool) -> np.ndarray:
    """Return sRGB colors for Diffuse or linear-light colors for PBR."""
    values = np.asarray(colors, dtype=np.float32)
    return srgb_to_linear_colors(values) if bool(pbr_enabled) else values

def normalize_vertex_normals(normals, count: int) -> np.ndarray | None:
    """Return normalized finite source normals, or None when data is unusable."""
    try:
        values = np.asarray(normals, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.shape != (int(count), 3) or not np.isfinite(values).all():
        return None
    lengths = np.linalg.norm(values, axis=1)
    if np.any(lengths <= 1e-12):
        return None
    return np.ascontiguousarray(values / lengths[:, None], dtype=np.float32)

def apply_studio_environment_intensity(
    texture: vtk.vtkTexture,
    base_radiance: np.ndarray,
    intensity_percent: float,
    color=(1.0, 1.0, 1.0),
    *,
    enabled: bool = True,
    reference_intensity: float = 40.0,
) -> None:
    """Scale the linear-light IBL texture independently from LightKit lights."""
    image = texture.GetInput() if texture is not None else None
    scalars = image.GetPointData().GetScalars() if image is not None else None
    if scalars is None:
        return
    base = np.asarray(base_radiance, dtype=np.float32).reshape(-1, 3)
    target = vtk_to_numpy(scalars)
    if len(base) != len(target):
        raise ValueError("Studio environment radiance does not match the VTK texture.")
    tint = np.asarray(color, dtype=np.float32).reshape(-1)
    if tint.size != 3 or not np.isfinite(tint).all():
        tint = np.ones(3, dtype=np.float32)
    tint = np.clip(tint, 0.0, 1.0)
    try:
        intensity = float(intensity_percent)
    except (TypeError, ValueError):
        intensity = float(reference_intensity)
    if not np.isfinite(intensity):
        intensity = float(reference_intensity)
    reference = max(float(reference_intensity), 1e-6)
    scale = max(0.0, intensity) / reference if enabled else 0.0
    target[:] = base * (scale * tint[None, :])
    scalars.Modified()
    image.Modified()
    texture.Modified()

def create_studio_environment_texture() -> vtk.vtkTexture:
    """Create a compact neutral linear-light studio environment for PBR IBL."""
    width, height = 128, 64
    u = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    longitude = (u * 2.0 - 1.0) * np.pi
    latitude = (0.5 - v) * np.pi
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)

    horizon = np.cos(lat_grid)
    base = 0.54 + 0.14 * np.clip(horizon, 0.0, 1.0) - 0.040 * np.clip(-horizon, 0.0, 1.0)

    def soft_panel(center_lon: float, center_lat: float, width_lon: float, width_lat: float):
        delta_lon = np.arctan2(
            np.sin(lon_grid - center_lon), np.cos(lon_grid - center_lon)
        )
        return np.exp(-0.5 * (
            (delta_lon / width_lon) ** 2 + ((lat_grid - center_lat) / width_lat) ** 2
        ))

    key_panel = soft_panel(-0.78, 0.42, 0.30, 0.24)
    fill_panel = soft_panel(1.32, 0.18, 0.43, 0.34)
    radiance = base + 1.10 * key_panel + 0.48 * fill_panel
    rgb = np.ascontiguousarray(np.repeat(radiance[..., None], 3, axis=2), dtype=np.float32)

    image = vtk.vtkImageData()
    image.SetDimensions(width, height, 1)
    scalars = numpy_to_vtk(
        rgb.reshape(-1, 3), deep=True, array_type=vtk.VTK_FLOAT
    )
    scalars.SetNumberOfComponents(3)
    scalars.SetName("StudioEnvironmentLinearRGB")
    image.GetPointData().SetScalars(scalars)

    texture = vtk.vtkTexture()
    texture.SetInputData(image)
    texture.SetColorModeToDirectScalars()
    texture.SetInterpolate(True)
    texture.SetMipmap(True)
    return texture

def configure_studio_environment(renderer: vtk.vtkRenderer) -> vtk.vtkTexture:
    """Enable neutral image-based lighting on the scene renderer."""
    texture = create_studio_environment_texture()

    renderer.SetEnvironmentTexture(texture, False)
    renderer.UseImageBasedLightingOn()
    return texture

def _vtk_rgba(colors: np.ndarray) -> vtk.vtkUnsignedCharArray:
    values = np.asarray(colors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] not in (3, 4):
        raise ValueError("VTK point colors must have shape (N, 3) or (N, 4).")
    if values.shape[1] == 3:
        values = np.column_stack((values, np.ones(len(values), dtype=np.float32)))
    if values.size and float(np.nanmax(values)) > 1.5:
        values = values / 255.0
    rgba = np.rint(np.clip(np.nan_to_num(values, nan=1.0), 0.0, 1.0) * 255.0).astype(np.uint8)
    result = numpy_to_vtk(np.ascontiguousarray(rgba), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    result.SetNumberOfComponents(4)
    result.SetName("RGBA")
    return result

def _vtk_points(points: np.ndarray) -> vtk.vtkPoints:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    result = vtk.vtkPoints()
    if len(values):
        result.SetData(numpy_to_vtk(np.ascontiguousarray(values), deep=True))
    return result

def _vtk_cells(indices: np.ndarray, vertices_per_cell: int) -> vtk.vtkCellArray:
    cells = np.asarray(indices, dtype=np.int64).reshape(-1, vertices_per_cell)
    result = vtk.vtkCellArray()
    if not len(cells):
        return result
    offsets = np.arange(len(cells) + 1, dtype=np.int64) * int(vertices_per_cell)
    connectivity = np.ascontiguousarray(cells.reshape(-1), dtype=np.int64)
    result.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )
    return result

def mesh_to_polydata(
    vertexes: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray | None = None,
    vertex_uvs: np.ndarray | None = None,
    vertex_normals: np.ndarray | None = None,
) -> vtk.vtkPolyData:
    """Convert geometry, optional appearance, UVs, and authored point normals to vtkPolyData."""
    points = np.asarray(vertexes, dtype=np.float32).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(triangles):
        valid = (triangles >= 0).all(axis=1) & (triangles < len(points)).all(axis=1)
        triangles = triangles[valid]
    poly = vtk.vtkPolyData()
    poly.SetPoints(_vtk_points(points))
    poly.SetPolys(_vtk_cells(triangles, 3))
    if vertex_colors is not None:
        colors = np.asarray(vertex_colors, dtype=np.float32)
        if len(colors) == len(points):
            poly.GetPointData().SetScalars(_vtk_rgba(colors))
    if vertex_uvs is not None:
        uv = np.asarray(vertex_uvs, dtype=np.float32)
        if uv.shape == (len(points), 2):
            tcoords = numpy_to_vtk(np.ascontiguousarray(uv), deep=True)
            tcoords.SetNumberOfComponents(2)
            tcoords.SetName("TCoords")
            poly.GetPointData().SetTCoords(tcoords)
    normals = normalize_vertex_normals(vertex_normals, len(points)) if vertex_normals is not None else None
    if normals is not None:
        vtk_normals = numpy_to_vtk(normals, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_normals.SetNumberOfComponents(3)
        vtk_normals.SetName("Normals")
        poly.GetPointData().SetNormals(vtk_normals)
    return poly

def texture_image_to_vtk_image(image) -> vtk.vtkImageData:
    """Convert a PIL/array image into VTK texture data with a lower-left UV origin.

    glTF UV coordinates use a lower-left origin while decoded PNG/JPEG buffers
    are conventionally top-left-first. Flip rows here so VTK texture v=0 samples
    the same texel that the GLB material defines.
    """
    from PIL import Image

    source = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.shape[0] < 1 or rgba.shape[1] < 1:
        raise ValueError("Texture image must be a non-empty RGB/RGBA image.")
    rgba = np.ascontiguousarray(np.flipud(rgba))
    height, width = rgba.shape[:2]
    scalars = numpy_to_vtk(rgba.reshape(-1, 4), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    scalars.SetNumberOfComponents(4)
    scalars.SetName("RGBA")
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(int(width), int(height), 1)
    vtk_image.GetPointData().SetScalars(scalars)
    return vtk_image

def lines_to_polydata(positions: np.ndarray) -> vtk.vtkPolyData:
    points = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    segment_count = len(points) // 2
    poly = vtk.vtkPolyData()
    poly.SetPoints(_vtk_points(points[: segment_count * 2]))
    if segment_count:
        indices = np.arange(segment_count * 2, dtype=np.int64).reshape(-1, 2)
        poly.SetLines(_vtk_cells(indices, 2))
    return poly
