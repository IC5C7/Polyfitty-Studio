from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from app.vtk_geometry import (
    configure_studio_environment,
    srgb_to_linear_colors,
    vertex_colors_for_render_pass,
)

def test_srgb_vertex_colors_decode_to_linear_for_pbr():
    source = np.asarray([[0.5, 0.7353569, 1.0, 0.4]], dtype=np.float32)
    linear = srgb_to_linear_colors(source)
    np.testing.assert_allclose(linear[0], [0.21404114, 0.5, 1.0, 0.4], atol=2e-6)

def test_vertex_color_buffer_switches_between_diffuse_and_pbr_spaces():
    colors = np.tile(
        np.asarray([[0.5, 0.7353569, 1.0, 1.0]], dtype=np.float32), (3, 1)
    )
    diffuse = vertex_colors_for_render_pass(colors, pbr_enabled=False)
    pbr = vertex_colors_for_render_pass(colors, pbr_enabled=True)
    np.testing.assert_allclose(diffuse, colors, atol=0.0)
    np.testing.assert_allclose(
        pbr[0], [0.21404114, 0.5, 1.0, 1.0], atol=2e-6
    )

def test_studio_pbr_preview_has_neutral_linear_environment_fill():
    renderer = vtk.vtkRenderer()
    texture = configure_studio_environment(renderer)

    assert renderer.GetUseImageBasedLighting()
    assert renderer.GetEnvironmentTexture() is texture
    assert not texture.GetUseSRGBColorSpace()
    assert texture.GetInterpolate()
    assert texture.GetMipmap()
    image = texture.GetInput()
    assert image.GetDimensions() == (128, 64, 1)
    radiance = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(64, 128, 3)
    assert np.isfinite(radiance).all()
    assert float(radiance.mean()) >= 0.6
    assert float(radiance.max()) > 1.0
