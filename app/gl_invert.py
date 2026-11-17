from __future__ import annotations

"""Small OpenGL state guard for real-time invert-color viewport overlays.

The pass uses the fixed-function color logic operation GL_INVERT. Unlike a
CPU color-sampling path, it inverts the live framebuffer fragments directly.
It is used for wireframe/face overlays and the glyph-only navigation text.
"""

import ctypes
import ctypes.util
import sys

GL_BLEND = 0x0BE2
GL_COLOR_LOGIC_OP = 0x0BF2
GL_DEPTH_TEST = 0x0B71
GL_INVERT = 0x150A
GL_DEPTH_FUNC = 0x0B74
GL_DEPTH_WRITEMASK = 0x0B72
GL_LOGIC_OP_MODE = 0x0BF0
GL_LESS = 0x0201

class GLInvertPass:
    """Enable hardware pixel inversion while a dedicated VTK renderer draws."""

    def __init__(self):
        self._gl = None
        self._saved_state = None

    def _load_gl(self):
        if self._gl is not None:
            return self._gl
        if sys.platform == "win32":
            library = ctypes.WinDLL("opengl32.dll")
        elif sys.platform == "darwin":
            library = ctypes.CDLL(
                "/System/Library/Frameworks/OpenGL.framework/OpenGL"
            )
        else:
            library_name = ctypes.util.find_library("GL") or "libGL.so.1"
            library = ctypes.CDLL(library_name)

        get_integer = library.glGetIntegerv
        get_integer.argtypes = (ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
        get_integer.restype = None

        get_boolean = library.glGetBooleanv
        get_boolean.argtypes = (ctypes.c_uint, ctypes.POINTER(ctypes.c_ubyte))
        get_boolean.restype = None

        is_enabled = library.glIsEnabled
        is_enabled.argtypes = (ctypes.c_uint,)
        is_enabled.restype = ctypes.c_ubyte

        enable = library.glEnable
        enable.argtypes = (ctypes.c_uint,)
        enable.restype = None

        disable = library.glDisable
        disable.argtypes = (ctypes.c_uint,)
        disable.restype = None

        logic_op = library.glLogicOp
        logic_op.argtypes = (ctypes.c_uint,)
        logic_op.restype = None

        self._gl = (library, get_integer, get_boolean, is_enabled, enable, disable, logic_op)
        return self._gl

    @staticmethod
    def _integer(get_integer, token: int) -> int:
        value = ctypes.c_int()
        get_integer(token, ctypes.byref(value))
        return int(value.value)

    @staticmethod
    def _boolean(get_boolean, token: int) -> bool:
        value = ctypes.c_ubyte()
        get_boolean(token, ctypes.byref(value))
        return bool(value.value)

    def begin(self, state) -> bool:
        """Invert only visible line fragments and suppress duplicate line hits.

        The renderer inherits the base mesh depth buffer when FACE OVERLAY is
        OFF. VTK's line mapper applies a small negative coincident-topology
        offset, allowing GL_LESS to pass coplanar edges while still rejecting
        back-side edges and subsequent duplicate fragments.
        """
        if self._saved_state is not None:
            return False
        try:
            _library, get_integer, get_boolean, is_enabled, enable, disable, logic_op = self._load_gl()
            saved = {
                "blend": bool(is_enabled(GL_BLEND)),
                "logic": bool(is_enabled(GL_COLOR_LOGIC_OP)),
                "depth_test": bool(is_enabled(GL_DEPTH_TEST)),
                "logic_mode": self._integer(get_integer, GL_LOGIC_OP_MODE),
                "depth_func": self._integer(get_integer, GL_DEPTH_FUNC),
                "depth_mask": self._boolean(get_boolean, GL_DEPTH_WRITEMASK),
            }
            self._saved_state = (state, saved, logic_op)

            state.vtkglDisable(GL_BLEND)
            state.vtkglEnable(GL_DEPTH_TEST)
            state.vtkglDepthFunc(GL_LESS)
            state.vtkglDepthMask(1)
            enable(GL_COLOR_LOGIC_OP)
            logic_op(GL_INVERT)
            if not is_enabled(GL_COLOR_LOGIC_OP):
                raise RuntimeError("The active OpenGL context does not support color logic operations.")
            return True
        except Exception:
            self.end()
            return False

    def end(self) -> None:
        """Restore the OpenGL state captured by :meth:`begin`."""
        active = self._saved_state
        self._saved_state = None
        if active is None:
            return
        state, saved, logic_op = active
        _library, _get_integer, _get_boolean, _is_enabled, enable, disable, _logic_op = self._gl
        try:
            logic_op(int(saved["logic_mode"]))
            state.vtkglDepthFunc(int(saved["depth_func"]))
            state.vtkglDepthMask(1 if saved["depth_mask"] else 0)
            (state.vtkglEnable if saved["blend"] else state.vtkglDisable)(GL_BLEND)
            (enable if saved["logic"] else disable)(GL_COLOR_LOGIC_OP)
            (state.vtkglEnable if saved["depth_test"] else state.vtkglDisable)(GL_DEPTH_TEST)
        except Exception:
            pass
