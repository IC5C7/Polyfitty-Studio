from __future__ import annotations

import os
from pathlib import Path

class StartupSplash:
    """Send startup status to the native Windows splash launcher."""

    def __init__(self) -> None:
        self._status_path = os.environ.get("POLYFITTY_STARTUP_STATUS_FILE", "").strip()
        self._ready_event = os.environ.get("POLYFITTY_STARTUP_READY_EVENT", "").strip()
        self._enabled = os.name == "nt" and bool(self._status_path) and bool(self._ready_event)
        self._finished = False
        if self._enabled:
            self.update("Loading Polyfitty Studio modules...")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self, message: str) -> None:
        if not self._enabled or self._finished:
            return
        try:
            target = Path(self._status_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(message).replace("\r", " ").replace("\n", " "), encoding="utf-8")
        except Exception:
            pass

    def finish(self, main_window) -> None:
        if not self._enabled or self._finished:
            return
        try:
            if main_window is not None and not main_window.isVisible():
                self.update("Waiting for the main Studio window...")
                return
        except Exception:
            pass
        self.update("Polyfitty Studio is ready.")
        self._signal_ready()
        self._finished = True

    def fail(self, message: str) -> None:
        if not self._enabled or self._finished:
            return
        safe_message = str(message).replace("\r", " ").replace("\n", " ")
        self.update("ERROR: " + safe_message)

    def close(self) -> None:
        if not self._enabled or self._finished:
            return
        self.update("Polyfitty Studio is ready.")
        self._signal_ready()
        self._finished = True

    def _signal_ready(self) -> None:
        if not self._enabled:
            return
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenEventW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
            kernel32.OpenEventW.restype = wintypes.HANDLE
            kernel32.SetEvent.argtypes = (wintypes.HANDLE,)
            kernel32.SetEvent.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenEventW(0x0002, False, self._ready_event)
            if handle:
                try:
                    kernel32.SetEvent(handle)
                finally:
                    kernel32.CloseHandle(handle)
        except Exception:
            pass
