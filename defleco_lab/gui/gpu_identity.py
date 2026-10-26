"""Read-only process, Windows GPU preference and display-adapter diagnostics.

Registry preferences are configuration evidence, not proof of context placement.
"""
from __future__ import annotations

import ctypes
import os
import re
import sys
from collections.abc import Iterable
from ctypes import wintypes


def process_identity() -> dict:
    """Return interpreter identity plus the actual Windows process image, if available."""
    result = {
        "executable": sys.executable,
        "base_executable": getattr(sys, "_base_executable", None),
        "process_image_path": None,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "prefix": sys.prefix,
        "status": "unsupported" if sys.platform != "win32" else "unknown",
    }
    if sys.platform == "win32":
        try:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            query = kernel.QueryFullProcessImageNameW
            query.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                              ctypes.POINTER(wintypes.DWORD)]
            query.restype = wintypes.BOOL
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if query(kernel.GetCurrentProcess(), 0, buffer, ctypes.byref(size)):
                result.update(process_image_path=buffer.value, status="ok")
        except (OSError, AttributeError):
            pass
    return result


def _parse_gpu_preference(raw: object) -> int | None:
    if not isinstance(raw, str):
        return None
    match = re.search(r"(?:^|;)\s*GpuPreference\s*=\s*(\d+)\s*(?:;|$)", raw)
    return int(match.group(1)) if match else None


def gpu_preferences(paths: Iterable[str | None]) -> list[dict]:
    """Read only the supplied executable value names from the current user's key."""
    rows = []
    seen = set()
    for path in paths:
        if not path or os.path.normcase(path) in seen:
            continue
        seen.add(os.path.normcase(path))
        row = {"path": path, "raw_value": None, "gpu_preference": None,
               "status": "unsupported" if sys.platform != "win32" else "missing"}
        if sys.platform == "win32":
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                    r"Software\Microsoft\DirectX\UserGpuPreferences",
                                    0, winreg.KEY_READ) as key:
                    raw, _ = winreg.QueryValueEx(key, path)
                row.update(raw_value=raw, gpu_preference=_parse_gpu_preference(raw), status="ok")
            except FileNotFoundError:
                pass
            except OSError as exc:
                row.update(status="unknown", error=str(exc))
        rows.append(row)
    return rows


def display_adapter_for_rect(x: int, y: int, width: int, height: int) -> dict:
    """Map an exact native-pixel monitor rectangle to its Windows display adapter.

    Partial, scaled or spanning rectangles deliberately remain unresolved.
    """
    result = {"status": "unsupported" if sys.platform != "win32" else "unknown",
              "adapter_name": None, "device_name": None,
              "mapping_method": "EnumDisplayMonitors/GetMonitorInfoW/EnumDisplayDevicesW"}
    if sys.platform != "win32" or width <= 0 or height <= 0:
        return result

    class MonitorInfo(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                    ("szDevice", wintypes.WCHAR * 32)]

    class DisplayDevice(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("DeviceName", wintypes.WCHAR * 32),
                    ("DeviceString", wintypes.WCHAR * 128), ("StateFlags", wintypes.DWORD),
                    ("DeviceID", wintypes.WCHAR * 128), ("DeviceKey", wintypes.WCHAR * 128)]

    try:
        user = ctypes.WinDLL("user32", use_last_error=True)
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.HDC,
                                         ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
        user.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
        user.GetMonitorInfoW.restype = wintypes.BOOL
        user.EnumDisplayMonitors.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                                            callback_type, wintypes.LPARAM]
        user.EnumDisplayMonitors.restype = wintypes.BOOL
        user.EnumDisplayDevicesW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                            ctypes.POINTER(DisplayDevice), wintypes.DWORD]
        user.EnumDisplayDevicesW.restype = wintypes.BOOL
        matches = []

        @callback_type
        def collect(monitor, _dc, _rect, _data):
            info = MonitorInfo()
            info.cbSize = ctypes.sizeof(info)
            if user.GetMonitorInfoW(monitor, ctypes.byref(info)):
                rect = info.rcMonitor
                if (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top) == (
                    x, y, width, height
                ):
                    matches.append(info.szDevice)
            return True

        if not user.EnumDisplayMonitors(None, None, collect, 0) or len(matches) != 1:
            return result
        result["device_name"] = matches[0]
        index = 0
        while True:
            device = DisplayDevice()
            device.cb = ctypes.sizeof(device)
            if not user.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
                break
            if device.DeviceName == matches[0] and device.DeviceString:
                result.update(status="ok", adapter_name=device.DeviceString)
                break
            index += 1
    except (OSError, AttributeError):
        pass
    return result


def _vendor(name: str | None) -> str | None:
    text = (name or "").lower()
    if "nvidia" in text:
        return "NVIDIA"
    if "intel" in text:
        return "Intel"
    if any(token in text for token in ("amd", "radeon", "advanced micro devices")):
        return "AMD"
    return None


def render_path_status(renderer: str | None, adapter: str | None) -> str:
    """Describe observed renderer identity without treating Intel as an error."""
    renderer_vendor = _vendor(renderer)
    adapter_vendor = _vendor(adapter)
    status = (f"{renderer_vendor} renderer active" if renderer_vendor
              else "Renderer vendor unresolved")
    if adapter_vendor is None:
        return status + "; display adapter unresolved"
    if renderer_vendor and renderer_vendor != adapter_vendor:
        return status + f"; hybrid path suspected (display adapter: {adapter_vendor})"
    return status + f"; display adapter: {adapter_vendor}"
