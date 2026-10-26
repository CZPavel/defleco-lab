from defleco_lab.gui import gpu_identity as gpu


def test_preferences_parse_only_exact_field():
    assert gpu._parse_gpu_preference("SwapEffectUpgradeEnable=1;GpuPreference=2;") == 2
    assert gpu._parse_gpu_preference("GpuPreference=1;") == 1
    assert gpu._parse_gpu_preference("GpuPreference=0;") == 0
    for raw in (None, 2, "OtherGpuPreference=2;", "GpuPreference=oops;", ""):
        assert gpu._parse_gpu_preference(raw) is None


def test_renderer_vendor_and_hybrid_evidence():
    assert gpu.render_path_status("NVIDIA RTX 2000", "Intel UHD") == (
        "NVIDIA renderer active; hybrid path suspected (display adapter: Intel)"
    )
    assert gpu.render_path_status("Intel UHD", "Intel UHD") == (
        "Intel renderer active; display adapter: Intel"
    )
    assert "hybrid path suspected" in gpu.render_path_status("AMD Radeon", "NVIDIA RTX")
    assert gpu.render_path_status("NVIDIA RTX", None).endswith("display adapter unresolved")
    assert gpu.render_path_status(None, "Intel UHD").startswith("Renderer vendor unresolved")


def test_non_windows_is_explicit(monkeypatch):
    monkeypatch.setattr(gpu.sys, "platform", "linux")
    assert gpu.process_identity()["status"] == "unsupported"
    assert gpu.process_identity()["process_image_path"] is None
    assert gpu.display_adapter_for_rect(0, 0, 100, 100)["status"] == "unsupported"
    rows = gpu.gpu_preferences([None, "python.exe", "python.exe"])
    assert len(rows) == 1
    assert rows[0]["status"] == "unsupported"


def test_registry_read_only_and_preserves_raw(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []

    class Key:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def open_key(*args):
        calls.append(args)
        return Key()

    def query(_key, path):
        if path == "missing.exe":
            raise FileNotFoundError
        return "GpuPreference=2;Other=1;", 1

    registry = SimpleNamespace(HKEY_CURRENT_USER=1, KEY_READ=0x20019,
                               OpenKey=open_key, QueryValueEx=query)
    monkeypatch.setitem(sys.modules, "winreg", registry)
    monkeypatch.setattr(gpu.sys, "platform", "win32")
    rows = gpu.gpu_preferences(["python.exe", "missing.exe"])
    assert rows[0]["raw_value"] == "GpuPreference=2;Other=1;"
    assert rows[0]["gpu_preference"] == 2
    assert rows[1]["status"] == "missing"
    assert all(call[2:] == (0, registry.KEY_READ) for call in calls)


def test_display_mapping_uses_gdi_device_identity(monkeypatch):
    import ctypes
    from types import SimpleNamespace

    def monitor_info(_monitor, pointer):
        info = pointer._obj
        info.rcMonitor.left, info.rcMonitor.top = 100, -20
        info.rcMonitor.right, info.rcMonitor.bottom = 900, 580
        info.szDevice = r"\\.\DISPLAY5"
        return True

    def monitors(_dc, _clip, callback, _data):
        callback(1, None, None, 0)
        return True

    def devices(_name, index, pointer, _flags):
        if index > 1:
            return False
        device = pointer._obj
        device.DeviceName = r"\\.\DISPLAY1" if index == 0 else r"\\.\DISPLAY5"
        device.DeviceString = "Intel UHD" if index == 0 else "NVIDIA RTX"
        return True

    user = SimpleNamespace(GetMonitorInfoW=monitor_info, EnumDisplayMonitors=monitors,
                           EnumDisplayDevicesW=devices)
    monkeypatch.setattr(gpu.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: user, raising=False)
    monkeypatch.setattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, raising=False)
    result = gpu.display_adapter_for_rect(100, -20, 800, 600)
    assert result["status"] == "ok"
    assert result["adapter_name"] == "NVIDIA RTX"
    assert result["device_name"] == r"\\.\DISPLAY5"
    assert gpu.display_adapter_for_rect(100, -20, 400, 600)["status"] == "unknown"
