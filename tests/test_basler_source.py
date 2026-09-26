import subprocess
import sys

from defleco_lab.camera.basler_source import BaslerSource, CameraDescriptor


def test_camera_module_imports_without_pypylon() -> None:
    code = (
        "import sys; sys.modules['pypylon'] = None; "
        "import defleco_lab.camera.basler_source as module; "
        "assert module.CameraDescriptor"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False)
    assert completed.returncode == 0


def test_public_camera_label_excludes_user_id_and_serial() -> None:
    descriptor = CameraDescriptor("MODEL", "PRIVATE USER", "BaslerGigE", serial="SECRET")
    assert descriptor.label == "MODEL (BaslerGigE)"
    assert "PRIVATE" not in repr(descriptor)
    assert "SECRET" not in repr(descriptor)


def test_integer_quantization_respects_increment() -> None:
    class Node:
        Min = 8
        Max = 100
        Inc = 4
        Value = 8

    assert BaslerSource._quantize(Node(), 19) == 20
    assert BaslerSource._quantize(Node(), 1000) == 100



class _ValueNode:
    def __init__(self, value, *, writable=True):
        self.Value = value
        self._writable = writable

    def IsReadable(self):
        return True

    def IsWritable(self):
        return self._writable


def test_manual_exposure_is_not_written_while_auto_exposure_is_active(monkeypatch) -> None:
    descriptor = CameraDescriptor("MODEL", "", "BaslerGigE", serial="hidden")
    source = BaslerSource(descriptor)
    source.camera = object()
    exposure_auto = _ValueNode("Continuous")
    exposure = _ValueNode(6754.0)
    nodes = {"ExposureAuto": exposure_auto, "ExposureTime": exposure}
    monkeypatch.setattr(source, "_node", lambda name: nodes.get(name))

    report = source.apply_temporary({"ExposureTime": 20000.0})

    assert exposure.Value == 6754.0
    assert "skipped" in report["ExposureTime"]


def test_auto_exposure_restore_does_not_replay_stale_runtime_exposure(monkeypatch) -> None:
    descriptor = CameraDescriptor("MODEL", "", "BaslerGigE", serial="hidden")
    source = BaslerSource(descriptor)
    source.camera = object()
    exposure_auto = _ValueNode("Off")
    exposure = _ValueNode(12000.0)
    nodes = {"ExposureAuto": exposure_auto, "ExposureTime": exposure}
    monkeypatch.setattr(source, "_node", lambda name: nodes.get(name))
    source._baseline = {"ExposureAuto": "Continuous", "ExposureTime": 5_000_000.0}

    failures = []
    source._restore_profile(failures)

    assert not failures
    assert exposure_auto.Value == "Continuous"
    # ExposureTime is the runtime output of AE in this baseline and must not be
    # forced back to the previously observed 5 s value.
    assert exposure.Value == 12000.0
