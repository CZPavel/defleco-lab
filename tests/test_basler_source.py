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
