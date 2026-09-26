from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any

import numpy as np

from .frame_packet import FramePacket


@dataclass(frozen=True, slots=True)
class CameraDescriptor:
    model: str
    user_id: str = field(repr=False)
    device_class: str
    serial: str = field(repr=False)

    @property
    def label(self) -> str:
        # Unique device labels are intentionally hidden from public-facing screenshots and logs.
        return f"{self.model} ({self.device_class})"


def _pylon():
    try:
        from pypylon import pylon
    except ImportError as exc:
        raise RuntimeError("Basler support requires pypylon 26.6 and pylon Runtime") from exc
    return pylon


def discover_cameras() -> list[CameraDescriptor]:
    pylon = _pylon()
    return [
        CameraDescriptor(
            model=d.GetModelName(),
            user_id=d.GetUserDefinedName(),
            device_class=d.GetDeviceClass(),
            serial=d.GetSerialNumber(),
        )
        for d in pylon.TlFactory.GetInstance().EnumerateDevices()
    ]


class BaslerSource:
    """Exact-target, session-only Basler acquisition adapter.

    The source establishes a temporary free-run acquisition state for live preview
    and restores every acquisition node that it changes when the camera is closed.
    This makes live mode robust against a camera left in software/hardware trigger
    mode by another tool without writing persistent User Sets.
    """

    SAFE_PARAMETERS = (
        "ExposureTime",
        "Gain",
        "AcquisitionFrameRate",
        "Width",
        "Height",
        "OffsetX",
        "OffsetY",
    )

    _ROI_ORDER = ("OffsetX", "OffsetY", "Width", "Height")

    def __init__(self, descriptor: CameraDescriptor) -> None:
        self.descriptor = descriptor
        self.camera: Any = None
        self._baseline: dict[str, Any] = {}
        self._free_run_baseline: dict[str, Any] = {}
        self._frame_id = 0

    def open(self) -> None:
        pylon = _pylon()
        matches = [
            d
            for d in pylon.TlFactory.GetInstance().EnumerateDevices()
            if d.GetSerialNumber() == self.descriptor.serial
            and d.GetModelName() == self.descriptor.model
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one explicitly selected camera, found {len(matches)}")
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(matches[0]))
        self.camera.Open()
        self._baseline = self.snapshot_parameters()
        self._free_run_baseline = {}

    def snapshot_parameters(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name in self.SAFE_PARAMETERS:
            node = self._node(name)
            if node is not None and node.IsReadable():
                values[name] = node.Value
        return values

    def prepare_free_run(self) -> dict[str, Any]:
        """Temporarily configure free-running continuous acquisition.

        Only volatile GenICam nodes are changed. Original values are remembered and
        restored in close(). Unsupported nodes are simply skipped.
        """
        report: dict[str, Any] = {}

        acquisition = self._node("AcquisitionMode")
        if acquisition is not None and acquisition.IsReadable() and acquisition.IsWritable():
            current = acquisition.Value
            values = self._enum_values(acquisition)
            if not values or "Continuous" in values:
                self._free_run_baseline["AcquisitionMode"] = current
                if str(current) != "Continuous":
                    acquisition.Value = "Continuous"
                report["AcquisitionMode"] = acquisition.Value

        selector = self._node("TriggerSelector")
        original_selector = None
        if selector is not None and selector.IsReadable():
            original_selector = selector.Value
            self._free_run_baseline["TriggerSelector"] = original_selector
            values = self._enum_values(selector)
            if selector.IsWritable() and (not values or "FrameStart" in values):
                selector.Value = "FrameStart"

        trigger_mode = self._node("TriggerMode")
        if trigger_mode is not None and trigger_mode.IsReadable() and trigger_mode.IsWritable():
            self._free_run_baseline["TriggerModeFrameStart"] = trigger_mode.Value
            values = self._enum_values(trigger_mode)
            if not values or "Off" in values:
                if str(trigger_mode.Value) != "Off":
                    trigger_mode.Value = "Off"
                report["TriggerMode"] = trigger_mode.Value

        # Keep FrameStart selected while the session is live. Some cameras expose
        # TriggerMode per TriggerSelector; restoring the original selector before
        # starting could re-enable a triggered acquisition path.
        if original_selector is not None:
            report["TriggerSelector"] = _read(self.camera, "TriggerSelector")

        return report

    def apply_temporary(self, changes: dict[str, Any]) -> dict[str, Any]:
        report: dict[str, Any] = {}
        unknown = set(changes) - set(self.SAFE_PARAMETERS)
        if unknown:
            raise ValueError(f"Unsupported temporary parameter: {min(unknown)}")

        ordered = [name for name in self.SAFE_PARAMETERS if name in changes]
        # Width/height can constrain offsets. Move offsets to zero first, then apply size,
        # then the requested offsets. All values remain session-only and are rolled back.
        roi_changes = any(name in changes for name in self._ROI_ORDER)
        if roi_changes:
            for name in ("OffsetX", "OffsetY"):
                node = self._node(name)
                if node is not None and node.IsWritable():
                    node.Value = self._quantize(node, 0)

        ordered = [name for name in ordered if name not in self._ROI_ORDER]
        ordered += [name for name in ("Width", "Height", "OffsetX", "OffsetY") if name in changes]
        for name in ordered:
            requested = changes[name]
            node = self._node(name)
            if node is None or not node.IsWritable():
                report[name] = "not writable"
                continue
            requested = self._quantize(node, requested)
            if hasattr(node, "TrySetValue"):
                ok = bool(node.TrySetValue(requested))
                report[name] = node.Value if ok else "rejected"
            else:
                node.Value = requested
                report[name] = node.Value
        return report

    def _node(self, name: str) -> Any | None:
        """Return a parameter proxy only when the live node map contains it."""
        if self.camera is None:
            return None
        try:
            raw = self.camera.GetNodeMap().GetNode(name)
            if raw is None:
                return None
            return getattr(self.camera, name)
        except Exception:
            return None

    @staticmethod
    def _enum_values(node: Any) -> list[str]:
        try:
            return [str(value) for value in node.Symbolics]
        except Exception:
            try:
                return [str(value) for value in node.GetSymbolics()]
            except Exception:
                return []

    @staticmethod
    def _quantize(node: Any, requested: Any) -> Any:
        """Clamp numeric requests and respect GenICam integer increments."""
        try:
            lo, hi = node.Min, node.Max
            value = min(hi, max(lo, requested))
            increment = getattr(node, "Inc", None)
            if increment and isinstance(value, (int, np.integer)):
                value = lo + round((value - lo) / increment) * increment
            return type(node.Value)(value)
        except (AttributeError, TypeError, ValueError):
            return requested

    def start(self) -> None:
        pylon = _pylon()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def grab(self, timeout_ms: int = 1000) -> FramePacket:
        pylon = _pylon()
        result = self.camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
        if result is None:
            raise TimeoutError(f"No frame within {timeout_ms} ms")
        with result:
            if not result.GrabSucceeded():
                raise RuntimeError(result.ErrorDescription)
            with result.GetFirstImageDataComponent() as component:
                image = component.Array.copy()
            packet = FramePacket(
                image=np.asarray(image),
                frame_id=self._frame_id,
                host_timestamp_ns=perf_counter_ns(),
                camera_timestamp=_read(result, "TimeStamp"),
                block_id=_read(result, "BlockID"),
                exposure_us=_read(self.camera, "ExposureTime"),
                gain=_read(self.camera, "Gain"),
                pixel_format=str(_read(self.camera, "PixelFormat") or ""),
                source="basler",
            )
            self._frame_id += 1
            return packet

    def _restore_free_run(self, failures: list[str]) -> None:
        if not self._free_run_baseline or self.camera is None:
            return

        selector = self._node("TriggerSelector")
        if "TriggerModeFrameStart" in self._free_run_baseline:
            if selector is not None and selector.IsWritable():
                values = self._enum_values(selector)
                if not values or "FrameStart" in values:
                    try:
                        selector.Value = "FrameStart"
                    except Exception as exc:
                        failures.append(f"TriggerSelector: {exc}")
            trigger_mode = self._node("TriggerMode")
            if trigger_mode is not None and trigger_mode.IsWritable():
                try:
                    trigger_mode.Value = self._free_run_baseline["TriggerModeFrameStart"]
                except Exception as exc:
                    failures.append(f"TriggerMode: {exc}")

        if "TriggerSelector" in self._free_run_baseline:
            selector = self._node("TriggerSelector")
            if selector is not None and selector.IsWritable():
                try:
                    selector.Value = self._free_run_baseline["TriggerSelector"]
                except Exception as exc:
                    failures.append(f"TriggerSelector restore: {exc}")

        if "AcquisitionMode" in self._free_run_baseline:
            acquisition = self._node("AcquisitionMode")
            if acquisition is not None and acquisition.IsWritable():
                try:
                    acquisition.Value = self._free_run_baseline["AcquisitionMode"]
                except Exception as exc:
                    failures.append(f"AcquisitionMode: {exc}")

    def close(self) -> list[str]:
        failures: list[str] = []
        if self.camera is None:
            return failures
        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            if any(name in self._baseline for name in self._ROI_ORDER):
                for name in ("OffsetX", "OffsetY"):
                    node = self._node(name)
                    if node is not None and node.IsWritable():
                        try:
                            node.Value = self._quantize(node, 0)
                        except Exception as exc:
                            failures.append(f"{name}: {exc}")
            restore_order = [
                name
                for name in reversed(self.SAFE_PARAMETERS)
                if name in self._baseline and name not in self._ROI_ORDER
            ]
            restore_order += [
                name
                for name in ("Width", "Height", "OffsetX", "OffsetY")
                if name in self._baseline
            ]
            for name in restore_order:
                value = self._baseline[name]
                node = self._node(name)
                if node is not None and node.IsWritable():
                    try:
                        node.Value = value
                    except Exception as exc:  # hardware/API error must be reported
                        failures.append(f"{name}: {exc}")
            self._restore_free_run(failures)
        finally:
            if self.camera.IsOpen():
                self.camera.Close()
            self.camera = None
            self._free_run_baseline = {}
        return failures


def _read(owner: Any, name: str) -> Any:
    try:
        if hasattr(owner, "GetNodeMap"):
            raw = owner.GetNodeMap().GetNode(name)
            node = getattr(owner, name) if raw is not None else None
        else:
            node = getattr(owner, name, None)
    except Exception:
        return None
    try:
        return node.Value if hasattr(node, "Value") else node
    except Exception:
        return None
