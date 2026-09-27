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

    Camera parameter handling intentionally follows the already proven patterns
    used by CZPavel/basler-camera-encoder and CZPavel/basler-ace2-gige-tester:
    auto-exposure/auto-gain state is treated separately from the current numeric
    ExposureTime/Gain value, ROI changes are ordered defensively, and temporary
    session changes are restored on close.
    """

    # User-editable temporary values exposed by Defleco LAB.
    SAFE_PARAMETERS = (
        "ExposureTime",
        "Gain",
        "AcquisitionFrameRate",
        "Width",
        "Height",
        "OffsetX",
        "OffsetY",
    )

    # Runtime camera state that must be understood for safe rollback/readback.
    PROFILE_PARAMETERS = (
        "ExposureAuto",
        "ExposureTime",
        "GainAuto",
        "Gain",
        "AcquisitionFrameRateEnable",
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
        for name in self.PROFILE_PARAMETERS:
            node = self._node(name)
            if node is not None and node.IsReadable():
                values[name] = node.Value
        return values

    def public_readback(self) -> dict[str, Any]:
        """Sanitized runtime state useful for diagnostics, never unique camera identity."""
        resulting_fps = None
        for name in ("BslResultingAcquisitionFrameRate", "ResultingFrameRate"):
            node = self._node(name)
            if node is not None and node.IsReadable():
                try:
                    value = float(node.Value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0:
                    resulting_fps = value
                    break
        return {
            "model": self.descriptor.model,
            "width": _read(self.camera, "Width"),
            "height": _read(self.camera, "Height"),
            "pixel_format": _read(self.camera, "PixelFormat"),
            "exposure_auto": _read(self.camera, "ExposureAuto"),
            "exposure_us": _read(self.camera, "ExposureTime"),
            "gain_auto": _read(self.camera, "GainAuto"),
            "gain": _read(self.camera, "Gain"),
            "configured_fps": _read(self.camera, "AcquisitionFrameRate"),
            "resulting_fps": resulting_fps,
        }

    def prepare_free_run(self) -> dict[str, Any]:
        """Temporarily configure free-running continuous acquisition.

        Only volatile GenICam nodes are changed. Original values are remembered and
        restored in close(). Unsupported nodes are skipped.
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

        if original_selector is not None:
            report["TriggerSelector"] = _read(self.camera, "TriggerSelector")

        return report

    def apply_temporary(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Apply user-requested session-only values defensively.

        ExposureTime/Gain are not written while their corresponding automatic
        control loop is active. This mirrors the proven camera-profile behavior
        from basler-ace2-gige-tester and prevents a runtime AE/AGC value from being
        mistaken for a manual configuration value.
        """
        report: dict[str, Any] = {}
        unknown = set(changes) - set(self.SAFE_PARAMETERS)
        if unknown:
            raise ValueError(f"Unsupported temporary parameter: {min(unknown)}")

        changes = dict(changes)
        exposure_auto = self._read_node_value("ExposureAuto")
        if "ExposureTime" in changes and exposure_auto not in (None, "Off"):
            report["ExposureTime"] = f"skipped (ExposureAuto={exposure_auto})"
            changes.pop("ExposureTime")

        gain_auto = self._read_node_value("GainAuto")
        if "Gain" in changes and gain_auto not in (None, "Off"):
            report["Gain"] = f"skipped (GainAuto={gain_auto})"
            changes.pop("Gain")

        if "AcquisitionFrameRate" in changes:
            enabled = self._node("AcquisitionFrameRateEnable")
            if enabled is not None and enabled.IsReadable() and enabled.IsWritable():
                try:
                    if not bool(enabled.Value):
                        enabled.Value = True
                        report["AcquisitionFrameRateEnable"] = True
                except Exception as exc:
                    # Some cameras expose the node but manage it implicitly.
                    report["AcquisitionFrameRateEnable"] = f"unchanged ({exc})"

        # Proven ROI ordering: offsets low first, size, then requested offsets.
        roi_changes = any(name in changes for name in self._ROI_ORDER)
        if roi_changes:
            for name in ("OffsetX", "OffsetY"):
                node = self._node(name)
                if node is not None and node.IsWritable():
                    node.Value = self._quantize(node, 0)

        ordered = [
            name
            for name in ("ExposureTime", "Gain", "AcquisitionFrameRate", "Width", "Height")
            if name in changes
        ]
        ordered += [name for name in ("OffsetX", "OffsetY") if name in changes]

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
        if self.camera is None:
            return None
        try:
            raw = self.camera.GetNodeMap().GetNode(name)
            if raw is None:
                return None
            return getattr(self.camera, name)
        except Exception:
            return None

    def _read_node_value(self, name: str) -> Any:
        node = self._node(name)
        if node is None or not node.IsReadable():
            return None
        try:
            return node.Value
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
        # Keep SDK delivery ordered and let the application's own bounded mailbox
        # decide where back-pressure is handled. LatestImageOnly can discard frames
        # inside pylon before Defleco can count/report them, which is undesirable for
        # temporal analysis and raw recording.
        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

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

    def _restore_profile(self, failures: list[str]) -> None:
        """Restore baseline using the same auto/manual semantics as proven tools."""
        if not self._baseline:
            return

        # Geometry first: zero offsets, restore size, then exact offsets.
        for name in ("OffsetX", "OffsetY"):
            node = self._node(name)
            if node is not None and node.IsWritable():
                try:
                    node.Value = self._quantize(node, 0)
                except Exception as exc:
                    failures.append(f"{name}: {exc}")

        for name in ("Width", "Height", "OffsetX", "OffsetY"):
            if name not in self._baseline:
                continue
            node = self._node(name)
            if node is not None and node.IsWritable():
                try:
                    node.Value = self._baseline[name]
                except Exception as exc:
                    failures.append(f"{name}: {exc}")

        # Restore automatic modes before deciding whether numeric values are manual.
        for name in ("ExposureAuto", "GainAuto"):
            if name not in self._baseline:
                continue
            node = self._node(name)
            if node is not None and node.IsWritable():
                try:
                    node.Value = self._baseline[name]
                except Exception as exc:
                    failures.append(f"{name}: {exc}")

        if self._baseline.get("ExposureAuto") in (None, "Off"):
            self._restore_value("ExposureTime", failures)
        if self._baseline.get("GainAuto") in (None, "Off"):
            self._restore_value("Gain", failures)

        self._restore_value("AcquisitionFrameRate", failures)
        self._restore_value("AcquisitionFrameRateEnable", failures)

    def _restore_value(self, name: str, failures: list[str]) -> None:
        if name not in self._baseline:
            return
        node = self._node(name)
        if node is not None and node.IsWritable():
            try:
                node.Value = self._baseline[name]
            except Exception as exc:
                failures.append(f"{name}: {exc}")

    def close(self) -> list[str]:
        failures: list[str] = []
        if self.camera is None:
            return failures
        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            self._restore_profile(failures)
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
