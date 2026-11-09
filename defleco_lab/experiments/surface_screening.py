"""Single-frame Surface screening, reproducible on a saved source frame."""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6 import QtCore

from defleco_lab.camera.frame_packet import FramePacket
from defleco_lab.gui.surface_io import _response, _settings, render_surface
from defleco_lab.processing.base import ProcessingResult
from defleco_lab.processing.surface import normalize_input, surface_registry

CAPTURE_SCHEMA = "defleco.surface.screening.capture.v1"
RECIPE_VERSION = "surface-screening.v1"
PROFILES = {"surface", "surface_extended"}
DEFAULT_DISPLAY = {
    "range": "percentile",
    "percentile_low": 1.0,
    "percentile_high": 99.0,
    "polarity": "absolute",
    "opacity": 0.5,
    "overlay_base": "enhanced",
    "colormap": "Turbo",
    "enhanced_polarity": "signed",
}


def _display_settings(settings):
    configured = DEFAULT_DISPLAY | deepcopy(settings or {})
    _settings(configured)
    _response(ProcessingResult(np.zeros((1, 1), np.float32)), configured)
    return configured


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def _unique_directory(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    # Never reuse an earlier capture or processing run.
    for index in range(1000):
        path = parent / f"{prefix}_{_timestamp()}_{index:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"Cannot create unique {prefix} directory")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    # A replace keeps the previous complete manifest available until the new one is ready.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    tmp.replace(path)


def _write_png(path: Path, image: np.ndarray, *, rgb: bool = False) -> None:
    values = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image
    ok, encoded = cv2.imencode(".png", values)
    if not ok:
        raise OSError(f"Could not encode {path}")
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())


def create_surface_workspace(
    parent: Path,
    packet: FramePacket,
    profile: str = "surface",
    save_response: bool = False,
    *,
    display_settings: dict | None = None,
) -> Path:
    """Save an owned, unmodified image once; caller establishes frame freshness."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown Surface profile: {profile}")
    display_settings = _display_settings(display_settings)
    image = np.array(packet.image, copy=True)
    _, input_info = normalize_input(image)  # validates supported type, finite data and shape
    root = _unique_directory(Path(parent).expanduser(), "Surface_screening")
    extension = ".png" if image.dtype in (np.uint8, np.uint16) else ".npy"
    raw = root / f"{root.name}__original{extension}"
    if raw.suffix == ".png":
        _write_png(raw, image)
    else:
        with raw.open("xb") as handle:
            np.save(handle, image, allow_pickle=False)
    _write_json(
        root / "capture_manifest.json",
        {
            "schema": CAPTURE_SCHEMA,
            "profile": profile,
            "save_response": bool(save_response),
            "display_settings": display_settings,
            "status": "captured",
            "original": raw.name,
            "source_shape": list(image.shape),
            "input": input_info,
            "frame": packet.public_metadata(),
        },
    )
    return root


def build_surface_recipes(profile: str = "surface") -> list[dict[str, Any]]:
    """Build the approved versioned standard/boosted configurations."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown Surface profile: {profile}")
    recipes = []
    for info in surface_registry.infos():
        default = {key: spec["default"] for key, spec in info.parameters.items()}
        variants = ["standard", "boosted"] if profile == "surface_extended" else ["standard"]
        for variant in variants:
            params = dict(default)
            gain = 1.0
            if variant == "boosted":
                for key in ("sigma", "large"):
                    if key in params:
                        spec = info.parameters[key]
                        params[key] = min(spec["maximum"], max(spec["minimum"], params[key] * 2))
                for key, value in (("size", 31), ("radius", 8), ("clip", 4), ("amount", 2)):
                    if key in params:
                        params[key] = value
                if info.id == "surface_brightness":
                    params.update(amount=1.4, offset=0.05)
                elif info.id == "surface_gamma":
                    params["gamma"] = 0.6
                elif info.id == "surface_log":
                    params["amount"] = 3
                elif info.id == "surface_canny":
                    params["threshold"] = 0.075
                elif info.id in ("surface_gabor", "surface_directional_residual"):
                    params["orientations"] = 16
                elif info.id == "surface_fft_bandpass":
                    params.update(low_cutoff=0.01, high_cutoff=0.25)
                elif info.id in ("surface_equalize", "surface_stretch"):
                    gain = 1.5
            surface_registry.create(info.id, **params)  # reject invalid published recipes early
            recipes.append(
                {
                    "version": RECIPE_VERSION,
                    "method_id": info.id,
                    "method_name": info.name,
                    "family": info.category,
                    "variant": variant,
                    "parameters": params,
                    "enhancement_gain": gain,
                    "display_only_boost": gain != 1.0,
                    "interpretation": "categorical texture code"
                    if info.id == "surface_lbp"
                    else "exploratory response",
                }
            )
    return recipes


def _read_capture(root: Path) -> tuple[dict[str, Any], np.ndarray]:
    payload = json.loads((root / "capture_manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema") != CAPTURE_SCHEMA or payload.get("status") != "captured":
        raise ValueError("Not a completed Surface screening capture")
    original = payload.get("original", "")
    original_path = Path(original)
    if (
        not original
        or original_path.name != original
        or original_path.suffix not in {".png", ".npy"}
        or not (original_path.stem == "original" or original_path.stem.endswith("__original"))
    ):
        raise ValueError("Invalid Surface original file")
    path = root / original
    if path.suffix == ".npy":
        image = np.load(path, allow_pickle=False)
    else:
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"Cannot read source image {path}")
    normalize_input(image)
    if (
        list(image.shape) != payload.get("source_shape")
        or str(image.dtype) != payload["input"]["input_dtype"]
    ):
        raise ValueError("Saved image does not match capture metadata")
    return payload, image


def _range(response: np.ndarray, settings: dict) -> tuple[float, float]:
    values = _response(ProcessingResult(response), settings)
    if settings["range"] == "fixed":
        return float(settings.get("low", 0)), float(settings.get("high", 1))
    if settings["range"] == "auto":
        low, high = float(values.min()), float(values.max())
    else:
        low, high = (
            float(x)
            for x in np.percentile(
                values, [settings["percentile_low"], settings["percentile_high"]]
            )
        )
    if high <= low:
        high = low + 1.0
    return low, high


def _gallery(run: Path, manifest: dict[str, Any]) -> None:
    entries = manifest["outputs"]

    def options(key):
        return "".join(
            f'<option value="{html.escape(v, quote=True)}">{html.escape(v)}</option>'
            for v in sorted({str(entry[key]) for entry in entries})
        )

    cards = []
    for entry in entries:
        attrs = " ".join(
            f'data-{k}="{html.escape(str(entry[k]), quote=True)}"'
            for k in ("family", "method_id", "variant")
        )
        if entry["status"] == "completed":
            images = entry["images"]
            body = "".join(
                f'<a href="{html.escape(images[k], quote=True)}">'
                f'<img src="{html.escape(images[k], quote=True)}" alt="{label}">'
                f"{label}</a>"
                for k, label in (("enhanced", "Enhanced"), ("overlay", "Enhanced + heatmap"))
            )
        else:
            body = html.escape(str(entry.get("error", "No result")))
        cards.append(
            f"<article {attrs}><h3>{html.escape(entry['method_name'])} — "
            f'{html.escape(entry["variant"])}</h3><div class="images">{body}</div></article>'
        )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Surface screening</title>
<style>body{{font:15px sans-serif;background:#15191d;color:#e8eef2;margin:2em}}
label{{margin-right:1em}}article{{border:1px solid #59636d;padding:1em;margin:1em 0}}
.images{{display:flex;gap:1em}}.images a{{color:#a9d9ff;flex:1;min-width:0}}
img{{display:block;max-width:100%;max-height:600px}}</style>
<h1>Surface screening</h1><p>Status: {html.escape(manifest["status"])}</p>
<label>Family <select id="family"><option value="">All</option>{options("family")}</select></label>
<label>Method <select id="method_id"><option value="">All</option>{options("method_id")}</select></label>
<label>Variant <select id="variant"><option value="">All</option>{options("variant")}</select></label>
{"".join(cards)}<script>const keys=['family','method_id','variant'];
function apply(){{document.querySelectorAll('article').forEach(a=>{{a.hidden=!keys.every(k=>
!document.getElementById(k).value||a.getAttribute('data-'+k)===document.getElementById(k).value)}})}}
keys.forEach(k=>document.getElementById(k).addEventListener('change',apply));</script></html>"""
    (run / "screening_results.html").write_text(page, encoding="utf-8")


def process_surface_workspace(
    root: Path,
    progress: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    *,
    profile: str | None = None,
    save_response: bool | None = None,
    display_settings: dict | None = None,
) -> Path:
    """Run each filter sequentially; retain completed entries on cancellation/error."""
    root = Path(root)
    capture, image = _read_capture(root)
    profile = profile or capture["profile"]
    save_response = capture["save_response"] if save_response is None else bool(save_response)
    configured_display = _display_settings(
        capture.get("display_settings") if display_settings is None else display_settings
    )
    recipes = build_surface_recipes(profile)
    normalized, input_info = normalize_input(image)
    run = _unique_directory(root, "run")
    manifest_path = run / "results_manifest.json"
    manifest: dict[str, Any] = {
        "schema": "defleco.surface.screening.results.v1",
        "source_manifest": "../capture_manifest.json",
        "source_original": "../" + capture["original"],
        "input": input_info,
        "frame": capture["frame"],
        "source_shape": list(image.shape),
        "profile": profile,
        "recipe_version": RECIPE_VERSION,
        "save_response": save_response,
        "requested_display_settings": configured_display,
        "status": "running",
        "total": len(recipes),
        "outputs": [],
        "gallery": "screening_results.html",
    }
    _write_json(manifest_path, manifest)
    baseline_range: dict[str, tuple[float, float]] = {}
    baseline_result = None
    last_method = None
    entry = None
    target = None
    sidecar_name = None

    def record_method_error(current, destination, message):
        destination.mkdir(parents=True, exist_ok=True)
        current.update(status="error", error=message)
        _write_json(destination / Path(current["sidecar"]).name, current)
        manifest["outputs"].append(current)
        _write_json(manifest_path, manifest)

    try:
        for index, recipe in enumerate(recipes):
            if cancelled is not None and cancelled():
                manifest["status"] = "cancelled"
                break
            target = run / recipe["method_id"] / recipe["variant"]
            rel = target.relative_to(run).as_posix()
            stem = f"{run.name}__{recipe['method_id']}__{recipe['variant']}"
            enhanced_name = f"{stem}__enhanced.png"
            overlay_name = f"{stem}__enhanced_heatmap.png"
            response_name = f"{stem}__response.npy"
            sidecar_name = f"{stem}__result.json"
            entry = dict(
                recipe,
                status="pending",
                input=input_info,
                frame=capture["frame"],
                source_original=manifest["source_original"],
                sidecar=f"{rel}/{sidecar_name}",
                processing_time_ms=None,
                display_settings=configured_display
                | {"enhancement_gain": recipe["enhancement_gain"]},
            )
            if progress:
                progress(index, len(recipes), f"{recipe['method_name']} / {recipe['variant']}")
            try:
                if recipe["method_id"] != last_method:
                    baseline_result = None
                    last_method = recipe["method_id"]
                if recipe["display_only_boost"] and baseline_result is not None:
                    result = baseline_result
                else:
                    result = surface_registry.create(
                        recipe["method_id"], **recipe["parameters"]
                    ).process([normalized], keep_intermediates=True)
                    if recipe["variant"] == "standard":
                        baseline_range[recipe["method_id"]] = _range(
                            result.primary, configured_display
                        )
                        if recipe["method_id"] in {"surface_equalize", "surface_stretch"}:
                            baseline_result = result
                result.intermediates["_source_normalized"] = normalized
            except Exception as exc:  # noqa: BLE001 - individual methods must not abort the sweep
                record_method_error(entry, target, f"{type(exc).__name__}: {exc}")
                continue

            # A failed standard variant cannot supply the common comparison range.
            # Record the boosted entry as an error rather than silently inventing a range.
            if recipe["method_id"] not in baseline_range:
                record_method_error(
                    entry, target, "Standard response unavailable; no comparison range"
                )
                continue
            low, high = baseline_range[recipe["method_id"]]
            base_settings = configured_display | {
                "mode": "overlay",
                "range": "fixed",
                "low": low,
                "high": high,
                "enhanced_polarity": "signed",
                "enhancement_gain": recipe["enhancement_gain"],
            }
            entry["display_settings"] = base_settings
            try:
                enhanced = render_surface(
                    image, result, base_settings | {"mode": "enhanced", "polarity": "signed"}
                )
                overlay = render_surface(image, result, base_settings | {"mode": "overlay"})
            except Exception as exc:  # noqa: BLE001 - per-filter rendering is recoverable
                record_method_error(entry, target, f"{type(exc).__name__}: {exc}")
                continue
            target.mkdir(parents=True)
            _write_png(target / enhanced_name, enhanced, rgb=True)
            _write_png(target / overlay_name, overlay, rgb=True)
            if save_response:
                with (target / response_name).open("xb") as handle:
                    np.save(handle, np.asarray(result.primary, np.float32), allow_pickle=False)
            entry.update(
                status="completed",
                processing_time_ms=result.processing_time_ms,
                response_dtype=str(result.primary.dtype),
                response_shape=list(result.primary.shape),
                response_min=float(np.min(result.primary)),
                response_max=float(np.max(result.primary)),
                input=input_info,
                frame=capture["frame"],
                source_original=manifest["source_original"],
                display_settings=base_settings,
                diagnostics=result.diagnostics,
                enhanced_display_settings=base_settings
                | {"mode": "enhanced", "polarity": "signed"},
                images={
                    "enhanced": f"{rel}/{enhanced_name}",
                    "overlay": f"{rel}/{overlay_name}",
                },
                response_file=f"{rel}/{response_name}" if save_response else None,
                sidecar=f"{rel}/{sidecar_name}",
            )
            _write_json(target / sidecar_name, entry)
            manifest["outputs"].append(entry)
            _write_json(manifest_path, manifest)
            del enhanced, overlay, result
        else:
            if cancelled is not None and cancelled():
                manifest["status"] = "cancelled"
            else:
                manifest["status"] = (
                    "completed_with_errors"
                    if any(x["status"] == "error" for x in manifest["outputs"])
                    else "completed"
                )
    except Exception as exc:  # noqa: BLE001 - save partial manifest before reporting failure
        manifest["status"] = "error"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if entry is not None and entry not in manifest["outputs"]:
            entry.update(status="error", error=manifest["error"])
            if target is not None and target.exists():
                entry["partial_files"] = [
                    p.relative_to(run).as_posix() for p in target.iterdir() if p.is_file()
                ]
                try:
                    _write_json(target / sidecar_name, entry)
                except OSError:
                    entry["sidecar"] = None
            else:
                entry["sidecar"] = None
            manifest["outputs"].append(entry)
    _write_json(manifest_path, manifest)
    _gallery(run, manifest)
    if progress:
        progress(len(manifest["outputs"]), len(recipes), manifest["status"])
    return manifest_path


class SurfaceScreeningWorker(QtCore.QThread):
    progressChanged = QtCore.Signal(int, int, str)
    completed = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        root: Path | None = None,
        parent=None,
        *,
        profile=None,
        save_response=None,
        capture=None,
        display_settings=None,
    ) -> None:
        super().__init__(parent)
        if (root is None) == (capture is None):
            raise ValueError("Provide exactly one of root or capture")
        self.root = Path(root) if root is not None else None
        self.profile, self.save_response, self.capture = profile, save_response, capture
        self.display_settings = deepcopy(display_settings)

    def cancel(self) -> None:
        self.requestInterruption()

    def run(self) -> None:
        try:
            if self.capture is not None:
                try:
                    self.root = create_surface_workspace(
                        *self.capture, display_settings=self.display_settings
                    )
                finally:
                    self.capture = None
            assert self.root is not None
            path = process_surface_workspace(
                self.root,
                self.progressChanged.emit,
                self.isInterruptionRequested,
                profile=self.profile,
                save_response=self.save_response,
                display_settings=self.display_settings,
            )
            self.completed.emit(str(path))
        except Exception as exc:  # noqa: BLE001 - exceptions must reach the GUI
            self.failed.emit(f"{type(exc).__name__}: {exc}")
