"""Deterministic pattern screening and offline response export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PySide6 import QtCore

from defleco_lab.gui.pattern_output import PatternSettings
from defleco_lab.gui.visualization import (
    RangeMode,
    VisualizationSettings,
    VisualizationTransform,
)
from defleco_lab.processing.pipeline import apply_postprocessing, apply_preprocessing
from defleco_lab.processing.registry import registry


@dataclass(frozen=True, slots=True)
class PatternCase:
    case_id: str
    group_id: str
    sequence_index: int
    settings: PatternSettings

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "group_id": self.group_id,
            "sequence_index": self.sequence_index,
            "settings": self.settings.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class AnalysisRecipe:
    recipe_id: str
    method_id: str
    parameters: dict[str, Any]
    scale: float = 0.5
    preprocessing: dict[str, Any] = field(default_factory=dict)
    postprocessing: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_pattern_cases(profile: str = "quick") -> list[PatternCase]:
    """Build a bounded, interpretable screening set rather than a Cartesian explosion."""

    if profile == "extended":
        periods = [14.0, 20.0, 28.0, 40.0]
        stripe_angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
        checker_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]
        spiral_angles = stripe_angles
        ring_phases = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    else:
        periods = [20.0, 32.0]
        stripe_angles = [0.0, 45.0, 90.0, 135.0]
        checker_angles = [0.0, 15.0, 30.0, 45.0]
        spiral_angles = [0.0, 45.0, 90.0, 135.0]
        ring_phases = [0.0, 90.0, 180.0, 270.0]

    cases: list[PatternCase] = []
    for period in periods:
        cases.extend(_angle_group("stripes", period, stripe_angles))
        cases.extend(_angle_group("checker", period, checker_angles))
        cases.extend(_angle_group("spiral", period, spiral_angles))
        cases.extend(_phase_group("rings", period, ring_phases))
    return cases


def _angle_group(family: str, period: float, values: list[float]) -> list[PatternCase]:
    group = f"{family}_p{_number(period)}"
    return [
        PatternCase(
            case_id=f"{group}_a{_number(angle)}",
            group_id=group,
            sequence_index=index,
            settings=PatternSettings(
                family=family,
                period_px=period,
                angle_deg=angle,
                mode="static",
                waveform="binary",
            ),
        )
        for index, angle in enumerate(values)
    ]


def _phase_group(family: str, period: float, values: list[float]) -> list[PatternCase]:
    group = f"{family}_p{_number(period)}"
    return [
        PatternCase(
            case_id=f"{group}_ph{_number(phase)}",
            group_id=group,
            sequence_index=index,
            settings=PatternSettings(
                family=family,
                period_px=period,
                phase_deg=phase,
                mode="static",
                waveform="binary",
            ),
        )
        for index, phase in enumerate(values)
    ]


def default_analysis_recipes() -> list[AnalysisRecipe]:
    """Initial recipes based on the first real-car observations and current hypotheses."""

    return [
        AnalysisRecipe(
            "scharr_magnitude",
            "gradient",
            {
                "operator": "scharr",
                "sigma": 0.8,
                "ksize": 3,
                "residual_sigma": 8.0,
                "output": "magnitude",
            },
        ),
        AnalysisRecipe(
            "scharr_vector_residual",
            "gradient",
            {
                "operator": "scharr",
                "sigma": 0.8,
                "ksize": 3,
                "residual_sigma": 8.0,
                "output": "vector_residual",
            },
        ),
        AnalysisRecipe(
            "tensor_orientation_residual",
            "structure_tensor",
            {
                "operator": "scharr",
                "derivative_sigma": 0.8,
                "tensor_sigma": 2.0,
                "orientation_smooth_sigma": 8.0,
                "output": "orientation_residual",
            },
        ),
        AnalysisRecipe(
            "tensor_linearity_suppressed",
            "structure_tensor",
            {
                "operator": "scharr",
                "derivative_sigma": 0.8,
                "tensor_sigma": 2.0,
                "orientation_smooth_sigma": 8.0,
                "output": "linearity_suppressed",
            },
        ),
        AnalysisRecipe(
            "tensor_junction",
            "structure_tensor",
            {
                "operator": "scharr",
                "derivative_sigma": 0.8,
                "tensor_sigma": 2.0,
                "orientation_smooth_sigma": 8.0,
                "output": "junction_response",
            },
        ),
        AnalysisRecipe(
            "frame_difference",
            "frame_difference",
            {"stride": 1, "blur_sigma": 0.0, "local_normalization": False},
        ),
        AnalysisRecipe(
            "temporal_std_4",
            "temporal_statistics",
            {
                "window": 4,
                "stride": 1,
                "percentile_low": 10.0,
                "percentile_high": 90.0,
                "output": "std",
            },
        ),
        AnalysisRecipe(
            "temporal_range_4",
            "temporal_statistics",
            {
                "window": 4,
                "stride": 1,
                "percentile_low": 10.0,
                "percentile_high": 90.0,
                "output": "range",
            },
        ),
    ]


def process_capture_workspace(
    root: Path,
    recipes: list[AnalysisRecipe] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Process an acquisition manifest and export numeric + presentation variants."""

    root = Path(root)
    capture_manifest = root / "capture_manifest.json"
    payload = json.loads(capture_manifest.read_text(encoding="utf-8"))
    records = payload.get("captures", [])
    if not records:
        raise ValueError("capture manifest contains no captured frames")

    recipes = recipes or default_analysis_recipes()
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["group_id"]), []).append(record)
    for group in groups.values():
        group.sort(key=lambda item: int(item["sequence_index"]))

    total = sum(len(group) for group in groups.values()) * len(recipes)
    done = 0
    outputs: list[dict[str, Any]] = []

    for group_id, group in groups.items():
        images = [_load_raw(root, record) for record in group]
        for recipe in recipes:
            method = registry.create(recipe.method_id, **recipe.parameters)
            needed = method.history_requirement()
            if len(images) < needed:
                done += len(group)
                continue
            for index, record in enumerate(group):
                done += 1
                if progress is not None:
                    progress(done, total, f"{group_id} / {recipe.recipe_id}")
                if index + 1 < needed:
                    continue
                source_frames = images[index + 1 - needed : index + 1]
                frames = [_prepare_frame(frame, recipe) for frame in source_frames]
                result = method.process(frames, keep_intermediates=False)
                response = apply_postprocessing(result.primary, recipe.postprocessing)
                output_entry = _save_response_variants(
                    root,
                    record,
                    recipe,
                    response,
                    source_frames[-1],
                    group[index + 1 - needed : index + 1],
                )
                outputs.append(output_entry)

    result_manifest = {
        "source_manifest": "capture_manifest.json",
        "recipes": [recipe.as_dict() for recipe in recipes],
        "outputs": outputs,
    }
    path = root / "results_manifest.json"
    path.write_text(json.dumps(result_manifest, indent=2), encoding="utf-8")
    return path


def _prepare_frame(image: np.ndarray, recipe: AnalysisRecipe) -> np.ndarray:
    frame = image
    if recipe.scale != 1.0:
        frame = cv2.resize(
            frame,
            None,
            fx=recipe.scale,
            fy=recipe.scale,
            interpolation=cv2.INTER_AREA,
        )
    return apply_preprocessing(frame, recipe.preprocessing)


def _save_response_variants(
    root: Path,
    record: dict[str, Any],
    recipe: AnalysisRecipe,
    response: np.ndarray,
    original: np.ndarray,
    source_records: list[dict[str, Any]],
) -> dict[str, Any]:
    target = root / "processed" / str(record["group_id"]) / recipe.recipe_id
    target.mkdir(parents=True, exist_ok=True)
    stem = f"{record['case_id']}__{recipe.recipe_id}"

    npy_path = target / f"{stem}__response.npy"
    np.save(npy_path, np.asarray(response, dtype=np.float32))

    variants = {
        "gray_full": (False, 1.0),
        "color_full": (True, 1.0),
        "gray_overlay50": (False, 0.5),
        "color_overlay50": (True, 0.5),
    }
    written: dict[str, str] = {}
    for name, (heatmap, alpha) in variants.items():
        settings = VisualizationSettings(
            range_mode=RangeMode.PERCENTILE,
            percentile_low=1.0,
            percentile_high=99.0,
            heatmap=heatmap,
            overlay_alpha=alpha,
        )
        rendered = VisualizationTransform().render(response, settings, original=original)
        path = target / f"{stem}__{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rendered.image, cv2.COLOR_RGB2BGR))
        written[name] = str(path.relative_to(root))

    return {
        "case_id": record["case_id"],
        "group_id": record["group_id"],
        "recipe_id": recipe.recipe_id,
        "source_cases": [item["case_id"] for item in source_records],
        "numeric_response": str(npy_path.relative_to(root)),
        "images": written,
    }


def _load_raw(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = root / str(record["raw_file"])
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _number(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


class ScreeningProcessingWorker(QtCore.QThread):
    progressChanged = QtCore.Signal(int, int, str)
    completed = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, root: Path, parent=None) -> None:
        super().__init__(parent)
        self.root = Path(root)

    def run(self) -> None:
        try:
            path = process_capture_workspace(
                self.root,
                progress=lambda done, total, label: self.progressChanged.emit(
                    done, total, label
                ),
            )
            self.completed.emit(str(path))
        except Exception as exc:  # keep batch failures outside the GUI event loop
            self.failed.emit(str(exc))
