"""Deterministic pattern screening and offline response export."""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
                mode="step",
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


def default_analysis_recipes(profile: str = "quick") -> list[AnalysisRecipe]:
    """Bounded recipes based on real-car observations and current hypotheses.

    Quick focuses on the responses that were already visually promising plus the
    new line-suppression hypotheses. Extended adds slower/study-backed alternatives
    without creating a free-form Cartesian parameter sweep.
    """

    recipes = [
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
        AnalysisRecipe(
            "directional_line_residual",
            "directional_residual",
            {"orientations": 8, "length": 21},
        ),
    ]

    if profile == "extended":
        recipes.extend(
            [
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
                    "dog_multiscale_baseline",
                    "dog",
                    {"sigma_small": 1.0, "sigma_large": 5.0},
                ),
                AnalysisRecipe(
                    "gabor_orientation_residual",
                    "gabor",
                    {
                        "periods": [16.0, 32.0],
                        "orientations": 8,
                        "sigma": 6.0,
                        "gamma": 0.5,
                        "psi": 0.0,
                        "scale": 1.0,
                        "output": "orientation_residual",
                    },
                    scale=0.5,
                ),
                AnalysisRecipe(
                    "temporal_median_residual_5",
                    "temporal_median_residual",
                    {"window": 5, "stride": 1},
                ),
                AnalysisRecipe(
                    "farneback_local_residual",
                    "farneback",
                    {"stride": 1, "output": "local_residual", "scale": 1.0},
                    scale=0.5,
                ),
            ]
        )
    return recipes


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

    profile = str(payload.get("profile", "quick"))
    recipes = recipes or default_analysis_recipes(profile)
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

    gallery_path = _write_results_gallery(root, outputs)
    result_manifest = {
        "source_manifest": "capture_manifest.json",
        "recipes": [recipe.as_dict() for recipe in recipes],
        "gallery": gallery_path.relative_to(root).as_posix(),
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
        written[name] = path.relative_to(root).as_posix()

    return {
        "case_id": record["case_id"],
        "group_id": record["group_id"],
        "recipe_id": recipe.recipe_id,
        "source_cases": [item["case_id"] for item in source_records],
        "numeric_response": npy_path.relative_to(root).as_posix(),
        "images": written,
    }


def _write_results_gallery(root: Path, outputs: list[dict[str, Any]]) -> Path:
    """Write a dependency-free visual index for fast human screening."""

    groups = sorted({str(item["group_id"]) for item in outputs})
    recipes = sorted({str(item["recipe_id"]) for item in outputs})
    cards = []
    for item in outputs:
        images = item["images"]
        preview = html.escape(str(images.get("color_overlay50") or images["color_full"]))
        links = " ".join(
            f'<a href="{html.escape(str(path))}">{html.escape(name)}</a>'
            for name, path in images.items()
        )
        numeric = html.escape(str(item["numeric_response"]))
        group_id = html.escape(str(item["group_id"]))
        recipe_id = html.escape(str(item["recipe_id"]))
        case_id = html.escape(str(item["case_id"]))
        cards.append(
            f"""
            <article class="card" data-group="{group_id}" data-recipe="{recipe_id}"
                    data-search="{case_id} {group_id} {recipe_id}">
              <img loading="lazy" src="{preview}" alt="{case_id}">
              <div class="meta">
                <strong>{case_id}</strong>
                <span>{group_id}</span>
                <span>{recipe_id}</span>
                <div class="links">{links} <a href="{numeric}">numeric .npy</a></div>
              </div>
            </article>
            """
        )

    group_options = "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in groups
    )
    recipe_options = "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in recipes
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Defleco LAB screening results</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#15181c;color:#e7e9eb}}
header{{position:sticky;top:0;z-index:3;background:#20242a;padding:14px 18px;border-bottom:1px solid #3a4048}}
.controls{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
select,input{{background:#111419;color:#e7e9eb;border:1px solid #4b535d;border-radius:5px;padding:7px}}
#count{{color:#aeb6bf;margin-left:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;padding:12px}}
.card{{background:#20242a;border:1px solid #363c44;border-radius:7px;overflow:hidden}}
.card img{{width:100%;height:230px;object-fit:contain;background:#090b0e}}
.meta{{padding:9px;display:grid;gap:4px}}
.meta span{{color:#b7bec7;font-size:13px}}
.links{{display:flex;gap:7px;flex-wrap:wrap;font-size:12px;margin-top:5px}}
a{{color:#8fc7ff}}
.hidden{{display:none}}
</style>
</head>
<body>
<header>
  <strong>Defleco LAB screening results</strong>
  <div class="controls">
    <label>Pattern group <select id="group"><option value="">All</option>{group_options}</select></label>
    <label>Recipe <select id="recipe"><option value="">All</option>{recipe_options}</select></label>
    <label>Search <input id="search" type="search" placeholder="case / group / recipe"></label>
    <span id="count"></span>
  </div>
</header>
<main class="grid">
{''.join(cards)}
</main>
<script>
const group=document.querySelector('#group');
const recipe=document.querySelector('#recipe');
const search=document.querySelector('#search');
const cards=[...document.querySelectorAll('.card')];
const count=document.querySelector('#count');
function apply(){{
  const g=group.value, r=recipe.value, q=search.value.trim().toLowerCase();
  let visible=0;
  for(const card of cards){{
    const ok=(!g||card.dataset.group===g)&&(!r||card.dataset.recipe===r)&&
      (!q||card.dataset.search.toLowerCase().includes(q));
    card.classList.toggle('hidden',!ok);
    if(ok) visible++;
  }}
  count.textContent=visible+' / '+cards.length;
}}
group.addEventListener('change',apply);
recipe.addEventListener('change',apply);
search.addEventListener('input',apply);
apply();
</script>
</body>
</html>
"""
    path = root / "screening_results.html"
    path.write_text(page, encoding="utf-8")
    return path


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
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, root: Path, parent=None) -> None:
        super().__init__(parent)
        self.root = Path(root)

    def cancel(self) -> None:
        self.requestInterruption()

    def _progress(self, done: int, total: int, label: str) -> None:
        if self.isInterruptionRequested():
            raise InterruptedError("Screening processing cancelled")
        self.progressChanged.emit(done, total, label)

    def run(self) -> None:
        try:
            path = process_capture_workspace(self.root, progress=self._progress)
            if self.isInterruptionRequested():
                self.cancelled.emit()
            else:
                self.completed.emit(str(path))
        except InterruptedError:
            self.cancelled.emit()
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, cv2.error) as exc:
            self.failed.emit(str(exc))
