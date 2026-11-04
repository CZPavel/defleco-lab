from __future__ import annotations

import json
import os

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from defleco_lab.experiments.screening import (
    AnalysisRecipe,
    build_pattern_cases,
    process_capture_workspace,
)
from defleco_lab.gui.experiment_panel import ExperimentPanel
from defleco_lab.gui.pattern_output import PatternSettings


def test_quick_screening_plan_is_bounded_and_identifiable() -> None:
    cases = build_pattern_cases("quick")
    assert 8 <= len(cases) <= 64
    assert len({case.case_id for case in cases}) == len(cases)
    assert {"stripes", "checker", "spiral", "rings"} <= {
        case.settings.family for case in cases
    }
    periods = {case.settings.period_px for case in cases}
    assert 48.0 in periods and 64.0 in periods
    assert min(periods) >= 48.0


def test_extended_screening_reuses_additional_v02_pattern_families() -> None:
    families = {case.settings.family for case in build_pattern_cases("extended")}
    assert {
        "composite",
        "nested_square",
        "squircle",
        "counter_spiral",
        "starburst",
        "solid",
    } <= families


def test_screening_color_override_and_manifest_compatibility() -> None:
    colors = ((12, 34, 56), (210, 180, 90))
    for profile in ("quick", "extended"):
        default = build_pattern_cases(profile)
        colored = build_pattern_cases(profile, colors)
        assert [case.case_id for case in colored] == [case.case_id for case in default]
        assert all((case.settings.dark_color, case.settings.light_color) == colors for case in colored)
        assert all(case.settings.dark_color == (0, 0, 0) for case in default)
        assert all(case.settings.light_color == (255, 255, 255) for case in default)
        serialized = json.loads(json.dumps(colored[0].as_dict()))
        assert serialized["settings"]["dark_color"] == list(colors[0])
        assert serialized["settings"]["light_color"] == list(colors[1])

    legacy_settings = {"family": "stripes", "period_px": 48.0}
    legacy = PatternSettings(**legacy_settings)
    assert legacy.dark_color == (0, 0, 0)
    assert legacy.light_color == (255, 255, 255)


def test_experiment_color_override_is_explicit() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = ExperimentPanel()
    assert not panel.config()["use_pattern_colors"]
    panel.use_pattern_colors.setChecked(True)
    assert panel.config()["use_pattern_colors"]
    panel.close()
    assert app is not None


def test_offline_screening_reuses_raw_frames_and_exports_variants(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    captures = []
    y, x = np.mgrid[:96, :128]

    for index, angle in enumerate((0, 20, 40, 60)):
        phase = np.deg2rad(angle)
        image = (
            127
            + 80
            * np.sin(2 * np.pi * (x * np.cos(phase) + y * np.sin(phase)) / 18)
        ).astype(np.uint8)
        case_id = f"stripes_p18_a{angle}"
        raw_file = f"raw/{case_id}.png"
        assert cv2.imwrite(str(tmp_path / raw_file), image)
        captures.append(
            {
                "case_id": case_id,
                "group_id": "stripes_p18",
                "sequence_index": index,
                "raw_file": raw_file,
            }
        )

    (tmp_path / "capture_manifest.json").write_text(
        json.dumps({"captures": captures}), encoding="utf-8"
    )
    recipes = [
        AnalysisRecipe(
            "scharr",
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
            "difference",
            "frame_difference",
            {"stride": 1, "blur_sigma": 0.0, "local_normalization": False},
        ),
    ]

    result_path = process_capture_workspace(tmp_path, recipes=recipes)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["gallery"] == "screening_results.html"
    gallery = tmp_path / result["gallery"]
    assert gallery.exists()
    gallery_text = gallery.read_text(encoding="utf-8")
    assert "Defleco LAB screening results" in gallery_text
    assert "color_overlay50" in gallery_text
    assert result["outputs"]
    assert {item["recipe_id"] for item in result["outputs"]} == {
        "scharr",
        "difference",
    }
    first = result["outputs"][0]
    assert (tmp_path / first["numeric_response"]).exists()
    assert set(first["images"]) == {
        "gray_full",
        "color_full",
        "gray_overlay50",
        "color_overlay50",
    }
    assert all((tmp_path / path).exists() for path in first["images"].values())
