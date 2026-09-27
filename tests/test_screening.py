from __future__ import annotations

import json

import cv2
import numpy as np

from defleco_lab.experiments.screening import (
    AnalysisRecipe,
    build_pattern_cases,
    process_capture_workspace,
)


def test_quick_screening_plan_is_bounded_and_identifiable() -> None:
    cases = build_pattern_cases("quick")
    assert 8 <= len(cases) <= 64
    assert len({case.case_id for case in cases}) == len(cases)
    assert {"stripes", "checker", "spiral", "rings"} <= {
        case.settings.family for case in cases
    }


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
