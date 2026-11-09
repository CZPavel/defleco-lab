import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from defleco_lab.camera.frame_packet import FramePacket
from defleco_lab.experiments import surface_screening as screening
from defleco_lab.processing.surface import surface_registry


def packet(dtype=np.uint16, color=False):
    y, x = np.mgrid[:40, :48]
    image = ((x * 1000 + y * 100) % 60000).astype(np.uint16)
    if dtype == np.uint8:
        image = (image // 256).astype(np.uint8)
    if color:
        image = np.stack([image, image // 2, image // 3], axis=-1)
    return FramePacket(image, frame_id=41, host_timestamp_ns=500, source="test")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_recipe_coverage_and_concrete_boosts():
    standard = screening.build_surface_recipes("surface")
    extended = screening.build_surface_recipes("surface_extended")
    assert len(standard) == len(surface_registry.infos()) == 50
    assert len(extended) == 100
    assert {x["method_id"] for x in standard} == {x.id for x in surface_registry.infos()}
    for entry in extended:
        surface_registry.create(entry["method_id"], **entry["parameters"])
    boosted = {x["method_id"]: x for x in extended if x["variant"] == "boosted"}
    assert boosted["surface_brightness"]["parameters"] == {"amount": 1.4, "offset": 0.05}
    assert boosted["surface_dog"]["parameters"]["large"] == 16
    assert boosted["surface_canny"]["parameters"]["threshold"] == 0.075
    assert boosted["surface_equalize"]["display_only_boost"]
    assert boosted["surface_stretch"]["enhancement_gain"] == 1.5


@pytest.mark.parametrize(
    "profile,dtype,color,count",
    [
        ("surface", np.uint16, False, 50),
        ("surface_extended", np.uint8, True, 100),
    ],
)
def test_complete_sweep_saves_correct_source_and_outputs(tmp_path, profile, dtype, color, count):
    source = packet(dtype, color)
    before = source.image.copy()
    root = screening.create_surface_workspace(tmp_path, source, profile, save_response=True)
    source.image[:] = 0  # saved capture is an owned snapshot
    result_path = screening.process_surface_workspace(root)
    result = read(result_path)
    assert result["status"] == "completed"
    assert result["total"] == len(result["outputs"]) == count
    assert len(list(result_path.parent.rglob("*.png"))) == count * 2
    raw = cv2.imdecode(
        np.fromfile(root / read(root / "capture_manifest.json")["original"], np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    np.testing.assert_array_equal(raw, before)
    assert result["input"]["input_dtype"] == str(dtype().dtype)
    assert (result_path.parent / "screening_results.html").is_file()
    filenames = []
    for row in result["outputs"]:
        assert row["status"] == "completed"
        response = np.load(result_path.parent / row["response_file"], allow_pickle=False)
        assert response.dtype == np.float32 and response.shape == before.shape[:2]
        assert np.isfinite(response).all()
        metadata = read(result_path.parent / row["sidecar"])
        assert metadata["parameters"] == row["parameters"]
        assert metadata["frame"]["frame_id"] == 41
        assert metadata["display_settings"]["overlay_base"] == "enhanced"
        assert metadata["display_settings"]["opacity"] == 0.5
        for image_file in row["images"].values():
            name = Path(image_file).name
            assert row["method_id"] in name and row["variant"] in name
            filenames.append(name)
            decoded = cv2.imdecode(np.fromfile(result_path.parent / image_file, np.uint8), 1)
            assert decoded.shape[:2] == before.shape[:2]
    assert len(set(filenames)) == count * 2  # flattening must never overwrite a method/variant
    if count == 100:
        for index in range(0, count, 2):
            first, second = result["outputs"][index : index + 2]
            assert first["method_id"] == second["method_id"]
            assert (first["display_settings"]["low"], first["display_settings"]["high"]) == (
                second["display_settings"]["low"],
                second["display_settings"]["high"],
            )


def test_rerun_does_not_overwrite_and_optional_response(tmp_path):
    root = screening.create_surface_workspace(tmp_path, packet(), "surface")
    first = screening.process_surface_workspace(root, cancelled=lambda: True)
    second = screening.process_surface_workspace(
        root, cancelled=lambda: True, profile="surface_extended", save_response=True
    )
    assert first != second and first.exists() and second.exists()
    assert read(first)["status"] == "cancelled"
    assert read(second)["status"] == "cancelled"
    assert read(second)["profile"] == "surface_extended"
    assert (first.parent / "screening_results.html").exists()


def test_flattened_reruns_have_unique_names_and_old_captures_still_load(tmp_path, monkeypatch):
    root = screening.create_surface_workspace(tmp_path, packet(), "surface")
    capture = read(root / "capture_manifest.json")
    (root / capture["original"]).rename(root / "original.png")
    capture["original"] = "original.png"
    capture.pop("display_settings")
    (root / "capture_manifest.json").write_text(json.dumps(capture), encoding="utf-8")
    recipe = screening.build_surface_recipes()[0]
    monkeypatch.setattr(screening, "build_surface_recipes", lambda profile: [recipe])
    first = screening.process_surface_workspace(root)
    second = screening.process_surface_workspace(root)
    names = [p.name for run in (first.parent, second.parent) for p in run.rglob("*.png")]
    assert len(names) == len(set(names)) == 4
    assert read(first)["requested_display_settings"]["opacity"] == 0.5
    assert read(first)["outputs"][0]["display_settings"]["polarity"] == "absolute"


@pytest.mark.parametrize(
    "settings",
    [
        {
            "range": "fixed",
            "low": -0.2,
            "high": 0.3,
            "polarity": "signed",
            "colormap": "Inferno",
            "opacity": 0.0,
            "overlay_base": "enhanced",
        },
        {
            "range": "percentile",
            "percentile_low": 10.0,
            "percentile_high": 90.0,
            "polarity": "absolute",
            "colormap": "Viridis",
            "opacity": 0.8,
            "overlay_base": "original",
        },
    ],
)
def test_custom_heatmap_settings_saved_and_shared_between_variants(tmp_path, monkeypatch, settings):
    recipes = [
        r
        for r in screening.build_surface_recipes("surface_extended")
        if r["method_id"] == "surface_dog"
    ]
    monkeypatch.setattr(screening, "build_surface_recipes", lambda profile: recipes)
    root = screening.create_surface_workspace(
        tmp_path, packet(), "surface_extended", display_settings=settings
    )
    result_path = screening.process_surface_workspace(root)
    result = read(result_path)
    assert result["status"] == "completed"
    for row in result["outputs"]:
        display = row["display_settings"]
        for key in ("polarity", "colormap", "opacity", "overlay_base"):
            assert display[key] == settings[key]
    standard, boosted = result["outputs"]
    for key in ("low", "high"):
        assert standard["display_settings"][key] == boosted["display_settings"][key]
        if settings["range"] == "fixed":
            assert standard["display_settings"][key] == settings[key]
    if settings["opacity"] == 0:
        for row in result["outputs"]:
            decoded = [
                cv2.imdecode(np.fromfile(result_path.parent / image, np.uint8), 1)
                for image in row["images"].values()
            ]
            np.testing.assert_array_equal(*decoded)


def test_worker_snapshots_settings_and_palette_validation(tmp_path):
    settings = {"opacity": 0.2, "colormap": "Magma"}
    worker = screening.SurfaceScreeningWorker(tmp_path, display_settings=settings)
    settings["opacity"] = 0.9
    assert worker.display_settings["opacity"] == 0.2
    with pytest.raises(ValueError, match="palette"):
        screening.create_surface_workspace(
            tmp_path, packet(), display_settings={"colormap": "unknown"}
        )


def test_stop_between_methods_keeps_partial_results(tmp_path):
    root = screening.create_surface_workspace(tmp_path, packet(), "surface")
    state = {"count": 0}

    def on_progress(done, _total, _label):
        state["count"] = done

    path = screening.process_surface_workspace(
        root, progress=on_progress, cancelled=lambda: state["count"] >= 2
    )
    result = read(path)
    assert result["status"] == "cancelled"
    assert len(result["outputs"]) == 3  # progress fires before each current method
    assert (path.parent / "screening_results.html").is_file()
    assert not list(path.parent.rglob("*.npy"))


def test_bad_method_does_not_stop_sweep(tmp_path, monkeypatch):
    root = screening.create_surface_workspace(tmp_path, packet(), "surface")
    method_type = type(surface_registry.create("surface_brightness"))
    original = method_type._process

    def fail(self, frames, **context):
        raise RuntimeError("test method failure")

    monkeypatch.setattr(method_type, "_process", fail)
    try:
        path = screening.process_surface_workspace(root)
    finally:
        monkeypatch.setattr(method_type, "_process", original)
    result = read(path)
    assert result["status"] == "completed_with_errors"
    assert result["outputs"][0]["status"] == "error"
    assert "test method failure" in result["outputs"][0]["error"]
    assert read(path.parent / result["outputs"][0]["sidecar"])["status"] == "error"
    assert result["outputs"][1]["status"] == "completed"


def test_write_error_aborts_and_keeps_manifest_and_gallery(tmp_path, monkeypatch):
    root = screening.create_surface_workspace(tmp_path, packet(), "surface")
    original = screening._write_png
    calls = []

    def fail(path, image, **kwargs):
        calls.append(path)
        if len(calls) == 3:
            raise OSError("disk failure")
        return original(path, image, **kwargs)

    monkeypatch.setattr(screening, "_write_png", fail)
    path = screening.process_surface_workspace(root)
    result = read(path)
    assert result["status"] == "error"
    assert "disk failure" in result["error"]
    assert len(result["outputs"]) == 2
    assert result["outputs"][-1]["status"] == "error"
    assert (path.parent / "screening_results.html").exists()


def test_equalize_boost_reuses_baseline_compute(tmp_path, monkeypatch):
    root = screening.create_surface_workspace(tmp_path, packet(), "surface_extended")
    method = type(surface_registry.create("surface_equalize"))
    original = method._process
    calls = []

    def process(self, frames, **kwargs):
        calls.append(1)
        return original(self, frames, **kwargs)

    monkeypatch.setattr(method, "_process", process)
    path = screening.process_surface_workspace(root)
    assert read(path)["status"] == "completed"
    assert len(calls) == 1
