import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6 import QtWidgets

from defleco_lab.camera.frame_packet import FramePacket
from defleco_lab.experiments.surface_screening import (
    build_surface_recipes,
    create_surface_workspace,
    process_surface_workspace,
)
from defleco_lab.gui.main_window import MainWindow


def synthetic():
    y, x = np.mgrid[:80, :96]
    image = (0.3 + 0.3 * x / 96).astype(np.float32)
    image[20:22, 12:80] += 0.2
    image[45:49, 12:80] -= 0.2
    image[(x - 55) ** 2 + (y - 60) ** 2 < 25] += 0.15
    return np.rint(image * 65535).astype(np.uint16)


def test_seven_recommended_fusion_recipes_export_one_shared_native_image(tmp_path):
    recipes = build_surface_recipes("surface_fusion")
    assert len(recipes) == 7
    assert [r["method_id"] for r in recipes] == [f"fusion_k{i:02d}" for i in range(1, 8)]
    assert {r["variant"] for r in recipes} == {"recommended"}
    image = synthetic()
    source = FramePacket(
        image.copy(),
        42,
        100,
        source="image-sequence",
        metadata={"source_filename": "paint_16bit.tiff"},
    )
    root = create_surface_workspace(tmp_path, source, "surface_fusion", True)
    manifest_path = process_surface_workspace(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["total"] == len(manifest["outputs"]) == 7
    assert manifest["source_filename"] == "paint_16bit.tiff"
    assert len(list(manifest_path.parent.rglob("*.png"))) == 14
    for output in manifest["outputs"]:
        assert output["status"] == "completed"
        assert output["fusion"]["recipe_id"]
        assert output["native_resolution"] and output["spatial_scale"] == 1
        assert output["source_shape"] == list(image.shape)
        assert output["source_filename"] == "paint_16bit.tiff"
        response = np.load(manifest_path.parent / output["response_file"])
        assert response.dtype == np.float32 and response.shape == image.shape
        assert np.isfinite(response).all()
        for path in output["images"].values():
            decoded = cv2.imdecode(np.fromfile(manifest_path.parent / path, np.uint8), 1)
            assert decoded.shape[:2] == image.shape
    np.testing.assert_array_equal(source.image, image)


def test_fusion_screening_uses_selected_file_without_basler(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        window.experiment.profile.setCurrentIndex(
            window.experiment.profile.findData("surface_fusion")
        )
        assert window.experiment.case_count.text() == "7"
        assert window.experiment.start_capture.text() == "Run Fusion screening"
        assert window.experiment.settle_ms.isHidden()
        window.fusion_lab.set_image(synthetic(), {"source_filename": "chosen.png"})
        assert window.camera_worker is None
        window._run_capture_experiment(
            {"profile": "surface_fusion", "parent": str(tmp_path), "save_response": False}
        )
        deadline = time.monotonic() + 20
        while window._experiment_processing_worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        assert window._experiment_processing_worker is None
        assert "completed" in window.experiment.status.text()
        assert window.experiment.open_results.isEnabled()
        assert not window.pattern_output.output.isVisible()
    finally:
        window.close()
        app.processEvents()
