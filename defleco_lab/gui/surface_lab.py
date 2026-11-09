"""Independent single-frame surface inspection workspace."""

from __future__ import annotations

from copy import deepcopy
from threading import Condition

import cv2
import numpy as np
from PySide6 import QtCore, QtWidgets

from defleco_lab.processing.surface import normalize_input, surface_registry

from .image_viewer import ImageViewer
from .parameter_panel import ParameterPanel
from .surface_io import export_surface, load_preset, render_surface, save_preset


class SurfaceWorker(QtCore.QThread):
    ready = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.condition = Condition()
        self.pending = None
        self.completed = None
        self.notified = False
        self.stopping = False
        self.generation = 0
        self.signature = None

    def submit(self, request):
        with self.condition:
            signature = (
                request["method_id"],
                repr(request["parameters"]),
                repr(request["settings"]),
                repr(request["roi"]),
                request["native"],
            )
            if signature != self.signature:
                self.generation += 1
                self.signature = signature
                self.completed = None
            request["generation"] = self.generation
            self.pending = request
            self.condition.notify()

    def invalidate(self):
        with self.condition:
            self.generation += 1
            self.pending = None
            self.completed = None
            self.signature = None

    def take(self):
        with self.condition:
            result, self.completed = self.completed, None
            self.notified = False
            return result

    def stop(self):
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify()
        return self.wait(30000)

    def run(self):
        while True:
            with self.condition:
                while self.pending is None and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    return
                request, self.pending = self.pending, None
            try:
                source = request["source"]
                normalized, source_input = normalize_input(source)
                request["source_metadata"]["source_input"] = source_input
                original_display = np.rint(np.clip(normalized, 0, 1) * 255).astype(np.uint8)
                if original_display.ndim == 3:
                    original_display = cv2.cvtColor(original_display, cv2.COLOR_BGR2RGB)
                roi = request["roi"]
                image = (
                    source
                    if roi is None
                    else source[
                        roi["y"] : roi["y"] + roi["height"], roi["x"] : roi["x"] + roi["width"]
                    ]
                )
                normalized = (
                    normalized
                    if roi is None
                    else normalized[
                        roi["y"] : roi["y"] + roi["height"], roi["x"] : roi["x"] + roi["width"]
                    ]
                )
                if not request["native"] and image.shape[0] * image.shape[1] > 1_000_000:
                    factor = (1_000_000 / (image.shape[0] * image.shape[1])) ** 0.5
                    size = (
                        max(1, int(image.shape[1] * factor)),
                        max(1, int(image.shape[0] * factor)),
                    )
                    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                    normalized = cv2.resize(normalized, size, interpolation=cv2.INTER_AREA)
                image = np.ascontiguousarray(image)
                result = surface_registry.create(
                    request["method_id"], **request["parameters"]
                ).process([normalized], keep_intermediates=True)
                result.intermediates["_source_normalized"] = normalized
                rendered = render_surface(image, result, request["settings"])
                canvas = (
                    cv2.cvtColor(original_display, cv2.COLOR_GRAY2RGB)
                    if original_display.ndim == 2
                    else original_display.copy()
                )
                if roi is None:
                    rendered = cv2.resize(
                        rendered,
                        (canvas.shape[1], canvas.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    x, y, width, height = (roi[k] for k in ("x", "y", "width", "height"))
                    canvas[y : y + height, x : x + width] = cv2.resize(
                        rendered, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                    rendered = canvas
                payload = (request, image, result, rendered, original_display, None)
            except Exception as exc:  # noqa: BLE001
                payload = (request, None, None, None, None, str(exc))
            with self.condition:
                if request["generation"] != self.generation or self.stopping:
                    continue
                self.completed = payload
                notify = not self.notified
                self.notified = True
            if notify:
                self.ready.emit()


class SurfaceLab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.source_metadata = {}
        self.snapshot = None
        self.active = False
        self._updating = False
        self.native_inflight = False
        self.pinned = False
        self.pinned_roi = None
        self.worker = SurfaceWorker(self)
        self.worker.ready.connect(self._completed)
        self.worker.start()
        layout = QtWidgets.QHBoxLayout(self)
        self.original, self.result_view = ImageViewer(), ImageViewer()
        views = QtWidgets.QSplitter()
        for title, viewer in (("Original", self.original), ("Result", self.result_view)):
            panel = QtWidgets.QWidget()
            column = QtWidgets.QVBoxLayout(panel)
            column.setContentsMargins(0, 0, 0, 0)
            column.addWidget(QtWidgets.QLabel(title))
            column.addWidget(viewer, 1)
            views.addWidget(panel)
        self.original.viewTransformChanged.connect(self.result_view.apply_view_transform)
        self.result_view.viewTransformChanged.connect(self.original.apply_view_transform)
        for axis in ("horizontalScrollBar", "verticalScrollBar"):
            a, b = getattr(self.original, axis)(), getattr(self.result_view, axis)()
            a.valueChanged.connect(b.setValue)
            b.valueChanged.connect(a.setValue)
        layout.addWidget(views, 1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(270)
        scroll.setMaximumWidth(340)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(controls)
        self.family, self.method = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for combo in (self.family, self.method):
            combo.setMinimumContentsLength(12)
            combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
        self.family.addItems(sorted({info.category for info in surface_registry.infos()}))
        form.addRow("Family", self.family)
        form.addRow("Method", self.method)
        self.params = ParameterPanel()
        form.addRow(self.params)
        self.mode, self.range_mode, self.polarity = (
            QtWidgets.QComboBox(),
            QtWidgets.QComboBox(),
            QtWidgets.QComboBox(),
        )
        self.mode.addItems(["enhanced", "grayscale", "heatmap", "overlay"])
        self.range_mode.addItems(["auto", "fixed", "percentile"])
        self.polarity.addItems(["signed", "bright", "dark", "absolute"])
        form.addRow("Output", self.mode)
        form.addRow("Range", self.range_mode)
        self.low, self.high = self._number(-1e9, 1e9, 0), self._number(-1e9, 1e9, 1)
        self.percentile_low, self.percentile_high = (
            self._number(0, 100, 1),
            self._number(0, 100, 99),
        )
        self.opacity = self._number(0, 1, 0.5)
        for label, widget in [
            ("Low", self.low),
            ("High", self.high),
            ("Percentile low", self.percentile_low),
            ("Percentile high", self.percentile_high),
            ("Polarity", self.polarity),
            ("Opacity", self.opacity),
        ]:
            form.addRow(label, widget)
        for label, callback in [
            ("Open image", self.open_image),
            ("Draw ROI", lambda: self.original.begin_roi("analysis")),
            ("Clear ROI", lambda: self.original.clear_rois("analysis")),
            ("Process native ROI", self.process_roi),
            ("Process full resolution", self.process_full),
            ("Resume preview", self.resume_preview),
            ("Fit views", self.fit_views),
            ("Actual size", self.actual_size),
            ("Export result", self.export),
            ("Save preset", self.save_preset),
            ("Load preset", self.load_preset),
        ]:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            form.addRow(button)
        self.save_response = QtWidgets.QCheckBox("Export float response (.npy)")
        self.save_response.setChecked(True)
        form.addRow(self.save_response)
        self.status = QtWidgets.QLabel("Open an image or select a source")
        self.status.setWordWrap(True)
        form.addRow(self.status)
        scroll.setWidget(controls)
        layout.addWidget(scroll)
        self.family.currentIndexChanged.connect(self._family_changed)
        self.method.currentIndexChanged.connect(self._method_changed)
        self.params.parametersChanged.connect(self._changed)
        for widget in (self.mode, self.range_mode, self.polarity):
            widget.currentIndexChanged.connect(self._changed)
        for widget in (
            self.low,
            self.high,
            self.percentile_low,
            self.percentile_high,
            self.opacity,
        ):
            widget.valueChanged.connect(self._changed)
        self._family_changed()

    @staticmethod
    def _number(low, high, value):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(4)
        widget.setValue(value)
        return widget

    def _family_changed(self, *_):
        self._updating = True
        self.method.clear()
        for info in surface_registry.infos():
            if info.category == self.family.currentText():
                self.method.addItem(info.name, info.id)
        self._updating = False
        self._method_changed()

    def _method_changed(self, *_):
        if self._updating or not self.method.currentData():
            return
        info = next(i for i in surface_registry.infos() if i.id == self.method.currentData())
        self._updating = True
        self.params.set_schema(info.parameters)
        self.method.setToolTip(f"{info.description}\n{info.limitations}")
        self._updating = False
        self._changed()

    def settings(self):
        return {
            "mode": self.mode.currentText(),
            "range": self.range_mode.currentText(),
            "low": self.low.value(),
            "high": self.high.value(),
            "percentile_low": self.percentile_low.value(),
            "percentile_high": self.percentile_high.value(),
            "polarity": self.polarity.currentText(),
            "opacity": self.opacity.value(),
        }

    def _changed(self, *_):
        if not self._updating:
            self.snapshot = None
            self.submit(self.pinned, self.pinned_roi if self.pinned else None)

    def set_active(self, active):
        self.active = active
        if not active:
            self.worker.invalidate()
            self.native_inflight = False
        elif self.source is not None:
            self.submit(self.pinned, self.pinned_roi if self.pinned else None)

    def set_image(self, image, metadata=None):
        if self.pinned:
            return
        shape_changed = self.source is None or self.source.shape[:2] != image.shape[:2]
        self.source = np.array(image, copy=True)
        self.source_metadata = dict(metadata or {})
        if shape_changed:
            self.original.clear_rois("analysis")
        if not self.native_inflight:
            self.submit()

    @staticmethod
    def _display_source(image):
        normalized, _ = normalize_input(image)
        display = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(display, cv2.COLOR_BGR2RGB) if display.ndim == 3 else display

    def submit(self, native=False, roi=None):
        if not self.active or self.source is None or not self.method.currentData():
            return
        self.native_inflight = native
        self.worker.submit(
            {
                "source": self.source,
                "source_metadata": deepcopy(self.source_metadata),
                "source_shape": list(self.source.shape),
                "roi": deepcopy(roi),
                "native": native,
                "method_id": self.method.currentData(),
                "parameters": deepcopy(self.params.values()),
                "settings": self.settings(),
            }
        )
        self.status.setText("Processing…")

    def process_roi(self):
        rois = [r for r in self.original.rois("analysis") if r["enabled"]]
        if len(rois) != 1:
            self.status.setText("Draw exactly one enabled analysis ROI.")
            return
        r = rois[0]
        self.pinned = True
        self.pinned_roi = {k: r[k] for k in ("x", "y", "width", "height")}
        self.snapshot = None
        self.submit(True, self.pinned_roi)

    def process_full(self):
        self.pinned = True
        self.pinned_roi = None
        self.snapshot = None
        self.submit(True)

    def resume_preview(self):
        self.pinned = False
        self.native_inflight = False
        self.snapshot = None
        self.submit()

    def fit_views(self):
        self.original.fit_to_window()
        self.result_view.fit_to_window()

    def actual_size(self):
        self.original.actual_size()
        self.result_view.actual_size()

    def _completed(self):
        payload = self.worker.take()
        if payload is None or not self.active:
            return
        request, image, result, rendered, original_display, error = payload
        self.native_inflight = False
        if error:
            self.status.setText(f"Processing error: {error}")
            return
        self.snapshot = (request, image, result)
        self.original.set_array(original_display)
        self.result_view.set_array(rendered)
        self.status.setText(
            f"{image.shape[1]} × {image.shape[0]}{' native' if request['native'] else ' preview'}"
        )

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.tif *.tiff *.bmp *.jpg *.jpeg);;All files (*)"
        )
        if not path:
            return
        try:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Cannot decode image")
            if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
                raise ValueError("Expected grayscale or three-channel BGR image")
            if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
                raise ValueError("Unsupported image data type")
        except (OSError, ValueError, cv2.error) as exc:
            self.status.setText(f"Cannot open image: {exc}")
            return
        self.worker.invalidate()
        self.native_inflight = False
        self.pinned = False
        self.snapshot = None
        self.set_image(image, {"source_filename": path})

    def export(self):
        if self.snapshot is None:
            self.status.setText("No current result to export")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export result", "", "PNG (*.png);;TIFF (*.tif *.tiff)"
        )
        if not path:
            return
        request, image, result = self.snapshot
        metadata = dict(
            request["source_metadata"],
            source_shape=request["source_shape"],
            roi=request["roi"],
            processing_shape=list(image.shape),
            native_resolution=request["native"],
            method_id=request["method_id"],
            parameters=request["parameters"],
            display_settings=request["settings"],
        )
        try:
            export_surface(
                path,
                image,
                result,
                metadata,
                request["settings"],
                save_response=self.save_response.isChecked(),
            )
            self.status.setText(f"Exported: {path}")
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Export error: {exc}")

    def save_preset(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save preset", "", "JSON (*.json)")
        if path:
            try:
                save_preset(
                    path,
                    {
                        "method_id": self.method.currentData(),
                        "parameters": self.params.values(),
                        "display_settings": self.settings(),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self.status.setText(f"Preset error: {exc}")

    def load_preset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load preset", "", "JSON (*.json)")
        if not path:
            return
        try:
            config = load_preset(path)
            info = next(i for i in surface_registry.infos() if i.id == config["method_id"])
            params, settings = config["parameters"], config["display_settings"]
            if set(params) != set(info.parameters):
                raise ValueError("Preset parameters do not match method schema")
            surface_registry.create(info.id, **params)
            self._updating = True
            self.family.setCurrentText(info.category)
            self._family_changed()
            self._updating = True
            self.method.setCurrentIndex(self.method.findData(info.id))
            self._updating = False
            self._method_changed()
            self._updating = True
            self.params.set_values(params)
            for key, widget in [
                ("mode", self.mode),
                ("range", self.range_mode),
                ("polarity", self.polarity),
            ]:
                if widget.findText(settings[key]) < 0:
                    raise ValueError(f"Invalid {key}")
                widget.setCurrentText(settings[key])
            for key in ("low", "high", "percentile_low", "percentile_high", "opacity"):
                getattr(self, key).setValue(float(settings[key]))
            self._updating = False
            self.submit()
        except Exception as exc:  # noqa: BLE001
            self._updating = False
            self.status.setText(f"Preset error: {exc}")

    def shutdown(self):
        return self.worker.stop()
