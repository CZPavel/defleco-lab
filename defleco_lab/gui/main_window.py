from __future__ import annotations

import json
from collections import deque
from importlib.resources import files
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from defleco_lab.camera.basler_source import discover_cameras
from defleco_lab.camera.image_sequence_source import load_image_sequence
from defleco_lab.camera.synthetic_source import SyntheticSource
from defleco_lab.processing.registry import registry
from defleco_lab.runtime.acquisition_worker import BaslerAcquisitionWorker
from defleco_lab.runtime.processing_worker import ProcessingRequest, ProcessingWorker
from defleco_lab.sessions import AsyncSessionRecorder, load_session

from .image_viewer import ImageViewer
from .parameter_panel import ParameterPanel


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Defleco LAB - Experimental Research Software")
        self.resize(1500, 900)
        self.source = SyntheticSource()
        self.history = deque(maxlen=64)
        self.replay = []
        self.replay_index = 0
        self.recorder: AsyncSessionRecorder | None = None
        self.camera_worker: BaslerAcquisitionWorker | None = None
        self.last_result = None
        self.processing_drops = 0
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._next_frame)
        self.processor = ProcessingWorker(self)
        self.processor.completed.connect(self._processing_complete)
        self.processor.failed.connect(
            lambda message: self.statusBar().showMessage(f"Processing: {message}")
        )
        self.processor.start()
        self._build_ui()
        self._load_methods()
        self.statusBar().showMessage("Ready - synthetic source - hardware NOT TESTED")
        self._next_frame()

    def _build_ui(self) -> None:
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.original = ImageViewer()
        self.processed = ImageViewer()
        self.intermediate = ImageViewer()
        self.compare_left = ImageViewer()
        self.compare_right = ImageViewer()
        self.motion_debug = ImageViewer()
        self.tabs.addTab(self.original, "Original")
        self.tabs.addTab(self.processed, "Processed")
        self.tabs.addTab(self.intermediate, "Intermediate")
        compare = QtWidgets.QSplitter()
        compare.addWidget(self.compare_left)
        compare.addWidget(self.compare_right)
        self.compare_left.viewTransformChanged.connect(self.compare_right.apply_view_transform)
        self.compare_right.viewTransformChanged.connect(self.compare_left.apply_view_transform)
        self.compare_left.horizontalScrollBar().valueChanged.connect(
            self.compare_right.horizontalScrollBar().setValue
        )
        self.compare_right.horizontalScrollBar().valueChanged.connect(
            self.compare_left.horizontalScrollBar().setValue
        )
        self.compare_left.verticalScrollBar().valueChanged.connect(
            self.compare_right.verticalScrollBar().setValue
        )
        self.compare_right.verticalScrollBar().valueChanged.connect(
            self.compare_left.verticalScrollBar().setValue
        )
        self.tabs.addTab(compare, "Compare")
        self.tabs.addTab(self.motion_debug, "Motion Debug")
        self.original.pixelHovered.connect(
            lambda x, y, v: self.statusBar().showMessage(f"Pixel x={x}, y={y}, value={v}")
        )

        toolbar = self.addToolBar("Status")
        self.state_label = QtWidgets.QLabel("FROZEN")
        self.metrics = QtWidgets.QLabel(
            "Camera 0.0 FPS | Processing 0.0 ms | Shift N/A | Q N/A | Drops 0"
        )
        toolbar.addWidget(self.state_label)
        toolbar.addSeparator()
        toolbar.addWidget(self.metrics)

        left = QtWidgets.QDockWidget("Input / Camera / Session", self)
        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["Synthetic", "Basler", "Recorded Session", "Image sequence"])
        self.source_combo.setToolTip(
            "Basler requires explicit discovery and selection; this V01 GUI defaults to hardware-free Synthetic."
        )
        form.addRow("Source", self.source_combo)
        camera_row = QtWidgets.QHBoxLayout()
        self.camera_combo = QtWidgets.QComboBox()
        discover = QtWidgets.QPushButton("Discover")
        discover.clicked.connect(self._discover_cameras)
        camera_row.addWidget(self.camera_combo)
        camera_row.addWidget(discover)
        form.addRow("Explicit camera", camera_row)
        self.camera_settings = {}
        for label, node in (
            ("Exposure us", "ExposureTime"),
            ("Gain", "Gain"),
            ("FPS", "AcquisitionFrameRate"),
            ("Width", "Width"),
            ("Height", "Height"),
            ("Offset X", "OffsetX"),
            ("Offset Y", "OffsetY"),
        ):
            editor = QtWidgets.QLineEdit()
            editor.setPlaceholderText("leave unchanged")
            editor.setToolTip(
                "Optional session-only value; applied with GenICam writability and readback checks"
            )
            form.addRow(label, editor)
            self.camera_settings[node] = editor
        self.pattern = QtWidgets.QComboBox()
        self.pattern.addItems(["fringes", "checker", "grid", "speckle"])
        self.pattern.currentTextChanged.connect(self._change_pattern)
        form.addRow("Synthetic pattern", self.pattern)
        row = QtWidgets.QHBoxLayout()
        self.live_btn = QtWidgets.QPushButton("Live")
        self.stop_btn = QtWidgets.QPushButton("Freeze")
        row.addWidget(self.live_btn)
        row.addWidget(self.stop_btn)
        form.addRow(row)
        self.live_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        roi_row = QtWidgets.QHBoxLayout()
        a = QtWidgets.QPushButton("Draw Analysis ROI")
        m = QtWidgets.QPushButton("Draw Motion ROI")
        roi_row.addWidget(a)
        roi_row.addWidget(m)
        form.addRow(roi_row)
        a.clicked.connect(lambda: self.original.begin_roi("analysis"))
        m.clicked.connect(lambda: self.original.begin_roi("motion"))
        rec_row = QtWidgets.QHBoxLayout()
        rec = QtWidgets.QPushButton("Record")
        rec_stop = QtWidgets.QPushButton("Stop record")
        rec_row.addWidget(rec)
        rec_row.addWidget(rec_stop)
        form.addRow(rec_row)
        rec.clicked.connect(self._record)
        rec_stop.clicked.connect(self._record_stop)
        load = QtWidgets.QPushButton("Load session...")
        load.clicked.connect(self._load_session)
        form.addRow(load)
        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setMaximumHeight(70)
        form.addRow("Session notes", self.notes)
        left.setWidget(panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, left)

        right = QtWidgets.QDockWidget("Method / Parameters", self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        rp = QtWidgets.QWidget()
        rform = QtWidgets.QVBoxLayout(rp)
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.currentIndexChanged.connect(self._method_changed)
        rform.addWidget(self.method_combo)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("Custom")
        self._load_presets()
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        rform.addWidget(self.preset_combo)
        options = QtWidgets.QFormLayout()
        self.stride = QtWidgets.QSpinBox()
        self.stride.setRange(1, 16)
        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(["100%", "50%", "25%"])
        self.comp = QtWidgets.QCheckBox("Enable one-axis compensation")
        self.analysis_only = QtWidgets.QCheckBox("Process enabled Analysis ROIs only")
        self.motion_axis = QtWidgets.QComboBox()
        self.motion_axis.addItems(["x", "y"])
        self.motion_mode = QtWidgets.QComboBox()
        self.motion_mode.addItems(["auto", "manual"])
        self.manual_shift = QtWidgets.QDoubleSpinBox()
        self.manual_shift.setRange(-1000, 1000)
        self.manual_shift.setDecimals(3)
        self.motion_preprocessing = QtWidgets.QComboBox()
        self.motion_preprocessing.addItems(["none", "gaussian", "gradient"])
        self.minimum_texture = QtWidgets.QDoubleSpinBox()
        self.minimum_texture.setRange(0, 255)
        self.minimum_texture.setValue(3.0)
        self.minimum_q = QtWidgets.QDoubleSpinBox()
        self.minimum_q.setRange(-1, 1)
        self.minimum_q.setDecimals(3)
        self.minimum_q.setValue(0.08)
        self.motion_deadband = QtWidgets.QDoubleSpinBox()
        self.motion_deadband.setRange(0, 100)
        self.motion_deadband.setValue(0.0)
        self.max_shift = QtWidgets.QDoubleSpinBox()
        self.max_shift.setRange(0.1, 10000)
        self.max_shift.setValue(50.0)
        options.addRow("Frame stride", self.stride)
        options.addRow("Processing scale", self.scale)
        options.addRow(self.comp)
        options.addRow(self.analysis_only)
        options.addRow("Motion axis", self.motion_axis)
        options.addRow("Motion mode", self.motion_mode)
        options.addRow("Manual px/frame", self.manual_shift)
        options.addRow("Motion preprocessing", self.motion_preprocessing)
        options.addRow("Minimum texture", self.minimum_texture)
        options.addRow("Minimum Q", self.minimum_q)
        options.addRow("Deadband px", self.motion_deadband)
        options.addRow("Maximum shift px", self.max_shift)
        rform.addLayout(options)
        self.params = ParameterPanel()
        rform.addWidget(self.params)
        apply = QtWidgets.QPushButton("Apply / Process")
        apply.clicked.connect(self._process)
        rform.addWidget(apply)
        self.help = QtWidgets.QTextBrowser()
        self.help.setMinimumHeight(240)
        rform.addWidget(self.help)
        scroll.setWidget(rp)
        right.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, right)

        transport = QtWidgets.QToolBar("Transport")
        self.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, transport)
        for text, slot in [
            ("Fit", self._fit),
            ("1:1", self._actual),
            ("Previous", lambda: self._seek(-1)),
            ("Next", lambda: self._seek(1)),
            ("Snapshot", self._snapshot),
            ("Fullscreen", self._fullscreen),
        ]:
            button = QtGui.QAction(text, self)
            button.triggered.connect(slot)
            transport.addAction(button)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._seek_absolute)
        transport.addWidget(self.slider)

    def _load_methods(self) -> None:
        self.method_combo.clear()
        for info in registry.infos():
            self.method_combo.addItem(f"{info.category}: {info.name}", info.id)
        self._method_changed()

    def _load_presets(self) -> None:
        self.presets = []
        for p in sorted(files("defleco_lab.presets").iterdir()):
            if p.name.endswith(".json"):
                data = json.loads(p.read_text(encoding="utf-8"))
                self.presets.append(data)
                self.preset_combo.addItem(data["name"])

    def _method_changed(self) -> None:
        mid = self.method_combo.currentData()
        if not mid:
            return
        info = next(i for i in registry.infos() if i.id == mid)
        self.params.set_schema(info.parameters)
        self.help.setMarkdown(
            f"### {info.name}\n\n{info.description}\n\n**Frames:** {info.required_frames}\n\n**Recommended use:** {info.recommended_use or 'Exploratory comparison.'}\n\n**Limitations:** {info.limitations or 'Scene-dependent; not a metrological result.'}\n\n**Motion compensation:** {'supported' if info.supports_motion_compensation else 'not normally required'}\n\n**References:** {', '.join(info.references) if info.references else 'See docs/references.md.'}"
        )
        self._process()

    def _preset_changed(self, index: int) -> None:
        if index <= 0:
            return
        p = self.presets[index - 1]
        idx = self.method_combo.findData(p["method"])
        if idx >= 0:
            self.method_combo.setCurrentIndex(idx)
        self.stride.setValue(int(p.get("frame_stride", 1)))
        self.scale.setCurrentText(f"{int(p.get('processing_scale', 1) * 100)}%")
        self.comp.setChecked(bool(p.get("motion_compensation", False)))
        for k, v in p.get("parameters", {}).items():
            if k in self.params.editors:
                self.params.editors[k].setText(json.dumps(v))
        self._process()

    def _change_pattern(self, name: str) -> None:
        self.source.config.pattern = name
        self.source._base = self.source._make_base()
        self.source.reset()
        self.history.clear()
        self._next_frame()

    def _start(self):
        selected_source = self.source_combo.currentText()
        self._stop()
        if selected_source == "Synthetic":
            self.replay = []
            self.history.clear()
        if selected_source == "Basler":
            self.replay = []
            descriptor = self.camera_combo.currentData()
            if descriptor is None:
                self.statusBar().showMessage("Discover and explicitly select a camera first")
                return
            temporary = {}
            for node, editor in self.camera_settings.items():
                text = editor.text().strip()
                if text:
                    temporary[node] = (
                        int(float(text))
                        if node in {"Width", "Height", "OffsetX", "OffsetY"}
                        else float(text)
                    )
            self.camera_worker = BaslerAcquisitionWorker(descriptor, temporary, self)
            self.camera_worker.frameReady.connect(self._receive_packet)
            self.camera_worker.failed.connect(
                lambda message: self.statusBar().showMessage(f"Camera: {message}")
            )
            self.camera_worker.restoreReport.connect(self._camera_restored)
            self.camera_worker.applyReport.connect(
                lambda report: self.statusBar().showMessage(f"Temporary camera readback: {report}")
            )
            self.camera_worker.start()
            self.state_label.setText("LIVE")
            return
        if selected_source == "Recorded Session":
            self.replay = []
            self._load_session()
            if not self.replay:
                return
        if selected_source == "Image sequence":
            self.replay = []
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open image sequence")
            if not folder:
                return
            try:
                self.replay = load_image_sequence(Path(folder))
            except ValueError as exc:
                self.statusBar().showMessage(str(exc))
                return
            self.replay_index = 0
            self.slider.setRange(0, len(self.replay) - 1)
            self.history.clear()
        self.state_label.setText("LIVE")
        self.timer.start()

    def _stop(self):
        self.timer.stop()
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker = None
        self.state_label.setText("FROZEN")

    def closeEvent(self, event):
        self._stop()
        try:
            self._record_stop()
        except (OSError, TimeoutError) as exc:
            self.statusBar().showMessage(f"Recorder shutdown error: {exc}")
        if not self.processor.stop():
            self.statusBar().showMessage("Processing worker did not stop; close postponed")
            event.ignore()
            return
        event.accept()

    def _next_frame(self) -> None:
        displayed_index = self.replay_index
        packet = self.replay[displayed_index] if self.replay else self.source.next_frame()
        self._receive_packet(packet)
        if self.replay:
            self.slider.blockSignals(True)
            self.slider.setValue(displayed_index)
            self.slider.blockSignals(False)
            if displayed_index >= len(self.replay) - 1:
                self.timer.stop()
                self.state_label.setText("REPLAY END")
            else:
                self.replay_index = displayed_index + 1

    @QtCore.Slot(object)
    def _receive_packet(self, packet) -> None:
        self.history.append(packet)
        self.original.set_array(packet.image)
        self.compare_left.set_array(packet.image)
        if self.recorder:
            self.recorder.append(packet)
        self._process()

    def _discover_cameras(self) -> None:
        self.camera_combo.clear()
        try:
            descriptors = discover_cameras()
        except RuntimeError as exc:
            self.statusBar().showMessage(str(exc))
            return
        for descriptor in descriptors:
            self.camera_combo.addItem(descriptor.label, descriptor)
        self.statusBar().showMessage(
            f"Discovered {len(descriptors)} camera(s); select one explicitly"
        )

    def _camera_restored(self, failures: list[str]) -> None:
        if failures:
            self.statusBar().showMessage(
                "Temporary camera settings restore warning: " + "; ".join(failures)
            )
        else:
            self.statusBar().showMessage(
                "Camera closed; temporary settings restored where applicable"
            )

    def _process(self) -> None:
        mid = self.method_combo.currentData() if hasattr(self, "method_combo") else None
        if not mid or not self.history:
            return
        stride = self.stride.value()
        parameters = self.params.values()
        parameters["stride"] = stride
        method = registry.create(mid, **parameters)
        needed = method.history_requirement()
        if len(self.history) < needed:
            return
        frames = [packet.image for packet in list(self.history)[-needed:]]
        scale = {"100%": 1.0, "50%": 0.5, "25%": 0.25}[self.scale.currentText()]
        motion_rois = [
            {k: roi[k] for k in ("x", "y", "width", "height", "name", "enabled")}
            for roi in self.original.rois("motion")
        ]
        analysis_rois = []
        if self.analysis_only.isChecked():
            analysis_rois = [
                {k: roi[k] for k in ("x", "y", "width", "height", "name", "enabled")}
                for roi in self.original.rois("analysis")
            ]
            if not analysis_rois:
                self.statusBar().showMessage("Draw and enable at least one Analysis ROI")
                return
        self.processor.submit(
            ProcessingRequest(
                method_id=mid,
                parameters=parameters,
                frames=frames,
                scale=scale,
                stride=stride,
                motion_compensation=self.comp.isChecked(),
                motion_axis=self.motion_axis.currentText(),
                motion_rois=motion_rois,
                analysis_rois=analysis_rois,
                motion_mode=self.motion_mode.currentText(),
                manual_shift_px=self.manual_shift.value(),
                motion_preprocessing=self.motion_preprocessing.currentText(),
                minimum_texture=self.minimum_texture.value(),
                minimum_q=self.minimum_q.value(),
                deadband=self.motion_deadband.value(),
                max_shift=self.max_shift.value(),
                frame_id=self.history[-1].frame_id,
            )
        )

    @QtCore.Slot(object, float, float, int, object, int)
    def _processing_complete(
        self, result, elapsed: float, scale: float, stride: int, motion, frame_id: int
    ) -> None:
        self.last_result = result
        self.processing_drops = self.processor.dropped
        for packet in reversed(self.history):
            if packet.frame_id == frame_id:
                packet.estimated_motion_px = float(motion["shift"])
                packet.motion_quality = float(motion["quality"])
                packet.cumulative_position_px = float(motion["cumulative"])
                break
        if self.recorder:
            self.recorder.update_motion(
                frame_id,
                float(motion["shift"]),
                float(motion["quality"]),
                float(motion["cumulative"]),
            )
        self.processed.set_array(result.primary, heatmap=True)
        if result.intermediates:
            self.intermediate.set_array(next(iter(result.intermediates.values())), heatmap=True)
        self.compare_right.set_array(result.primary, heatmap=True)
        if motion.get("mask") is not None:
            self.motion_debug.set_array(motion["mask"].astype(np.uint8) * 255)
        roi_summary = ""
        if motion.get("roi_results"):
            roi_summary = " | " + "; ".join(
                f"{item.roi.name}: d={item.selected_shift:.2f}px tex={item.texture:.1f} Q={item.quality:.3f} {'OK' if item.valid else item.reason}"
                for item in motion["roi_results"]
            )
        recorder_state = ""
        if self.recorder:
            recorder_state = (
                f" | Recorder q={self.recorder.queue_size} drop={self.recorder.dropped}"
            )
        self.metrics.setText(
            f"Source {self.source_combo.currentText()} | Processing {elapsed:.1f} ms | Scale {scale:.2f} | Stride {stride} | Shift {motion['shift']:.2f} px | Q {motion['quality']:.3f} | valid ROI {motion['valid']} | Processing drops {self.processing_drops}{recorder_state}{roi_summary}"
        )

    def _record(self):
        root = QtWidgets.QFileDialog.getExistingDirectory(self, "Session root")
        if root:
            self.recorder = AsyncSessionRecorder(Path(root), self.notes.toPlainText())
            self.statusBar().showMessage("Recording raw frames")

    def _record_stop(self):
        if self.recorder:
            path = self.recorder.close()
            self.recorder = None
            self.statusBar().showMessage(f"Session saved: {path}")

    def _load_session(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open recorded session")
        if path:
            self._stop()
            self.replay = load_session(Path(path))
            self.replay_index = 0
            self.slider.setRange(0, max(0, len(self.replay) - 1))
            self.history.clear()
            self.state_label.setText("REPLAY")
            self._seek_absolute(0)
            if len(self.replay) > 1:
                self.replay_index = 1

    def _seek(self, delta):
        self._seek_absolute(self.replay_index + delta)

    def _seek_absolute(self, index):
        if not self.replay:
            return
        self.replay_index = max(0, min(index, len(self.replay) - 1))
        self.history.clear()
        packet = self.replay[self.replay_index]
        self.history.append(packet)
        self.original.set_array(packet.image)
        self.compare_left.set_array(packet.image)
        self._process()

    def _fit(self):
        [
            v.fit_to_window()
            for v in (
                self.original,
                self.processed,
                self.intermediate,
                self.compare_left,
                self.compare_right,
                self.motion_debug,
            )
        ]

    def _actual(self):
        [
            v.actual_size()
            for v in (
                self.original,
                self.processed,
                self.intermediate,
                self.compare_left,
                self.compare_right,
                self.motion_debug,
            )
        ]

    def _snapshot(self):
        if self.last_result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export result", "result.png", "PNG (*.png)"
        )
        if path:
            a = np.nan_to_num(self.last_result.primary)
            a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(path, a)

    def _fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()
