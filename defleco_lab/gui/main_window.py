from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from defleco_lab.camera.basler_source import discover_cameras
from defleco_lab.camera.image_sequence_source import load_image_sequence
from defleco_lab.camera.synthetic_source import SyntheticSource
from defleco_lab.experiments import ScreeningProcessingWorker, build_pattern_cases
from defleco_lab.processing.registry import registry
from defleco_lab.runtime.acquisition_worker import BaslerAcquisitionWorker
from defleco_lab.runtime.processing_worker import ProcessingRequest, ProcessingWorker
from defleco_lab.sessions import AsyncSessionRecorder, load_session, replay_history

from .experiment_panel import ExperimentPanel
from .image_viewer import ImageViewer
from .parameter_panel import ParameterPanel
from .pattern_output import PatternControlPanel
from .pipeline_panel import PostprocessingPanel, PreprocessingPanel
from .visualization import VisualizationPanel, VisualizationSettings, VisualizationTransform


class MainWindow(QtWidgets.QMainWindow):
    LIVE_HISTORY_CAPACITY = 64
    DISPLAY_MAX_PIXELS = 1_500_000

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Defleco LAB - Experimental Research Software")
        self.resize(1500, 900)
        self.source = SyntheticSource()
        # Keep live memory bounded. At 2448x2048 Mono8, 512 frames exceed 2.5 GB.
        self.history = deque(maxlen=self.LIVE_HISTORY_CAPACITY)
        self.replay = []
        self.replay_index = 0
        self.recorder: AsyncSessionRecorder | None = None
        self.camera_worker: BaslerAcquisitionWorker | None = None
        self.last_result = None
        self.last_original: np.ndarray | None = None
        self.last_motion_mask: np.ndarray | None = None
        self._last_display_scale = 1.0
        self.compare_reference: np.ndarray | None = None
        self.compare_reference_mask: np.ndarray | None = None
        self.visualization_transform = VisualizationTransform()
        self.visualization_settings = VisualizationSettings()
        self.processing_drops = 0
        self.received_fps = 0.0
        self.processed_fps = 0.0
        self._source_times: deque[float] = deque(maxlen=120)
        self._processed_times: deque[float] = deque(maxlen=120)
        self.camera_metrics: dict[str, float | int] = {}
        self._experiment_cases = []
        self._experiment_records: list[dict] = []
        self._experiment_index = -1
        self._experiment_root: Path | None = None
        self._experiment_baseline_frame_id: int | None = None
        self._experiment_case_started = 0.0
        self._experiment_settle_ms = 250
        self._experiment_auto_process = True
        self._experiment_profile = "quick"
        self._experiment_processing_worker: ScreeningProcessingWorker | None = None
        self.experiment_timer = QtCore.QTimer(self)
        self.experiment_timer.setInterval(30)
        self.experiment_timer.timeout.connect(self._experiment_poll)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._next_frame)
        self.camera_timer = QtCore.QTimer(self)
        self.camera_timer.setInterval(33)
        self.camera_timer.timeout.connect(self._poll_camera_frames)
        self.visualization_timer = QtCore.QTimer(self)
        self.visualization_timer.setSingleShot(True)
        self.visualization_timer.setInterval(60)
        self.visualization_timer.timeout.connect(self._render_visualization)
        self.processor = ProcessingWorker(self)
        self.processor.resultAvailable.connect(self._consume_processing_result)
        self.processor.failed.connect(
            lambda message: self.statusBar().showMessage(f"Processing: {message}")
        )
        self.processor.start()
        self._build_ui()
        self._load_methods()
        self.statusBar().showMessage("Ready - synthetic source - camera idle")
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
        self.tabs.currentChanged.connect(self._active_view_changed)
        self.original.pixelHovered.connect(
            lambda x, y, v: self.statusBar().showMessage(f"Pixel x={x}, y={y}, value={v}")
        )

        toolbar = self.addToolBar("Status")
        self.state_label = QtWidgets.QLabel("FROZEN")
        self.metrics = QtWidgets.QLabel(
            "Source 0.0 FPS | Processed 0.0 FPS | Processing 0.0 ms | Drops 0"
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
        self.camera_status = QtWidgets.QLabel(
            "Camera image acquisition only. Configure exposure, gain, ROI and other "
            "camera parameters in Basler pylon Viewer."
        )
        self.camera_status.setWordWrap(True)
        self.camera_status.setToolTip(
            "Defleco LAB intentionally does not duplicate pylon camera-parameter controls."
        )
        form.addRow(self.camera_status)
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

        pattern_dock = QtWidgets.QDockWidget("Pattern generator", self)
        self.pattern_output = PatternControlPanel()
        pattern_dock.setWidget(self.pattern_output)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, pattern_dock)

        experiment_dock = QtWidgets.QDockWidget("Screening experiment", self)
        self.experiment = ExperimentPanel()
        self.experiment.captureRequested.connect(self._run_capture_experiment)
        self.experiment.processRequested.connect(self._start_offline_processing)
        self.experiment.stopRequested.connect(self._stop_experiment)
        experiment_dock.setWidget(self.experiment)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, experiment_dock)

        right = QtWidgets.QDockWidget("Analysis pipeline", self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        rp = QtWidgets.QWidget()
        rform = QtWidgets.QVBoxLayout(rp)

        self.preprocessing = PreprocessingPanel()
        rform.addWidget(self.preprocessing)

        method_group = QtWidgets.QGroupBox("2. Analysis method")
        method_layout = QtWidgets.QVBoxLayout(method_group)
        method_hint = QtWidgets.QLabel(
            "Only parameters used by the selected method are shown. Hover a control "
            "for a short explanation."
        )
        method_hint.setWordWrap(True)
        method_layout.addWidget(method_hint)

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.currentIndexChanged.connect(self._method_changed)
        method_layout.addWidget(self.method_combo)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("Custom")
        self._load_presets()
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        method_layout.addWidget(self.preset_combo)

        self.method_options = QtWidgets.QFormLayout()
        self.stride = QtWidgets.QSpinBox()
        self.stride.setRange(1, 16)
        self.stride.setToolTip(
            "Frame spacing used only by multi-frame methods. 1 means adjacent frames."
        )
        self.analysis_only = QtWidgets.QCheckBox("Process enabled Analysis ROIs only")
        self.analysis_only.setToolTip(
            "Restrict numerical output to manually drawn Analysis ROIs."
        )
        self.method_options.addRow("Frame stride", self.stride)
        self.method_options.addRow(self.analysis_only)
        method_layout.addLayout(self.method_options)

        self.params = ParameterPanel()
        method_layout.addWidget(self.params)
        rform.addWidget(method_group)

        self.motion_group = QtWidgets.QGroupBox("Motion compensation (advanced)")
        motion_form = QtWidgets.QFormLayout(self.motion_group)
        motion_hint = QtWidgets.QLabel(
            "Use only when the vehicle/object moves between frames. For the current "
            "static-car experiments this can stay disabled."
        )
        motion_hint.setWordWrap(True)
        motion_form.addRow(motion_hint)
        self.comp = QtWidgets.QCheckBox("Enable one-axis compensation")
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
        motion_form.addRow(self.comp)
        motion_form.addRow("Motion axis", self.motion_axis)
        motion_form.addRow("Motion mode", self.motion_mode)
        motion_form.addRow("Manual px/frame", self.manual_shift)
        motion_form.addRow("Motion preprocessing", self.motion_preprocessing)
        motion_form.addRow("Minimum texture", self.minimum_texture)
        motion_form.addRow("Minimum Q", self.minimum_q)
        motion_form.addRow("Deadband px", self.motion_deadband)
        motion_form.addRow("Maximum shift px", self.max_shift)
        reset_motion = QtWidgets.QPushButton("Reset motion position")
        reset_motion.clicked.connect(self._reset_motion_position)
        motion_form.addRow(reset_motion)
        rform.addWidget(self.motion_group)

        self.postprocessing = PostprocessingPanel()
        rform.addWidget(self.postprocessing)

        self.visualization = VisualizationPanel()
        self.visualization.settingsChanged.connect(self._visualization_changed)
        self.visualization.rangeReset.connect(self.visualization_transform.reset_range)
        self.visualization.compareReferenceRequested.connect(self._set_compare_reference)
        rform.addWidget(self.visualization)

        apply = QtWidgets.QPushButton("Apply / Process current frame")
        apply.setToolTip(
            "Recalculate the current frozen/replay frame. During live acquisition "
            "new frames automatically use the current settings."
        )
        apply.clicked.connect(self._process)
        rform.addWidget(apply)

        help_group = QtWidgets.QGroupBox("Help for selected method")
        help_layout = QtWidgets.QVBoxLayout(help_group)
        self.help = QtWidgets.QTextBrowser()
        self.help.setMinimumHeight(220)
        help_layout.addWidget(self.help)
        rform.addWidget(help_group)
        rform.addStretch(1)

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
            ("Play / Pause", self._toggle_replay),
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
        self.params.set_schema(
            {key: value for key, value in info.parameters.items() if key != "stride"}
        )
        self.help.setMarkdown(
            f"### {info.name}\n\n{info.description}\n\n"
            f"**Frames:** {info.required_frames}\n\n"
            f"**Recommended use:** {info.recommended_use or 'Exploratory comparison.'}\n\n"
            f"**Limitations:** {info.limitations or 'Scene-dependent; not a metrological result.'}\n\n"
            f"**Motion compensation:** {'supported' if info.supports_motion_compensation else 'not normally required'}\n\n"
            "**Tip:** Start from default parameters. Change one family of settings at a time "
            "and use recorded frames when comparing methods.\n\n"
            f"**References:** {', '.join(info.references) if info.references else 'See docs/references.md.'}"
        )
        multi_frame = info.required_frames > 1 or "stride" in info.parameters
        self.stride.setVisible(multi_frame)
        stride_label = self.method_options.labelForField(self.stride)
        if stride_label is not None:
            stride_label.setVisible(multi_frame)
        self.motion_group.setVisible(info.supports_motion_compensation)
        if self.replay:
            self._seek_absolute(self.slider.value())
        else:
            self._process()

    def _preset_changed(self, index: int) -> None:
        if index <= 0:
            return
        p = self.presets[index - 1]
        idx = self.method_combo.findData(p["method"])
        if idx >= 0:
            self.method_combo.setCurrentIndex(idx)
        self.stride.setValue(int(p.get("frame_stride", 1)))
        self.preprocessing.set_processing_scale(float(p.get("processing_scale", 1)))
        self.comp.setChecked(bool(p.get("motion_compensation", False)))
        if "motion_mode" in p:
            self.motion_mode.setCurrentText(str(p["motion_mode"]))
        if "axis" in p:
            self.motion_axis.setCurrentText(str(p["axis"]))
        self.params.set_values(p.get("parameters", {}))
        self._process()

    def _change_pattern(self, name: str) -> None:
        self.source.config.pattern = name
        self.source._base = self.source._make_base()
        self.source.reset()
        self.history.clear()
        self.compare_reference = None
        self.compare_reference_mask = None
        self.visualization_transform.reset_range()
        self.processor.reset_state()
        self._source_times.clear()
        self._processed_times.clear()
        self.received_fps = 0.0
        self.processed_fps = 0.0
        self.camera_metrics = {}
        self._next_frame()

    def _start(self):
        selected_source = self.source_combo.currentText()
        self._stop()
        self.compare_reference = None
        self.compare_reference_mask = None
        self.visualization_transform.reset_range()
        self.processor.reset_state()
        if selected_source == "Synthetic":
            self.replay = []
            self.history.clear()
        if selected_source == "Basler":
            self.replay = []
            descriptor = self.camera_combo.currentData()
            if descriptor is None:
                self.statusBar().showMessage("Discover and explicitly select a camera first")
                return
            # Camera parameters are intentionally configured in pylon Viewer.
            # Defleco LAB only owns acquisition for the experiment.
            self.camera_worker = BaslerAcquisitionWorker(descriptor, {}, self)
            self.camera_worker.failed.connect(self._camera_failed)
            self.camera_worker.metricsReady.connect(self._camera_metrics)
            self.camera_worker.restoreReport.connect(self._camera_restored)
            self.camera_worker.applyReport.connect(
                lambda report: self.statusBar().showMessage(f"Temporary camera readback: {report}")
            )
            self.camera_worker.start()
            self.camera_timer.start()
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
        self.camera_timer.stop()
        if self.camera_worker is not None:
            worker = self.camera_worker
            if worker.stop():
                self.camera_worker = None
            else:
                self.state_label.setText("ERROR")
                self.statusBar().showMessage(
                    "Camera worker did not stop within 5 s; camera state restoration is not confirmed"
                )
                return False
        self.state_label.setText("FROZEN")
        return True

    def closeEvent(self, event):
        self._stop_experiment()
        self.pattern_output.shutdown()
        if not self._stop():
            self.statusBar().showMessage(
                "Camera worker did not stop cleanly; close postponed to protect camera state"
            )
            event.ignore()
            return
        try:
            self._record_stop()
        except (OSError, TimeoutError) as exc:
            self.statusBar().showMessage(f"Recorder shutdown error: {exc}")
        if not self.processor.stop():
            self.statusBar().showMessage("Processing worker did not stop; close postponed")
            event.ignore()
            return
        event.accept()

    @QtCore.Slot(object)
    def _run_capture_experiment(self, config: dict) -> None:
        if self.source_combo.currentText() != "Basler":
            self.experiment.failed("Select Basler as Source and start Live acquisition first.")
            return
        if self.camera_worker is None or not self.camera_worker.isRunning():
            self.experiment.failed("Basler camera is not running. Press Live first.")
            return
        if not self.history:
            self.experiment.failed("No camera frame is available yet.")
            return

        parent_text = str(config.get("parent", "")).strip()
        if not parent_text:
            self.experiment.failed("Choose an experiment parent folder.")
            return
        parent = Path(parent_text).expanduser()
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        root = parent / f"Defleco_screening_{stamp}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "raw").mkdir()

        self._experiment_profile = str(config.get("profile", "quick"))
        self._experiment_cases = build_pattern_cases(self._experiment_profile)
        self._experiment_records = []
        self._experiment_index = -1
        self._experiment_root = root
        self._experiment_settle_ms = max(50, int(config.get("settle_ms", 250)))
        self._experiment_auto_process = bool(config.get("auto_process", True))
        self.tabs.setCurrentIndex(0)
        self.experiment.set_running(True)
        self.experiment.progress.setValue(0)
        self._write_capture_manifest(status="capturing")
        self._experiment_next_case()

    def _experiment_next_case(self) -> None:
        if not self._experiment_cases or self._experiment_root is None:
            return
        self._experiment_index += 1
        if self._experiment_index >= len(self._experiment_cases):
            self._finish_capture_experiment()
            return

        case = self._experiment_cases[self._experiment_index]
        self.pattern_output.output.set_settings(case.settings)
        if not self.pattern_output.output.isVisible():
            screen = self.pattern_output.screen.currentData()
            self.pattern_output.output.show_on_screen(0 if screen is None else int(screen))

        # Do not choose the capture baseline until the requested display-settle
        # interval has elapsed. Then require one strictly newer camera frame.
        # This prevents a pre-settle frame that arrived during the delay from being
        # accepted as the case image.
        self._experiment_baseline_frame_id = None
        self._experiment_case_started = perf_counter()
        self.experiment.update_progress(
            self._experiment_index,
            len(self._experiment_cases),
            f"Settling pattern {self._experiment_index + 1}/{len(self._experiment_cases)}: "
            f"{case.case_id}",
        )
        self.experiment_timer.start()

    def _experiment_poll(self) -> None:
        if self._experiment_root is None or self._experiment_index < 0:
            self.experiment_timer.stop()
            return
        if not self.history:
            return

        elapsed_ms = (perf_counter() - self._experiment_case_started) * 1000.0
        if elapsed_ms < self._experiment_settle_ms:
            return

        packet = self.history[-1]
        if self._experiment_baseline_frame_id is None:
            self._experiment_baseline_frame_id = packet.frame_id
            self.experiment.status.setText(
                f"Pattern settled; waiting for a fresh camera frame: "
                f"{self._experiment_cases[self._experiment_index].case_id}"
            )
            return

        if packet.frame_id == self._experiment_baseline_frame_id:
            if elapsed_ms > max(5000.0, self._experiment_settle_ms + 3000.0):
                self.experiment_timer.stop()
                self.experiment.failed("Timed out waiting for a fresh post-settle camera frame.")
                self._write_capture_manifest(status="error")
            return

        self.experiment_timer.stop()
        case = self._experiment_cases[self._experiment_index]
        raw_name = f"raw/{case.case_id}.png"
        raw_path = self._experiment_root / raw_name
        if not cv2.imwrite(str(raw_path), packet.image):
            self.experiment.failed(f"Could not save {raw_path}")
            self._write_capture_manifest(status="error")
            return

        record = case.as_dict()
        record["raw_file"] = raw_name
        record["frame"] = packet.public_metadata()
        record["capture_delay_ms"] = round(elapsed_ms, 3)
        self._experiment_records.append(record)
        self._write_capture_manifest(status="capturing")
        self._experiment_next_case()

    def _write_capture_manifest(self, status: str) -> None:
        if self._experiment_root is None:
            return
        screens = QtGui.QGuiApplication.screens()
        screen_index = self.pattern_output.screen.currentData()
        display = None
        if screens:
            index = 0 if screen_index is None else max(0, min(int(screen_index), len(screens) - 1))
            screen = screens[index]
            geometry = screen.geometry()
            display = {
                "index": index,
                "name": screen.name(),
                "width": geometry.width(),
                "height": geometry.height(),
                "refresh_hz_reported": round(float(screen.refreshRate()), 3),
            }
        payload = {
            "version": 1,
            "status": status,
            "profile": self._experiment_profile,
            "settle_ms": self._experiment_settle_ms,
            "display": display,
            "captured_count": len(self._experiment_records),
            "planned_count": len(self._experiment_cases),
            "captures": self._experiment_records,
        }
        (self._experiment_root / "capture_manifest.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _finish_capture_experiment(self) -> None:
        self.experiment_timer.stop()
        self.pattern_output.output.hide_output()
        self._write_capture_manifest(status="captured")
        root = self._experiment_root
        self.experiment.update_progress(
            len(self._experiment_records),
            max(1, len(self._experiment_cases)),
            f"Captured {len(self._experiment_records)} RAW pattern cases.",
        )
        if root is not None and self._experiment_auto_process:
            self._start_offline_processing(str(root))
        elif root is not None:
            self.experiment.finished(f"Capture complete: {root}")

    @QtCore.Slot(str)
    def _start_offline_processing(self, folder: str) -> None:
        root = Path(folder)
        if not (root / "capture_manifest.json").exists():
            self.experiment.failed("capture_manifest.json was not found.")
            return
        if (
            self._experiment_processing_worker is not None
            and self._experiment_processing_worker.isRunning()
        ):
            self.experiment.failed("A screening processing job is already running.")
            return

        self.experiment.set_running(True)
        worker = ScreeningProcessingWorker(root, self)
        self._experiment_processing_worker = worker
        worker.progressChanged.connect(self.experiment.update_progress)
        worker.completed.connect(self._experiment_processing_complete)
        worker.cancelled.connect(self._experiment_processing_cancelled)
        worker.failed.connect(self._experiment_processing_failed)
        worker.start()

    @QtCore.Slot(str)
    def _experiment_processing_complete(self, manifest: str) -> None:
        gallery = Path(manifest).parent / "screening_results.html"
        self.experiment.set_results_path(str(gallery))
        self.experiment.finished(f"Screening results ready: {gallery}")
        self.statusBar().showMessage("Automated screening processing complete")
        self._experiment_processing_worker = None

    @QtCore.Slot()
    def _experiment_processing_cancelled(self) -> None:
        self.experiment.finished("Offline processing cancelled; captured RAW data were preserved.")
        self.statusBar().showMessage("Screening processing cancelled")
        self._experiment_processing_worker = None

    @QtCore.Slot(str)
    def _experiment_processing_failed(self, message: str) -> None:
        self.experiment.failed(message)
        self.statusBar().showMessage(f"Screening processing: {message}")
        self._experiment_processing_worker = None

    def _stop_experiment(self) -> None:
        if self.experiment_timer.isActive():
            self.experiment_timer.stop()
            self._write_capture_manifest(status="cancelled")
            self.pattern_output.output.hide_output()
            self.experiment.failed("Capture cancelled; already saved RAW cases were preserved.")
        worker = self._experiment_processing_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            self.experiment.status.setText("Stopping offline processing...")
        self._experiment_cases = []
        self._experiment_index = -1

    def _poll_camera_frames(self) -> None:
        """Drain the bounded camera mailbox and process only the newest display frame."""
        worker = self.camera_worker
        if worker is None:
            self.camera_timer.stop()
            return
        packets = worker.drain_frames()
        if not packets:
            return
        for packet in packets:
            self.history.append(packet)
            if self.recorder:
                self.recorder.append(packet)
        latest = packets[-1]
        self.last_original = latest.image
        if self.tabs.currentIndex() == 0:
            self.original.set_array(latest.image)
        else:
            self._process()

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
        now = perf_counter()
        self._source_times.append(now)
        if len(self._source_times) > 1:
            self.received_fps = (len(self._source_times) - 1) / max(
                self._source_times[-1] - self._source_times[0], 1e-9
            )
        self.history.append(packet)
        self.last_original = packet.image
        if self.tabs.currentIndex() == 0:
            self.original.set_array(packet.image)
        if self.recorder:
            self.recorder.append(packet)
        if self.tabs.currentIndex() != 0:
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

    @QtCore.Slot(str)
    def _camera_failed(self, message: str) -> None:
        self.state_label.setText("ERROR")
        self.statusBar().showMessage(f"Camera: {message}")

    @QtCore.Slot(object)
    def _camera_metrics(self, metrics: dict[str, float | int]) -> None:
        self.camera_metrics = dict(metrics)
        if self.source_combo.currentText() == "Basler":
            self.received_fps = float(metrics.get("received_fps", 0.0))
            descriptor = self.camera_combo.currentData()
            label = descriptor.label if descriptor is not None else "Basler"
            self.camera_status.setText(
                f"{label}\nAcquisition: {self.received_fps:.2f} FPS | "
                f"buffer drops: {int(metrics.get('buffer_drops', 0))}\n"
                "Camera parameters are managed in pylon Viewer."
            )
        if self.tabs.currentIndex() == 0:
            self._show_original_metrics()

    def _process(self) -> None:
        mid = self.method_combo.currentData() if hasattr(self, "method_combo") else None
        if not mid or not self.history:
            return
        stride = self.stride.value()
        parameters = self.params.values()
        info = next(item for item in registry.infos() if item.id == mid)
        if "stride" in info.parameters:
            parameters["stride"] = stride
        method = registry.create(mid, **parameters)
        needed = method.history_requirement()
        if not self.replay and needed > self.LIVE_HISTORY_CAPACITY:
            self.last_result = None
            self._clear_processed_views()
            self.statusBar().showMessage(
                f"Method needs {needed} frames, but live history is bounded to "
                f"{self.LIVE_HISTORY_CAPACITY}; reduce window/stride or use replay"
            )
            return
        if len(self.history) < needed:
            self.last_result = None
            self._clear_processed_views()
            self.statusBar().showMessage(
                f"Insufficient history: {len(self.history)}/{needed} frames"
            )
            return
        frames = [packet.image for packet in list(self.history)[-needed:]]
        scale = self.preprocessing.processing_scale()
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
                preprocessing=self.preprocessing.values(),
                postprocessing=self.postprocessing.values(),
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
                motion_frames=list(self.history) if self.comp.isChecked() else None,
                keep_intermediates=self.tabs.currentIndex() == 2,
            )
        )

    def _reset_motion_position(self) -> None:
        self.processor.reset_state()
        self.statusBar().showMessage("Motion position reset to 0 px")

    @QtCore.Slot()
    def _consume_processing_result(self) -> None:
        payload = self.processor.take_latest_result()
        if payload is not None:
            self._processing_complete(*payload)

    def _processing_complete(
        self, result, elapsed: float, scale: float, stride: int, motion, frame_id: int
    ) -> None:
        self.last_result = result
        now = perf_counter()
        self._processed_times.append(now)
        if len(self._processed_times) > 1:
            self.processed_fps = (len(self._processed_times) - 1) / max(
                self._processed_times[-1] - self._processed_times[0], 1e-9
            )
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
        self.last_motion_mask = motion.get("mask")
        self._schedule_visualization()
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
            f"{self.source_combo.currentText()} {self.received_fps:.1f} FPS"
            f" | Processed {self.processed_fps:.1f} FPS | Processing {elapsed:.1f} ms"
            f" | Scale {scale:.2f} | Stride {stride} | Shift {motion['shift']:.2f} px"
            f" | Position {motion['cumulative']:.2f} px | Q {motion['quality']:.3f}"
            f" | valid Motion ROI {motion['valid']} | Processing drops {self.processing_drops}"
            f" | Result drops {self.processor.result_dropped}"
            f" | Cam-buffer drops {int(self.camera_metrics.get('buffer_drops', 0))}"
            f" | Display x{self._last_display_scale:.2f}"
            f"{recorder_state}{roi_summary}"
        )

    def _show_original_metrics(self) -> None:
        self.metrics.setText(
            f"{self.source_combo.currentText()} {self.received_fps:.1f} FPS"
            " | Original view | Processing idle"
            f" | Cam-buffer drops {int(self.camera_metrics.get('buffer_drops', 0))}"
        )

    @QtCore.Slot(object)
    def _visualization_changed(self, settings: VisualizationSettings) -> None:
        self.visualization_settings = settings
        self._schedule_visualization()

    @QtCore.Slot(int)
    def _active_view_changed(self, index: int) -> None:
        if index == 0:
            if self.last_original is not None:
                self.original.set_array(self.last_original)
            self._show_original_metrics()
            return

        # Original is deliberately a camera/recording baseline. Processing starts
        # only when a processed/debug view is requested.
        if self.history:
            self._process()
        else:
            self._schedule_visualization()

    def _schedule_visualization(self) -> None:
        if self.tabs.currentIndex() == 0:
            return
        self.visualization_timer.start()

    def _display_inputs(
        self,
        response: np.ndarray,
        valid_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        source = np.asarray(response)
        height, width = source.shape[:2]
        pixels = height * width
        if pixels <= self.DISPLAY_MAX_PIXELS:
            self._last_display_scale = 1.0
            return source, valid_mask
        scale = float(np.sqrt(self.DISPLAY_MAX_PIXELS / pixels))
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        preview = cv2.resize(source, target, interpolation=cv2.INTER_AREA)
        mask = None
        if valid_mask is not None:
            mask = cv2.resize(
                np.asarray(valid_mask, dtype=np.uint8),
                target,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        self._last_display_scale = scale
        return preview, mask

    def _render_response(
        self,
        response: np.ndarray,
        valid_mask: np.ndarray | None,
    ):
        preview, preview_mask = self._display_inputs(response, valid_mask)
        return self.visualization_transform.render(
            preview,
            self.visualization_settings,
            self.last_original,
            preview_mask,
        )

    def _render_visualization(self) -> None:
        if self.last_result is None:
            return
        index = self.tabs.currentIndex()
        if index == 0:
            return
        if index == 1:
            primary = self._render_response(
                self.last_result.primary, self.last_result.valid_mask
            )
            self.processed.set_array(primary.image)
            return
        if index == 2:
            if not self.last_result.intermediates:
                return
            intermediate = self._render_response(
                next(iter(self.last_result.intermediates.values())),
                self.last_result.valid_mask,
            )
            self.intermediate.set_array(intermediate.image)
            return
        if index == 3:
            primary = self._render_response(
                self.last_result.primary, self.last_result.valid_mask
            )
            self.compare_right.set_array(primary.image)
            if self.compare_reference is not None:
                reference = self._render_response(
                    self.compare_reference, self.compare_reference_mask
                )
                self.compare_left.set_array(reference.image)
            elif self.last_original is not None:
                preview, _ = self._display_inputs(self.last_original, None)
                self.compare_left.set_array(preview)
            return
        if index == 4 and self.last_motion_mask is not None:
            preview, _ = self._display_inputs(
                self.last_motion_mask.astype(np.uint8) * 255, None
            )
            self.motion_debug.set_array(preview)

    def _set_compare_reference(self) -> None:
        if self.last_result is None:
            self.statusBar().showMessage("Process a response before setting a compare reference")
            return
        self.compare_reference = np.asarray(self.last_result.primary).copy()
        self.compare_reference_mask = (
            None
            if self.last_result.valid_mask is None
            else np.asarray(self.last_result.valid_mask, dtype=bool).copy()
        )
        if not self.visualization.lock_range.isChecked():
            self.visualization.lock_range.setChecked(True)
        self.visualization_transform.reset_range()
        self._render_visualization()
        self.statusBar().showMessage("Compare reference stored with a shared display range")

    def _record(self):
        root = QtWidgets.QFileDialog.getExistingDirectory(self, "Session root")
        if root:
            self.recorder = AsyncSessionRecorder(Path(root), self.notes.toPlainText())
            self.statusBar().showMessage("Recording raw frames")

    def _record_stop(self):
        if self.recorder:
            recorder = self.recorder
            path = recorder.close()
            self.recorder = None
            if recorder.dropped:
                self.statusBar().showMessage(
                    f"Session saved: {path} | WARNING: recorder dropped {recorder.dropped} item(s)"
                )
            else:
                self.statusBar().showMessage(f"Session saved: {path} | recorder drops: 0")

    def _load_session(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open recorded session")
        if path:
            self._stop()
            self.source_combo.setCurrentText("Recorded Session")
            self.replay = load_session(Path(path))
            self.replay_index = 0
            self.slider.setRange(0, max(0, len(self.replay) - 1))
            self.history.clear()
            self.compare_reference = None
            self.compare_reference_mask = None
            self.visualization_transform.reset_range()
            self.processor.reset_state()
            self.state_label.setText("REPLAY")
            self._seek_absolute(0)

    def _seek(self, delta):
        self._seek_absolute(self.slider.value() + delta)

    def _toggle_replay(self) -> None:
        if not self.replay:
            self.statusBar().showMessage("Load a recorded session before replay")
            return
        if self.timer.isActive():
            self.timer.stop()
            self.state_label.setText("FROZEN")
        else:
            if (
                self.replay_index >= len(self.replay) - 1
                or self.slider.value() >= len(self.replay) - 1
            ):
                self.replay_index = 0
                self.history.clear()
                self.processor.reset_state()
            else:
                self.replay_index = self.slider.value() + 1
            self.state_label.setText("REPLAY")
            self.timer.start()

    def _seek_absolute(self, index):
        if not self.replay:
            return
        self.replay_index = max(0, min(index, len(self.replay) - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(self.replay_index)
        self.slider.blockSignals(False)
        self.processor.reset_state()
        self.history.clear()
        packet = self.replay[self.replay_index]
        self.last_original = packet.image
        self.original.set_array(packet.image)
        self.compare_left.set_array(packet.image)
        mid = self.method_combo.currentData()
        parameters = self.params.values()
        if mid:
            info = next(item for item in registry.infos() if item.id == mid)
            if "stride" in info.parameters:
                parameters["stride"] = self.stride.value()
        needed = registry.create(mid, **parameters).history_requirement() if mid else 1
        reconstructed = replay_history(self.replay, self.replay_index, needed)
        if not reconstructed:
            self.last_result = None
            self._clear_processed_views()
            self.statusBar().showMessage(
                f"Insufficient history: frame {self.replay_index} needs {needed} frames"
            )
            return
        self.history.extend(reconstructed)
        self._process()

    def _clear_processed_views(self) -> None:
        for viewer in (self.processed, self.intermediate, self.compare_right, self.motion_debug):
            viewer.clear_image()

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
