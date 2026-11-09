"""Operator controls for reproducible screening experiments."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from defleco_lab.experiments import build_pattern_cases


class ExperimentPanel(QtWidgets.QGroupBox):
    captureRequested = QtCore.Signal(object)
    processRequested = QtCore.Signal(str)
    stopRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Automated screening", parent)
        form = QtWidgets.QFormLayout(self)

        intro = self.intro = QtWidgets.QLabel(
            "Captures a deterministic set of pattern states first, then processes the "
            "saved RAW frames offline. This avoids recomputing the camera acquisition "
            "for every analysis/display combination."
        )
        intro.setWordWrap(True)
        form.addRow(intro)

        self.profile = QtWidgets.QComboBox()
        self.profile.addItem("Quick screening", "quick")
        self.profile.addItem("Extended screening", "extended")
        self.profile.addItem("Surface screening", "surface")
        self.profile.addItem("Surface screening Extended", "surface_extended")
        self.profile.addItem("GPixel Surface Fusion", "surface_fusion")
        self.profile.setToolTip(
            "Quick is the recommended first pass. Extended adds more periods/angles "
            "after the geometry has been checked."
        )
        self.profile.currentIndexChanged.connect(self._update_count)

        self.case_count = QtWidgets.QLabel()
        self.settle_ms = QtWidgets.QSpinBox()
        self.settle_ms.setRange(50, 5000)
        self.settle_ms.setValue(250)
        self.settle_ms.setSingleStep(50)
        self.settle_ms.setSuffix(" ms")
        self.settle_ms.setToolTip(
            "Minimum delay after changing the display pattern before accepting a new "
            "camera frame. At ~6 FPS, 250 ms normally guarantees at least one new frame."
        )

        self.root = QtWidgets.QLineEdit()
        self.root.setPlaceholderText("Choose parent folder for experiment data")
        self.root.setToolTip(
            "A timestamped Defleco screening folder will be created here. RAW frames "
            "are preserved and processed results are written separately."
        )
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        root_row = QtWidgets.QHBoxLayout()
        root_row.addWidget(self.root)
        root_row.addWidget(browse)

        self.auto_process = QtWidgets.QCheckBox("Process captured data automatically")
        self.auto_process.setChecked(True)
        self.auto_process.setToolTip(
            "After RAW capture, run the current bounded screening recipe set offline "
            "and export grayscale/colour plus 50% overlay variants."
        )

        self.use_pattern_colors = QtWidgets.QCheckBox("Use selected pattern colours")
        self.use_pattern_colors.setToolTip(
            "Apply the dark and light colours selected in the pattern controls to every "
            "screening case. Leave unchecked for the standard screening recipes."
        )
        self.surface_npy = QtWidgets.QCheckBox("Save numerical response (.npy)")
        self.surface_npy.setToolTip("Save the unmodified float32 response for each Surface result.")

        self.start_capture = QtWidgets.QPushButton("Run pattern capture sweep")
        self.start_capture.setToolTip(
            "Requires Basler Live acquisition. The app switches to Original view during capture."
        )
        self.start_capture.clicked.connect(self._request_capture)

        self.process_existing = QtWidgets.QPushButton("Process existing capture...")
        self.process_existing.setToolTip(
            "Select a completed screening workspace containing capture_manifest.json."
        )
        self.process_existing.clicked.connect(self._request_existing)

        self.stop = QtWidgets.QPushButton("Stop")
        self.stop.clicked.connect(self.stopRequested)
        self.stop.setEnabled(False)
        self.open_results = QtWidgets.QPushButton("Open results")
        self.open_results.setEnabled(False)
        self.open_results.setToolTip(
            "Open the generated screening_results.html gallery in the default browser."
        )
        self.open_results.clicked.connect(self._open_results)
        self._results_path = ""

        buttons = QtWidgets.QGridLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.start_capture, 0, 0)
        buttons.addWidget(self.process_existing, 0, 1)
        buttons.addWidget(self.open_results, 1, 0)
        buttons.addWidget(self.stop, 1, 1)
        buttons.setColumnStretch(0, 1)
        buttons.setColumnStretch(1, 1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status = QtWidgets.QLabel("Idle")
        self.status.setWordWrap(True)

        form.addRow("Plan", self.profile)
        form.addRow("Pattern cases", self.case_count)
        form.addRow("Settle time", self.settle_ms)
        form.addRow("Experiment parent", root_row)
        form.addRow(self.use_pattern_colors)
        form.addRow(self.auto_process)
        form.addRow(self.surface_npy)
        form.addRow(buttons)
        form.addRow(self.progress)
        form.addRow(self.status)
        self._update_count()

    def config(self) -> dict:
        return {
            "profile": str(self.profile.currentData()),
            "settle_ms": int(self.settle_ms.value()),
            "parent": self.root.text().strip(),
            "auto_process": self.auto_process.isChecked(),
            "use_pattern_colors": self.use_pattern_colors.isChecked(),
            "save_response": self.surface_npy.isChecked(),
        }

    def set_running(self, running: bool) -> None:
        self.start_capture.setEnabled(not running)
        self.process_existing.setEnabled(not running)
        self.stop.setEnabled(running)
        self.profile.setEnabled(not running)

    def set_results_path(self, path: str) -> None:
        self._results_path = path
        self.open_results.setEnabled(bool(path) and Path(path).exists())

    def update_progress(self, done: int, total: int, label: str) -> None:
        total = max(1, int(total))
        self.progress.setValue(round(100 * int(done) / total))
        self.status.setText(label)

    def finished(self, message: str) -> None:
        self.set_running(False)
        self.progress.setValue(100)
        self.status.setText(message)

    def failed(self, message: str) -> None:
        self.set_running(False)
        self.status.setText(f"Error: {message}")

    def _open_results(self) -> None:
        if self._results_path and Path(self._results_path).exists():
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(Path(self._results_path).resolve()))
            )

    def _browse(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Experiment parent folder")
        if folder:
            self.root.setText(folder)

    def _request_capture(self) -> None:
        config = self.config()
        if not config["parent"]:
            self._browse()
            config = self.config()
        if config["parent"]:
            self.captureRequested.emit(config)

    def _request_existing(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open screening workspace")
        if folder and (Path(folder) / "capture_manifest.json").exists():
            self.processRequested.emit(folder)
        elif folder:
            self.failed("Selected folder does not contain capture_manifest.json")

    def _update_count(self, *_args: object) -> None:
        profile = str(self.profile.currentData())
        surface = profile in {"surface", "surface_extended", "surface_fusion"}
        self.settle_ms.setVisible(not surface)
        self.form_label(self.settle_ms).setVisible(not surface)
        self.use_pattern_colors.setVisible(not surface)
        self.auto_process.setVisible(not surface)
        self.surface_npy.setVisible(surface)
        self.start_capture.setText(
            "Run Surface screening" if surface else "Run pattern capture sweep"
        )
        if profile == "surface_fusion":
            self.start_capture.setText("Run Fusion screening")
        self.intro.setText(
            "Capture one fresh live image, then compare all surface filters at full resolution."
            if surface
            else "Captures a deterministic set of pattern states first, then processes the "
            "saved RAW frames offline. This avoids recomputing the camera acquisition "
            "for every analysis/display combination."
        )
        self.form_label(self.case_count).setText("Filter variants" if surface else "Pattern cases")
        if profile == "surface_fusion":
            self.intro.setText(
                "Run K01–K07 on the image selected in GPixel Surface Fusion at full resolution."
            )
            count = 7
        elif surface:
            from defleco_lab.processing.surface import surface_registry

            count = len(surface_registry.infos()) * (2 if profile == "surface_extended" else 1)
        else:
            count = len(build_pattern_cases(profile))
        self.case_count.setText(str(count))

    def form_label(self, widget):
        return self.layout().labelForField(widget)
