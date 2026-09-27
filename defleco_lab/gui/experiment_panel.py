"""Operator controls for reproducible screening experiments."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from defleco_lab.experiments import build_pattern_cases


class ExperimentPanel(QtWidgets.QGroupBox):
    captureRequested = QtCore.Signal(object)
    processRequested = QtCore.Signal(str)
    stopRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Automated screening", parent)
        form = QtWidgets.QFormLayout(self)

        intro = QtWidgets.QLabel(
            "Captures a deterministic set of pattern states first, then processes the "
            "saved RAW frames offline. This avoids recomputing the camera acquisition "
            "for every analysis/display combination."
        )
        intro.setWordWrap(True)
        form.addRow(intro)

        self.profile = QtWidgets.QComboBox()
        self.profile.addItem("Quick screening", "quick")
        self.profile.addItem("Extended screening", "extended")
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

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.start_capture)
        buttons.addWidget(self.process_existing)
        buttons.addWidget(self.stop)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status = QtWidgets.QLabel("Idle")
        self.status.setWordWrap(True)

        form.addRow("Plan", self.profile)
        form.addRow("Pattern cases", self.case_count)
        form.addRow("Settle time", self.settle_ms)
        form.addRow("Experiment parent", root_row)
        form.addRow(self.auto_process)
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
        }

    def set_running(self, running: bool) -> None:
        self.start_capture.setEnabled(not running)
        self.process_existing.setEnabled(not running)
        self.stop.setEnabled(running)

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
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open screening workspace"
        )
        if folder and (Path(folder) / "capture_manifest.json").exists():
            self.processRequested.emit(folder)
        elif folder:
            self.failed("Selected folder does not contain capture_manifest.json")

    def _update_count(self, *_args: object) -> None:
        count = len(build_pattern_cases(str(self.profile.currentData())))
        self.case_count.setText(str(count))
