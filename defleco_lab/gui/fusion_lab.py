"""GPixel recipes presented with the existing Surface viewers and bounded worker."""

from PySide6 import QtWidgets

from defleco_lab.processing.surface.fusion import fusion_registry, recipe_metadata

from .surface_lab import SurfaceLab


class FusionLab(SurfaceLab):
    def __init__(self, parent=None):
        super().__init__(parent, method_registry=fusion_registry, scale_parameters=True)
        self.family.hide()
        self.controls_form.labelForField(self.family).hide()
        self.controls_form.labelForField(self.method).setText("Recipe")
        self.purpose = QtWidgets.QLabel()
        self.purpose.setWordWrap(True)
        self.controls_form.insertRow(1, self.purpose)
        self.scale_hint = QtWidgets.QLabel(
            "Spatial parameters are native-image pixels. Preview adjusts filter scales "
            "to its reduced raster; native ROI/full resolution preserves the requested scale. "
            "K03/K07 exact float median can be slow on large native images: try a native ROI first. "
            "Fused responses are experimental, not defect confidence."
        )
        self.scale_hint.setWordWrap(True)
        self.controls_form.insertRow(2, self.scale_hint)
        self.reset_button = QtWidgets.QPushButton("Restore recommended defaults")
        self.reset_button.clicked.connect(self.restore_defaults)
        self.controls_form.insertRow(3, self.reset_button)
        self._update_purpose()

    def _method_changed(self, *_):
        super()._method_changed()
        if hasattr(self, "purpose"):
            self._update_purpose()

    def _update_purpose(self):
        info = next(i for i in self.method_registry.infos() if i.id == self.method.currentData())
        self.purpose.setText(info.description)

    def restore_defaults(self):
        self._method_changed()

    def _completed(self):
        previous = self.snapshot
        super()._completed()
        if self.snapshot is not None and self.snapshot is not previous:
            clipped = self.snapshot[2].diagnostics.get("enhanced_clip_fraction", 0.0)
            self.status.setText(f"{self.status.text()} | Clipped pixels: {float(clipped):.1%}")

    def experiment_metadata(self, method_id, parameters):
        return recipe_metadata(method_id, parameters)
