from __future__ import annotations

import json
from typing import Any

from PySide6 import QtCore, QtWidgets


class ParameterPanel(QtWidgets.QWidget):
    parametersChanged = QtCore.Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.form = QtWidgets.QFormLayout(self)
        self.editors: dict[str, QtWidgets.QWidget] = {}

    def set_schema(self, schema: dict[str, Any]) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.editors.clear()
        for name, spec in schema.items():
            default = spec.get("default") if isinstance(spec, dict) else spec
            editor = QtWidgets.QLineEdit(
                json.dumps(default) if isinstance(default, (list, dict)) else str(default)
            )
            editor.setToolTip(f"Method parameter: {name}")
            editor.editingFinished.connect(self._emit)
            self.form.addRow(name.replace("_", " ").title(), editor)
            self.editors[name] = editor

    def values(self) -> dict[str, Any]:
        result = {}
        for name, editor in self.editors.items():
            text = editor.text().strip()
            try:
                result[name] = json.loads(
                    text.lower() if text.lower() in ("true", "false", "null") else text
                )
            except (ValueError, json.JSONDecodeError):
                result[name] = text
        return result

    def _emit(self) -> None:
        self.parametersChanged.emit(self.values())
