"""Typed, metadata-driven processing parameter editors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from PySide6 import QtCore, QtWidgets


@dataclass(frozen=True, slots=True)
class NormalizedSpec:
    kind: str
    default: Any
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    decimals: int | None = None
    units: str = ""
    choices: tuple[Any, ...] = ()
    tooltip: str = ""
    item_type: str | None = None
    visible_if: dict[str, Any] | None = None


class ListEditor(QtWidgets.QLineEdit):
    """Compact editor accepting JSON arrays or comma-separated values."""

    def __init__(self, spec: NormalizedSpec) -> None:
        super().__init__()
        self.spec = spec
        self.set_value(spec.default)
        self.editingFinished.connect(self.validate)

    def set_value(self, value: Any) -> None:
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        self.setText(", ".join(str(item) for item in values))
        self.validate()

    def value(self) -> list[Any]:
        text = self.text().strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("value must be a list")
            values = parsed
        else:
            values = [part.strip() for part in text.split(",") if part.strip()]
        return [_convert_list_item(item, self.spec.item_type, self.spec.default) for item in values]

    def validate(self) -> bool:
        try:
            self.value()
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.setProperty("invalid", True)
            self.setToolTip(f"Invalid list: {exc}")
            self.style().unpolish(self)
            self.style().polish(self)
            return False
        self.setProperty("invalid", False)
        self.setToolTip(self.spec.tooltip)
        self.style().unpolish(self)
        self.style().polish(self)
        return True


class ParameterPanel(QtWidgets.QWidget):
    parametersChanged = QtCore.Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.form = QtWidgets.QFormLayout(self)
        self.editors: dict[str, QtWidgets.QWidget] = {}
        self.specs: dict[str, NormalizedSpec] = {}

    def set_schema(self, schema: dict[str, Any]) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.editors.clear()
        self.specs = {name: _normalize_spec(spec) for name, spec in schema.items()}
        for name, spec in self.specs.items():
            editor = self._create_editor(spec)
            label = name.replace("_", " ").title()
            if spec.units:
                label += f" ({spec.units})"
            tooltip = spec.tooltip or f"Method parameter: {name}"
            editor.setToolTip(tooltip)
            self.form.addRow(label, editor)
            label_widget = self.form.labelForField(editor)
            if label_widget is not None:
                label_widget.setToolTip(tooltip)
            self.editors[name] = editor
        self._update_visibility()

    def _create_editor(self, spec: NormalizedSpec) -> QtWidgets.QWidget:
        if spec.kind == "bool":
            editor = QtWidgets.QCheckBox()
            editor.setChecked(bool(spec.default))
            editor.toggled.connect(self._emit)
            return editor
        if spec.kind == "enum":
            editor = QtWidgets.QComboBox()
            for choice in spec.choices:
                editor.addItem(str(choice), choice)
            index = editor.findData(spec.default)
            editor.setCurrentIndex(max(0, index))
            editor.currentIndexChanged.connect(self._emit)
            return editor
        if spec.kind == "int":
            editor = QtWidgets.QSpinBox()
            editor.setRange(
                int(spec.minimum if spec.minimum is not None else -2_147_483_647),
                int(spec.maximum if spec.maximum is not None else 2_147_483_647),
            )
            editor.setSingleStep(int(spec.step or 1))
            editor.setValue(int(spec.default))
            editor.valueChanged.connect(self._emit)
            return editor
        if spec.kind == "float":
            editor = QtWidgets.QDoubleSpinBox()
            editor.setRange(
                float(spec.minimum if spec.minimum is not None else -1e12),
                float(spec.maximum if spec.maximum is not None else 1e12),
            )
            editor.setSingleStep(float(spec.step or 0.1))
            editor.setDecimals(spec.decimals if spec.decimals is not None else 4)
            editor.setValue(float(spec.default))
            editor.valueChanged.connect(self._emit)
            return editor
        if spec.kind == "list":
            editor = ListEditor(spec)
            editor.editingFinished.connect(self._emit)
            return editor
        editor = QtWidgets.QLineEdit(str(spec.default))
        editor.editingFinished.connect(self._emit)
        return editor

    def values(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, editor in self.editors.items():
            if isinstance(editor, QtWidgets.QCheckBox):
                result[name] = editor.isChecked()
            elif isinstance(editor, QtWidgets.QComboBox):
                result[name] = editor.currentData()
            elif isinstance(editor, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                result[name] = editor.value()
            elif isinstance(editor, ListEditor):
                try:
                    result[name] = editor.value()
                except (TypeError, ValueError, json.JSONDecodeError):
                    result[name] = self.specs[name].default
            elif isinstance(editor, QtWidgets.QLineEdit):
                result[name] = editor.text()
        return result

    def set_values(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            editor = self.editors.get(name)
            if editor is None:
                continue
            blocker = QtCore.QSignalBlocker(editor)
            if isinstance(editor, QtWidgets.QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, QtWidgets.QComboBox):
                index = editor.findData(value)
                if index >= 0:
                    editor.setCurrentIndex(index)
            elif isinstance(editor, QtWidgets.QSpinBox):
                editor.setValue(int(value))
            elif isinstance(editor, QtWidgets.QDoubleSpinBox):
                editor.setValue(float(value))
            elif isinstance(editor, ListEditor):
                editor.set_value(value)
            elif isinstance(editor, QtWidgets.QLineEdit):
                editor.setText(str(value))
            del blocker
        self._emit()

    def _emit(self, *_args: object) -> None:
        self._update_visibility()
        self.parametersChanged.emit(self.values())

    def _update_visibility(self) -> None:
        current = self.values()
        for name, editor in self.editors.items():
            condition = self.specs[name].visible_if
            visible = True
            if condition:
                for dependency, expected in condition.items():
                    actual = current.get(dependency)
                    allowed = expected if isinstance(expected, (list, tuple, set)) else [expected]
                    if actual not in allowed:
                        visible = False
                        break
            editor.setVisible(visible)
            label = self.form.labelForField(editor)
            if label is not None:
                label.setVisible(visible)


def _normalize_spec(raw: Any) -> NormalizedSpec:
    if not isinstance(raw, dict) or "default" not in raw:
        default = raw
        return NormalizedSpec(_infer_kind(default), default)
    default = raw.get("default")
    choices = tuple(raw.get("choices", ()))
    kind = raw.get("type") or ("enum" if choices else _infer_kind(default))
    if isinstance(kind, type):
        kind = kind.__name__
    return NormalizedSpec(
        kind=str(kind).lower(),
        default=default,
        minimum=raw.get("minimum", raw.get("min")),
        maximum=raw.get("maximum", raw.get("max")),
        step=raw.get("step"),
        decimals=raw.get("decimals"),
        units=str(raw.get("units", "")),
        choices=choices,
        tooltip=str(raw.get("tooltip", raw.get("explanation", ""))),
        item_type=raw.get("item_type"),
        visible_if=raw.get("visible_if"),
    )


def _infer_kind(default: Any) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, (list, tuple)):
        return "list"
    return "str"


def _convert_list_item(value: Any, item_type: str | None, default: Any) -> Any:
    kind = item_type
    if kind is None and default:
        kind = _infer_kind(default[0])
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "bool":
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().casefold()
        if normalized not in {"true", "false"}:
            raise ValueError(f"{value!r} is not a boolean")
        return normalized == "true"
    return str(value)
