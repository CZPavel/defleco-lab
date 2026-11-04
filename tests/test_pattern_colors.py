import json

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from defleco_lab.gui.pattern_output import (
    PatternControlPanel,
    PatternSettings,
    render_pattern,
    render_pattern_rgb,
)


@pytest.mark.parametrize('invalid', [(1, 2), (0, 0, 256), (-1, 0, 0), (1.2, 0, 0), '#123456'])
def test_rgb888_rejects_invalid_channels(invalid):
    with pytest.raises(ValueError):
        PatternSettings(dark_color=invalid)


def test_manifest_rgb_normalization_and_legacy_defaults():
    settings = PatternSettings(dark_color=(12, 34, 56), light_color=(254, 128, 0))
    assert PatternSettings(**json.loads(json.dumps(settings.as_dict()))) == settings
    assert PatternSettings(family='rings').light_color == (255, 255, 255)


@pytest.mark.parametrize('family', ['stripes', 'checker', 'rings', 'composite', 'nested_square', 'squircle', 'spiral', 'counter_spiral', 'starburst', 'speckle', 'solid'])
@pytest.mark.parametrize('brightness', [0, 17, 50, 100])
@pytest.mark.parametrize('invert', [False, True])
def test_default_rgb_exactly_repeats_mono(family, brightness, invert):
    settings = PatternSettings(family=family, waveform='sinusoidal', brightness=brightness, invert=invert, phase_deg=31)
    mono = render_pattern(131, 97, settings)
    rgb = render_pattern_rgb(131, 97, settings)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (*mono.shape, 3)
    for channel in range(3):
        np.testing.assert_array_equal(rgb[..., channel], mono)


def test_sinusoidal_maps_unrounded_signal():
    settings = PatternSettings(waveform='sinusoidal', period_px=32, dark_color=(0, 0, 255), light_color=(255, 255, 0))
    rgb = render_pattern_rgb(129, 30, settings)
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 12
    assert np.max(np.abs(rgb[..., 0].astype(int) + rgb[..., 2].astype(int) - 255)) <= 1
    assert np.array_equal(rgb[..., 0], rgb[..., 1])


def test_color_controls_dialog_swap_reset_and_persistence(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    storage = tmp_path / 'colors.ini'
    original = QtCore.QSettings
    monkeypatch.setattr(QtCore, 'QSettings', lambda *args: original(str(storage), original.Format.IniFormat))
    panel = PatternControlPanel()
    panel.dark_color.set_color(QtGui.QColor('#123456'))
    monkeypatch.setattr(QtWidgets.QColorDialog, 'getColor', lambda *args: QtGui.QColor('#abcdef'))
    panel.light_color.custom.click()
    assert panel.settings().dark_color == (18, 52, 86)
    assert panel.settings().light_color == (171, 205, 239)
    assert panel.output.effective_settings.light_color == (171, 205, 239)
    restored = PatternControlPanel()
    assert restored.settings().light_color == panel.settings().light_color
    assert restored.settings().dark_color == panel.settings().dark_color
    panel.swap_colors.click()
    assert panel.settings().dark_color == (171, 205, 239)
    panel.reset_colors.click()
    assert panel.settings().dark_color == (0, 0, 0)
    assert panel.settings().light_color == (255, 255, 255)
    monkeypatch.setattr(QtWidgets.QColorDialog, 'getColor', lambda *args: QtGui.QColor())
    panel.light_color.custom.click()
    assert panel.settings().light_color == (255, 255, 255)
    panel.shutdown()
    restored.shutdown()
    assert app is not None
