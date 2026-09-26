from __future__ import annotations

import json
from importlib.resources import files

from defleco_lab.processing.registry import load_builtin_methods, registry


def test_all_presets_reference_valid_method_parameters() -> None:
    load_builtin_methods()
    infos = {info.id: info for info in registry.infos()}
    for path in files("defleco_lab.presets").iterdir():
        if not path.name.endswith(".json"):
            continue
        preset = json.loads(path.read_text(encoding="utf-8"))
        assert preset["method"] in infos, f"{path.name}: unknown method"
        allowed = set(infos[preset["method"]].parameters)
        supplied = set(preset.get("parameters", {}))
        unknown = supplied - allowed
        assert not unknown, f"{path.name}: unknown parameters {sorted(unknown)}"
