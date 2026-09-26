from __future__ import annotations

import json
from importlib.resources import files

from defleco_lab.processing import registry


def test_all_presets_reference_known_methods_and_parameters() -> None:
    infos = {info.id: info for info in registry.infos()}
    for path in files("defleco_lab.presets").iterdir():
        if not path.name.endswith(".json"):
            continue
        preset = json.loads(path.read_text(encoding="utf-8"))
        method_id = preset["method"]
        assert method_id in infos, f"{path.name}: unknown method {method_id!r}"
        allowed = set(infos[method_id].parameters)
        supplied = set(preset.get("parameters", {}))
        unknown = supplied - allowed
        assert not unknown, f"{path.name}: unknown parameter(s) {sorted(unknown)}"
