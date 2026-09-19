from pathlib import Path

import numpy as np

from modules.Navigable_Curiosity_Field.inference import smoke_curiosity
from modules.Panoramic_Place_Compass.arrival import (
    resolve_arrival_model_path,
    resolve_protocol_resource,
)
from modules.Panoramic_Place_Compass.geometry import resolve_protocol_path
from modules.Panoramic_Place_Compass.inference import smoke_compass
from modules.Topological_Belief_Cascade.inference import smoke_topology


def test_public_module_smoke_contracts():
    image = np.zeros((224, 448, 3), dtype=np.uint8)
    assert smoke_compass(image, image)["ok"]
    assert smoke_topology(image)["ok"]
    assert smoke_curiosity(image, image)["ok"]


def test_external_protocol_paths_are_portable():
    protocol = Path("/tmp/pivotnav-test/protocol.json")
    assert resolve_protocol_path(protocol, "assets/input.json") == (protocol.parent / "assets/input.json").resolve()
    assert resolve_protocol_path(protocol, "/var/tmp/input.json") == Path("/var/tmp/input.json")
    assert resolve_protocol_resource("models/arrival.joblib").name == "arrival.joblib"


def test_public_sources_have_no_deleted_runtime_imports():
    inference_source = Path("modules/Panoramic_Place_Compass/inference.py").read_text()
    assert ".runtime.evidence" not in inference_source


def test_relative_arrival_model_override_is_not_resolved_against_cwd(monkeypatch, tmp_path):
    protocol = tmp_path / "protocol" / "arrival_protocol.json"
    protocol.parent.mkdir()
    monkeypatch.chdir(tmp_path)
    resolved = resolve_arrival_model_path("models/arrival.joblib", protocol, "fallback.joblib")
    assert resolved == (protocol.parent / "models/arrival.joblib").resolve()
