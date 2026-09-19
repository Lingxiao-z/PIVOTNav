from pathlib import Path

import numpy as np

from modules.Navigable_Curiosity_Field.inference import smoke_curiosity
from modules.Panoramic_Place_Compass.arrival import resolve_protocol_resource
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
