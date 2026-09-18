import numpy as np

from modules.Navigable_Curiosity_Field.inference import smoke_curiosity
from modules.Panoramic_Place_Compass.inference import smoke_compass
from modules.Topological_Belief_Cascade.inference import smoke_topology


def test_public_module_smoke_contracts():
    image = np.zeros((224, 448, 3), dtype=np.uint8)
    assert smoke_compass(image, image)["ok"]
    assert smoke_topology(image)["ok"]
    assert smoke_curiosity(image, image)["ok"]
