"""Collect the R3 panoramic retrieval dataset from Habitat-GS.

Set ``PIVOTNAV_HABITAT_ROOT`` to the Habitat-GS checkout and
``PIVOTNAV_PPC_TRAINING_ROOT`` to the output workspace. All dataset builder
arguments are forwarded to the released R3 collector.
"""
from modules.Panoramic_Place_Compass.training.retrieval.build_dataset import main


if __name__ == "__main__":
    main()
