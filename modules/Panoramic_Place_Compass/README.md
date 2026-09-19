# Panoramic Place Compass

This module provides panoramic VPR, relative bearing estimation, and the
LightGlue + RANSAC goal-image verifier.

Runtime weights are external. Pass their parent directory through
`--weights-root`; expected subdirectories are `r361/`, `r363/`, and
`lightglue/`. Online code is grouped into `localization.py`, `geometry.py`, and
`arrival.py`; checkpoint-compatible networks are grouped by responsibility in
`model/`, and LightGlue is vendored once in `third_party/`.
DINOv2 is an external dependency: set `PIVOTNAV_DINOV2_REPO` to the pinned
official checkout and `PIVOTNAV_DINOV2_WEIGHT` to its checkpoint. The unified
training entry is under `training/`; historical revision scripts are not part
of the release. Training data is not included.
