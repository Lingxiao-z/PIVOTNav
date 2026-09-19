# Panoramic Place Compass

This module provides panoramic VPR, relative bearing estimation, and the
LightGlue + RANSAC goal-image verifier.

Runtime weights are external. Pass their parent directory through
`--weights-root`; expected subdirectories are `r361/`, `r363/`, and
`lightglue/`. The online implementation is in `runtime/`; the reusable
LightGlue source is vendored once in `third_party/lightglue/`. Training data
is not included in this repository.
