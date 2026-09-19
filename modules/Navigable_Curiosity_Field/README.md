# Navigable Curiosity Field

This module contains the unified curiosity-exploration runtime. The released
model keeps its internal FG compatibility branch inside
`runtime/curiosity/curiosity_model.py` and
`runtime/curiosity/curiosity_inference.py`; FG is not exposed as a separate training or
inference task.

Runtime inputs are current ERP RGB, goal ERP RGB, and 360 OmniTrav distances.
The public output is a validity-filtered candidate-frontier score field.
Training and feature preparation live in `training/` and `data/`.
OmniTrav and OmniGuard are external runtime dependencies; the repository keeps
only the worker/client boundary in `runtime/`.
