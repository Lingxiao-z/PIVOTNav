# Topological Belief Cascade

This module owns regular-node memory, candidate-frontier lifecycle, original
global VPR+BPL graph updates, and the optional accelerated backend. The
online implementation is split by responsibility: `topology.py` owns graph
state, candidate-frontier events, scheduling, and graph backends, while
`localization.py` owns known-node localization and Habitat action conversion.

The default is `topology_backend: original`. Use `accelerated` only for the
online-index benchmark or an explicitly selected navigation run.
