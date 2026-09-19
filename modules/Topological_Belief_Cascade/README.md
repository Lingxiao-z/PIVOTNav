# Topological Belief Cascade

This module owns regular-node memory, candidate-frontier lifecycle, original
global VPR+BPL graph updates, and the optional accelerated backend. The
online implementation is kept in `runtime/`; `reference_graph.py` contains
the original and accelerated graph backends behind one interface.

The default is `topology_backend: original`. Use `accelerated` only for the
online-index benchmark or an explicitly selected navigation run.
