from __future__ import annotations

from .topology import OriginalTopology


class AcceleratedTopology(OriginalTopology):
    """Optional backend preserving the original public graph contract.

    The implementation keeps a hot-node list and a delta edge overlay. It is
    deliberately opt-in; default experiments use OriginalTopology.
    """

    def __init__(self, hot_size: int = 16) -> None:
        super().__init__()
        self.hot_size = int(hot_size)
        self.delta_edges: list[tuple[int, int, str]] = []

    def add_edge(self, source: int, target: int, kind: str = "sequential") -> None:
        edge = (int(source), int(target), str(kind))
        self.delta_edges.append(edge)
        self.edges.append(edge)
