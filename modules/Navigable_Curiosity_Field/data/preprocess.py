from pathlib import Path

import numpy as np


def validate_shard(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"images", "pair_fs_scores", "pair_fg_labels"}
        missing = sorted(required - set(archive.files))
        return {"path": str(path), "valid": not missing, "missing": missing}
