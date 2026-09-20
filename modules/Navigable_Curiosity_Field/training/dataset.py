from __future__ import annotations

import bisect
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_ARRAYS = {
    "images", "positions", "fg_labels", "pair_source_ids", "pair_goal_ids",
    "pair_fg_labels", "pair_fs_scores", "pair_fs_valid_mask", "pair_unreachable_mask",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FGFSPairDatasetV2(Dataset):
    """Read only v2 joint-label shards and fail closed on v1 schemas."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str | None = None,
        scene_records: str | Path = "scene_records.json",
        cache_size: int = 2,
        verify_shards: bool = True,
    ) -> None:
        self.root = Path(dataset_root)
        self.cache_size = cache_size
        records_path = self.root / scene_records
        records = json.loads(records_path.read_text())
        if not isinstance(records, list):
            raise ValueError("scene_records.json must contain a list")
        self.manifest_sha256 = file_sha256(records_path)
        self.entries = []
        cumulative = 0
        for record in records:
            if record.get("label_version") != "nts_inspired_habitat_gs_fgfs_v2":
                raise ValueError(f"ineligible label version for {record.get('scene_id')}")
            if split is not None and record.get("split") != split:
                continue
            path = self.root / "shards" / record["shard_name"]
            if verify_shards and file_sha256(path) != record["shard_sha256"]:
                raise ValueError(f"shard SHA256 mismatch: {path}")
            pair_count = int(record["pair_count"])
            cumulative += pair_count
            self.entries.append({**record, "path": path, "cumulative": cumulative})
        self.cumulative = [entry["cumulative"] for entry in self.entries]
        self.block_ranges = [
            (self.cumulative[index - 1] if index else 0, value)
            for index, value in enumerate(self.cumulative)
        ]
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def _load(self, scene_index: int) -> dict[str, np.ndarray]:
        if scene_index in self._cache:
            self._cache.move_to_end(scene_index)
            return self._cache[scene_index]
        with np.load(self.entries[scene_index]["path"], allow_pickle=False) as archive:
            missing = REQUIRED_ARRAYS - set(archive.files)
            if missing:
                raise ValueError(f"v2 shard missing arrays: {sorted(missing)}")
            shard = {name: archive[name] for name in archive.files}
        self._cache[scene_index] = shard
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return shard

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        scene_index = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[scene_index - 1] if scene_index else 0
        pair_index = index - previous
        shard = self._load(scene_index)
        source = int(shard["pair_source_ids"][pair_index])
        goal = int(shard["pair_goal_ids"][pair_index])
        fs_valid = shard["pair_fs_valid_mask"][pair_index].astype(bool, copy=True)
        unreachable = shard["pair_unreachable_mask"][pair_index].astype(bool, copy=True)
        if np.any(fs_valid & unreachable):
            raise ValueError("unreachable directions cannot be FS-valid")
        return {
            "source": torch.from_numpy(shard["images"][source].copy()).permute(2, 0, 1).float().div_(255.0),
            "goal": torch.from_numpy(shard["images"][goal].copy()).permute(2, 0, 1).float().div_(255.0),
            "fg_labels": torch.from_numpy(shard["pair_fg_labels"][pair_index].astype(np.float32, copy=True)),
            "fs_scores": torch.from_numpy(shard["pair_fs_scores"][pair_index].astype(np.float32, copy=True)),
            "fs_valid_mask": torch.from_numpy(fs_valid),
            "unreachable_mask": torch.from_numpy(unreachable),
            "scene_id": self.entries[scene_index]["scene_id"],
            "scene_index": torch.tensor(scene_index),
            "pair_index": torch.tensor(pair_index),
            "source_id": torch.tensor(source),
            "goal_id": torch.tensor(goal),
        }
