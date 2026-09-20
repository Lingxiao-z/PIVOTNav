from argparse import Namespace
import json

import numpy as np
import torch

from modules.Navigable_Curiosity_Field.data.labels import decide_fg, fs_targets, target_offset
from modules.Navigable_Curiosity_Field.training.dataset import FGFSPairDatasetV2
from modules.Navigable_Curiosity_Field.training.train_curiosity_exploration import synthetic_smoke as curiosity_smoke
from modules.Panoramic_Place_Compass.evaluation.evaluate import evaluate_arrival, evaluate_pair, evaluate_v7
from modules.Panoramic_Place_Compass.training.train import (
    package_r361,
    package_r363,
    synthetic_smoke as compass_smoke,
    train_arrival,
    train_formal_bearing,
    train_private_bearing,
    train_r363_head,
    train_v7,
)


def test_fgfs_label_contract():
    requested = target_offset(0)
    decision = decide_fg(
        direction_index=0,
        requested_target=requested,
        snapped_target=requested,
        navigable=True,
        path_found=True,
        shortest_path_distance_m=3.0,
    )
    assert decision.fg_label
    scores, valid, unreachable = fs_targets(
        np.asarray([[5.0, np.inf] + [0.0] * 10]),
        np.asarray([[True, True] + [False] * 10]),
        np.asarray([[True, True] + [False] * 10]),
    )
    assert np.isclose(scores[0, 0], 0.75)
    assert valid[0, 0] and unreachable[0, 1]


def test_fgfs_shard_dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "shards").mkdir(parents=True)
    shard = root / "shards/scene.npz"
    images = np.zeros((2, 224, 448, 3), dtype=np.uint8)
    np.savez_compressed(
        shard,
        images=images,
        positions=np.zeros((2, 3), dtype=np.float32),
        fg_labels=np.ones((2, 12), dtype=bool),
        pair_source_ids=np.asarray([0], dtype=np.int32),
        pair_goal_ids=np.asarray([1], dtype=np.int32),
        pair_fg_labels=np.ones((1, 12), dtype=bool),
        pair_fs_scores=np.ones((1, 12), dtype=np.float32),
        pair_fs_valid_mask=np.ones((1, 12), dtype=bool),
        pair_unreachable_mask=np.zeros((1, 12), dtype=bool),
    )
    import hashlib
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    (root / "scene_records.json").write_text(json.dumps([
        {"scene_id": "scene", "label_version": "nts_inspired_habitat_gs_fgfs_v2", "shard_name": shard.name, "shard_sha256": digest, "pair_count": 1}
    ]))
    dataset = FGFSPairDatasetV2(root)
    assert len(dataset) == 1
    assert dataset[0]["source"].shape == (3, 224, 448)


def _args(data, output):
    return Namespace(data=data, output=output, device="cpu", steps=1, batch_size=1, lr=1e-4, seed=3)


def test_compass_component_checkpoint_roundtrip(tmp_path):
    pair_data = tmp_path / "pairs.npz"
    np.savez_compressed(
        pair_data,
        source_tokens=np.random.randn(1, 16, 32, 384).astype(np.float32),
        target_tokens=np.random.randn(1, 16, 32, 384).astype(np.float32),
        source_descriptor_ring=np.random.randn(1, 32, 128).astype(np.float32),
        target_descriptor_ring=np.random.randn(1, 32, 128).astype(np.float32),
        yaw_degrees=np.asarray([5.0], dtype=np.float32),
        bearing_degrees=np.asarray([20.0], dtype=np.float32),
        bearing_valid=np.asarray([True]),
        distance_bucket=np.asarray([0], dtype=np.int64),
    )
    private_checkpoint = tmp_path / "private.pt"
    formal_checkpoint = tmp_path / "formal.pt"
    r363_checkpoint = tmp_path / "r363-head.pt"
    train_private_bearing(_args(pair_data, private_checkpoint))
    train_formal_bearing(_args(pair_data, formal_checkpoint))
    train_r363_head(_args(pair_data, r363_checkpoint))
    assert evaluate_pair("private-bearing", pair_data, private_checkpoint, "cpu")["samples"] == 1
    assert evaluate_pair("formal-bearing", pair_data, formal_checkpoint, "cpu")["samples"] == 1
    assert evaluate_pair("r363-bearing", pair_data, r363_checkpoint, "cpu")["samples"] == 1

    arrival_data = tmp_path / "arrival.npz"
    np.savez_compressed(
        arrival_data,
        features=np.random.randn(1, 3, 280).astype(np.float32),
        target_arrived=np.asarray([[False, False, True]]),
        distance_m=np.asarray([[1.5, 1.1, 0.7]], dtype=np.float32),
        frame_mask=np.ones((1, 3), dtype=bool),
    )
    arrival_checkpoint = tmp_path / "arrival.pt"
    train_arrival(_args(arrival_data, arrival_checkpoint))
    assert evaluate_arrival(arrival_data, arrival_checkpoint, "cpu")["frames"] == 3

    r361_source = tmp_path / "r361-source.pt"
    torch.save({
        "encoder_system": {
            "backbone.model.blocks.11.fake": torch.ones(1),
            "backbone.model.norm.fake": torch.ones(1),
        }
    }, r361_source)
    final_r363 = tmp_path / "r363.pt"
    package_r363(Namespace(r361_checkpoint=r361_source, bearing_checkpoint=r363_checkpoint, output=final_r363))
    payload = torch.load(final_r363, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "r363_limited_unfreeze_checkpoint_v1"
    assert sorted(payload["bearing_tail"]) == ["blocks.11.fake", "norm.fake"]


def test_v7_training_roundtrip(tmp_path):
    data = tmp_path / "v7.npz"
    features = np.arange(80, dtype=np.float32).reshape(20, 4)
    target = features[:, 0] * 0.1
    np.savez_compressed(data, features=features, target_m=target)
    output = tmp_path / "v7.joblib"
    args = _args(data, output)
    args.estimators = 5
    args.max_depth = 2
    train_v7(args)
    assert evaluate_v7(data, output)["samples"] == 20


def test_synthetic_training_smokes():
    assert curiosity_smoke()["finite"]
    assert compass_smoke()["finite"]
