from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


PROJECT = Path(os.environ.get("PIVOTNAV_HABITAT_ROOT", "")).expanduser().resolve()
ROOT = Path(os.environ.get("PIVOTNAV_PPC_TRAINING_ROOT", Path.cwd())).expanduser().resolve()
SEED = 20260801


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(scene_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{scene_id}".encode()).hexdigest()


def main() -> None:
    train_root = PROJECT / "data/scene_datasets/gs_scenes/train"
    val_root = PROJECT / "data/scene_datasets/gs_scenes/val"
    all_training = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    final_test = sorted(path.name for path in val_root.iterdir() if path.is_dir())
    ranked = sorted(all_training, key=stable_key)
    development = sorted(ranked[:14])
    training = sorted(ranked[14:])
    if len(training) != 96 or len(development) != 14 or len(final_test) != 19:
        raise RuntimeError("unexpected Habitat-GS scene inventory")
    if set(training) & set(development) or (set(training) | set(development)) != set(all_training):
        raise RuntimeError("train/development split is invalid")
    if (set(training) | set(development)) & set(final_test):
        raise RuntimeError("official train and validation scenes overlap")

    inherited = {}

    payload = {
        "schema_version": "3.0.0",
        "protocol": "r3_spatial_multi_positive_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "splits": {"train": training, "development": development, "test": final_test},
        "counts": {"train": len(training), "development": len(development), "test": len(final_test)},
        "sources": {"train": "Habitat-GS official train", "development": "Habitat-GS official train", "test": "Habitat-GS official val"},
        "scene_disjoint": True,
        "r2_evidence": inherited,
        "test_policy_zh": "Test 使用 19 个官方 validation scenes。可在训练前只渲染并冻结数据与哈希，但在 Phase A/B/C checkpoint、代码、阈值和 Development 结论冻结前，禁止模型读取或评测 Test。最终 Test 只允许执行一次。",
        "note_zh": "110 个 training scenes 已全部被历史 R2/V1 使用，无法从中构造未访问 Test；R3 因此使用官方 validation split 作为一次性 Test，并如实记录。",
    }
    manifest = ROOT / "data/manifests/v3_scene_split_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "created_at": payload["created_at"],
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "training_started": False,
        "test_model_accessed": False,
        "note_zh": "场景划分在 R3 数据生成与训练前冻结。",
    }
    freeze_path = ROOT / "runtime/r3_scene_split_freeze_record.json"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "sha256": freeze["manifest_sha256"], "counts": payload["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
