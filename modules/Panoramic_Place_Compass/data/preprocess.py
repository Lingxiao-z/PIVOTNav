"""Prepare images and frozen pair features for Compass training."""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..model.checkpoint import load_r361_modular_package
from ..localization import make_features


def preprocess_directory(source: Path, destination: Path, size: tuple[int, int] = (448, 224)) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.glob("*.png")):
        Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR).save(destination / path.name)
        count += 1
    return count


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _image(path: str | Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB").resize((448, 224), Image.Resampling.BILINEAR)).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def _pair_image(row: dict, side: str, dataset_root: Path | None) -> torch.Tensor:
    direct = row.get(f"{side}_image")
    if direct:
        return _image(direct)
    payload = row[side]
    shard = Path(payload["sample_shard"])
    if not shard.is_absolute():
        if dataset_root is None:
            raise ValueError("--dataset-root is required for tar-backed pair manifests")
        candidates = (dataset_root / shard, dataset_root / "data/shards/samples" / shard.name)
        shard = next((path for path in candidates if path.is_file()), candidates[0])
    with tarfile.open(shard, "r") as archive:
        member = archive.extractfile(payload["jpeg_member"])
        if member is None:
            raise FileNotFoundError(f"{payload['jpeg_member']} in {shard}")
        image = Image.open(io.BytesIO(member.read())).convert("RGB").resize((448, 224), Image.Resampling.BILINEAR)
        array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def _pair_key(row: dict, side: str) -> tuple[str, ...]:
    direct = row.get(f"{side}_image")
    if direct:
        return ("path", str(Path(direct).expanduser().resolve()))
    payload = row[side]
    return ("tar", str(payload["sample_shard"]), str(payload["jpeg_member"]))


def _wrapped(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _cpu_encoding(value: dict[str, object]) -> dict[str, object]:
    return {
        key: item.detach().cpu() if torch.is_tensor(item) else item
        for key, item in value.items()
    }


def _device_encoding(value: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: item.to(device) if torch.is_tensor(item) else item
        for key, item in value.items()
    }


def _remember(cache: dict, key: tuple[str, ...], value: dict, cache_size: int) -> None:
    cache[key] = value
    if len(cache) > max(2, cache_size):
        cache.pop(next(iter(cache)))


def encode_pairs(manifest: Path, checkpoint: Path, output: Path, device: str, dataset_root: Path | None = None, cache_size: int = 512) -> None:
    runtime = load_r361_modular_package(checkpoint, device)
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    cache: dict[tuple[str, ...], dict[str, torch.Tensor]] = {}
    for row in _rows(manifest):
        source_key, target_key = _pair_key(row, "source"), _pair_key(row, "target")
        if source_key not in cache:
            _remember(cache, source_key, _cpu_encoding(runtime.encode_panorama(_pair_image(row, "source", dataset_root))), cache_size)
        if target_key not in cache:
            _remember(cache, target_key, _cpu_encoding(runtime.encode_panorama(_pair_image(row, "target", dataset_root))), cache_size)
        source, target = cache[source_key], cache[target_key]
        values["source_tokens"].append(source["tokens"][0].float().cpu().numpy())
        values["target_tokens"].append(target["tokens"][0].float().cpu().numpy())
        values["source_descriptor_ring"].append(source["descriptor_ring"][0].float().cpu().numpy())
        values["target_descriptor_ring"].append(target["descriptor_ring"][0].float().cpu().numpy())
        yaw = row.get("yaw_degrees")
        if yaw is None and isinstance(row.get("source"), dict):
            yaw = _wrapped(row["target"]["render_yaw_degrees"] - row["source"]["render_yaw_degrees"])
        values["yaw_degrees"].append(np.asarray(yaw or 0.0, dtype=np.float32))
        values["bearing_degrees"].append(np.asarray(row.get("bearing_degrees", 0.0), dtype=np.float32))
        values["bearing_valid"].append(np.asarray(row.get("bearing_valid", True), dtype=bool))
        bucket = row.get("distance_bucket", 0)
        if isinstance(bucket, str):
            bucket = {"0.5-1m": 0, "1-2m": 1, "2-3m": 2, "3-4m": 3, "4-6m_diagnostic": 4}[bucket]
        values["distance_bucket"].append(np.asarray(bucket, dtype=np.int64))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{key: np.stack(items) for key, items in values.items()})


def encode_arrival_sequences(manifest: Path, checkpoint: Path, output: Path, device: str, dataset_root: Path | None = None, cache_size: int = 512) -> None:
    runtime = load_r361_modular_package(checkpoint, device)
    grouped: dict[str, list[tuple[int, np.ndarray, bool, float]]] = defaultdict(list)
    cache: dict[tuple[str, ...], dict[str, torch.Tensor]] = {}
    for row in _rows(manifest):
        source_key, target_key = _pair_key(row, "source"), _pair_key(row, "target")
        if source_key not in cache:
            _remember(cache, source_key, _cpu_encoding(runtime.encode_panorama(_pair_image(row, "source", dataset_root))), cache_size)
        if target_key not in cache:
            _remember(cache, target_key, _cpu_encoding(runtime.encode_panorama(_pair_image(row, "target", dataset_root))), cache_size)
        source = _device_encoding(cache[source_key], runtime.device)
        target = _device_encoding(cache[target_key], runtime.device)
        feature = runtime._pair_outputs(source, target)["arrival_feature"][0].float().cpu().numpy()
        grouped[str(row["sequence_id"])].append(
            (int(row["frame_index"]), feature, bool(row["arrived"]), float(row["distance_m"]))
        )
    ordered = [sorted(items) for _, items in sorted(grouped.items())]
    maximum = max(map(len, ordered))
    features = np.zeros((len(ordered), maximum, 280), dtype=np.float32)
    target = np.zeros((len(ordered), maximum), dtype=bool)
    distance = np.full((len(ordered), maximum), np.inf, dtype=np.float32)
    mask = np.zeros((len(ordered), maximum), dtype=bool)
    for sequence_index, items in enumerate(ordered):
        for frame_index, (_, feature, arrived, distance_m) in enumerate(items):
            features[sequence_index, frame_index] = feature
            target[sequence_index, frame_index] = arrived
            distance[sequence_index, frame_index] = distance_m
            mask[sequence_index, frame_index] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=features, target_arrived=target, distance_m=distance, frame_mask=mask)


def prepare_v7(manifest: Path, output: Path) -> None:
    rows = _rows(manifest)
    features = np.asarray(
        [
            row["features"]
            if "features" in row
            else make_features(row["parallax_features"], row["active_evidence"])
            for row in rows
        ],
        dtype=np.float32,
    )
    target = np.asarray([row["target_m"] for row in rows], dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"features": features, "target_m": target}
    if any("sample_weight" in row for row in rows):
        arrays["sample_weight"] = np.asarray([row.get("sample_weight", 1.0) for row in rows], dtype=np.float32)
    np.savez_compressed(output, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    resize = sub.add_parser("resize")
    resize.add_argument("--source", type=Path, required=True)
    resize.add_argument("--output", type=Path, required=True)
    for name in ("pairs", "arrival"):
        child = sub.add_parser(name)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--r361-checkpoint", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--device", default="cuda")
        child.add_argument("--dataset-root", type=Path)
        child.add_argument("--cache-size", type=int, default=512)
    v7 = sub.add_parser("v7")
    v7.add_argument("--manifest", type=Path, required=True)
    v7.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "resize":
        print(json.dumps({"images": preprocess_directory(args.source, args.output)}))
    elif args.command == "pairs":
        encode_pairs(args.manifest, args.r361_checkpoint, args.output, args.device, args.dataset_root, args.cache_size)
    elif args.command == "arrival":
        encode_arrival_sequences(args.manifest, args.r361_checkpoint, args.output, args.device, args.dataset_root, args.cache_size)
    else:
        prepare_v7(args.manifest, args.output)


if __name__ == "__main__":
    main()
