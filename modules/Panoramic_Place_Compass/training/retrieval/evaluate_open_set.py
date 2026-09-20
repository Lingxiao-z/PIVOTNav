#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(os.environ.get("PIVOTNAV_PPC_TRAINING_ROOT", Path.cwd())).expanduser().resolve()
sys.path.insert(0, str(ROOT / "src"))

from modules.Panoramic_Place_Compass.training.dataset import TarImageStore, image_to_tensor, is_same_place_positive_view, read_cached_sample_index, roll_tensor_by_degrees, v2_sample_shards
from modules.Panoramic_Place_Compass.training.spatial_protocol import positive_place_ids
from modules.Panoramic_Place_Compass.training.evidence import build_vpr_evidence
from modules.Panoramic_Place_Compass.model.core import PanoramicVPRV2System


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def anchor_sample_id(place: Dict[str, Any]) -> str:
    for view in place.get("views", []):
        if view.get("view_id") == "anchor_0m" or view.get("base_distance_m") == 0.0:
            return str(view["sample_id"])
    return str(place["views"][0]["sample_id"])


def distance_group(row: Dict[str, Any]) -> str:
    d = row.get("base_distance_m")
    if d is None:
        return "absent"
    d = float(d)
    for label, value in (("0m", 0.0), ("0.25m", 0.25), ("0.5m", 0.5), ("1m", 1.0), ("2m", 2.0)):
        if abs(d - value) < 1e-6:
            return label
    return f"{d:.2f}m"


def circular_error(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks for ties.
    for value in np.unique(s):
        idx = np.where(s == value)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    pos = int((y == 1).sum())
    if pos == 0:
        return None
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * (y == 1)).sum() / pos)


def load_checkpoint(system: PanoramicVPRV2System, checkpoint: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "system" not in ckpt:
        raise RuntimeError(f"Phase C evaluation requires a full system checkpoint, got keys={sorted(ckpt.keys())}")
    missing, unexpected = system.load_state_dict(ckpt["system"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    return ckpt


def stack_images(sample_ids: Sequence[str], store: TarImageStore, decoder: str) -> torch.Tensor:
    return torch.stack([image_to_tensor(store.read_jpeg(sid), decoder=decoder) for sid in sample_ids])


def encode_ids(system: PanoramicVPRV2System, sample_ids: Sequence[str], store: TarImageStore, decoder: str, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    chunks: Dict[str, List[torch.Tensor]] = {"global": [], "ring": [], "cluster_mass": [], "dustbin_fraction": []}
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
        for start in range(0, len(sample_ids), batch_size):
            ids = sample_ids[start : start + batch_size]
            if not ids:
                continue
            images = stack_images(ids, store, decoder).to(device, non_blocking=True)
            out = system.encode(images)
            for key in chunks:
                chunks[key].append(out[key].float().cpu())
    return {k: torch.cat(v, dim=0) for k, v in chunks.items()}


def to_device_encoding(enc: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in enc.items()}


def slice_encoding(enc: Dict[str, torch.Tensor], indices: Sequence[int], device: torch.device) -> Dict[str, torch.Tensor]:
    idx = torch.tensor(list(indices), dtype=torch.long)
    return {k: v.index_select(0, idx).to(device, non_blocking=True) for k, v in enc.items()}


def topk_entropy(prob: torch.Tensor) -> float:
    p = prob.detach().float().cpu().clamp_min(1e-8)
    return float((-(p * p.log()).sum() / math.log(max(2, p.numel()))).item())


def evaluate_query_batch(
    system: PanoramicVPRV2System,
    query_enc: Dict[str, torch.Tensor],
    gallery_dev: Dict[str, torch.Tensor],
    gallery_place_ids: Sequence[str],
    query_meta: Sequence[Dict[str, Any]],
    top_k: int,
    threshold: float,
    query_type: str,
) -> List[Dict[str, Any]]:
    q_dev = to_device_encoding(query_enc, next(system.parameters()).device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=next(system.parameters()).is_cuda):
        out = system.match_gallery(q_dev, gallery_dev, top_k=top_k)
    records: List[Dict[str, Any]] = []
    scores = out["retrieval_scores"].detach().float().cpu()
    indices = out["candidate_indices"].detach().cpu()
    node_prob = out["node_probabilities"].detach().float().cpu()
    unknown = out["unknown_probability"].detach().float().cpu()
    best_candidate = out["best_candidate_index"].detach().cpu()
    yaw_bins = out["yaw_shift_bins"].detach().cpu()
    ring_peak = out["ring_peak"].detach().float().cpu()
    ring_margin = out["ring_peak_second_margin"].detach().float().cpu()
    for i, meta in enumerate(query_meta):
        top_indices = indices[i].tolist()
        top_places = [gallery_place_ids[j] for j in top_indices]
        target = meta.get("target_place_id")
        accepted_places = set(positive_place_ids(meta))
        if target is not None and not accepted_places:
            accepted_places.add(str(target))
        is_absent = target is None and not accepted_places
        retrieval_miss = bool(not any(place in accepted_places for place in top_places)) if not is_absent else False
        best_idx = int(best_candidate[i].item())
        best_place = top_places[best_idx]
        unknown_probability = float(unknown[i].item())
        predicted_unknown = unknown_probability >= threshold
        predicted_place = None if predicted_unknown else best_place
        spatial_anchor_correct = (not is_absent and predicted_place in accepted_places) or (is_absent and predicted_unknown)
        strict_anchor_correct = (not is_absent and predicted_place == target) or (is_absent and predicted_unknown)
        top1_top2_margin = float(scores[i, 0].item() - scores[i, 1].item()) if scores.shape[1] > 1 else float("nan")
        target_probability = 0.0
        valid_probabilities = [float(node_prob[i, top_places.index(place)].item()) for place in accepted_places if place in top_places]
        if valid_probabilities:
            target_probability = max(valid_probabilities)
        yaw_shift_degrees = float(yaw_bins[i, best_idx].item()) * 360.0 / 32.0
        expected_yaw = meta.get("expected_yaw_degrees")
        yaw_error = circular_error(yaw_shift_degrees, float(expected_yaw)) if expected_yaw is not None else None
        records.append({
            "query_type": query_type,
            "sample_id": meta.get("sample_id"),
            "place_id": meta.get("place_id"),
            "scene_id": meta.get("scene_id"),
            "view_id": meta.get("view_id"),
            "base_distance_m": meta.get("base_distance_m"),
            "yaw_degrees": meta.get("yaw_degrees"),
            "distance_group": meta.get("distance_group", distance_group(meta)),
            "target_place_id": target,
            "spatial_positive_place_ids": sorted(accepted_places),
            "spatial_positive_count": len(accepted_places),
            "ambiguous_spatial_positive": len(accepted_places) > 1,
            "top1_place_id": top_places[0],
            "best_matcher_place_id": best_place,
            "predicted_place_id": predicted_place,
            "predicted_unknown": bool(predicted_unknown),
            "correct_anchor": bool(spatial_anchor_correct),
            "strict_correct_anchor": bool(strict_anchor_correct),
            "retrieval_miss": bool(retrieval_miss),
            "strict_correct_at_1": bool((not is_absent) and top_places[0] == target),
            "strict_correct_at_5": bool((not is_absent) and target in top_places[: min(5, len(top_places))]),
            "correct_at_1": bool((not is_absent) and top_places[0] in accepted_places),
            "correct_at_5": bool((not is_absent) and any(place in accepted_places for place in top_places[: min(5, len(top_places))])),
            "top1_score": float(scores[i, 0].item()),
            "top1_top2_margin": top1_top2_margin,
            "target_node_probability": target_probability,
            "present_probability": float(1.0 - unknown_probability),
            "unknown_probability": unknown_probability,
            "topk_entropy": topk_entropy(node_prob[i]),
            "yaw_shift_degrees": yaw_shift_degrees,
            "expected_yaw_degrees": expected_yaw,
            "yaw_error_degrees": yaw_error,
            "ring_peak": float(ring_peak[i, best_idx].item()),
            "ring_peak_second_margin": float(ring_margin[i, best_idx].item()),
        })
    return records


def encoding_storage_bytes(enc: Dict[str, torch.Tensor], cache_keys: Sequence[str] = ("global", "ring", "cluster_mass", "dustbin_fraction")) -> Dict[str, Any]:
    fields: Dict[str, int] = {}
    for key in cache_keys:
        tensor = enc.get(key)
        if tensor is not None:
            fields[key] = int(tensor.numel() * tensor.element_size())
    total = int(sum(fields.values()))
    count = int(next(iter(enc.values())).shape[0]) if enc else 0
    return {
        "field_bytes": fields,
        "total_bytes": total,
        "count": count,
        "bytes_per_item": (total / count) if count else None,
        "mib": total / 1048576.0,
        "cache_schema": "pano_vpr_v2_erp224_salad2112_ring32x128_v1",
        "note_zh": "按 float32 tensor 估算缓存 global/ring/cluster_mass/dustbin_fraction；不包括 raw JPEG 或 Python 容器开销。",
    }


def evaluate_hard_negative_pairs(
    system: PanoramicVPRV2System,
    gallery_enc: Dict[str, torch.Tensor],
    gallery_sample_ids: Sequence[str],
    hard_pairs: Sequence[Dict[str, Any]],
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sample_to_index = {str(sample_id): i for i, sample_id in enumerate(gallery_sample_ids)}
    usable_pairs = []
    for row in hard_pairs:
        qid = str(row.get("query_sample_id") or "")
        nid = str(row.get("negative_sample_id") or "")
        if qid in sample_to_index and nid in sample_to_index:
            usable_pairs.append(row)

    records: List[Dict[str, Any]] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
        for start in range(0, len(usable_pairs), max(1, batch_size)):
            chunk = usable_pairs[start : start + max(1, batch_size)]
            query_indices = [sample_to_index[str(r["query_sample_id"])] for r in chunk]
            negative_indices = [sample_to_index[str(r["negative_sample_id"])] for r in chunk]
            query = slice_encoding(gallery_enc, query_indices, device)
            positive = slice_encoding(gallery_enc, query_indices, device)
            negative = slice_encoding(gallery_enc, negative_indices, device)
            candidates = {k: torch.stack([positive[k], negative[k]], dim=1) for k in positive}
            out = system.matcher(query, candidates)
            node_prob = out["node_probabilities"].detach().float().cpu()
            unknown = out["unknown_probability"].detach().float().cpu()
            global_similarity = out["global_similarity"].detach().float().cpu()
            candidate_logits = out["candidate_logits"].detach().float().cpu()
            best_candidate = out["best_candidate_index"].detach().cpu()
            ring_peak = out["ring_peak"].detach().float().cpu()
            ring_margin = out["ring_peak_second_margin"].detach().float().cpu()
            yaw_bins = out["yaw_shift_bins"].detach().cpu()
            for i, pair in enumerate(chunk):
                unknown_probability = float(unknown[i].item())
                predicted_unknown = unknown_probability >= threshold
                best_idx = int(best_candidate[i].item())
                predicted_negative = (not predicted_unknown) and best_idx == 1
                records.append({
                    "query_sample_id": pair.get("query_sample_id"),
                    "query_place_id": pair.get("query_place_id"),
                    "negative_sample_id": pair.get("negative_sample_id"),
                    "negative_place_id": pair.get("negative_place_id"),
                    "scene_id": pair.get("scene_id"),
                    "geodesic_distance_m": pair.get("geodesic_distance_m"),
                    "euclidean_distance_m": pair.get("euclidean_distance_m"),
                    "positive_node_probability": float(node_prob[i, 0].item()),
                    "negative_node_probability": float(node_prob[i, 1].item()),
                    "unknown_probability": unknown_probability,
                    "predicted_unknown": bool(predicted_unknown),
                    "best_candidate_role": "negative" if best_idx == 1 else "positive",
                    "hard_negative_false_positive": bool(predicted_negative),
                    "positive_global_similarity": float(global_similarity[i, 0].item()),
                    "negative_global_similarity": float(global_similarity[i, 1].item()),
                    "positive_candidate_logit": float(candidate_logits[i, 0].item()),
                    "negative_candidate_logit": float(candidate_logits[i, 1].item()),
                    "negative_minus_positive_probability": float(node_prob[i, 1].item() - node_prob[i, 0].item()),
                    "negative_minus_positive_similarity": float(global_similarity[i, 1].item() - global_similarity[i, 0].item()),
                    "negative_yaw_shift_degrees": float(yaw_bins[i, 1].item()) * 360.0 / 32.0,
                    "negative_ring_peak": float(ring_peak[i, 1].item()),
                    "negative_ring_peak_second_margin": float(ring_margin[i, 1].item()),
                    "rgb_only_model_input": True,
                    "gt_usage_zh": "geodesic/euclidean 仅用于离线 hard-negative 标签；评测输入只使用 RGB descriptor cache。",
                })

    candidate_labels: List[int] = []
    probability_scores: List[float] = []
    similarity_scores: List[float] = []
    logit_scores: List[float] = []
    for row in records:
        candidate_labels.extend([1, 0])
        probability_scores.extend([float(row["positive_node_probability"]), float(row["negative_node_probability"])])
        similarity_scores.extend([float(row["positive_global_similarity"]), float(row["negative_global_similarity"])])
        logit_scores.extend([float(row["positive_candidate_logit"]), float(row["negative_candidate_logit"])])
    by_query: Dict[str, List[bool]] = defaultdict(list)
    by_negative: Dict[str, List[bool]] = defaultdict(list)
    for row in records:
        by_query[str(row.get("query_place_id"))].append(bool(row["hard_negative_false_positive"]))
        by_negative[str(row.get("negative_place_id"))].append(bool(row["hard_negative_false_positive"]))
    metrics = {
        "count": len(records),
        "top1_false_positive_rate": sum(bool(r["hard_negative_false_positive"]) for r in records) / max(1, len(records)),
        "unique_query_false_positive_rate": sum(1 for vals in by_query.values() if any(vals)) / max(1, len(by_query)),
        "unique_negative_false_positive_rate": sum(1 for vals in by_negative.values() if any(vals)) / max(1, len(by_negative)),
        "candidate_probability_roc_auc": binary_roc_auc(candidate_labels, probability_scores),
        "candidate_probability_pr_auc": average_precision(candidate_labels, probability_scores),
        "candidate_similarity_roc_auc": binary_roc_auc(candidate_labels, similarity_scores),
        "candidate_similarity_pr_auc": average_precision(candidate_labels, similarity_scores),
        "candidate_logit_roc_auc": binary_roc_auc(candidate_labels, logit_scores),
        "candidate_logit_pr_auc": average_precision(candidate_labels, logit_scores),
        "negative_node_probability": summary([float(r["negative_node_probability"]) for r in records]),
        "negative_global_similarity": summary([float(r["negative_global_similarity"]) for r in records]),
        "negative_minus_positive_probability": summary([float(r["negative_minus_positive_probability"]) for r in records]),
        "negative_minus_positive_similarity": summary([float(r["negative_minus_positive_similarity"]) for r in records]),
        "hard_negative_unique_far_le_0_0333": (sum(1 for vals in by_query.values() if any(vals)) / max(1, len(by_query))) <= 0.0333,
        "note_zh": "Hard-negative 评测使用候选集 [同地点 anchor, hard negative anchor]；pose/geodesic 只用于离线标签，不进入模型输入。",
    }
    return records, metrics


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metrics_for_present(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    present = [r for r in rows if r.get("target_place_id")]
    if not present:
        return {"count": 0}
    return {
        "count": len(present),
        "recall_at_1": sum(bool(r["correct_at_1"]) for r in present) / len(present),
        "recall_at_5": sum(bool(r["correct_at_5"]) for r in present) / len(present),
        "strict_owner_recall_at_1": sum(bool(r["strict_correct_at_1"]) for r in present) / len(present),
        "strict_owner_recall_at_5": sum(bool(r["strict_correct_at_5"]) for r in present) / len(present),
        "ambiguous_spatial_query_rate": sum(bool(r.get("ambiguous_spatial_positive")) for r in present) / len(present),
        "retrieval_miss_rate": sum(bool(r["retrieval_miss"]) for r in present) / len(present),
        "goal_anchor_recall": sum(bool(r["correct_anchor"]) for r in present) / len(present),
        "unknown_false_negative_rate": sum(bool(r["predicted_unknown"]) for r in present) / len(present),
        "top1_score": summary([float(r["top1_score"]) for r in present]),
        "present_probability": summary([float(r["present_probability"]) for r in present]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-tag", default="development_native_erp224_r3")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-name", default="phase_c_native_erp224_r3_development")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--decoder", default="pil")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--presence-threshold", type=float, default=0.5)
    ap.add_argument("--synthetic-yaws", default="15,30,45,60,90,135,180,225,270,315")
    ap.add_argument("--allow-one-time-test", action="store_true")
    ap.add_argument("--threshold-freeze", default="")
    args = ap.parse_args()

    is_test = args.split_tag.startswith("test_")
    test_access_path = ROOT / "runtime/r3_test_evaluation_access_record.json"
    if is_test:
        if not args.allow_one_time_test:
            raise SystemExit("拒绝 Test：必须在模型、代码和 Development 阈值冻结后显式传入 --allow-one-time-test。")
        if test_access_path.exists():
            raise SystemExit(f"拒绝重复 Test：一次性访问记录已存在 {test_access_path}")
        if not args.threshold_freeze:
            raise SystemExit("拒绝 Test：缺少 --threshold-freeze。")
        threshold_payload = json.loads(Path(args.threshold_freeze).read_text(encoding="utf-8"))
        if abs(float(threshold_payload["selected_unknown_probability_threshold"]) - float(args.presence_threshold)) > 1e-12:
            raise SystemExit("拒绝 Test：命令阈值与 Development 冻结阈值不一致。")
        if str(threshold_payload.get("checkpoint_sha256")) != sha_path(Path(args.checkpoint)):
            raise SystemExit("拒绝 Test：checkpoint SHA256 与阈值冻结记录不一致。")
        test_access_path.parent.mkdir(parents=True, exist_ok=True)
        test_access_path.write_text(json.dumps({
            "started_at": now(), "completed": False, "argv": sys.argv[:],
            "checkpoint_sha256": sha_path(Path(args.checkpoint)),
            "threshold_freeze": str(Path(args.threshold_freeze)),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    started_at = now()
    wall_start = time.perf_counter()
    timing: Dict[str, float] = {}
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    checkpoint = Path(args.checkpoint)
    system = PanoramicVPRV2System().to(device).eval()
    ckpt = load_checkpoint(system, checkpoint, device)
    system.eval()

    manifest_dir = ROOT / "data/manifests"
    places = load_jsonl(manifest_dir / f"{args.split_tag}_place_manifest.jsonl")
    samples = load_jsonl(manifest_dir / f"{args.split_tag}_sample_manifest.jsonl")
    sample_by_id = {str(r["sample_id"]): r for r in samples}
    gallery_places = [p for p in places if p.get("role") == "gallery_place" and p.get("views")]
    absent_places = [p for p in places if p.get("role") == "absent_outside_full_gallery" and p.get("views")]
    gallery_ids = [anchor_sample_id(p) for p in gallery_places]
    gallery_place_ids = [str(p["place_id"]) for p in gallery_places]
    sample_index = read_cached_sample_index(v2_sample_shards(ROOT, args.split_tag))
    if sample_index is None:
        raise RuntimeError(f"missing compact sample index for {args.split_tag}")
    store = TarImageStore(sample_index)

    rendered_records: List[Dict[str, Any]] = []
    yaw_records: List[Dict[str, Any]] = []
    absent_records: List[Dict[str, Any]] = []
    hard_negative_records: List[Dict[str, Any]] = []
    hard_negative_metrics: Dict[str, Any] = {"count": 0}
    try:
        t = time.perf_counter()
        gallery_enc = encode_ids(system, gallery_ids, store, args.decoder, args.batch_size, device)
        timing["gallery_encoding_seconds"] = time.perf_counter() - t
        gallery_dev = to_device_encoding(gallery_enc, device)

        query_ids: List[str] = []
        query_meta: List[Dict[str, Any]] = []
        for p in gallery_places:
            for view in p.get("views", []):
                sid = str(view["sample_id"])
                if sid not in sample_by_id:
                    continue
                row = dict(sample_by_id[sid])
                row["target_place_id"] = p["place_id"]
                row["distance_group"] = distance_group(row)
                query_ids.append(sid)
                query_meta.append(row)
        t = time.perf_counter()
        for start in range(0, len(query_ids), args.batch_size):
            ids = query_ids[start : start + args.batch_size]
            metas = query_meta[start : start + args.batch_size]
            enc = encode_ids(system, ids, store, args.decoder, args.batch_size, device)
            rendered_records.extend(evaluate_query_batch(system, enc, gallery_dev, gallery_place_ids, metas, args.top_k, args.presence_threshold, "rendered_distance"))

        timing["rendered_query_encoding_and_match_seconds"] = time.perf_counter() - t

        yaws = [float(x) for x in args.synthetic_yaws.split(",") if x.strip()]
        pending_imgs: List[torch.Tensor] = []
        pending_meta: List[Dict[str, Any]] = []
        yaw_t = time.perf_counter()
        def flush_yaw() -> None:
            nonlocal pending_imgs, pending_meta, yaw_records
            if not pending_imgs:
                return
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                images = torch.stack(pending_imgs).to(device, non_blocking=True)
                out = system.encode(images)
                enc = {k: v.float().cpu() for k, v in out.items() if k in ("global", "ring", "cluster_mass", "dustbin_fraction")}
            yaw_records.extend(evaluate_query_batch(system, enc, gallery_dev, gallery_place_ids, pending_meta, args.top_k, args.presence_threshold, "synthetic_same_place_yaw"))
            pending_imgs = []
            pending_meta = []
        for p, sid in zip(gallery_places, gallery_ids):
            base_img = image_to_tensor(store.read_jpeg(sid), decoder=args.decoder)
            for yaw in yaws:
                rolled, shift = roll_tensor_by_degrees(base_img, yaw)
                pending_imgs.append(rolled)
                pending_meta.append({"sample_id": f"{sid}__eval_roll{int(round(yaw)) % 360:03d}", "place_id": p["place_id"], "owner_place_id": p["place_id"], "spatial_positive_place_ids": [p["place_id"]], "scene_id": p["scene_id"], "target_place_id": p["place_id"], "expected_yaw_degrees": yaw, "distance_group": "0m_synthetic_yaw", "shift_pixels": shift})
                if len(pending_imgs) >= args.batch_size:
                    flush_yaw()
        flush_yaw()
        timing["synthetic_yaw_generation_encoding_and_match_seconds"] = time.perf_counter() - yaw_t

        absent_ids = [anchor_sample_id(p) for p in absent_places]
        absent_meta = []
        for p in absent_places:
            sid = anchor_sample_id(p)
            row = dict(sample_by_id[sid])
            row["target_place_id"] = None
            row["distance_group"] = "absent"
            absent_meta.append(row)
        t = time.perf_counter()
        for start in range(0, len(absent_ids), args.batch_size):
            ids = absent_ids[start : start + args.batch_size]
            metas = absent_meta[start : start + args.batch_size]
            enc = encode_ids(system, ids, store, args.decoder, args.batch_size, device)
            absent_records.extend(evaluate_query_batch(system, enc, gallery_dev, gallery_place_ids, metas, args.top_k, args.presence_threshold, "absent"))
        timing["absent_query_encoding_and_match_seconds"] = time.perf_counter() - t

        hard_pair_path = manifest_dir / f"{args.split_tag}_hard_negative_pairs.jsonl"
        hard_pairs = load_jsonl(hard_pair_path) if hard_pair_path.is_file() else []
        t = time.perf_counter()
        hard_negative_records, hard_negative_metrics = evaluate_hard_negative_pairs(
            system,
            gallery_enc,
            gallery_ids,
            hard_pairs,
            args.presence_threshold,
            args.batch_size,
            device,
        )
        timing["hard_negative_forced_pair_match_seconds"] = time.perf_counter() - t
    finally:
        store.close()

    all_rows = rendered_records + yaw_records + absent_records
    anchor_rendered_records = [r for r in rendered_records if is_same_place_positive_view(r)]
    anchor_present_records = anchor_rendered_records + yaw_records
    anchor_eval_rows = anchor_present_records + absent_records
    labels = [0 if r.get("target_place_id") is None else 1 for r in anchor_eval_rows]
    scores = [float(r["present_probability"]) for r in anchor_eval_rows]
    tp = sum(1 for r in anchor_present_records if r.get("predicted_place_id") is not None and r.get("correct_anchor"))
    strict_tp = sum(1 for r in anchor_present_records if r.get("predicted_place_id") == r.get("target_place_id"))
    pred_anchor = sum(1 for r in anchor_eval_rows if r.get("predicted_place_id") is not None)
    present_count = len(anchor_present_records)
    absent_count = len(absent_records)
    absent_fp = sum(1 for r in absent_records if r.get("predicted_place_id") is not None)
    unique_absent = defaultdict(list)
    for r in absent_records:
        unique_absent[str(r.get("place_id") or r.get("sample_id"))].append(r)
    unique_absent_fp = sum(1 for rows in unique_absent.values() if any(r.get("predicted_place_id") is not None for r in rows))

    by_group = defaultdict(list)
    for r in rendered_records:
        by_group[str(r.get("distance_group"))].append(r)
    metrics_by_distance = {k: metrics_for_present(v) for k, v in sorted(by_group.items())}
    yaw_errors = [float(r["yaw_error_degrees"]) for r in yaw_records if r.get("yaw_error_degrees") is not None]
    evidence_sample = None
    for r in all_rows:
        if r.get("predicted_place_id") is not None:
            evidence_sample = build_vpr_evidence(
                candidate_id=str(r["predicted_place_id"]),
                normalized_similarity=float(r["top1_score"]),
                rank=1,
                top1_top2_margin=float(r["top1_top2_margin"]),
                yaw_shift_degrees=float(r["yaw_shift_degrees"]),
                yaw_alignment_confidence=float(r["ring_peak"]),
                ring_peak_score=float(r["ring_peak"]),
                ring_peak_second_margin=float(r["ring_peak_second_margin"]),
                topk_entropy=float(r["topk_entropy"]),
                open_set_probability=float(r["unknown_probability"]),
                backend_version=f"phase_c:{checkpoint.name}:sha256={sha_path(checkpoint)[:16]}",
            )
            break

    out_dir = ROOT / "outputs/evidence" / args.output_name
    write_csv(out_dir / "rendered_queries.csv", rendered_records)
    write_csv(out_dir / "synthetic_yaw_queries.csv", yaw_records)
    write_csv(out_dir / "absent_queries.csv", absent_records)
    write_csv(out_dir / "hard_negative_pairs.csv", hard_negative_records)

    result = {
        "created_at": now(),
        "started_at": started_at,
        "argv": sys.argv[:],
        "split_tag": args.split_tag,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha_path(checkpoint),
        "checkpoint_phase": str(ckpt.get("phase", "unknown")),
        "top_k": args.top_k,
        "presence_threshold": args.presence_threshold,
        "gallery_count": len(gallery_ids),
        "rendered_query_count": len(rendered_records),
        "synthetic_yaw_query_count": len(yaw_records),
        "absent_query_count": len(absent_records),
        "hard_negative_pair_count": len(hard_negative_records),
        "rendered_metrics": metrics_for_present(rendered_records),
        "metrics_by_distance_group": metrics_by_distance,
        "synthetic_yaw_metrics": {**metrics_for_present(yaw_records), "yaw_error_degrees": summary(yaw_errors), "yaw_p90_le_15": (summary(yaw_errors).get("p90", 999.0) <= 15.0 if yaw_errors else False)},
        "hard_negative_metrics": hard_negative_metrics,
        "open_set_metrics": {
            "present_probability_auroc": binary_roc_auc(labels, scores),
            "present_probability_pr_auc": average_precision(labels, scores),
            "absent_far": absent_fp / max(1, absent_count),
            "unique_absent_far": unique_absent_fp / max(1, len(unique_absent)),
            "present_unknown_false_negative_rate": sum(1 for r in anchor_present_records if r.get("predicted_unknown")) / max(1, present_count),
            "unknown_probability_present": summary([float(r["unknown_probability"]) for r in anchor_present_records]),
            "unknown_probability_absent": summary([float(r["unknown_probability"]) for r in absent_records]),
        },
        "latency_and_storage": {
            "timing_seconds": {**timing, "total_eval_wall_seconds": time.perf_counter() - wall_start},
            "counts": {
                "gallery_count": len(gallery_ids),
                "rendered_query_count": len(rendered_records),
                "synthetic_yaw_query_count": len(yaw_records),
                "absent_query_count": len(absent_records),
                "hard_negative_pair_count": len(hard_negative_records),
            },
            "gallery_descriptor_cache": encoding_storage_bytes(gallery_enc),
        },
        "goal_anchor_metrics": {
            "precision": tp / max(1, pred_anchor),
            "recall": tp / max(1, present_count),
            "f1": (2 * tp / max(1, pred_anchor) * tp / max(1, present_count) / max(1e-9, (tp / max(1, pred_anchor)) + (tp / max(1, present_count)))) if present_count else 0.0,
            "true_positive_count": tp,
            "predicted_anchor_count": pred_anchor,
            "present_query_count": present_count,
            "strict_owner_precision": strict_tp / max(1, pred_anchor),
            "strict_owner_recall": strict_tp / max(1, present_count),
            "ambiguous_spatial_query_count": sum(bool(r.get("ambiguous_spatial_positive")) for r in anchor_present_records),
            "rendered_positive_max_distance_m": 1.0,
            "excluded_boundary_and_far_rendered_query_count": len(rendered_records) - len(anchor_rendered_records),
        },
        "development_gate_check": {
            "anchor_eligible_recall_at_1_ge_0_95": metrics_for_present(anchor_rendered_records).get("recall_at_1", 0.0) >= 0.95,
            "anchor_eligible_recall_at_5_ge_0_99": metrics_for_present(anchor_rendered_records).get("recall_at_5", 0.0) >= 0.99,
            "yaw_p90_le_15": summary(yaw_errors).get("p90", 999.0) <= 15.0 if yaw_errors else False,
            "goal_anchor_precision_ge_0_95": (tp / max(1, pred_anchor)) >= 0.95,
            "goal_anchor_recall_ge_0_90": (tp / max(1, present_count)) >= 0.90,
            "absent_unique_far_le_0_0333": (unique_absent_fp / max(1, len(unique_absent))) <= 0.0333,
            "hard_negative_unique_far_le_0_0333": bool(hard_negative_metrics.get("hard_negative_unique_far_le_0_0333")),
        },
        "evidence_sample": evidence_sample,
        "outputs": {
            "rendered_queries_csv": str(out_dir / "rendered_queries.csv"),
            "synthetic_yaw_queries_csv": str(out_dir / "synthetic_yaw_queries.csv"),
            "absent_queries_csv": str(out_dir / "absent_queries.csv"),
            "hard_negative_pairs_csv": str(out_dir / "hard_negative_pairs.csv"),
        },
        "note_zh": "Phase C Development/Test 评测：0-1m rendered query 使用 spatial_positive_place_ids 多正样本集合；严格 owner-ID 指标单独保留但不作为唯一主指标。1m 以外只报告距离退化曲线，不进入 UNKNOWN/Anchor 主指标。自然 Top-K 未召回任何空间有效正节点单独计为 retrieval_miss；hard-negative forced-pair 只用 RGB descriptor，pose/geodesic 不进入模型输入。阈值由 --presence-threshold 指定，进入 Test 前必须冻结。",
    }
    result["development_gate_check"]["all_listed_gates_passed"] = all(result["development_gate_check"].values())
    out_metric = ROOT / "outputs/metrics" / f"{args.output_name}.json"
    atomic_json(out_metric, result)
    if is_test:
        access = json.loads(test_access_path.read_text(encoding="utf-8"))
        access.update({"completed": True, "completed_at": now(), "metric": str(out_metric), "metric_sha256": sha_path(out_metric)})
        atomic_json(test_access_path, access)
    with (ROOT / "runtime/command_records_v3.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"created_at": now(), "step": "evaluate_phase_c_v2", "argv": sys.argv[:], "outputs": [str(out_metric), str(out_dir / "rendered_queries.csv"), str(out_dir / "synthetic_yaw_queries.csv"), str(out_dir / "absent_queries.csv"), str(out_dir / "hard_negative_pairs.csv")], "returncode": 0}, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"metric": str(out_metric), "gate_check": result["development_gate_check"], "open_set": result["open_set_metrics"], "hard_negative": result["hard_negative_metrics"], "goal_anchor": result["goal_anchor_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
