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
from torch.nn import functional as F

ROOT = Path(os.environ.get("PIVOTNAV_PPC_TRAINING_ROOT", Path.cwd())).expanduser().resolve()

from modules.Panoramic_Place_Compass.training.dataset import TarImageStore, image_to_tensor, read_cached_sample_index, roll_tensor_by_degrees, v2_sample_shards
from modules.Panoramic_Place_Compass.training.evidence import build_vpr_evidence
from modules.Panoramic_Place_Compass.model.core import circular_correlation
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
    rows = []
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
    if abs(d - 0.0) < 1e-6:
        return "0m"
    if abs(d - 0.25) < 1e-6:
        return "0.25m"
    if abs(d - 0.5) < 1e-6:
        return "0.5m"
    if abs(d - 1.0) < 1e-6:
        return "1m"
    if abs(d - 2.0) < 1e-6:
        return "2m"
    return f"{d:.2f}m"


class Encoder:
    def __init__(self, checkpoint: Path, device: torch.device, decoder: str):
        self.device = device
        self.decoder = decoder
        self.system = PanoramicVPRV2System().to(device).eval()
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        load_mode = "unknown"
        if "system" in ckpt:
            missing, unexpected = self.system.load_state_dict(ckpt["system"], strict=False)
            critical_missing = [k for k in missing if not k.startswith("matcher.")]
            if critical_missing or unexpected:
                raise RuntimeError(f"checkpoint system load mismatch: missing={missing}, unexpected={unexpected}")
            load_mode = "full_system"
        elif "head" in ckpt:
            self.system.descriptor_head.load_state_dict(ckpt["head"], strict=True)
            load_mode = "descriptor_head_only"
        elif "descriptor_head" in ckpt:
            self.system.descriptor_head.load_state_dict(ckpt["descriptor_head"], strict=True)
            load_mode = "descriptor_head_only"
        else:
            raise RuntimeError(f"unsupported checkpoint keys: {sorted(ckpt.keys())}")
        self.system.eval()
        phase = str(ckpt.get("phase", "unknown"))
        self.backend_version = f"{phase}:{load_mode}:{checkpoint.name}:sha256={sha_path(checkpoint)[:16]}"

    def encode_images(self, images: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        chunks: Dict[str, List[torch.Tensor]] = {"global": [], "ring": [], "cluster_mass": [], "dustbin_mass": [], "dustbin_fraction": []}
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
            for start in range(0, images.shape[0], batch_size):
                batch = images[start : start + batch_size].to(self.device, non_blocking=True)
                out = self.system.encode(batch)
                for key in chunks:
                    if key in out:
                        chunks[key].append(out[key].float().cpu())
        return {key: torch.cat(value, dim=0) for key, value in chunks.items() if value}


def load_sample_tensors(sample_ids: Sequence[str], store: TarImageStore, decoder: str) -> torch.Tensor:
    return torch.stack([image_to_tensor(store.read_jpeg(sid), decoder=decoder) for sid in sample_ids])


def concat_encoding_chunks(chunks: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not chunks:
        return {}
    keys = chunks[0].keys()
    return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in keys}


def encode_sample_ids(encoder: Encoder, sample_ids: Sequence[str], store: TarImageStore, decoder: str, batch_size: int) -> Dict[str, torch.Tensor]:
    chunks: List[Dict[str, torch.Tensor]] = []
    for start in range(0, len(sample_ids), batch_size):
        batch_ids = sample_ids[start : start + batch_size]
        if not batch_ids:
            continue
        images = load_sample_tensors(batch_ids, store, decoder)
        chunks.append(encoder.encode_images(images, batch_size))
    return concat_encoding_chunks(chunks)


def encode_synthetic_roll_queries(
    encoder: Encoder,
    gallery_ids: Sequence[str],
    gallery_place_ids: Sequence[str],
    gallery_scene_ids: Sequence[str],
    store: TarImageStore,
    decoder: str,
    batch_size: int,
    yaws: Sequence[float],
) -> Tuple[Dict[str, torch.Tensor] | None, List[Dict[str, Any]]]:
    chunks: List[Dict[str, torch.Tensor]] = []
    roll_meta: List[Dict[str, Any]] = []
    pending_images: List[torch.Tensor] = []

    def flush() -> None:
        nonlocal pending_images
        if pending_images:
            images = torch.stack(pending_images)
            chunks.append(encoder.encode_images(images, batch_size))
            pending_images = []

    for i, sid in enumerate(gallery_ids):
        base_img = image_to_tensor(store.read_jpeg(sid), decoder=decoder)
        for yaw in yaws:
            rolled, shift = roll_tensor_by_degrees(base_img, yaw)
            pending_images.append(rolled)
            roll_meta.append({
                "sample_id": f"{sid}__eval_roll{int(round(yaw)) % 360:03d}",
                "target_place_id": gallery_place_ids[i],
                "scene_id": gallery_scene_ids[i],
                "yaw_degrees": yaw,
                "shift_pixels": shift,
            })
            if len(pending_images) >= batch_size:
                flush()
    flush()
    return (concat_encoding_chunks(chunks) if chunks else None), roll_meta


def retrieve(query_global: torch.Tensor, gallery_global: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    q = F.normalize(query_global.float(), dim=-1)
    g = F.normalize(gallery_global.float(), dim=-1)
    sims = q @ g.T
    vals, idx = torch.topk(sims, k=min(top_k, gallery_global.shape[0]), dim=1)
    return vals, idx


def yaw_from_rings(query_ring: torch.Tensor, cand_ring: torch.Tensor) -> Dict[str, float]:
    scores = circular_correlation(query_ring.unsqueeze(0).float(), cand_ring.unsqueeze(0).float())[0]
    top = torch.topk(scores, k=min(2, scores.numel()))
    pred_bin = int(top.indices[0].item())
    second = float(top.values[1].item()) if top.values.numel() > 1 else float("-inf")
    peak = float(top.values[0].item())
    return {
        "pred_bin": pred_bin,
        "pred_degrees": pred_bin * 360.0 / 32.0,
        "peak": peak,
        "second_margin": peak - second,
    }


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


def metrics_for_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "recall_at_1": float(sum(r["correct_at_1"] for r in records) / len(records)),
        "recall_at_5": float(sum(r["correct_at_5"] for r in records) / len(records)),
        "top1_score": summary([r["top1_score"] for r in records]),
        "top1_top2_margin": summary([r["top1_top2_margin"] for r in records]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-tag", default="development_full_v2_r0")
    ap.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/phase_a_v2_r0/last.pt"))
    ap.add_argument("--output-name", default="phase_a_v2_r0_development")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--decoder", default="torchvision")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--synthetic-yaws", default="15,30,45,60,90,135,180,225,270,315")
    args = ap.parse_args()
    started_at = now()
    wall_start = time.perf_counter()
    timing: Dict[str, float] = {}
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    checkpoint = Path(args.checkpoint)
    place_rows = load_jsonl(ROOT / "data/manifests" / f"{args.split_tag}_place_manifest.jsonl")
    sample_rows = load_jsonl(ROOT / "data/manifests" / f"{args.split_tag}_sample_manifest.jsonl")
    sample_by_id = {r["sample_id"]: r for r in sample_rows}
    gallery_places = [p for p in place_rows if p.get("role") == "gallery_place"]
    absent_places = [p for p in place_rows if p.get("role") == "dynamic_gallery_prefix_absent"]
    gallery_ids = [anchor_sample_id(p) for p in gallery_places]
    gallery_place_ids = [p["place_id"] for p in gallery_places]
    gallery_scene_ids = [p["scene_id"] for p in gallery_places]
    sample_shards = v2_sample_shards(ROOT, args.split_tag)
    sample_index = read_cached_sample_index(sample_shards)
    if sample_index is None:
        raise RuntimeError(f"missing compact index for {args.split_tag}")
    store = TarImageStore(sample_index)
    encoder = Encoder(checkpoint, device, args.decoder)
    try:
        t = time.perf_counter()
        gallery_desc = encode_sample_ids(encoder, gallery_ids, store, args.decoder, args.batch_size)
        timing["gallery_encoding_seconds"] = time.perf_counter() - t
        # Near/distance queries use real rendered samples from place bundles.
        query_ids, query_meta = [], []
        for p in gallery_places:
            for view in p.get("views", []):
                sid = str(view["sample_id"])
                if sid not in sample_by_id:
                    continue
                row = dict(sample_by_id[sid])
                row["target_place_id"] = p["place_id"]
                query_ids.append(sid)
                query_meta.append(row)
        t = time.perf_counter()
        query_desc = encode_sample_ids(encoder, query_ids, store, args.decoder, args.batch_size)
        timing["rendered_query_encoding_seconds"] = time.perf_counter() - t
        # Synthetic same-place yaw queries are generated from anchor images and never stored as JPEGs.
        yaws = [float(x) for x in args.synthetic_yaws.split(",") if x.strip()]
        t = time.perf_counter()
        roll_desc, roll_meta = encode_synthetic_roll_queries(
            encoder,
            gallery_ids,
            gallery_place_ids,
            gallery_scene_ids,
            store,
            args.decoder,
            args.batch_size,
            yaws,
        )
        timing["synthetic_yaw_encoding_seconds"] = time.perf_counter() - t
        absent_ids = [anchor_sample_id(p) for p in absent_places]
        absent_meta = []
        for p in absent_places:
            sid = anchor_sample_id(p)
            row = dict(sample_by_id[sid])
            row["target_place_id"] = None
            row["gallery_prefix_size"] = p.get("gallery_prefix_size")
            absent_meta.append(row)
        t = time.perf_counter()
        absent_desc = encode_sample_ids(encoder, absent_ids, store, args.decoder, args.batch_size) if absent_ids else None
        timing["absent_query_encoding_seconds"] = time.perf_counter() - t
    finally:
        store.close()

    records: List[Dict[str, Any]] = []
    t = time.perf_counter()
    vals, idx = retrieve(query_desc["global"], gallery_desc["global"], args.top_k)
    timing["rendered_retrieval_seconds"] = time.perf_counter() - t
    for qi, sid in enumerate(query_ids):
        target = query_meta[qi]["target_place_id"]
        top_indices = idx[qi].tolist()
        top_scores = vals[qi].tolist()
        top_places = [gallery_place_ids[j] for j in top_indices]
        yaw_info = yaw_from_rings(query_desc["ring"][qi], gallery_desc["ring"][top_indices[0]])
        rec = {
            "query_type": "rendered_distance",
            "sample_id": sid,
            "scene_id": query_meta[qi]["scene_id"],
            "distance_group": distance_group(query_meta[qi]),
            "target_place_id": target,
            "top1_place_id": top_places[0],
            "correct_at_1": top_places[0] == target,
            "correct_at_5": target in top_places,
            "top1_score": float(top_scores[0]),
            "top1_top2_margin": float(top_scores[0] - top_scores[1]) if len(top_scores) > 1 else float("nan"),
            "pred_yaw_degrees": yaw_info["pred_degrees"],
            "ring_peak": yaw_info["peak"],
            "ring_second_margin": yaw_info["second_margin"],
            "dustbin_mass": float(query_desc["dustbin_mass"][qi].mean().item()),
            "cluster_entropy": float(-(query_desc["cluster_mass"][qi].clamp_min(1e-8) * query_desc["cluster_mass"][qi].clamp_min(1e-8).log()).sum().item()),
        }
        records.append(rec)

    yaw_records: List[Dict[str, Any]] = []
    if roll_desc is not None:
        t = time.perf_counter()
        vals_y, idx_y = retrieve(roll_desc["global"], gallery_desc["global"], args.top_k)
        timing["synthetic_yaw_retrieval_seconds"] = time.perf_counter() - t
        for qi, meta in enumerate(roll_meta):
            target = meta["target_place_id"]
            top_indices = idx_y[qi].tolist()
            top_scores = vals_y[qi].tolist()
            top_places = [gallery_place_ids[j] for j in top_indices]
            yaw_info = yaw_from_rings(roll_desc["ring"][qi], gallery_desc["ring"][top_indices[0]])
            expected = float(meta["yaw_degrees"])
            rec = {
                "query_type": "synthetic_same_place_yaw",
                "sample_id": meta["sample_id"],
                "scene_id": meta["scene_id"],
                "distance_group": "0m_synthetic_yaw",
                "target_place_id": target,
                "top1_place_id": top_places[0],
                "correct_at_1": top_places[0] == target,
                "correct_at_5": target in top_places,
                "top1_score": float(top_scores[0]),
                "top1_top2_margin": float(top_scores[0] - top_scores[1]) if len(top_scores) > 1 else float("nan"),
                "expected_yaw_degrees": expected,
                "pred_yaw_degrees": yaw_info["pred_degrees"],
                "yaw_error_degrees": circular_error(yaw_info["pred_degrees"], expected),
                "ring_peak": yaw_info["peak"],
                "ring_second_margin": yaw_info["second_margin"],
                "dustbin_mass": float(roll_desc["dustbin_mass"][qi].mean().item()),
                "cluster_entropy": float(-(roll_desc["cluster_mass"][qi].clamp_min(1e-8) * roll_desc["cluster_mass"][qi].clamp_min(1e-8).log()).sum().item()),
            }
            yaw_records.append(rec)

    absent_records: List[Dict[str, Any]] = []
    if absent_desc is not None:
        t = time.perf_counter()
        vals_a, idx_a = retrieve(absent_desc["global"], gallery_desc["global"], args.top_k)
        timing["absent_retrieval_seconds"] = time.perf_counter() - t
        for qi, sid in enumerate(absent_ids):
            top_indices = idx_a[qi].tolist()
            top_scores = vals_a[qi].tolist()
            top_places = [gallery_place_ids[j] for j in top_indices]
            rec = {
                "query_type": "absent",
                "sample_id": sid,
                "scene_id": absent_meta[qi]["scene_id"],
                "top1_place_id": top_places[0],
                "top1_score": float(top_scores[0]),
                "top1_top2_margin": float(top_scores[0] - top_scores[1]) if len(top_scores) > 1 else float("nan"),
                "dustbin_mass": float(absent_desc["dustbin_mass"][qi].mean().item()),
                "cluster_entropy": float(-(absent_desc["cluster_mass"][qi].clamp_min(1e-8) * absent_desc["cluster_mass"][qi].clamp_min(1e-8).log()).sum().item()),
            }
            absent_records.append(rec)

    # Hard-negative evidence uses manifest pairs and encoded descriptors where available.
    hard_pair_path = ROOT / "data/manifests" / f"{args.split_tag}_hard_negative_pairs.jsonl"
    hard_pairs = load_jsonl(hard_pair_path) if hard_pair_path.is_file() else []
    t = time.perf_counter()
    global_by_sample = {sid: query_desc["global"][i] for i, sid in enumerate(query_ids)}
    hard_scores = []
    for pair in hard_pairs:
        qid = pair["query_sample_id"]
        nid = pair["negative_sample_id"]
        if qid not in global_by_sample or nid not in global_by_sample:
            continue
        score = float((F.normalize(global_by_sample[qid].float(), dim=-1) * F.normalize(global_by_sample[nid].float(), dim=-1)).sum().item())
        hard_scores.append({
            "query_sample_id": qid,
            "negative_sample_id": nid,
            "scene_id": pair["scene_id"],
            "geodesic_distance_m": pair["geodesic_distance_m"],
            "similarity": score,
        })

    timing["hard_negative_scoring_seconds"] = time.perf_counter() - t

    by_group = defaultdict(list)
    for rec in records:
        by_group[rec["distance_group"]].append(rec)
    metrics_by_group = {k: metrics_for_records(v) for k, v in sorted(by_group.items())}
    rendered_metrics = metrics_for_records(records)
    yaw_metrics = metrics_for_records(yaw_records)
    yaw_metrics["yaw_error_degrees"] = summary([r["yaw_error_degrees"] for r in yaw_records])
    yaw_metrics["within_11_25_degrees"] = float(sum(r["yaw_error_degrees"] <= 11.25 + 1e-6 for r in yaw_records) / max(1, len(yaw_records)))
    absent_metrics = {
        "count": len(absent_records),
        "top1_score": summary([r["top1_score"] for r in absent_records]),
        "top1_top2_margin": summary([r["top1_top2_margin"] for r in absent_records]),
        "note_zh": "尚未训练 open-set verifier；这里只报告 raw score 分布，不能解释为 UNKNOWN 通过。",
    }
    anchor_positive_groups = {"0m", "0.25m", "0.5m", "1m"}
    positive_reference_scores = [float(r["top1_score"]) for r in records if str(r.get("distance_group")) in anchor_positive_groups]
    positive_reference_scores.extend(float(r["top1_score"]) for r in yaw_records)
    hard_negative_scores = [float(r["similarity"]) for r in hard_scores]
    hard_labels = [1] * len(positive_reference_scores) + [0] * len(hard_negative_scores)
    hard_eval_scores = positive_reference_scores + hard_negative_scores
    hard_metrics = {
        "count": len(hard_scores),
        "similarity": summary([r["similarity"] for r in hard_scores]),
        "positive_reference_count": len(positive_reference_scores),
        "positive_reference_similarity": summary(positive_reference_scores),
        "similarity_roc_auc": binary_roc_auc(hard_labels, hard_eval_scores),
        "similarity_pr_auc": average_precision(hard_labels, hard_eval_scores),
        "forced_pair_false_positive_rate": 1.0 if hard_scores else None,
        "note_zh": "hard-negative pair 没有开放集阈值时强制检索必然产生候选；ROC/PR 使用 0-1m 与 synthetic yaw 正样本 top1 score 对比 hard-negative pair similarity。",
    }
    # A minimal backend-neutral evidence sample for integration shape checks.
    evidence_sample = None
    if records:
        r = records[0]
        evidence_sample = build_vpr_evidence(
            candidate_id=r["top1_place_id"],
            normalized_similarity=r["top1_score"],
            rank=1,
            top1_top2_margin=r["top1_top2_margin"],
            yaw_shift_degrees=r["pred_yaw_degrees"],
            yaw_alignment_confidence=r["ring_peak"],
            ring_peak_score=r["ring_peak"],
            ring_peak_second_margin=r["ring_second_margin"],
            topk_entropy=0.0,
            open_set_probability=0.0,
            backend_version=encoder.backend_version,
        )

    out_dir = ROOT / "outputs/evidence" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query_csv = out_dir / "per_query_rendered.csv"
    with per_query_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["empty"])
        writer.writeheader()
        writer.writerows(records)
    yaw_csv = out_dir / "per_query_synthetic_yaw.csv"
    with yaw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(yaw_records[0].keys()) if yaw_records else ["empty"])
        writer.writeheader()
        writer.writerows(yaw_records)
    absent_csv = out_dir / "absent_queries.csv"
    with absent_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(absent_records[0].keys()) if absent_records else ["empty"])
        writer.writeheader()
        writer.writerows(absent_records)
    hard_csv = out_dir / "hard_negative_pairs.csv"
    with hard_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(hard_scores[0].keys()) if hard_scores else ["empty"])
        writer.writeheader()
        writer.writerows(hard_scores)

    result = {
        "created_at": now(),
        "started_at": started_at,
        "argv": sys.argv[:],
        "split_tag": args.split_tag,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha_path(checkpoint),
        "backend_version": encoder.backend_version,
        "gallery_count": len(gallery_ids),
        "rendered_query_count": len(records),
        "synthetic_yaw_query_count": len(yaw_records),
        "absent_query_count": len(absent_records),
        "hard_negative_pair_count": len(hard_scores),
        "rendered_metrics": rendered_metrics,
        "metrics_by_distance_group": metrics_by_group,
        "synthetic_yaw_metrics": yaw_metrics,
        "absent_metrics": absent_metrics,
        "hard_negative_metrics": hard_metrics,
        "latency_and_storage": {
            "timing_seconds": {**timing, "total_eval_wall_seconds": time.perf_counter() - wall_start},
            "counts": {
                "gallery_count": len(gallery_ids),
                "rendered_query_count": len(records),
                "synthetic_yaw_query_count": len(yaw_records),
                "absent_query_count": len(absent_records),
            },
            "gallery_descriptor_cache": encoding_storage_bytes(gallery_desc),
        },
        "development_gate_check": {
            "overall_recall_at_1_ge_0_95": rendered_metrics.get("recall_at_1", 0.0) >= 0.95,
            "overall_recall_at_5_ge_0_99": rendered_metrics.get("recall_at_5", 0.0) >= 0.99,
            "yaw_p90_le_11_25": yaw_metrics["yaw_error_degrees"].get("p90", 999.0) <= 11.25,
            "open_set_not_trained": True,
            "all_development_gates_passed": False,
        },
        "evidence_sample": evidence_sample,
        "outputs": {
            "per_query_rendered_csv": str(per_query_csv),
            "synthetic_yaw_csv": str(yaw_csv),
            "absent_csv": str(absent_csv),
            "hard_negative_csv": str(hard_csv),
        },
        "full_goal_complete": False,
        "note_zh": "Development-only retrieval/yaw evaluation；支持加载 full-system checkpoint，因此 Phase B 微调 backbone 后也按真实权重评测。open-set verifier 若未训练，则 absent/hard-negative 只作为 raw evidence 分布。",
    }
    out_metric = ROOT / "outputs/metrics" / f"{args.output_name}.json"
    atomic_json(out_metric, result)
    with (ROOT / "runtime/command_records_v2.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "created_at": now(),
            "step": "evaluate_phase_a_v2_development",
            "argv": sys.argv[:],
            "outputs": [str(out_metric), str(per_query_csv), str(yaw_csv), str(absent_csv), str(hard_csv)],
            "returncode": 0,
        }, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"metric": str(out_metric), "gate_check": result["development_gate_check"], "rendered": rendered_metrics, "yaw": yaw_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
