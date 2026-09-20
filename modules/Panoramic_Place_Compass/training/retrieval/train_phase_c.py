#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, random, sys, tempfile, time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(os.environ.get("PIVOTNAV_PPC_TRAINING_ROOT", Path.cwd())).expanduser().resolve()
sys.path.insert(0, str(ROOT / "src"))

from modules.Panoramic_Place_Compass.training.dataset import TarImageStore, image_to_tensor, is_same_place_positive_view, read_cached_sample_index, v2_sample_shards
from modules.Panoramic_Place_Compass.training.losses import multi_positive_open_set_loss, open_set_brier_loss, pairwise_same_place_loss
from modules.Panoramic_Place_Compass.model.core import TrainingPhase, optimizer_parameter_groups
from modules.Panoramic_Place_Compass.model.core import PanoramicVPRV2System


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp); os.replace(tmp, path)


def summary(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0: return {"count": 0}
    return {"count": int(arr.size), "mean": float(arr.mean()), "median": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "min": float(arr.min()), "max": float(arr.max())}


def dist_setup() -> tuple[bool, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank); torch.distributed.init_process_group("nccl")
    return world > 1, rank, local_rank, world


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def unwrap_system(model):
    core = unwrap(model)
    return core.system if hasattr(core, "system") else core


class PhaseCTrainModule(torch.nn.Module):
    def __init__(self, system: PanoramicVPRV2System):
        super().__init__()
        self.system = system

    def forward(self, images: torch.Tensor, batch_queries: int, top_k: int) -> Dict[str, torch.Tensor]:
        query, candidates = encode_query_candidates(self.system, images, int(batch_queries), int(top_k))
        return self.system.matcher(query, candidates)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows


def anchor_sample_id(place: Dict[str, Any]) -> str:
    for view in place.get("views", []):
        if view.get("view_id") == "anchor_0m" or view.get("base_distance_m") == 0.0:
            return str(view["sample_id"])
    return str(place["views"][0]["sample_id"])


class PhaseCSampler:
    def __init__(self, split_tag: str, seed: int, top_k: int, present_fraction: float):
        self.split_tag = split_tag; self.rng = random.Random(seed); self.top_k = int(top_k); self.present_fraction = float(present_fraction)
        manifest_dir = ROOT / "data/manifests"
        places = load_jsonl(manifest_dir / f"{split_tag}_place_manifest.jsonl")
        self.gallery = [p for p in places if p.get("role") == "gallery_place" and p.get("views")]
        self.absent = [p for p in places if p.get("role") == "absent_outside_full_gallery" and p.get("views")]
        if len(self.gallery) < self.top_k + 1:
            raise RuntimeError(f"not enough gallery places for top_k={self.top_k}")
        self.anchor_by_place = {p["place_id"]: anchor_sample_id(p) for p in self.gallery}
        self.gallery_place_ids = list(self.anchor_by_place)
        self.hard_by_place: Dict[str, List[str]] = {}
        hard_path = manifest_dir / f"{split_tag}_hard_negative_pairs.jsonl"
        if hard_path.is_file():
            for row in load_jsonl(hard_path):
                q = row.get("query_place_id"); n = row.get("negative_place_id")
                if q in self.anchor_by_place and n in self.anchor_by_place:
                    self.hard_by_place.setdefault(q, []).append(n)

    def _query_view(self, place: Dict[str, Any]) -> Dict[str, Any]:
        views = list(place.get("views", []))
        eligible = [v for v in views if is_same_place_positive_view(v)]
        non_anchor = [v for v in eligible if float(v.get("base_distance_m", 0.0)) > 0.0]
        pool = non_anchor or eligible
        if not pool:
            raise RuntimeError(f"place {place.get('place_id')} has no <=1m positive query view")
        return dict(self.rng.choice(pool))

    def _negative_places(self, positive_place_ids: set[str], count: int) -> List[str]:
        choices = [p for p in self.gallery_place_ids if p not in positive_place_ids]
        return self.rng.sample(choices, k=min(count, len(choices)))

    def sample_batch(self, batch_queries: int) -> Dict[str, Any]:
        query_ids=[]; candidates=[]; same_targets=[]; absent_target=[]; query_place_ids=[]; positive_place_ids=[]
        for _ in range(batch_queries):
            make_present = (self.rng.random() < self.present_fraction) or not self.absent
            if make_present:
                place = self.rng.choice(self.gallery); pid = str(place["place_id"])
                query_view = self._query_view(place); qid = str(query_view["sample_id"])
                valid_positive_ids = {
                    str(value) for value in query_view.get("spatial_positive_place_ids", [pid])
                    if str(value) in self.anchor_by_place
                }
                if not valid_positive_ids:
                    raise RuntimeError(f"present query {qid} has no valid spatial-positive gallery node")
                if len(valid_positive_ids) > self.top_k:
                    raise RuntimeError(f"query {qid} has {len(valid_positive_ids)} positives but top_k={self.top_k}")
                cand_places = sorted(valid_positive_ids)
                for hp in self.hard_by_place.get(pid, [])[: max(0, self.top_k - 1)]:
                    if hp not in cand_places and hp not in valid_positive_ids: cand_places.append(hp)
                for npid in self._negative_places(valid_positive_ids, self.top_k - len(cand_places)):
                    if npid not in cand_places: cand_places.append(npid)
                cand_places = cand_places[:self.top_k]
                self.rng.shuffle(cand_places)
                cids = [self.anchor_by_place[x] for x in cand_places]
                st = [1.0 if x in valid_positive_ids else 0.0 for x in cand_places]
                query_place_ids.append(pid); positive_place_ids.append(sorted(valid_positive_ids)); absent_target.append(0.0)
            else:
                place = self.rng.choice(self.absent); pid = str(place["place_id"])
                qid = anchor_sample_id(place)
                cand_places = self._negative_places(set(), self.top_k)
                cids = [self.anchor_by_place[x] for x in cand_places]
                st = [0.0] * len(cids)
                query_place_ids.append(pid); positive_place_ids.append([]); absent_target.append(1.0)
            while len(cids) < self.top_k:
                choices = [value for value in self.gallery_place_ids if self.anchor_by_place[value] not in cids]
                if not choices:
                    raise RuntimeError("cannot construct a unique top-k candidate set")
                extra = self.rng.choice(choices)
                cids.append(self.anchor_by_place[extra]); st.append(1.0 if extra in set(positive_place_ids[-1]) else 0.0)
            query_ids.append(qid); candidates.append(cids[:self.top_k]); same_targets.append(st[:self.top_k])
        return {"query_ids": query_ids, "candidate_ids": candidates, "same_targets": same_targets, "absent_target": absent_target, "query_place_ids": query_place_ids, "positive_place_ids": positive_place_ids}


def load_checkpoint(system: PanoramicVPRV2System, path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "system" in ckpt:
        system.load_state_dict(ckpt["system"], strict=False)
    elif "head" in ckpt:
        system.descriptor_head.load_state_dict(ckpt["head"], strict=True)
    else:
        raise RuntimeError(f"unsupported checkpoint keys: {sorted(ckpt.keys())}")
    return ckpt


def encode_query_candidates(system: PanoramicVPRV2System, images: torch.Tensor, batch_queries: int, top_k: int) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    # Phase C freezes DINOv2; descriptor head and matcher still receive gradients.
    with torch.no_grad():
        tokens = system.backbone.forward_tokens(images)
    enc = system.descriptor_head.forward_tokens(tokens.float())
    query = {k: v[:batch_queries] for k, v in enc.items() if k in ("global", "ring", "cluster_mass", "dustbin_fraction")}
    cand = {}
    for k in ("global", "ring", "cluster_mass", "dustbin_fraction"):
        tail = enc[k][batch_queries:]
        cand[k] = tail.reshape(batch_queries, top_k, *tail.shape[1:])
    return query, cand


def save_checkpoint(path: Path, system, optimizer, scheduler, scaler, args, step: int, best_metric: float, metrics: Dict[str, Any], sampler_rng_state: Any) -> None:
    core = unwrap(system)
    atomic_torch_save(path, {"phase": TrainingPhase.C.value, "system": core.state_dict(), "descriptor_head": core.descriptor_head.state_dict(), "matcher": core.matcher.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "global_step": int(step), "best_metric": float(best_metric), "sampler_rng_state": sampler_rng_state, "args": vars(args), "metrics": metrics, "created_at": now(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "data_manifest_sha256": sha_path(ROOT / "data/manifests" / f"{args.train_split_tag}_shard_manifest.json"), "architecture_validation_sha256": sha_path(ROOT / "runtime/architecture_v1_validation.json")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split-tag", default="train_native_erp224_r3")
    ap.add_argument("--checkpoint", required=True, help="Phase B best/last checkpoint")
    ap.add_argument("--output-name", default="phase_c_native_erp224_r3")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-queries", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--present-fraction", type=float, default=0.65)
    ap.add_argument("--decoder", default="pil")
    ap.add_argument("--base-lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--positive-weight", type=float, default=2.0)
    ap.add_argument("--pairwise-weight", type=float, default=1.0)
    ap.add_argument("--open-set-weight", type=float, default=1.0)
    ap.add_argument("--brier-weight", type=float, default=0.25)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp-init-scale", type=float, default=1024.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    ddp, rank, local_rank, world = dist_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank); np.random.seed((args.seed + rank) % (2**32 - 1))
    sampler = PhaseCSampler(args.train_split_tag, args.seed + rank * 100003, args.top_k, args.present_fraction)
    sample_index = read_cached_sample_index(v2_sample_shards(ROOT, args.train_split_tag))
    if sample_index is None: raise RuntimeError(f"missing compact sample index for {args.train_split_tag}")
    store = TarImageStore(sample_index)
    system = PanoramicVPRV2System().to(device); load_checkpoint(system, Path(args.checkpoint), device)
    phase_info = system.configure_phase(TrainingPhase.C)
    groups = optimizer_parameter_groups(system.backbone, system.descriptor_head, system.matcher, TrainingPhase.C, base_lr=args.base_lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(groups); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps), eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=args.amp_init_scale)
    step = 0; best = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False); system.load_state_dict(ckpt["system"], strict=False); optimizer.load_state_dict(ckpt["optimizer"]); scheduler.load_state_dict(ckpt["scheduler"]); scaler.load_state_dict(ckpt["scaler"]); step=int(ckpt.get("global_step",0)); best=float(ckpt.get("best_metric",best))
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu() if hasattr(ckpt["torch_rng"], "cpu") else ckpt["torch_rng"])
        if device.type == "cuda" and ckpt.get("cuda_rng"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        if ckpt.get("sampler_rng_state") is not None:
            sampler.rng.setstate(ckpt["sampler_rng_state"])
    train_module = PhaseCTrainModule(system).to(device)
    model = DDP(train_module, device_ids=[local_rank], find_unused_parameters=False) if ddp else train_module
    model.train(True)

    out_dir = ROOT / "outputs/checkpoints" / args.output_name; metric_path = ROOT / "outputs/metrics" / f"{args.output_name}_train_metrics.jsonl"; summary_path = ROOT / "outputs/metrics" / f"{args.output_name}_summary.json"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True); metric_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(ROOT / "outputs/logs" / args.output_name / "train_config.json", {"created_at": now(), "argv": sys.argv[:], "args": vars(args), "phase_info": phase_info, "world_size": world, "note_zh": "Phase C：候选集合显式包含 present 正节点；absent query 全候选为负，target=UNKNOWN。"})
    rows=[]; step_times=[]
    try:
        while step < args.steps:
            t0=time.perf_counter(); spec=sampler.sample_batch(args.batch_queries)
            flat_candidate_ids=[sid for row in spec["candidate_ids"] for sid in row]
            image_ids=list(spec["query_ids"])+flat_candidate_ids
            images=torch.stack([image_to_tensor(store.read_jpeg(sid), decoder=args.decoder) for sid in image_ids]).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type=="cuda"):
                out = model(images, args.batch_queries, args.top_k)
                same_targets=torch.tensor(spec["same_targets"], dtype=torch.float32, device=device)
                absent_target=torch.tensor(spec["absent_target"], dtype=torch.float32, device=device)
                pair_loss=pairwise_same_place_loss(out["candidate_logits"], same_targets, positive_weight=args.positive_weight)
                os_loss=multi_positive_open_set_loss(out["logits_with_unknown"], same_targets.bool(), absent_target.bool())
                br_loss=open_set_brier_loss(out["unknown_probability"], absent_target)
                loss=args.pairwise_weight*pair_loss + args.open_set_weight*os_loss + args.brier_weight*br_loss
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(system.parameters(), args.grad_clip)
            scale_before = scaler.get_scale(); scaler.step(optimizer); scaler.update(); optimizer_updated = scaler.get_scale() >= scale_before
            if optimizer_updated: scheduler.step()
            if device.type=="cuda": torch.cuda.synchronize(device)
            step += 1; dt=time.perf_counter()-t0; step_times.append(dt)
            with torch.no_grad():
                pred=out["logits_with_unknown"].argmax(dim=1); unknown_idx=args.top_k
                present_mask=(absent_target < 0.5); absent_mask=~present_mask
                if present_mask.any():
                    present_prediction = pred[present_mask]
                    valid_candidate = present_prediction < args.top_k
                    candidate_index = present_prediction.clamp(max=args.top_k - 1)
                    candidate_correct = same_targets[present_mask].gather(1, candidate_index.unsqueeze(1)).squeeze(1).bool()
                    present_acc = (valid_candidate & candidate_correct).float().mean().item()
                else:
                    present_acc = None
                absent_unknown=((pred[absent_mask]==unknown_idx).float().mean().item() if absent_mask.any() else None)
            row={"created_at":now(),"global_step":step,"rank":rank,"loss":float(loss.detach().cpu()),"pairwise_same_place_loss":float(pair_loss.detach().cpu()),"open_set_node_or_unknown_loss":float(os_loss.detach().cpu()),"open_set_brier_loss":float(br_loss.detach().cpu()),"present_candidate_accuracy":present_acc,"absent_unknown_accuracy":absent_unknown,"unknown_probability_mean":float(out["unknown_probability"].detach().float().mean().cpu()),"step_seconds":dt,"lr_descriptor":optimizer.param_groups[0]["lr"],"optimizer_step_skipped":not optimizer_updated}
            rows.append(row)
            if rank==0 and (step==1 or step%args.log_every==0):
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
                with metric_path.open("a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")
            if rank==0 and (step%args.checkpoint_every==0 or step==args.steps):
                metric=float(row["loss"]); save_checkpoint(out_dir/"last.pt", system, optimizer, scheduler, scaler, args, step, min(best, metric), row, sampler.rng.getstate())
                if metric < best: best=metric; save_checkpoint(out_dir/"best.pt", system, optimizer, scheduler, scaler, args, step, best, row, sampler.rng.getstate())
    finally:
        store.close()
    if rank==0:
        payload={"created_at":now(),"phase":TrainingPhase.C.value,"steps":step,"best_metric":best,"sampler_rng_state_recorded": True,"last_metrics":rows[-1] if rows else {},"timing":{"step_seconds":summary(step_times)},"checkpoint_last":str(out_dir/"last.pt"),"checkpoint_best":str(out_dir/"best.pt"),"architecture_validation_sha256":sha_path(ROOT/"runtime/architecture_v1_validation.json")}
        atomic_json(summary_path,payload)
        with (ROOT/"runtime/command_records_v3.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps({"created_at":now(),"step":"train_phase_c_v2","argv":sys.argv[:],"outputs":[str(summary_path),str(out_dir/"last.pt"),str(out_dir/"best.pt")],"returncode":0},ensure_ascii=False,sort_keys=True)+"\n")
        print(json.dumps({"summary":str(summary_path),"last":str(out_dir/"last.pt"),"best":str(out_dir/"best.pt")},ensure_ascii=False,indent=2))
    if ddp: torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
