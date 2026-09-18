from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from .checkpoint_io import load_model_state_dict
from .fgfs_models import NTSInspiredResNet18FGFS, build_resnet18_fgfs_encoder
from .inference import FGFSInference, normalize_erp, validate_online_inputs
from .models import DINOv2PanoramaFGFS
from .projection import twelve_sector_views


RESNET = "nts_inspired_resnet18_fgfs_habitat_gs_v2"
DINO_TRAIN = "dinov2_panorama_fgfs_train_scenes_habitat_gs_v2"
DINO_ALL = "dinov2_panorama_fgfs_all_scenes_habitat_gs_v2"
MODELS = (RESNET, DINO_TRAIN, DINO_ALL)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_dino_sectors(backbone: torch.nn.Module, erp: torch.Tensor) -> torch.Tensor:
    views = twelve_sector_views(erp)
    mean = torch.tensor([0.485, 0.456, 0.406], device=erp.device)[None, None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=erp.device)[None, None, :, None, None]
    batch, sectors = views.shape[:2]
    features = backbone(((views - mean) / std).flatten(0, 1))
    if isinstance(features, dict):
        features = features["x_norm_clstoken"]
    if features.ndim != 2 or features.shape != (batch * sectors, 384):
        raise RuntimeError("DINO backbone did not return one 384-D token per sector")
    return features.reshape(batch, sectors, 384)


class PackagedFGFSPredictor:
    """Load a frozen v2 joint model while preserving the RGB-only contract."""

    def __init__(
        self,
        package_root: str | Path,
        model_name: str,
        device: str | torch.device = "cuda:0",
        *,
        fg_threshold: float = 0.5,
    ) -> None:
        self.root = Path(package_root).resolve()
        manifest_path = self.root / "inference_package_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "pure_rgb_panorama_fgfs_inference_package_v2":
            raise RuntimeError("package is not an FG/FS v2 inference package")
        if set(manifest.get("models", {})) != set(MODELS):
            raise RuntimeError("package does not contain the exact three frozen FG/FS models")
        if model_name not in manifest["models"]:
            raise KeyError(model_name)
        self.device = torch.device(device)
        record = manifest["models"][model_name]
        checkpoint = self.root / record["checkpoint_relative_path"]
        if sha256(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError("packaged checkpoint hash mismatch")
        model_state = load_model_state_dict(checkpoint, map_location=self.device)
        self.model_name = model_name
        self.external_dino: torch.nn.Module | None = None

        if model_name == RESNET:
            model = NTSInspiredResNet18FGFS(build_resnet18_fgfs_encoder()).to(self.device)
            model.load_state_dict(model_state, strict=True)
            self.inference = FGFSInference(model.eval(), "resnet", fg_threshold)
            return

        has_backbone = any(key.startswith("backbone.") for key in model_state)
        if has_backbone:
            backbone = self._load_official_dino(manifest)
            model = DINOv2PanoramaFGFS(backbone).to(self.device)
            model.load_state_dict(model_state, strict=True)
            self.inference = FGFSInference(model.eval(), "dino", fg_threshold)
        else:
            self.external_dino = self._load_official_dino(manifest).eval()
            model = DINOv2PanoramaFGFS(torch.nn.Identity()).to(self.device)
            model.load_state_dict(model_state, strict=True)
            self.inference = FGFSInference(model.eval(), "dino", fg_threshold)

    def _load_official_dino(self, manifest: dict) -> torch.nn.Module:
        record = manifest["official_dino"]
        weight = self.root / record["weight_relative_path"]
        repository = self.root / record["repository_relative_path"]
        if sha256(weight) != record["weight_sha256"]:
            raise RuntimeError("packaged official DINO weight hash mismatch")
        if not repository.is_dir():
            raise FileNotFoundError(repository)
        model = torch.hub.load(
            str(repository), record["torch_hub_entry"], pretrained=False, source="local",
        ).to(self.device)
        state = torch.load(weight, map_location=self.device, weights_only=True)
        model.load_state_dict(state, strict=True)
        return model

    @torch.inference_mode()
    def __call__(self, inputs: Mapping[str, object]) -> dict[str, torch.Tensor]:
        if self.external_dino is None or "current_features" in inputs:
            return self.inference(inputs)
        validate_online_inputs(inputs)
        source = normalize_erp(inputs["current_erp_rgb"], device=self.device)
        goal = normalize_erp(inputs["goal_erp_rgb"], device=self.device)
        if source.shape != goal.shape:
            raise ValueError("Current and Goal ERP RGB batch shapes must match")
        return self.inference({
            "current_features": encode_dino_sectors(self.external_dino, source),
            "goal_features": encode_dino_sectors(self.external_dino, goal),
        })


# Compatibility name; legacy FS-only package schemas are still rejected.
PackagedFSPredictor = PackagedFGFSPredictor
