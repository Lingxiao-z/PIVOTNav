from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
IMPL = ROOT / "_impl"
for _path in (IMPL,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


class NavigableCuriosityField:
    """Unified FS inference with the internal FG checkpoint-compatible branch."""

    def __init__(self, weights_root: Path, config: dict[str, Any]):
        self.config = config
        self.weights_root = Path(weights_root).expanduser().resolve()
        self.device = config.get("device", "cuda")
        self.model = None
        self.backbone = None
        if str(weights_root) not in ("", "."):
            self._load()

    def _load(self) -> None:
        import torch

        from panorama_fs.dino_fgfs_revision3 import RevisedDINOv2PanoramaFGFSV3
        from panorama_fs.dino_fgfs_revision3_b2 import RevisedDINOv2PanoramaFGFSV3LastBlockB2

        repository = str(IMPL)
        dino_weight = self.weights_root / "dinov2/dinov2_vits14_pretrain.pth"
        checkpoint = self.weights_root / "fs/step_019000.pt"
        self.backbone = torch.hub.load(
            repository, "dinov2_vits14", pretrained=True,
            weights=str(dino_weight), source="local",
        ).to(self.device).eval()
        decoder = RevisedDINOv2PanoramaFGFSV3().to(self.device)
        self.model = RevisedDINOv2PanoramaFGFSV3LastBlockB2(
            self.backbone.blocks[-1], self.backbone.norm, decoder,
        ).to(self.device).eval()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model"], strict=True)

    def _tensor(self, image: np.ndarray):
        import torch
        import torch.nn.functional as F

        value = torch.from_numpy(np.ascontiguousarray(image[..., :3])).permute(2, 0, 1).float().div_(255.0)
        value = F.interpolate(value[None], size=(224, 448), mode="bilinear", align_corners=False).to(self.device)
        mean = value.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = value.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (value - mean) / std

    def predict(self, current_rgb: np.ndarray, goal_rgb: np.ndarray) -> dict[str, np.ndarray]:
        if self.model is None:
            return {
                "fg_probability": np.ones(12, dtype=np.float32),
                "fs_scores": np.zeros(12, dtype=np.float32),
            }
        import torch

        current, goal = self._tensor(current_rgb), self._tensor(goal_rgb)
        with torch.inference_mode():
            tokens = self.backbone.prepare_tokens_with_masks(torch.cat((current, goal), dim=0))
            for block in self.backbone.blocks[:-1]:
                tokens = block(tokens)
            current_tokens, goal_tokens = tokens.float().chunk(2, dim=0)
            output = self.model(current_tokens, goal_tokens)
        return {
            "fg_probability": torch.sigmoid(output.fg_logits).cpu().numpy()[0],
            "fs_scores": output.fs_scores.cpu().numpy()[0],
        }

    def select(self, scores: dict[str, np.ndarray], distances: np.ndarray) -> dict[str, np.ndarray | int | None]:
        fg = np.asarray(scores["fg_probability"], dtype=np.float32).reshape(12)
        fs = np.asarray(scores["fs_scores"], dtype=np.float32).reshape(12)
        distance = np.asarray(distances, dtype=np.float32).reshape(-1)
        if distance.size != 360:
            raise ValueError("OmniTrav must provide 360 distances")
        valid = fg >= float(self.config.get("fg_threshold", 0.5))
        clearance = np.asarray([
            np.percentile(distance[(i * 30 + np.arange(-10, 11)) % 360], 20)
            for i in range(12)
        ])
        valid &= clearance >= float(self.config.get("frontier_clearance_m", 0.70))
        masked = np.where(valid, fs, -np.inf)
        return {
            "fg_probability": fg,
            "fs_scores": fs,
            "valid_mask": valid,
            "selected_sector": int(np.argmax(masked)) if valid.any() else None,
        }


def smoke_curiosity(current: np.ndarray, goal: np.ndarray) -> dict[str, bool]:
    field = NavigableCuriosityField(Path(), {})
    scores = field.predict(current, goal)
    selected = field.select(scores, np.full(360, 8.0, dtype=np.float32))
    return {"ok": selected["valid_mask"].shape == (12,)}
