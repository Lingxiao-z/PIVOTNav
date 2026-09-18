from __future__ import annotations

import copy
import hashlib
import sys
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
R363_PYTHON = ROOT / "r363_python"
sys.path.insert(0, str(R363_PYTHON))
EXPECTED_SHA256 = "dd6326d3306b24e8538e42cd5ee0378b118b19e0045fc89aa2567dc7337a8353"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class R363BearingRuntime:
    """R36.3 private bearing tail/head over the frozen R36.1 backbone."""

    def __init__(self, vpr: Any, device: str = "cuda") -> None:
        from r362_bearing_head_v2 import R362BearingHeadConfig, R362PatchBearingHead

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint = Path(os.environ.get("PIVOTNAV_R363_CHECKPOINT", str(ROOT / "models" / "r363" / "r363_bearing.pt")))
        if _sha256(checkpoint) != EXPECTED_SHA256:
            raise RuntimeError("R36.3 bearing checkpoint SHA256 mismatch")
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        if payload.get("schema_version") != "r363_limited_unfreeze_checkpoint_v1":
            raise RuntimeError("unsupported R36.3 checkpoint schema")

        self.backbone = copy.deepcopy(vpr.runtime.encoder_system.backbone)
        state = self.backbone.model.state_dict()
        tail = payload["bearing_tail"]
        expected = {name for name in state if name.startswith(("blocks.11.", "norm."))}
        if set(tail) != expected:
            raise RuntimeError("R36.3 private backbone tail does not match R36.1")
        state.update(tail)
        self.backbone.model.load_state_dict(state, strict=True)
        self.backbone.requires_grad_(False).eval().to(self.device)
        self.head = R362PatchBearingHead(R362BearingHeadConfig())
        self.head.load_state_dict(payload["bearing_head"], strict=True)
        self.head.requires_grad_(False).eval().to(self.device)
        self.target_tokens: dict[int, torch.Tensor] = {}

    def _tokens(self, rgb: np.ndarray, prepare_erp: Any) -> torch.Tensor:
        image = prepare_erp(rgb).to(self.device, dtype=torch.float32).div_(255.0)
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        with torch.inference_mode():
            return self.backbone.forward_tokens(((image - mean) / std).unsqueeze(0)).float()

    def add_node(self, node_id: int, rgb: np.ndarray, prepare_erp: Any) -> None:
        self.target_tokens[int(node_id)] = self._tokens(rgb, prepare_erp)

    def predict_to_node(self, rgb: np.ndarray, node_id: int, prepare_erp: Any) -> dict[str, float]:
        source = self._tokens(rgb, prepare_erp)
        target = self.target_tokens[int(node_id)]
        with torch.inference_mode():
            output = self.head(source, target)
        return {
            "bearing_deg": float(output["bearing_angle_degrees"].float().reshape(-1)[0]),
            "bearing_confidence": float(output["bearing_confidence"].float().reshape(-1)[0]),
        }
