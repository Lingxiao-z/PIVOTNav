from __future__ import annotations

import __future__
import importlib.abc
import importlib.machinery
import json
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


OMNIGUARD_ROOT = Path(__file__).resolve().parents[4]
UNIK3D_ROOT = OMNIGUARD_ROOT / "third_party" / "unik3d"
DEFAULT_UNIK3D_MODEL_DIR = OMNIGUARD_ROOT / "deployment" / "checkpoints" / "unik3d-vitl"

_UNIK3D_IMPORTS: tuple[Any, ...] | None = None


class _UniK3DFutureAnnotationsLoader(importlib.machinery.SourceFileLoader):
    """Compile UniK3D sources with postponed annotation evaluation on Python 3.8."""

    def source_to_code(self, data, path, *, _optimize=-1):
        return compile(
            data,
            path,
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
            optimize=_optimize,
        )

    def get_code(self, fullname):
        source_path = self.get_filename(fullname)
        source_data = self.get_data(source_path)
        return self.source_to_code(source_data, source_path)


class _UniK3DFutureAnnotationsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "unik3d" and not fullname.startswith("unik3d."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        loader = spec.loader
        if not isinstance(loader, importlib.machinery.SourceFileLoader):
            return spec
        spec.loader = _UniK3DFutureAnnotationsLoader(fullname, loader.path)
        return spec


def _install_unik3d_py38_compat() -> None:
    if sys.version_info >= (3, 10):
        return
    for finder in sys.meta_path:
        if isinstance(finder, _UniK3DFutureAnnotationsFinder):
            return
    sys.meta_path.insert(0, _UniK3DFutureAnnotationsFinder())


def _load_unik3d_imports() -> tuple[Any, ...]:
    global _UNIK3D_IMPORTS
    if _UNIK3D_IMPORTS is not None:
        return _UNIK3D_IMPORTS
    if not UNIK3D_ROOT.is_dir():
        raise FileNotFoundError(f"UniK3D code directory not found inside OmniGuard: {UNIK3D_ROOT}")
    if str(UNIK3D_ROOT) not in sys.path:
        sys.path.insert(0, str(UNIK3D_ROOT))
    _install_unik3d_py38_compat()
    if "wandb" not in sys.modules:
        wandb_stub = types.ModuleType("wandb")
        wandb_stub.log = lambda *args, **kwargs: None
        wandb_stub.Image = lambda *args, **kwargs: args[0] if args else None
        sys.modules["wandb"] = wandb_stub
    try:
        import torchvision.transforms.v2.functional as TF  # type: ignore
        from unik3d.models import UniK3D  # type: ignore
        from unik3d.models.unik3d import _postprocess, get_paddings, get_resize_factor  # type: ignore
        from unik3d.utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Cannot import the OmniGuard-local UniK3D runtime. "
            "This is only required for fisheye ray/depth generation.\n"
            f"Expected code at: {UNIK3D_ROOT}\n"
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc
    _UNIK3D_IMPORTS = (
        UniK3D,
        _postprocess,
        get_paddings,
        get_resize_factor,
        IMAGENET_DATASET_MEAN,
        IMAGENET_DATASET_STD,
        TF,
    )
    return _UNIK3D_IMPORTS


def normalise_rays(rays: np.ndarray) -> np.ndarray:
    rays = np.asarray(rays, dtype=np.float32)
    if rays.ndim != 3 or rays.shape[2] != 3:
        raise ValueError(f"Expected rays with shape (H, W, 3), got {rays.shape}")
    norm = np.linalg.norm(rays, axis=2, keepdims=True)
    rays = rays / np.clip(norm, 1e-8, None)
    return np.nan_to_num(rays, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def resize_rays(rays: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    rays = normalise_rays(rays)
    if rays.shape[:2] == (target_h, target_w):
        return rays
    tensor = torch.from_numpy(rays).permute(2, 0, 1).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
    resized = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return normalise_rays(resized)


def resize_scalar_map(values: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected scalar map with shape (H, W), got {values.shape}")
    if values.shape == (target_h, target_w):
        return values.astype(np.float32, copy=False)
    tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def rays_to_meta_payload(rays: np.ndarray, *, meta_format: str = "rays_lr8_fp16") -> dict[str, np.ndarray]:
    rays = normalise_rays(rays)
    h, w = rays.shape[:2]
    if meta_format == "rays_fp32":
        return {
            "meta_format": np.array(meta_format),
            "rays": rays.astype(np.float32, copy=False),
            "orig_hw": np.array([h, w], dtype=np.int32),
        }
    if meta_format == "rays_lr8_fp16":
        factor = 8
        low_h = max(1, h // factor)
        low_w = max(1, w // factor)
        rays_lr = resize_rays(rays, (low_h, low_w)).astype(np.float16, copy=False)
        return {
            "meta_format": np.array(meta_format),
            "rays_lr": rays_lr,
            "orig_hw": np.array([h, w], dtype=np.int32),
            "factor": np.array([factor], dtype=np.int32),
        }
    raise ValueError(f"Unsupported UniK3D rays meta_format: {meta_format}")


def save_rays_npz(path: str | Path, rays: np.ndarray, *, meta_format: str = "rays_lr8_fp16") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = rays_to_meta_payload(rays, meta_format=meta_format)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp_path.replace(path)


def load_rays_npz(path: str | Path, *, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Fisheye rays NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "rays_lr" in data:
            rays = np.asarray(data["rays_lr"], dtype=np.float32)
        elif "rays" in data:
            rays = np.asarray(data["rays"], dtype=np.float32)
        else:
            raise KeyError(f"Fisheye rays NPZ has no 'rays_lr' or 'rays': {path}")
    if target_hw is not None:
        return resize_rays(rays, target_hw)
    return normalise_rays(rays)


class UniK3DRayGenerator:
    def __init__(
        self,
        *,
        model_dir: str | Path = DEFAULT_UNIK3D_MODEL_DIR,
        device: str | torch.device = "cuda",
        resolution_level: int = 9,
        interpolation_mode: str = "bilinear",
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"UniK3D model directory not found: {self.model_dir}")
        self.device = torch.device(device)
        self.resolution_level = int(resolution_level)
        self.interpolation_mode = str(interpolation_mode)
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        UniK3D = _load_unik3d_imports()[0]
        model = UniK3D.from_pretrained(str(self.model_dir))
        model.resolution_level = self.resolution_level
        model.interpolation_mode = self.interpolation_mode
        self._model = model.to(self.device).eval()
        return self._model

    def infer_rays(self, frame_bgr: np.ndarray) -> np.ndarray:
        _distance, rays = self.infer_depth_and_rays(frame_bgr)
        return rays

    def infer_depth_and_rays(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected BGR frame HxWx3 for UniK3D inference, got {frame.shape}")
        rgb = cv2.cvtColor(frame.astype(np.uint8, copy=False), cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(self.device)
        distance_t, rays_t = self._infer_depth_and_rays_with_model(self._load_model(), rgb_t)
        if distance_t is None:
            raise RuntimeError("UniK3D output has no metric point/range depth; cannot build RGB geo overlay.")
        if rays_t is None:
            raise RuntimeError("UniK3D output has no per-pixel rays; cannot build fisheye azimuth/overlay.")
        distance = distance_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        rays = rays_t[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32, copy=False)
        target_hw = rgb.shape[:2]
        return resize_scalar_map(distance, target_hw), resize_rays(rays, target_hw)

    @staticmethod
    def _infer_with_model(model: Any, rgb: torch.Tensor) -> torch.Tensor:
        _distance, rays = UniK3DRayGenerator._infer_depth_and_rays_with_model(model, rgb)
        if rays is None:
            raise RuntimeError("UniK3D output has no rays or points from which rays can be computed.")
        return rays

    @staticmethod
    def _infer_depth_and_rays_with_model(model: Any, rgb: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        (
            _UniK3D,
            _postprocess,
            get_paddings,
            get_resize_factor,
            IMAGENET_DATASET_MEAN,
            IMAGENET_DATASET_STD,
            TF,
        ) = _load_unik3d_imports()
        ratio_bounds = model.shape_constraints["ratio_bounds"]
        pixels_bounds = [
            model.shape_constraints["pixels_min"],
            model.shape_constraints["pixels_max"],
        ]
        if hasattr(model, "resolution_level"):
            if not (0 <= int(model.resolution_level) < 10):
                raise ValueError("UniK3D resolution_level must be in [0, 10)")
            pixels_range = pixels_bounds[1] - pixels_bounds[0]
            interval = pixels_range / 10
            pixels_bounds = (
                model.resolution_level * interval + pixels_bounds[0],
                (model.resolution_level + 1) * interval + pixels_bounds[0],
            )

        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        _, _, h, w = rgb.shape
        rgb = rgb.to(model.device)

        paddings, (padded_h, padded_w) = get_paddings((h, w), ratio_bounds)
        pad_left, pad_right, pad_top, pad_bottom = paddings
        _, (new_h, new_w) = get_resize_factor((padded_h, padded_w), pixels_bounds)

        rgb = TF.normalize(rgb.float() / 255.0, mean=IMAGENET_DATASET_MEAN, std=IMAGENET_DATASET_STD)
        rgb = F.pad(rgb, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        rgb = F.interpolate(rgb, size=(new_h, new_w), mode="bilinear", align_corners=False)

        device_type = "cuda" if model.device.type == "cuda" else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type=device_type,
            enabled=device_type == "cuda",
            dtype=torch.float16,
        ):
            _, model_outputs = model.encode_decode({"image": rgb}, image_metas={})

        distance = None
        rays = None
        if "rays" in model_outputs:
            rays = _postprocess(
                model_outputs["rays"],
                (padded_h, padded_w),
                paddings=paddings,
                interpolation_mode=model.interpolation_mode,
            )
            rays = rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-5)

        if "points" in model_outputs:
            points = _postprocess(
                model_outputs["points"],
                (padded_h, padded_w),
                paddings=paddings,
                interpolation_mode=model.interpolation_mode,
            )
            distance = torch.norm(points, dim=1, keepdim=True).clamp(min=1e-5)
            if rays is None:
                rays = points / distance

        if distance is None and rays is None:
            available = sorted(str(key) for key in model_outputs.keys())
            raise RuntimeError(f"UniK3D output has no rays or points. Available keys: {available}")

        if rays is not None:
            rays = rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-5)
        return distance, rays


def camera_signature_payload(camera_info: dict[str, Any], camera_model: str) -> dict[str, Any]:
    keys = ["frame_id", "width", "height", "fx", "fy", "cx", "cy", "distortion_model", "d", "k"]
    payload = {"camera_model": str(camera_model)}
    for key in keys:
        if key in camera_info:
            value = camera_info[key]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            payload[key] = value
    return payload


def stable_camera_signature(camera_info: dict[str, Any], camera_model: str) -> str:
    import hashlib

    payload = camera_signature_payload(camera_info, camera_model)
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
