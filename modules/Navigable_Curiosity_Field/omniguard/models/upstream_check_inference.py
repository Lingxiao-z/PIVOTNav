from __future__ import annotations

import __future__
import importlib.abc
import importlib.machinery
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as error:
    if getattr(error, "name", None) == "torch":
        isaaclab_root = Path(__file__).resolve().parents[9]
        diagnostic_message = (
            "omniguard_esdf.models.upstream_check_inference 导入失败：当前 Python 环境缺少 torch。\n"
            f"sys.executable: {sys.executable}\n"
            f"sys.version: {sys.version}\n"
            "建议优先使用 IsaacLab 对应解释器运行算法层脚本。\n"
            f"  cd {isaaclab_root}\n"
            "  ./isaaclab.sh -p scripts/reinforcement_learning/navigation/algorithm_layer/scripts/run_scene_diffusion_guidance.py\n"
        )
        raise ModuleNotFoundError(diagnostic_message) from error
    raise

from .traversability import infer_architecture_from_checkpoint


_LOCAL_UNIFIED_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "unified-depth-processor"
_ENV_UNIFIED_ROOT = os.environ.get("OMNIGUARD_UNIFIED_ROOT")
UNIFIED_ROOT = Path(_ENV_UNIFIED_ROOT).expanduser() if _ENV_UNIFIED_ROOT else _LOCAL_UNIFIED_ROOT
if not UNIFIED_ROOT.is_dir():
    raise FileNotFoundError(f"Upstream unified-depth-processor not found: {UNIFIED_ROOT}")
if str(UNIFIED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIFIED_ROOT))


class _Dinov3FutureAnnotationsLoader(importlib.machinery.SourceFileLoader):
    """Compile dinov3 sources with postponed annotation evaluation on Python < 3.10.

    The upstream dinov3 package uses PEP 604 annotations such as `float | None`.
    Those annotations are valid for the training code path, but importing them from
    Python 3.8 raises `TypeError: unsupported operand type(s) for |`.
    We keep the upstream model implementation intact and only enable the
    `annotations` future flag at import time so the original forward path remains
    unchanged.
    """

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


class _Dinov3FutureAnnotationsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "dinov3" and not fullname.startswith("dinov3."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        loader = spec.loader
        if not isinstance(loader, importlib.machinery.SourceFileLoader):
            return spec
        spec.loader = _Dinov3FutureAnnotationsLoader(fullname, loader.path)
        return spec


def _install_dinov3_py38_compat() -> None:
    if sys.version_info >= (3, 10):
        return
    for finder in sys.meta_path:
        if isinstance(finder, _Dinov3FutureAnnotationsFinder):
            return
    sys.meta_path.insert(0, _Dinov3FutureAnnotationsFinder())


_install_dinov3_py38_compat()

from scripts.traversability.dataset import compute_azimuth_map as upstream_compute_azimuth_map  # noqa: E402
from scripts.traversability.model import TraversabilityModel as UpstreamTraversabilityModel  # noqa: E402

from .unik3d_runtime import (  # noqa: E402
    DEFAULT_UNIK3D_MODEL_DIR,
    UniK3DRayGenerator,
    load_rays_npz,
    normalise_rays,
    save_rays_npz,
    stable_camera_signature,
)


@dataclass
class ModelMetadata:
    checkpoint_path: str
    checkpoint_epoch: int | None
    backbone_variant: str
    embed_dim: int
    num_blocks: int
    multi_scale: bool
    layer_indices: tuple[int, ...]
    decoder_dim: int
    angle_dim: int
    num_angles: int
    checkpoint_args: dict[str, Any]
    strict_load_ok: bool


@dataclass
class ModelOutput:
    image_tensor: torch.Tensor
    azimuth_tensor: torch.Tensor
    camera_model: str
    camera_intrinsics: dict[str, Any]
    camera_geometry: dict[str, Any]
    observed_angle_mask: np.ndarray
    raw_distance_m: np.ndarray
    combined_distance_m: np.ndarray
    effective_distance_m: np.ndarray
    exist_logit: np.ndarray
    exist_probability: np.ndarray
    latency_ms: float
    resize_hw: tuple[int, int]
    original_hw: tuple[int, int]


def _append_saved_weight_candidates(candidates: list[Path], saved: Any) -> None:
    if not saved:
        return
    saved_path = Path(str(saved)).expanduser()
    candidates.append(UNIFIED_ROOT / "model" / saved_path.name)
    if not saved_path.is_absolute():
        candidates.append(UNIFIED_ROOT / saved_path)
    candidates.append(saved_path)


def _default_backbone_weight_name(backbone_type: str) -> str:
    value = str(backbone_type or "vitb16").strip().lower()
    if value in {"vits16", "vits"}:
        return "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    if value in {"vitb16", "vitb"}:
        return "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    if value in {"vitl16", "vitl"}:
        return "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    return "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


def _checkpoint_has_full_backbone_state(state_dict: dict[str, Any]) -> bool:
    return any(str(key).startswith("backbone.model.") for key in state_dict.keys())


def _resolve_backbone_weights(
    cli_path: str | None,
    checkpoint_args: dict[str, Any],
    *,
    allow_missing_full_checkpoint: bool = False,
) -> Path | None:
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    _append_saved_weight_candidates(candidates, checkpoint_args.get("backbone_weights"))
    _append_saved_weight_candidates(candidates, checkpoint_args.get("student_backbone_weights"))
    candidates.append(UNIFIED_ROOT / "model" / _default_backbone_weight_name(str(checkpoint_args.get("backbone_type", "vitb16"))))

    for path in candidates:
        if path.exists():
            return path
    if allow_missing_full_checkpoint:
        print(
            "[WARN] Could not resolve DINOv3 pretrained backbone weights; "
            "continuing because checkpoint contains full backbone state. "
            f"Candidates: {candidates}",
            flush=True,
        )
        return None
    raise FileNotFoundError(f"Could not resolve upstream backbone weights from candidates: {candidates}")


def _compute_target_hw(orig_h: int, orig_w: int, img_size: int) -> tuple[int, int]:
    scale = min(float(img_size) / float(max(orig_h, orig_w)), 1.0)
    resize_h = max(16, int(round(orig_h * scale / 16.0) * 16))
    resize_w = max(16, int(round(orig_w * scale / 16.0) * 16))
    return resize_h, resize_w


def _camera_model_from_distortion(distortion_model: str | None) -> str | None:
    distortion = str(distortion_model or "").strip().lower()
    if not distortion:
        return None
    if distortion in {"plumb_bob", "pinhole", "perspective"}:
        return "pinhole"
    if "spherical" in distortion or "equirect" in distortion or "pano" in distortion:
        return "equirectangular"
    if "fisheye" in distortion:
        return "fisheye"
    return None


def _resolve_camera_model(camera_cfg: dict[str, Any], runtime_camera_info: dict[str, Any] | None) -> str:
    configured = str(camera_cfg.get("model", "auto")).strip().lower()
    if configured in {"", "auto"}:
        source_info = runtime_camera_info if runtime_camera_info is not None else camera_cfg
        inferred = _camera_model_from_distortion(source_info.get("distortion_model"))
        if inferred is None:
            raise ValueError(
                "Cannot infer camera model from "
                f"{'CameraInfo' if runtime_camera_info is not None else 'model.camera'}.distortion_model="
                f"{source_info.get('distortion_model')!r}. Expected plumb_bob/pinhole, "
                "fisheyePolynomial, or fisheyeSpherical."
            )
        return inferred
    if configured not in {"auto", ""}:
        if configured in {"pano", "panoramic", "spherical"}:
            return "equirectangular"
        if configured not in {"pinhole", "fisheye", "equirectangular"}:
            raise ValueError(f"Unsupported configured camera model: {configured!r}")
        return configured
    raise ValueError(f"Unsupported configured camera model: {configured!r}")


def _load_rays_array(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Fisheye rays value is None.")
    if isinstance(value, (str, os.PathLike)):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Configured fisheye rays file does not exist: {path}")
        return load_rays_npz(path)
    return normalise_rays(np.asarray(value, dtype=np.float32))


def _camera_info_intrinsics(runtime_camera_info: dict[str, Any]) -> dict[str, float]:
    required = ("fx", "fy", "cx", "cy")
    missing = [key for key in required if runtime_camera_info.get(key) is None]
    if missing:
        raise ValueError(f"Runtime CameraInfo is missing required intrinsics: {missing}")
    return {key: float(runtime_camera_info[key]) for key in required}


def _configured_intrinsics(camera_cfg: dict[str, Any]) -> dict[str, float]:
    required = ("fx", "fy", "cx", "cy")
    missing = [key for key in required if camera_cfg.get(key) is None]
    if missing:
        raise ValueError(f"model.camera is missing required intrinsics: {missing}")
    return {key: float(camera_cfg[key]) for key in required}


def _resolve_intrinsics(
    camera_cfg: dict[str, Any],
    runtime_camera_info: dict[str, Any] | None,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    del image_width, image_height
    camera_model = _resolve_camera_model(camera_cfg, runtime_camera_info)
    if bool(camera_cfg.get("use_ros_camera_info", False)) and runtime_camera_info is not None:
        intrinsics: dict[str, Any] = _camera_info_intrinsics(runtime_camera_info)
    else:
        intrinsics = _configured_intrinsics(camera_cfg)

    if camera_model == "pinhole":
        return intrinsics

    if camera_model == "fisheye":
        if runtime_camera_info is not None:
            rays_value = (
                runtime_camera_info.get("rays")
                if runtime_camera_info.get("rays") is not None
                else runtime_camera_info.get("rays_lr")
                if runtime_camera_info.get("rays_lr") is not None
                else None
            )
            if rays_value is not None:
                intrinsics["rays"] = _load_rays_array(rays_value)
                intrinsics["rays_source"] = str(
                    runtime_camera_info.get("depth_meta_path")
                    or runtime_camera_info.get("rays_path")
                    or "runtime_camera_info"
                )
                return intrinsics
            rays_source = runtime_camera_info.get("rays_path") or runtime_camera_info.get("depth_meta_path")
            if rays_source:
                intrinsics["rays_source"] = str(Path(str(rays_source)).expanduser())
                return intrinsics
        rays_cfg = camera_cfg.get("rays_path") or camera_cfg.get("depth_meta_path")
        if rays_cfg:
            intrinsics["rays_source"] = str(Path(str(rays_cfg)).expanduser())
            return intrinsics
        return intrinsics

    if camera_model == "equirectangular":
        return intrinsics

    raise ValueError(f"Unsupported camera model for upstream check inference: {camera_model}")


def _serialise_geometry_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": [int(v) for v in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (str, type(None))):
        return value
    return str(value)


def _camera_geometry_dict(
    *,
    camera_model: str,
    intrinsics: dict[str, Any],
    original_hw: tuple[int, int],
    resize_hw: tuple[int, int],
) -> dict[str, Any]:
    geometry = {
        "camera_model": str(camera_model),
        "original_hw": [int(original_hw[0]), int(original_hw[1])],
        "resize_hw": [int(resize_hw[0]), int(resize_hw[1])],
    }
    for key, value in intrinsics.items():
        geometry[key] = _serialise_geometry_value(value)
    return geometry


def _build_azimuth_tensor(
    *,
    camera_model: str,
    intrinsics: dict[str, Any],
    orig_h: int,
    orig_w: int,
    resize_h: int,
    resize_w: int,
) -> torch.Tensor:
    if camera_model == "fisheye":
        if "rays" not in intrinsics:
            raise RuntimeError(
                "Fisheye camera requires per-pixel rays before building azimuth_map. "
                "Provide camera.depth_meta_path/rays_path or enable OmniGuard-local UniK3D ray generation."
            )
        rays = np.asarray(intrinsics["rays"], dtype=np.float32)
        if rays.shape[:2] != (orig_h, orig_w):
            raise ValueError(
                f"Fisheye rays shape {rays.shape[:2]} does not match RGB shape {(orig_h, orig_w)}"
            )
        rays_t = torch.from_numpy(rays).permute(2, 0, 1).unsqueeze(0).float()
        rays_t = F.interpolate(rays_t, size=(resize_h, resize_w), mode="bilinear", align_corners=False)
        rays_t = rays_t / torch.linalg.norm(rays_t, dim=1, keepdim=True).clamp(min=1e-8)
        return torch.atan2(rays_t[:, 0:1], rays_t[:, 2:3]).contiguous()

    azimuth_row = upstream_compute_azimuth_map(orig_w, camera_model, intrinsics).astype(np.float32, copy=False)
    azimuth_1d = torch.from_numpy(azimuth_row).unsqueeze(0).unsqueeze(0)
    azimuth_1d = F.interpolate(azimuth_1d, size=(1, resize_w), mode="bilinear", align_corners=False)
    return azimuth_1d.expand(-1, -1, resize_h, -1).contiguous()


def _azimuth_cache_key(
    *,
    camera_model: str,
    intrinsics: dict[str, Any],
    orig_h: int,
    orig_w: int,
    resize_h: int,
    resize_w: int,
) -> tuple[Any, ...]:
    frozen_intrinsics: list[tuple[str, Any]] = []
    for key in sorted(intrinsics):
        if key == "rays":
            value = np.asarray(intrinsics[key])
            frozen_intrinsics.append(("rays_shape", tuple(int(v) for v in value.shape)))
            frozen_intrinsics.append(("rays_mean", round(float(np.nanmean(value)), 6)))
            continue
        value = intrinsics[key]
        if isinstance(value, float):
            frozen_value: Any = round(value, 6)
        elif isinstance(value, (int, str, bool)) or value is None:
            frozen_value = value
        else:
            frozen_value = str(value)
        frozen_intrinsics.append((str(key), frozen_value))
    return (
        str(camera_model),
        int(orig_h),
        int(orig_w),
        int(resize_h),
        int(resize_w),
        tuple(frozen_intrinsics),
    )


def _observed_angle_mask_from_azimuth_tensor(azimuth_tensor: torch.Tensor, num_angles: int) -> np.ndarray:
    """Return model angle bins covered by the runtime camera azimuth map."""
    if num_angles <= 0:
        raise ValueError("num_angles must be positive")
    azimuth = azimuth_tensor.detach().float().cpu().numpy().reshape(-1)
    azimuth = azimuth[np.isfinite(azimuth)]
    if azimuth.size == 0:
        return np.ones((num_angles,), dtype=bool)

    two_pi = 2.0 * np.pi
    wrapped = ((azimuth.astype(np.float64, copy=False) + np.pi) % two_pi) - np.pi
    bin_width = two_pi / float(num_angles)
    half_bin = 0.5 * bin_width
    centers = np.deg2rad(np.arange(num_angles, dtype=np.float64) - (num_angles // 2))

    observed = np.zeros((num_angles,), dtype=bool)
    nearest_bins = (np.rint(np.rad2deg(wrapped)).astype(np.int32) + (num_angles // 2)) % num_angles
    observed[np.unique(nearest_bins)] = True

    if wrapped.size <= 1:
        return observed

    sorted_azimuth = np.sort(wrapped)
    extended = np.concatenate([sorted_azimuth, sorted_azimuth[:1] + two_pi])
    gaps = np.diff(extended)
    largest_gap_index = int(np.argmax(gaps))
    covered_width = float(max(0.0, two_pi - float(gaps[largest_gap_index])))
    if covered_width >= two_pi - (2.0 * half_bin):
        return np.ones((num_angles,), dtype=bool)

    arc_start = float(extended[largest_gap_index + 1])
    relative_centers = (centers - arc_start) % two_pi
    observed |= relative_centers <= covered_width + half_bin
    return observed.astype(bool, copy=False)


class UpstreamTraversabilityInference:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device(config["model"]["device"])
        self._runtime_camera_info: dict[str, Any] | None = None
        self._azimuth_tensor_cache_enabled = bool(
            config.get("model", {}).get("azimuth_tensor_cache", {}).get("enabled", True)
        )
        self._azimuth_tensor_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._fisheye_rays_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._unik3d_ray_generator: UniK3DRayGenerator | None = None

        checkpoint_path = Path(config["model"]["checkpoint_path"]).expanduser()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        checkpoint_args = dict(checkpoint.get("args", {}))
        backbone_type = str(checkpoint_args.get("backbone_type", "vitb16"))
        architecture_override = config.get("model", {}).get("architecture", {})
        force_multi_scale = architecture_override.get("force_multi_scale")
        multiscale_layers = architecture_override.get("multiscale_layer_indices")
        override_layers = tuple(int(item) for item in multiscale_layers) if multiscale_layers else None
        architecture = infer_architecture_from_checkpoint(
            checkpoint_state=state_dict,
            checkpoint_args=checkpoint_args,
            override_multi_scale=None if force_multi_scale is None else bool(force_multi_scale),
            override_layers=override_layers,
        )
        use_multi_scale = bool(architecture.multi_scale)
        backbone_weights = _resolve_backbone_weights(
            None,
            checkpoint_args,
            allow_missing_full_checkpoint=_checkpoint_has_full_backbone_state(state_dict),
        )

        self.model = UpstreamTraversabilityModel(
            backbone_path="" if backbone_weights is None else str(backbone_weights),
            backbone_type=backbone_type,
            freeze_backbone=True,
            num_angles=360,
            use_multi_scale=use_multi_scale,
            decoder_dim=architecture.decoder_dim,
            decoder_hidden_dim=architecture.decoder_hidden_dim,
            angle_dim=architecture.angle_dim,
            fpn_dim=architecture.fpn_dim,
            decoder_dropout=architecture.decoder_dropout,
        ).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.metadata = ModelMetadata(
            checkpoint_path=str(checkpoint_path),
            checkpoint_epoch=checkpoint.get("epoch"),
            backbone_variant=architecture.backbone_variant,
            embed_dim=architecture.embed_dim,
            num_blocks=architecture.num_blocks,
            multi_scale=architecture.multi_scale,
            layer_indices=architecture.layer_indices,
            decoder_dim=architecture.decoder_dim,
            angle_dim=architecture.angle_dim,
            num_angles=architecture.num_angles,
            checkpoint_args=checkpoint_args,
            strict_load_ok=True,
        )
        self._img_size = int(checkpoint_args.get("img_size", config["model"]["input"]["long_edge"]))
        self._mean = np.array(config["model"]["normalize"]["mean"], dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(config["model"]["normalize"]["std"], dtype=np.float32).reshape(1, 1, 3)

    def _runtime_rays_cache_path(self, camera_model: str) -> Path | None:
        if self._runtime_camera_info is None:
            return None
        camera_cfg = self.config["model"]["camera"]
        rays_cache_dir = str(camera_cfg.get("rays_cache_dir", "")).strip()
        if not rays_cache_dir:
            return None
        signature = stable_camera_signature(self._runtime_camera_info, camera_model)
        return Path(rays_cache_dir).expanduser() / f"{camera_model}_{signature}.npz"

    def _fisheye_rays_cache_key(
        self,
        *,
        camera_model: str,
        intrinsics: dict[str, Any],
        orig_h: int,
        orig_w: int,
    ) -> tuple[Any, ...]:
        source = intrinsics.get("rays_source")
        if source:
            return ("source", str(source), int(orig_h), int(orig_w))
        if self._runtime_camera_info is not None:
            return (
                "runtime",
                stable_camera_signature(self._runtime_camera_info, camera_model),
                int(orig_h),
                int(orig_w),
            )
        return (
            "configured",
            str(self.config["model"]["camera"].get("rays_path") or self.config["model"]["camera"].get("depth_meta_path") or ""),
            int(orig_h),
            int(orig_w),
        )

    def _load_cached_fisheye_rays(
        self,
        *,
        path: str | Path,
        key: tuple[Any, ...],
        target_hw: tuple[int, int],
    ) -> np.ndarray:
        cached = self._fisheye_rays_cache.get(key)
        if cached is not None:
            return cached
        rays = load_rays_npz(path, target_hw=target_hw)
        self._fisheye_rays_cache[key] = rays
        return rays

    def _unik3d_generator(self) -> UniK3DRayGenerator:
        if self._unik3d_ray_generator is not None:
            return self._unik3d_ray_generator
        camera_cfg = self.config["model"]["camera"]
        unik3d_cfg = dict(camera_cfg.get("unik3d") or {})
        model_dir = Path(str(unik3d_cfg.get("model_dir") or DEFAULT_UNIK3D_MODEL_DIR)).expanduser()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"UniK3D checkpoint directory not found: {model_dir}")
        device_value = str(unik3d_cfg.get("device") or "").strip()
        device = device_value or str(self.device)
        self._unik3d_ray_generator = UniK3DRayGenerator(
            model_dir=model_dir,
            device=device,
            resolution_level=int(unik3d_cfg.get("resolution_level", 9)),
            interpolation_mode=str(unik3d_cfg.get("interpolation_mode", "bilinear")),
        )
        return self._unik3d_ray_generator

    def _ensure_fisheye_rays(
        self,
        *,
        frame_bgr: np.ndarray,
        camera_model: str,
        intrinsics: dict[str, Any],
        orig_h: int,
        orig_w: int,
    ) -> dict[str, Any]:
        if camera_model != "fisheye":
            return intrinsics
        if "rays" in intrinsics:
            rays = np.asarray(intrinsics["rays"], dtype=np.float32)
            if rays.shape[:2] != (orig_h, orig_w):
                source = intrinsics.get("rays_source")
                if not source:
                    raise ValueError(
                        f"Fisheye rays shape {rays.shape[:2]} does not match RGB shape {(orig_h, orig_w)}, "
                        "and no rays_source is available for resizing."
                    )
                key = self._fisheye_rays_cache_key(
                    camera_model=camera_model,
                    intrinsics=intrinsics,
                    orig_h=orig_h,
                    orig_w=orig_w,
                )
                rays = self._load_cached_fisheye_rays(path=source, key=key, target_hw=(orig_h, orig_w))
            else:
                rays = normalise_rays(rays)
                key = self._fisheye_rays_cache_key(
                    camera_model=camera_model,
                    intrinsics=intrinsics,
                    orig_h=orig_h,
                    orig_w=orig_w,
                )
                self._fisheye_rays_cache.setdefault(key, rays)
            intrinsics["rays"] = rays
            return intrinsics

        camera_cfg = self.config["model"]["camera"]
        source = intrinsics.get("rays_source")
        if source:
            key = self._fisheye_rays_cache_key(
                camera_model=camera_model,
                intrinsics=intrinsics,
                orig_h=orig_h,
                orig_w=orig_w,
            )
            intrinsics["rays"] = self._load_cached_fisheye_rays(path=source, key=key, target_hw=(orig_h, orig_w))
            return intrinsics

        cache_path = self._runtime_rays_cache_path(camera_model)
        if cache_path is not None and cache_path.is_file():
            intrinsics["rays_source"] = str(cache_path)
            key = self._fisheye_rays_cache_key(
                camera_model=camera_model,
                intrinsics=intrinsics,
                orig_h=orig_h,
                orig_w=orig_w,
            )
            intrinsics["rays"] = self._load_cached_fisheye_rays(path=cache_path, key=key, target_hw=(orig_h, orig_w))
            return intrinsics

        if not bool(camera_cfg.get("generate_fisheye_rays_with_unik3d", True)):
            raise RuntimeError(
                "Fisheye camera has no rays, and model.camera.generate_fisheye_rays_with_unik3d is false."
            )

        generator = self._unik3d_generator()
        rays = generator.infer_rays(frame_bgr)
        if rays.shape[:2] != (orig_h, orig_w):
            raise RuntimeError(f"UniK3D rays shape {rays.shape[:2]} does not match RGB shape {(orig_h, orig_w)}")
        rays = normalise_rays(rays)
        intrinsics["rays"] = rays
        if cache_path is not None:
            save_rays_npz(
                cache_path,
                rays,
                meta_format=str(camera_cfg.get("rays_cache_format", "rays_lr8_fp16")),
            )
            intrinsics["rays_source"] = str(cache_path)
        else:
            intrinsics["rays_source"] = "unik3d_runtime_uncached"
        key = self._fisheye_rays_cache_key(
            camera_model=camera_model,
            intrinsics=intrinsics,
            orig_h=orig_h,
            orig_w=orig_w,
        )
        self._fisheye_rays_cache[key] = rays
        return intrinsics

    def set_runtime_camera_info(self, camera_info: dict[str, Any] | None) -> None:
        if camera_info is None:
            if self._runtime_camera_info is not None and self._azimuth_tensor_cache_enabled:
                self._azimuth_tensor_cache.clear()
            self._runtime_camera_info = None
            self._fisheye_rays_cache.clear()
            return
        new_runtime_camera_info = {
            "fx": float(camera_info["fx"]),
            "fy": float(camera_info["fy"]),
            "cx": float(camera_info["cx"]),
            "cy": float(camera_info["cy"]),
            "width": int(camera_info["width"]),
            "height": int(camera_info["height"]),
            "frame_id": camera_info.get("frame_id"),
            "distortion_model": camera_info.get("distortion_model"),
            "d": camera_info.get("d"),
            "k": camera_info.get("k"),
            "r": camera_info.get("r"),
            "p": camera_info.get("p"),
            "depth_meta_path": camera_info.get("depth_meta_path"),
            "rays_path": camera_info.get("rays_path"),
        }
        if self._runtime_camera_info == new_runtime_camera_info:
            return
        self._runtime_camera_info = new_runtime_camera_info
        if self._azimuth_tensor_cache_enabled:
            self._azimuth_tensor_cache.clear()
        self._fisheye_rays_cache.clear()

    def preprocess(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int], str, dict[str, Any]]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR frame with shape HxWx3, got {frame_bgr.shape}")

        orig_h, orig_w = frame_bgr.shape[:2]
        resize_h, resize_w = _compute_target_hw(orig_h, orig_w, self._img_size)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self._mean) / self._std
        image_tensor = torch.from_numpy(normalized.transpose(2, 0, 1)[None, ...]).to(self.device)

        camera_cfg = self.config["model"]["camera"]
        camera_model = _resolve_camera_model(camera_cfg, self._runtime_camera_info)
        intrinsics = _resolve_intrinsics(
            camera_cfg=camera_cfg,
            runtime_camera_info=self._runtime_camera_info,
            image_width=orig_w,
            image_height=orig_h,
        )
        intrinsics = self._ensure_fisheye_rays(
            frame_bgr=frame_bgr,
            camera_model=camera_model,
            intrinsics=intrinsics,
            orig_h=orig_h,
            orig_w=orig_w,
        )
        cache_key = _azimuth_cache_key(
            camera_model=camera_model,
            intrinsics=intrinsics,
            orig_h=orig_h,
            orig_w=orig_w,
            resize_h=resize_h,
            resize_w=resize_w,
        )
        azimuth_tensor = self._azimuth_tensor_cache.get(cache_key) if self._azimuth_tensor_cache_enabled else None
        if azimuth_tensor is None:
            azimuth_tensor = _build_azimuth_tensor(
                camera_model=camera_model,
                intrinsics=intrinsics,
                orig_h=orig_h,
                orig_w=orig_w,
                resize_h=resize_h,
                resize_w=resize_w,
            ).to(self.device)
            if self._azimuth_tensor_cache_enabled:
                self._azimuth_tensor_cache[cache_key] = azimuth_tensor
        return image_tensor, azimuth_tensor, (resize_h, resize_w), (orig_h, orig_w), camera_model, intrinsics

    def apply_exist_usage(
        self,
        combined_distance_m: np.ndarray,
        raw_distance_m: np.ndarray,
        exist_probability: np.ndarray,
    ) -> np.ndarray:
        exist_cfg = self.config["model"]["exist_logit"]
        usage = str(exist_cfg["usage"])
        free_space_distance = float(exist_cfg["free_space_distance_m"])
        if usage == "combined_distance":
            return combined_distance_m
        if usage == "probability_threshold":
            threshold = float(exist_cfg["probability_threshold"])
            return np.where(exist_probability >= threshold, raw_distance_m, free_space_distance).astype(np.float32)
        raise ValueError(f"Unsupported exist_logit usage: {usage}")

    @torch.inference_mode()
    def run(self, frame_bgr: np.ndarray) -> ModelOutput:
        image_tensor, azimuth_tensor, resize_hw, original_hw, camera_model, intrinsics = self.preprocess(frame_bgr)
        start = time.perf_counter()
        combined_distance, exist_logit, raw_distance = self.model(image_tensor, azimuth_tensor)
        latency_ms = (time.perf_counter() - start) * 1000.0

        combined_np = combined_distance[0].detach().cpu().numpy().astype(np.float32)
        raw_np = raw_distance[0].detach().cpu().numpy().astype(np.float32)
        logit_np = exist_logit[0].detach().cpu().numpy().astype(np.float32)
        exist_probability = 1.0 / (1.0 + np.exp(-logit_np))
        effective = self.apply_exist_usage(combined_np, raw_np, exist_probability)
        observed_angle_mask = _observed_angle_mask_from_azimuth_tensor(
            azimuth_tensor,
            num_angles=int(effective.shape[0]),
        )
        return ModelOutput(
            image_tensor=image_tensor.detach().cpu(),
            azimuth_tensor=azimuth_tensor.detach().cpu(),
            camera_model=camera_model,
            camera_intrinsics={k: v for k, v in intrinsics.items() if k != "rays"},
            camera_geometry=_camera_geometry_dict(
                camera_model=camera_model,
                intrinsics=intrinsics,
                original_hw=original_hw,
                resize_hw=resize_hw,
            ),
            observed_angle_mask=observed_angle_mask.astype(bool, copy=False),
            raw_distance_m=raw_np,
            combined_distance_m=combined_np,
            effective_distance_m=effective,
            exist_logit=logit_np,
            exist_probability=exist_probability.astype(np.float32),
            latency_ms=latency_ms,
            resize_hw=resize_hw,
            original_hw=original_hw,
        )
