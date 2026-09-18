from __future__ import annotations

import __future__
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from .unik3d_runtime import (
    DEFAULT_UNIK3D_MODEL_DIR,
    UniK3DRayGenerator,
    load_rays_npz,
    resize_scalar_map,
    save_rays_npz,
    stable_camera_signature,
)


OMNIGUARD_ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT_ROOT = OMNIGUARD_ROOT / "deployment" / "checkpoints"
DA3_ROOT = OMNIGUARD_ROOT / "third_party" / "depth_anything_3"
DAP_ROOT = OMNIGUARD_ROOT / "third_party" / "dap"

DEFAULT_DA3_MODEL_DIR = CHECKPOINT_ROOT / "DA3-LARGE"
DEFAULT_DAP_WEIGHTS_DIR = CHECKPOINT_ROOT / "DAP-weights"
DEFAULT_DAP_CONFIG_PATH = DAP_ROOT / "config" / "infer.yaml"


@dataclass
class DepthInferenceResult:
    depth_m: np.ndarray
    camera_model: str
    backend: str
    rays: np.ndarray | None = None
    depth_cache_path: str | None = None
    rays_cache_path: str | None = None


class _DapDinov3FutureAnnotationsLoader(importlib.machinery.SourceFileLoader):
    """Compile DAP-local DINOv3 sources with postponed annotations on Python 3.8."""

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


class _DapDinov3FutureAnnotationsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "dinov3" and not fullname.startswith("dinov3."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        loader = spec.loader
        if not isinstance(loader, importlib.machinery.SourceFileLoader):
            return spec
        spec.loader = _DapDinov3FutureAnnotationsLoader(fullname, loader.path)
        return spec


def _install_dap_dinov3_py38_compat() -> None:
    if sys.version_info >= (3, 10):
        return
    for finder in sys.meta_path:
        if isinstance(finder, _DapDinov3FutureAnnotationsFinder):
            return
    sys.meta_path.insert(0, _DapDinov3FutureAnnotationsFinder())


def camera_model_from_distortion(distortion_model: str | None) -> str | None:
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


def normalise_camera_model(value: str) -> str:
    camera_model = str(value).strip().lower()
    if camera_model in {"pano", "panoramic", "spherical"}:
        return "equirectangular"
    if camera_model not in {"pinhole", "fisheye", "equirectangular"}:
        raise ValueError(f"Unsupported camera model for depth visualization: {value!r}")
    return camera_model


def resolve_visualization_camera_model(
    config: dict[str, Any],
    camera_info: dict[str, Any] | None,
    output_camera_model: str | None = None,
) -> str:
    if output_camera_model:
        return normalise_camera_model(output_camera_model)
    camera_cfg = config.get("model", {}).get("camera", {})
    configured = str(camera_cfg.get("model", "auto")).strip().lower()
    if configured in {"", "auto"}:
        if camera_info is None:
            raise ValueError("Cannot infer visualization depth camera model: CameraInfo is missing.")
        inferred = camera_model_from_distortion(camera_info.get("distortion_model"))
        if inferred is None:
            raise ValueError(
                "Cannot infer visualization depth camera model from distortion_model="
                f"{camera_info.get('distortion_model')!r}."
            )
        return inferred
    return normalise_camera_model(configured)


def intrinsics_matrix(intrinsics: dict[str, Any]) -> np.ndarray:
    required = ("fx", "fy", "cx", "cy")
    missing = [key for key in required if intrinsics.get(key) is None]
    if missing:
        raise ValueError(f"Pinhole depth projection requires intrinsics: missing {missing}")
    return np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _frame_hash(frame_bgr: np.ndarray) -> str:
    frame = np.ascontiguousarray(frame_bgr)
    digest = hashlib.sha1()
    digest.update(str(frame.shape).encode("ascii"))
    digest.update(frame.view(np.uint8))
    return digest.hexdigest()[:16]


def _camera_signature(camera_info: dict[str, Any] | None, intrinsics: dict[str, Any], camera_model: str) -> str:
    if camera_info is not None:
        return stable_camera_signature(camera_info, camera_model)
    payload = {
        "camera_model": camera_model,
        "fx": intrinsics.get("fx"),
        "fy": intrinsics.get("fy"),
        "cx": intrinsics.get("cx"),
        "cy": intrinsics.get("cy"),
    }
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _depth_cache_paths(
    *,
    cache_dir: str | Path | None,
    frame_bgr: np.ndarray,
    camera_model: str,
    intrinsics: dict[str, Any],
    camera_info: dict[str, Any] | None,
) -> tuple[Path | None, Path | None]:
    if cache_dir is None or not str(cache_dir).strip():
        return None, None
    cache_root = Path(cache_dir).expanduser()
    signature = _camera_signature(camera_info, intrinsics, camera_model)
    image_hash = _frame_hash(frame_bgr)
    stem = f"first_frame_{camera_model}_{signature}_{image_hash}"
    return cache_root / f"{stem}_depth.npz", cache_root / f"{stem}_rays.npz"


def _write_cache_metadata(
    cache_dir: str | Path | None,
    *,
    camera_model: str,
    backend: str,
    camera_info: dict[str, Any] | None,
    intrinsics: dict[str, Any],
) -> None:
    if cache_dir is None or not str(cache_dir).strip():
        return
    cache_root = Path(cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "camera_model": camera_model,
        "backend": backend,
        "camera_info": camera_info,
        "intrinsics": {key: value for key, value in intrinsics.items() if key != "rays"},
    }
    (cache_root / "camera_info_used.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


class VisualizationDepthRuntime:
    """First-frame depth runtime used only for RGB geo-interp debug panels."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        depth_cfg = self._depth_cfg
        device_value = str(depth_cfg.get("device") or "").strip()
        if not device_value:
            device_value = str(config.get("model", {}).get("device", "cuda"))
        self.device = torch.device(device_value)
        self._da3_model: Any | None = None
        self._dap_model: nn.Module | None = None
        self._unik3d_generator: UniK3DRayGenerator | None = None

    @property
    def _depth_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("visualization", {}).get("depth", {}) or {})

    def infer(
        self,
        *,
        frame_bgr: np.ndarray,
        camera_model: str,
        intrinsics: dict[str, Any],
        camera_info: dict[str, Any] | None = None,
        cache_dir: str | Path | None = None,
    ) -> DepthInferenceResult:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected BGR frame HxWx3 for depth visualization, got {frame.shape}")
        camera_model = normalise_camera_model(camera_model)
        depth_cfg = self._depth_cfg
        cache_dir = cache_dir or depth_cfg.get("cache_dir")
        depth_cache_path, rays_cache_path = _depth_cache_paths(
            cache_dir=cache_dir,
            frame_bgr=frame,
            camera_model=camera_model,
            intrinsics=intrinsics,
            camera_info=camera_info,
        )
        if depth_cache_path is not None and depth_cache_path.is_file():
            with np.load(depth_cache_path, allow_pickle=False) as data:
                depth_m = np.asarray(data["depth_m"], dtype=np.float32)
                backend = str(data["backend"].item() if np.asarray(data["backend"]).shape == () else data["backend"])
            rays = None
            if camera_model == "fisheye":
                if rays_cache_path is None or not rays_cache_path.is_file():
                    raise FileNotFoundError(
                        "Cached fisheye depth exists but cached rays are missing: "
                        f"{rays_cache_path}"
                    )
                rays = load_rays_npz(rays_cache_path, target_hw=frame.shape[:2])
            _write_cache_metadata(
                cache_dir,
                camera_model=camera_model,
                backend=backend,
                camera_info=camera_info,
                intrinsics=intrinsics,
            )
            return DepthInferenceResult(
                depth_m=resize_scalar_map(depth_m, frame.shape[:2]),
                camera_model=camera_model,
                backend=backend,
                rays=rays,
                depth_cache_path=str(depth_cache_path),
                rays_cache_path=None if rays_cache_path is None else str(rays_cache_path),
            )

        if camera_model == "pinhole":
            result = self._infer_da3(frame, intrinsics)
        elif camera_model == "equirectangular":
            result = self._infer_dap(frame)
        elif camera_model == "fisheye":
            result = self._infer_unik3d(frame)
        else:
            raise ValueError(f"Unsupported camera model for depth visualization: {camera_model}")

        if depth_cache_path is not None:
            depth_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                depth_cache_path,
                depth_m=result.depth_m.astype(np.float32, copy=False),
                camera_model=np.array(camera_model),
                backend=np.array(result.backend),
            )
            result.depth_cache_path = str(depth_cache_path)
        if rays_cache_path is not None and result.rays is not None:
            save_rays_npz(
                rays_cache_path,
                result.rays,
                meta_format=str(self._depth_cfg.get("rays_cache_format", "rays_lr8_fp16")),
            )
            result.rays_cache_path = str(rays_cache_path)

        _write_cache_metadata(
            cache_dir,
            camera_model=camera_model,
            backend=result.backend,
            camera_info=camera_info,
            intrinsics=intrinsics,
        )
        return result

    def _infer_da3(self, frame_bgr: np.ndarray, intrinsics: dict[str, Any]) -> DepthInferenceResult:
        model = self._load_da3()
        rgb = cv2.cvtColor(frame_bgr.astype(np.uint8, copy=False), cv2.COLOR_BGR2RGB)
        da3_cfg = dict(self._depth_cfg.get("da3") or {})
        k = intrinsics_matrix(intrinsics)[None, ...]
        prediction = model.inference(
            [rgb],
            intrinsics=k,
            export_dir=None,
            export_format="mini_npz",
            process_res=int(da3_cfg.get("process_res", 504)),
            process_res_method=str(da3_cfg.get("process_res_method", "upper_bound_resize")),
        )
        if prediction.depth is None or len(prediction.depth) == 0:
            raise RuntimeError("DA3 inference returned no depth map.")
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        depth = resize_scalar_map(depth, rgb.shape[:2])
        return DepthInferenceResult(depth_m=depth, camera_model="pinhole", backend="DA3")

    def _infer_dap(self, frame_bgr: np.ndarray) -> DepthInferenceResult:
        model = self._load_dap()
        rgb = cv2.cvtColor(frame_bgr.astype(np.uint8, copy=False), cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = model(tensor)
            if isinstance(outputs, dict) and "pred_depth" in outputs:
                pred = outputs["pred_depth"][0]
            elif isinstance(outputs, (tuple, list)):
                pred = outputs[0]
            else:
                pred = outputs
            depth = pred.detach().cpu().squeeze().numpy().astype(np.float32, copy=False)
        dap_cfg = dict(self._depth_cfg.get("dap") or {})
        depth_scale = float(dap_cfg.get("depth_scale", 100.0))
        if depth_scale != 1.0:
            depth = depth * depth_scale
        depth = resize_scalar_map(depth, rgb.shape[:2])
        return DepthInferenceResult(depth_m=depth, camera_model="equirectangular", backend="DAP")

    def _infer_unik3d(self, frame_bgr: np.ndarray) -> DepthInferenceResult:
        depth, rays = self._load_unik3d().infer_depth_and_rays(frame_bgr)
        return DepthInferenceResult(depth_m=depth, camera_model="fisheye", backend="UniK3D", rays=rays)

    def _load_da3(self) -> Any:
        if self._da3_model is not None:
            return self._da3_model
        da3_cfg = dict(self._depth_cfg.get("da3") or {})
        model_dir = Path(str(da3_cfg.get("model_dir") or DEFAULT_DA3_MODEL_DIR)).expanduser()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"DA3 model directory not found: {model_dir}")
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"DA3 config.json not found in: {model_dir}")
        if not ((model_dir / "model.safetensors").is_file() or (model_dir / "pytorch_model.bin").is_file()):
            raise FileNotFoundError(f"DA3 weights not found in: {model_dir}")
        da3_src = DA3_ROOT / "src"
        if not da3_src.is_dir():
            raise FileNotFoundError(f"OmniGuard-local DA3 source directory not found: {da3_src}")
        if str(da3_src) not in sys.path:
            sys.path.insert(0, str(da3_src))
        try:
            from depth_anything_3.api import DepthAnything3  # type: ignore
        except Exception as exc:
            raise ImportError(f"Cannot import OmniGuard-local DepthAnything3 from {da3_src}") from exc
        model = DepthAnything3.from_pretrained(str(model_dir)).to(device=self.device)
        model.eval()
        self._da3_model = model
        return model

    def _load_dap(self) -> nn.Module:
        if self._dap_model is not None:
            return self._dap_model
        dap_cfg = dict(self._depth_cfg.get("dap") or {})
        weights_dir = Path(str(dap_cfg.get("weights_dir") or DEFAULT_DAP_WEIGHTS_DIR)).expanduser()
        model_path = weights_dir / "model.pth"
        if not model_path.is_file():
            raise FileNotFoundError(f"DAP model.pth not found: {model_path}")
        config_path = Path(str(dap_cfg.get("config_path") or DEFAULT_DAP_CONFIG_PATH)).expanduser()
        if not config_path.is_absolute():
            config_path = DAP_ROOT / config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"DAP config not found: {config_path}")
        if not DAP_ROOT.is_dir():
            raise FileNotFoundError(f"OmniGuard-local DAP source directory not found: {DAP_ROOT}")
        if str(DAP_ROOT) not in sys.path:
            sys.path.insert(0, str(DAP_ROOT))
        dinov3_root = DAP_ROOT / "depth_anything_v2_metric" / "depth_anything_v2" / "dinov3"
        if not dinov3_root.is_dir():
            raise FileNotFoundError(f"DAP DINOv3 source directory not found: {dinov3_root}")
        if str(dinov3_root) not in sys.path:
            sys.path.insert(0, str(dinov3_root))
        _install_dap_dinov3_py38_compat()
        try:
            from networks.models import make  # type: ignore
        except Exception as exc:
            raise ImportError(f"Cannot import OmniGuard-local DAP networks from {DAP_ROOT}") from exc
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"DAP checkpoint must be a state_dict mapping: {model_path}")
        state = {
            key: value
            for key, value in state.items()
            if key not in {"epoch", "global_step", "optimizer", "scheduler"}
        }
        original_torch_hub_load = torch.hub.load

        def _load_dinov3_backbone_only(repo_or_dir: Any, model_name: str, *args: Any, source: str | None = None, **kwargs: Any) -> Any:
            repo_path = Path(str(repo_or_dir)).expanduser()
            if not repo_path.is_absolute():
                repo_path = (Path.cwd() / repo_path).resolve()
            if source == "local" and repo_path == dinov3_root.resolve() and str(model_name).startswith("dinov3_"):
                from dinov3.hub import backbones  # type: ignore

                if hasattr(backbones, str(model_name)):
                    return getattr(backbones, str(model_name))(*args, **kwargs)
            if source is None:
                return original_torch_hub_load(repo_or_dir, model_name, *args, **kwargs)
            return original_torch_hub_load(repo_or_dir, model_name, *args, source=source, **kwargs)

        previous_cwd = Path.cwd()
        try:
            torch.hub.load = _load_dinov3_backbone_only
            os.chdir(DAP_ROOT)
            model = make(config["model"])
        finally:
            torch.hub.load = original_torch_hub_load
            os.chdir(previous_cwd)
        if any(str(key).startswith("module.") for key in state.keys()):
            model = nn.DataParallel(model)
        model = model.to(self.device)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "DAP checkpoint did not match model exactly. "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        model.eval()
        self._dap_model = model
        return model

    def _load_unik3d(self) -> UniK3DRayGenerator:
        if self._unik3d_generator is not None:
            return self._unik3d_generator
        unik3d_cfg = dict(self._depth_cfg.get("unik3d") or self.config.get("model", {}).get("camera", {}).get("unik3d") or {})
        model_dir = Path(str(unik3d_cfg.get("model_dir") or DEFAULT_UNIK3D_MODEL_DIR)).expanduser()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"UniK3D model directory not found: {model_dir}")
        self._unik3d_generator = UniK3DRayGenerator(
            model_dir=model_dir,
            device=str(unik3d_cfg.get("device") or self.device),
            resolution_level=int(unik3d_cfg.get("resolution_level", 9)),
            interpolation_mode=str(unik3d_cfg.get("interpolation_mode", "bilinear")),
        )
        return self._unik3d_generator
