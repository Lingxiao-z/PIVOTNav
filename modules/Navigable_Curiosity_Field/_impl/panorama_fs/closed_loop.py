from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Protocol, Sequence

import numpy as np

from .inference import FGFSInference


class RGBController(Protocol):
    def reset(self, episode: dict) -> np.ndarray: ...
    def traversable_directions(self, current_erp_rgb: np.ndarray) -> np.ndarray: ...
    def execute_short_horizon(self, direction_index: int) -> np.ndarray: ...
    def post_action_audit(self) -> dict: ...


@dataclass
class FGFSDirectionController:
    """Adapter that selects with predicted FG, FS, and external traversability."""

    predictor: FGFSInference
    controller: RGBController
    goal_erp_rgb: np.ndarray
    action_log: list[dict]

    def reset(self, episode: dict) -> np.ndarray:
        self.goal_erp_rgb = np.asarray(episode["goal_erp_rgb"])
        return self.controller.reset(episode)

    def traversable_directions(self, current_erp_rgb: np.ndarray) -> np.ndarray:
        return np.asarray(self.controller.traversable_directions(current_erp_rgb), dtype=bool)

    def execute_short_horizon(self, direction_index: int) -> np.ndarray:
        self.action_log.append({"phase": "before_action", "direction_index": int(direction_index), "privileged_inputs_to_predictor": False})
        next_erp = self.controller.execute_short_horizon(direction_index)
        self.action_log[-1]["phase"] = "after_action"
        self.action_log[-1]["post_action_audit_allowed"] = True
        return next_erp

    def post_action_audit(self) -> dict:
        audit = dict(self.controller.post_action_audit())
        audit["prediction_input_contract"] = "current_erp_rgb + goal_erp_rgb only"
        audit["privileged_values_read_before_prediction"] = False
        return audit

    def predict(self, current_erp_rgb: np.ndarray) -> np.ndarray:
        output = self.predictor({"current_erp_rgb": current_erp_rgb, "goal_erp_rgb": self.goal_erp_rgb})
        return output["masked_fs_scores"][0].detach().cpu().numpy()


@dataclass
class ClosedLoopEvaluator:
    predictor: FGFSInference
    controller_factory: Callable[[], RGBController]

    def run_episode(self, episode: dict, *, max_actions: int) -> dict:
        controller=self.controller_factory(); current=controller.reset(episode); goal=episode["goal_erp_rgb"]; steps=[]
        for action_index in range(max_actions):
            inference_started=time.perf_counter()
            output=self.predictor({"current_erp_rgb":current,"goal_erp_rgb":goal})
            fs_scores=output["fs_scores"][0].detach().cpu().numpy()
            fg_logits=output["fg_logits"][0].detach().cpu().numpy()
            fg_probabilities=output["fg_probabilities"][0].detach().cpu().numpy()
            model_mask=output["fs_valid_mask"][0].detach().cpu().numpy().astype(bool)
            inference_latency_ms=(time.perf_counter()-inference_started)*1000.0
            traversable=np.asarray(controller.traversable_directions(current),dtype=bool)
            if traversable.shape!=(12,):raise ValueError("OmniTrav mask must contain 12 directions")
            selection_mask=traversable & model_mask
            direction=select_masked_direction(fs_scores,selection_mask)
            if direction is None:
                steps.append({"action_index":action_index,"fg_logits":fg_logits.tolist(),"fg_probabilities":fg_probabilities.tolist(),"model_fg_mask":model_mask.tolist(),"fs_scores":fs_scores.tolist(),"traversable_mask":traversable.tolist(),"selection_mask":selection_mask.tolist(),"selected_direction":None,"inference_latency_ms":inference_latency_ms,"audit":{"no_jointly_valid_direction":True}});break
            next_erp=controller.execute_short_horizon(direction)
            # Privileged values are requested only after the RGB-only prediction
            # and short-horizon action have both completed.
            audit=controller.post_action_audit(); steps.append({"action_index":action_index,"fg_logits":fg_logits.tolist(),"fg_probabilities":fg_probabilities.tolist(),"model_fg_mask":model_mask.tolist(),"fs_scores":fs_scores.tolist(),"traversable_mask":traversable.tolist(),"selection_mask":selection_mask.tolist(),"selected_direction":direction,"inference_latency_ms":inference_latency_ms,"audit":audit}); current=next_erp
        return summarize_episode(episode["episode_id"],steps)


def aggregate_episodes(episodes: Sequence[dict]) -> dict:
    """Aggregate episode summaries without treating missing audits as zeros."""
    if not episodes:
        raise ValueError("at least one closed-loop episode is required")

    def mean_present(key: str) -> float | None:
        values = [float(item[key]) for item in episodes if item.get(key) is not None]
        return float(np.mean(values)) if values else None

    latency_values = [
        float(item["inference_latency_ms"]["mean"])
        for item in episodes
        if item.get("inference_latency_ms", {}).get("mean") is not None
    ]
    return {
        "episode_count": len(episodes),
        "total_action_count": int(sum(int(item["action_count"]) for item in episodes)),
        "mean_geodesic_progress_m": mean_present("geodesic_progress_m"),
        "mean_positive_progress_rate": mean_present("positive_progress_rate"),
        "mean_effective_progress_per_100_actions_m": mean_present("effective_progress_per_100_actions_m"),
        "entered_goal_3m_rate": float(np.mean([bool(item["entered_goal_3m"]) for item in episodes])),
        "entered_goal_2m_rate": float(np.mean([bool(item["entered_goal_2m"]) for item in episodes])),
        "entered_goal_1m_rate": float(np.mean([bool(item["entered_goal_1m"]) for item in episodes])),
        "mean_invalid_direction_selection_rate": mean_present("invalid_direction_selection_rate"),
        "mean_final_coverage_fraction": mean_present("final_coverage_fraction"),
        "mean_coverage_auc": mean_present("coverage_auc"),
        "mean_repeated_region_return_rate": mean_present("repeated_region_return_rate"),
        "mean_episode_inference_latency_ms": float(np.mean(latency_values)) if latency_values else None,
        "scope_note": "controlled short-horizon FG/FS progress audit; not full ImageNav SR/SPL",
    }


def select_masked_direction(scores: np.ndarray, traversable: np.ndarray) -> int | None:
    scores = np.asarray(scores, dtype=np.float64)
    traversable = np.asarray(traversable, dtype=bool)
    if scores.shape != (12,) or traversable.shape != (12,):
        raise ValueError("scores and traversability must each contain 12 directions")
    finite = traversable & np.isfinite(scores)
    return int(np.where(finite, scores, -np.inf).argmax()) if finite.any() else None


def rgb_information_gain_scores(erp_rgb: np.ndarray) -> np.ndarray:
    """Deterministic RGB entropy proxy over 90-degree wrapped direction views."""
    image = np.asarray(erp_rgb)
    if image.shape != (224, 448, 3):
        raise ValueError("information-gain baseline requires a 224x448 RGB ERP")
    gray = np.rint(image.astype(np.float32).mean(axis=2)).astype(np.uint8)
    width = gray.shape[1]
    half_window = width // 8
    scores = []
    for direction in range(12):
        center = int(round(direction * width / 12.0))
        columns = np.arange(center - half_window, center + half_window) % width
        histogram = np.bincount(gray[:, columns].reshape(-1), minlength=256).astype(np.float64)
        probability = histogram[histogram > 0] / histogram.sum()
        scores.append(float(-(probability * np.log2(probability)).sum()))
    return np.asarray(scores, dtype=np.float32)


def summarize_episode(episode_id: str, steps: list[dict]) -> dict:
    progress=[]; after_distances=[]; invalid=0; coverage=[]; returns=[]; latencies=[]
    for step in steps:
        audit=step["audit"]
        before=audit.get("goal_geodesic_before_m"); after=audit.get("goal_geodesic_after_m")
        if before is not None and after is not None and np.isfinite(before) and np.isfinite(after):
            progress.append(float(before-after)); after_distances.append(float(after))
        invalid+=int(bool(audit.get("selected_direction_invalid",False)))
        if audit.get("coverage_fraction") is not None: coverage.append(float(audit["coverage_fraction"]))
        if audit.get("returned_to_visited_region") is not None: returns.append(bool(audit["returned_to_visited_region"]))
        if step.get("inference_latency_ms") is not None: latencies.append(float(step["inference_latency_ms"]))
    action_count=sum(step.get("selected_direction") is not None for step in steps)
    coverage_auc=float(np.trapezoid(coverage,dx=1.0)) if len(coverage)>1 else (coverage[0] if coverage else None)
    return {"episode_id":episode_id,"steps":steps,"action_count":action_count,"geodesic_progress_m":float(sum(progress)),"positive_progress_rate":float(np.mean(np.asarray(progress)>0)) if progress else None,"effective_progress_per_100_actions_m":float(sum(progress))*100.0/max(action_count,1),"entered_goal_3m":bool(after_distances and min(after_distances)<=3),"entered_goal_2m":bool(after_distances and min(after_distances)<=2),"entered_goal_1m":bool(after_distances and min(after_distances)<=1),"invalid_direction_selection_rate":invalid/max(action_count,1),"final_coverage_fraction":coverage[-1] if coverage else None,"coverage_auc":coverage_auc,"repeated_region_return_rate":float(np.mean(returns)) if returns else None,"inference_latency_ms":{"mean":float(np.mean(latencies)) if latencies else None,"p90":float(np.quantile(latencies,.9)) if latencies else None},"scope_note":"controlled short-horizon FG/FS progress audit; not full ImageNav SR/SPL"}


class FSDirectionController(FGFSDirectionController):
    """Test-only compatibility adapter for the historical tensor-returning stub."""

    def predict(self, current_erp_rgb: np.ndarray) -> np.ndarray:
        output = self.predictor({"current_erp_rgb": current_erp_rgb, "goal_erp_rgb": self.goal_erp_rgb})
        if isinstance(output, dict):
            tensor = output["masked_fs_scores"]
        else:
            tensor = output
        return tensor[0].detach().cpu().numpy()
