# PIVOTNav

PIVOTNav is an RGB-only panoramic image-goal navigation system. The public
implementation follows the formal Phase 6 runner used for the reported
experiments and is organized around:

- `Panoramic_Place_Compass`
- `Topological_Belief_Cascade`
- `Navigable_Curiosity_Field`

The repository contains algorithm code only. Habitat-GS/Habitat-Lab,
Habitat-Sim, scene assets, datasets, model checkpoints, and OmniGuard/FS
worker resources are external prerequisites.

## Requirements

Install a compatible Habitat-GS environment with its matching Habitat-Lab and
Habitat-Sim interfaces. The runtime must provide:

- `habitat.Env` and the Habitat-GS panoramic evaluation configuration;
- the `velocity_control` action and `allow_sliding=False` contract;
- the external PIVOTNav checkpoints and worker packages;
- Python packages listed in `requirements.txt`.

The public runner does not use GT pose, GT distance, NavMesh, or GT success as
online policy inputs. GT fields in task manifests are post-run audit fields.

## External Configuration

Set these paths in `config.yaml` or as environment variables:

```yaml
habitat_root: /path/to/habitat-gs
weights_root: /path/to/pivotnav-models
fs_worker: /path/to/expanded_v6_worker.py
fs_package_root: /path/to/fs-package
fs_checkpoint: /path/to/step_019000.pt
dinov2_weight: /path/to/dinov2_vits14_pretrain.pth
dinov2_repo: /path/to/dinov2-source
omniguard_root: /path/to/OmniGuard
omniguard_worker: /path/to/omniguard_worker.py
omniguard_checkpoint: /path/to/best_origin.pth
arrival_frozen_root: /path/to/arrival-verifier-runtime
arrival_dependency_root: /path/to/arrival-runtime-dependencies
arrival_protocol: /path/to/arrival_protocol.json
arrival_model: /path/to/GRADIENT_BOOSTING_SEQUENCE_V7.joblib
```

Weights are expected under `weights_root`, including the R36.1, R36.3 and FS
checkpoints. No checkpoints or scene data are committed to this repository.

## Run

Run the smoke test:

```bash
python main.py --smoke
```

Run one frozen task:

```bash
python main.py \
  --scene /absolute/path/to/task.json \
  --goal /absolute/path/to/goal_erp.png \
  --habitat-root /absolute/path/to/habitat-gs \
  --weights-root /absolute/path/to/pivotnav-models \
  --gpu 0
```

The task JSON must provide `scene_id`, `difficulty`, `runtime_inputs.episode_path`
and `runtime_inputs.goal_erp_path`, or the episode and goal can be supplied by
the CLI. The formal runner preserves the server action budgets: Easy 1200,
Medium 1500, Hard 2000, Hard+ 3000 and Hard++ 4000.

Training and evaluation entry points are kept inside their corresponding
paper-named module directories.
