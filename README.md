# PIVOTNav

PIVOTNav is an RGB-only panoramic image-goal navigation system for Habitat-GS.
The repository contains the navigation code, module training/evaluation entry
points, and tests. Model weights, datasets, and simulator scenes are external
artifacts and are never committed here.

## Install

Use Python 3.10+ with the Habitat-GS environment, then install this package:

```bash
git clone https://github.com/Lingxiao-z/PIVOTNav.git
cd PIVOTNav
pip install -e .
```

Set external artifacts in `config.yaml` or pass `--weights-root`. The external
directory must contain `r361/r361_modular.pt`, `r363/r363_bearing.pt`,
`fs/step_019000.pt`, LightGlue weights, and the OmniTrav checkpoint.

## Run a smoke test

```bash
python main.py --smoke
```

## Run Habitat-GS

`--scene` accepts a frozen task JSON containing `scene_id` and the task start
position. The task's goal ERP is passed separately.

```bash
python main.py \
  --config config.yaml \
  --scene /absolute/path/to/task.json \
  --goal /absolute/path/to/goal_erp.png \
  --weights-root /absolute/path/to/external/models
```

Set `habitat_root` in the YAML or `PIVOTNAV_HABITAT_ROOT` in the environment
to the Habitat-GS checkout. The adapter uses the native Habitat-Sim
`EquirectangularSensorSpec` and velocity control. OmniTrav is loaded from the
external `omnitrav/best_origin.pth` checkpoint.

The default topology backend is the original global VPR+BPL and graph update
implementation. The optional accelerated implementation can be selected with
`--topology-backend accelerated`.

Training and module evaluation commands are documented in the three module
README files. Dataset directories are intentionally empty in the repository.
The `third_party/` directory contains code only; all model weights and scene
assets remain external.
