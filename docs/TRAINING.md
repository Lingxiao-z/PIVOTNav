# Training PIVOTNav Models

PIVOTNav keeps Habitat-GS, scenes, DINOv2, OmniGuard, and trained weights as
external resources. This repository contains the data contracts, collectors,
model definitions, optimization loops, evaluation code, and checkpoint
packagers needed to recreate the learned PIVOTNav components.

## Model inventory

| Module | Artifact | Learned | Training entry |
| --- | --- | --- | --- |
| Navigable Curiosity Field | FG/FS last-block checkpoint | yes | `modules.Navigable_Curiosity_Field.training.train_curiosity_exploration` |
| Panoramic Place Compass | R3 retrieval encoder | yes | `modules.Panoramic_Place_Compass.training.retrieval.train_phase_{a,b,c}` |
| Panoramic Place Compass | Track-Y relative yaw | yes | `modules.Panoramic_Place_Compass.training.train yaw` |
| Panoramic Place Compass | arrival-private bearing | yes | `modules.Panoramic_Place_Compass.training.train private-bearing` |
| Panoramic Place Compass | R36.3 navigation bearing | yes | `modules.Panoramic_Place_Compass.training.train r363-bearing` |
| Panoramic Place Compass | temporal arrival head | yes | `modules.Panoramic_Place_Compass.training.train arrival` |
| Panoramic Place Compass | V7 sequence distance model | yes | `modules.Panoramic_Place_Compass.training.train v7` |
| Topological Belief Cascade | graph, scheduler, and BPL | no | no checkpoint is trained |

All online GT restrictions apply to inference. Data collection and training may
use Habitat-Sim NavMesh/geodesic information to construct offline labels.

## Verified environment

The server reproduction used Python 3.12.13, PyTorch 2.5.1+cu121,
torchvision 0.20.1+cu121, NumPy 2.2.6, OpenCV 5.0.0, Kornia 0.7.3,
and scikit-learn 1.7.1. `requirements.txt` specifies compatible lower bounds;
an exact lock file is not required by the pipeline.

## Navigable Curiosity Field

### 1. Collect FG/FS labels

The collector renders native 224x448 ERP RGB, samples navigable positions,
computes 12-sector FG labels, finds Source-connected local endpoints, and
computes FS targets from offline geodesic distance.

```bash
python -m modules.Navigable_Curiosity_Field.data.collect \
  --mode scene \
  --habitat-root /path/to/habitat-gs \
  --partition train \
  --scene-id interior_0007_840137 \
  --split train \
  --positions 300 \
  --output /path/to/fgfs/data/formal_v2/train
```

Each split contains `scene_records.json` plus `shards/*.fgfs_v2.npz`. The
required arrays and SHA validation are implemented in
`training/dataset.py`. Collect development scenes into a separate directory.

### 2. Train the released model

Provide the official pinned DINOv2 checkout and ViT-S/14 weight:

```bash
python -m modules.Navigable_Curiosity_Field.training.train_curiosity_exploration \
  --data-root /path/to/fgfs \
  --split train \
  --dinov2-repo /path/to/dinov2 \
  --dinov2-weight /path/to/dinov2_vits14_pretrain.pth \
  --output /path/to/output/fgfs \
  --steps 19000 \
  --batch-size 16 \
  --device cuda
```

Blocks 0-10 are frozen; block 11 and the 12-direction decoder are optimized.
The output uses schema
`revised_dino_fgfs_v3_last_block_b2_exact_resume_v1`, which is accepted by
the public inference class.

### 3. Evaluate

```bash
python -m modules.Navigable_Curiosity_Field.evaluation.evaluate \
  --data-root /path/to/fgfs \
  --split development \
  --checkpoint /path/to/output/fgfs/last.pt \
  --dinov2-repo /path/to/dinov2 \
  --dinov2-weight /path/to/dinov2_vits14_pretrain.pth \
  --output /path/to/output/fgfs/development_metrics.json
```

Set `fs_checkpoint`, `fs_sha256`, `dinov2_repo`, and `dinov2_weight` in the
runtime config to use a newly trained checkpoint. After `pip install -e .`, set
`fs_worker: pivotnav-fs-worker` to use the included persistent worker.

## Panoramic Place Compass

Set the two roots used by the retrieval data/training scripts:

```bash
export PIVOTNAV_HABITAT_ROOT=/path/to/habitat-gs
export PIVOTNAV_PPC_TRAINING_ROOT=/path/to/ppc-training-workspace
export PANORAMIC_VPR_DINOV2_CHECKOUT=/path/to/dinov2
export PANORAMIC_VPR_DINOV2_WEIGHT=/path/to/dinov2_vits14_pretrain.pth
```

### 1. Retrieval dataset and Phase A/B/C

Freeze the scene split, render samples, and train the released staged model:

```bash
python -m modules.Panoramic_Place_Compass.training.retrieval.prepare_scene_split
python -m modules.Panoramic_Place_Compass.data.collect \
  --split train --split-tag train_native_erp224_r3 --max-scenes 110
python -m modules.Panoramic_Place_Compass.data.collect \
  --split development --split-tag development_native_erp224_r3 --max-scenes 14

python -m modules.Panoramic_Place_Compass.training.retrieval.train_phase_a \
  --train-split-tag train_native_erp224_r3 \
  --output-name phase_a_native_erp224_r3
python -m modules.Panoramic_Place_Compass.training.retrieval.train_phase_b \
  --train-split-tag train_native_erp224_r3 \
  --checkpoint /path/to/phase_a/best.pt \
  --output-name phase_b_native_erp224_r3
python -m modules.Panoramic_Place_Compass.training.retrieval.train_phase_c \
  --train-split-tag train_native_erp224_r3 \
  --checkpoint /path/to/phase_b/best.pt \
  --output-name phase_c_native_erp224_r3
```

Phase A trains the descriptor/ring head with a frozen backbone. Phase B
unfreezes the final DINO blocks. Phase C trains candidate matching and the
explicit UNKNOWN output. The scripts retain checkpoint resume, DDP, hard
negative, circular roll, and test-isolation logic from the formal pipeline.

### 2. Pair-label manifest

After retrieval data collection, build scene-disjoint reciprocal bearing pairs
directly from the R3 tar shards:

```bash
python -m modules.Panoramic_Place_Compass.data.build_bearing_manifest \
  --repo "$PIVOTNAV_PPC_TRAINING_ROOT" \
  --output-dir /path/to/bearing-manifests
```

The resulting `train_bearing_pairs.jsonl` stores tar shard/member references,
offline Habitat position/yaw labels, reciprocal IDs, distance buckets, and
72-bin bearing labels. The preprocessor reads this format directly. A minimal
custom JSONL record may instead use absolute image paths:

```json
{"source_image":"/data/a.png","target_image":"/data/b.png","yaw_degrees":15.0,"bearing_degrees":-32.0,"bearing_valid":true,"distance_bucket":1}
```

Encode the frozen pair features:

```bash
python -m modules.Panoramic_Place_Compass.data.preprocess pairs \
  --manifest /path/to/bearing-manifests/train_bearing_pairs.jsonl \
  --dataset-root "$PIVOTNAV_PPC_TRAINING_ROOT" \
  --r361-checkpoint /path/to/base_r361.pt \
  --output /path/to/pairs.npz
```

Train the heads:

```bash
python -m modules.Panoramic_Place_Compass.training.train yaw \
  --data /path/to/pairs.npz --output /path/to/yaw.pt
python -m modules.Panoramic_Place_Compass.training.train private-bearing \
  --data /path/to/pairs.npz --output /path/to/private_bearing.pt
python -m modules.Panoramic_Place_Compass.training.train formal-bearing \
  --data /path/to/pairs.npz --output /path/to/formal_bearing.pt
python -m modules.Panoramic_Place_Compass.training.train r363-bearing \
  --data /path/to/pairs.npz --output /path/to/r363_head.pt
```

### 3. Temporal arrival and V7

Temporal arrival JSONL records add `sequence_id`, `frame_index`, `arrived`, and
offline `distance_m` to the source/target image fields. Convert and train:

```bash
python -m modules.Panoramic_Place_Compass.data.preprocess arrival \
  --manifest /path/to/arrival_sequences.jsonl \
  --r361-checkpoint /path/to/base_r361.pt \
  --output /path/to/arrival_sequences.npz
python -m modules.Panoramic_Place_Compass.training.train arrival \
  --data /path/to/arrival_sequences.npz --output /path/to/arrival.pt
```

For the final V7 regressor, provide JSONL rows containing either the frozen
visual feature vector or the raw `parallax_features` and `active_evidence`
dictionaries accepted by the released feature builder, plus the
remaining-distance target:

```json
{"features":[0.1,0.2,0.3],"target_m":0.72}
```

```bash
python -m modules.Panoramic_Place_Compass.data.preprocess v7 \
  --manifest /path/to/v7.jsonl --output /path/to/v7.npz
python -m modules.Panoramic_Place_Compass.training.train v7 \
  --data /path/to/v7.npz --output /path/to/GRADIENT_BOOSTING_SEQUENCE_V7.joblib
```

### 4. Package checkpoints

Build the modular R36.1 checkpoint from selected component checkpoints:

```bash
python -m modules.Panoramic_Place_Compass.training.train package-r361 \
  --retrieval-checkpoint /path/to/phase_c/best.pt \
  --yaw-checkpoint /path/to/yaw.pt \
  --formal-bearing-checkpoint /path/to/formal_bearing.pt \
  --private-bearing-checkpoint /path/to/private_bearing.pt \
  --arrival-checkpoint /path/to/arrival.pt \
  --output /path/to/r361/r361_modular.pt
```

Package the R36.3 navigation-bearing head with its private copy of the R36.1
tail:

```bash
python -m modules.Panoramic_Place_Compass.training.train package-r363 \
  --r361-checkpoint /path/to/r361/r361_modular.pt \
  --bearing-checkpoint /path/to/r363_head.pt \
  --output /path/to/r363/r363_bearing.pt
```

Record the generated SHA256 values as `r361_sha256`, `r363_sha256`, and
`fs_sha256` in `config.yaml`. Default hashes continue to protect the reported
paper checkpoints.

## Topological Belief Cascade

This module is deterministic graph state, candidate-frontier lifecycle,
event scheduling, known-node localization logic, and BPL updates. It does not
own a separately trained model. Its learned visual inputs are the Compass
retrieval/bearing checkpoints and Curiosity FG/FS checkpoint above.

## Training smoke tests

The smoke tests do not need scenes or pretrained weights:

```bash
python -m modules.Navigable_Curiosity_Field.training.train_curiosity_exploration --smoke
python -m modules.Panoramic_Place_Compass.training.train smoke
pytest -q tests/test_training_pipeline.py
```

They verify label semantics, dataset loading, finite losses, nonzero gradients,
checkpoint save/reload, V7 serialization, and R36.3 packaging. They are not a
replacement for full Habitat-GS collection and multi-GPU training.
