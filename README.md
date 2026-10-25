# SpitTrack

SpitTrack is a research-first, person-centric pipeline for spitting-event detection in railway-style surveillance video. Correlated synthetic variants are isolated as one **generation family**; a source video is never split into extracted-frame, track, or window units. The submission deadline is **30 August 2026**, so implementation is prioritized as follows:

1. leakage-safe folds and deterministic windows;
2. ResNet50 baseline results;
3. VideoMAE track-crop and whole-frame results;
4. complete-matrix cross-validation aggregation;
5. one-time staged-real stress testing.

No metric, label, checkpoint, or dataset artifact is fabricated. Missing or malformed inputs produce descriptive failures.

## Current milestone

The repository implements the empty data contract, strict validation, balanced immutable folds, deterministic track-crop and full-frame windows, all required model families, event/CV aggregation, policy-frozen final retraining, CPU-default full-video inference and evidence rendering, and a durable one-time staged-real runner.

## Scientific pipeline

```mermaid
flowchart LR
    A["Source MP4"] --> B["Normalize to 25 FPS"]
    B --> C["Frozen YOLOv8s person detection"]
    C --> D["ByteTrack per-video identities"]
    D --> E["Complete 2.0 s track windows"]
    E --> F["16 deterministic frames"]
    F --> G["ResNet50 / ResNet50-LSTM / VideoMAE"]
    G --> H["Validation-selected threshold"]
    H --> I["Temporal aggregation"]
    I --> J["Track, interval, box, confidence, evidence"]
```

## Local Flask video demonstration

The presentation UI uses the existing frozen final model to process an uploaded video:
normalization → YOLOv8s/ByteTrack → person windows → classification → event aggregation → evidence.
It displays the original video, annotated playback, event clips, confidence and timestamps,
processing logs, and downloadable JSON/CSV results. Predictions are not ground-truth labels.

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[web]"
python scripts/demo.py
```

Open http://127.0.0.1:5000 and choose a video (up to 512 MB). Use a clip containing
a clearly visible person for at least two continuous seconds. CPU processing may
take longer than the clip duration. The final model, provenance and config snapshot
must be present in `outputs/final_model/`; detector weights must also be available.
Browser playback uses bundled FFmpeg to convert outputs to H.264. Only one run is
processed at a time, in an isolated subprocess; closing the server interrupts work.
Refreshing the result URL resumes viewing its saved status. Files remain under
`outputs/web_demo/<run-id>/`; remove old run directories when no longer needed.
This is a local presentation server; keep it bound to localhost.
The demo calls arbitrary-video inference and does not invoke the one-time staged-real
evaluation runner or alter its receipt, annotations, training, or evaluation outputs.

## Install for local contract and CPU tests

Python 3.11 is the target research runtime. Create an environment, then install the package in editable mode:

```bash
python -m pip install -e ".[test]"
pytest
```

GPU/model dependencies in `requirements.txt` use compatible minimum versions until the mandatory Kaggle smoke test establishes a working environment. The actual resolved Kaggle environment will then be frozen in `requirements-lock.txt`; dependency versions will not be guessed in advance.

For a local CUDA installation, select the PyTorch wheel matching the installed driver from the official PyTorch installer before installing the remaining requirements. The target paper environment is Python 3.11; do not assume a CUDA 12.6 wheel is compatible until the Kaggle smoke gate confirms the actual image.

## Data placement

```text
data/
├── raw/
│   ├── synthetic/       # count N is read from metadata, never hard-coded
│   └── real_staged/     # external holdout; never enters folds or tuning
├── metadata/source_metadata.csv
├── annotations/action_intervals.csv
├── tracking/manifests/
├── processed/windows/
└── splits/              # generated once, reviewed, then committed
```

The committed CSV files are header-only templates. Confirm the empty schema without pretending that data exists:

```bash
python scripts/validate_sources.py --schema-only
```

Normal validation requires populated rows and referenced files:

```bash
python scripts/validate_sources.py
```

`family_id` is required and non-empty for every synthetic row. All variants generated from the same shot-list specification must use the same `family_id`, and a family may contain only one `contains_spit` class. Staged-real rows must leave `family_id` blank. Source IDs and filenames are opaque values read from the CSV—there is no `S<NN>` naming requirement. Optional metadata columns from the project plan are accepted when present: `location_type`, `permission_reference`, `consent_reference`, `main_actor_count`, and `crowd_count`. They are not required by the loader. Synthetic generator provenance and source hashes are required for populated synthetic rows.

## Tracking manifest contract

Each per-source CSV in `data/tracking/manifests/` must contain:

```text
source_id, tracking_run_id, frame_index, timestamp_s, track_id,
x1, y1, x2, y2, det_conf
```

`tracking_run_id` binds annotations and windows to the exact detector/tracker configuration. A source cannot mix run IDs. Missing track frames cause affected windows to be rejected; the window builder never interpolates boxes or hallucinates frames.

Annotation happens only after the detector/tracker configuration is frozen. Run tracking, review the trajectories, then enter action intervals with the emitted `tracking_run_id` and `track_id` in `action_intervals.csv`. If detector weights or tracker configuration changes, regenerate tracking artifacts and annotations rather than reusing stale IDs.

Generate CPU-only visual checks for every completed tracking manifest with:

```bash
python scripts/make_track_previews.py
```

This writes one complete timestamped MP4 per source under `data/tracking/previews/`, with each tracked person outlined and labelled by track ID. Use `--source-id SOURCE_ID` for a single preview and `--overwrite` only when deliberately regenerating derived previews.

Prepare track IDs for staged-real annotation without touching synthetic artifacts or training inputs:

```bash
python scripts/extract_tracks_real.py
python scripts/make_track_previews.py --real-staged
```

The first command passes the unchanged frozen detector and tracker configurations over normalized real-holdout clips, writes the common manifest schema under `data/tracking_real/manifests/`, and prints the shared configuration-bound `real_tracking_run_id`. Rejections and previews remain under `data/tracking_real/rejections/` and `data/tracking_real/previews/`. This tree is evaluation-input preparation only: fold construction and `build_windows.py` continue to read only synthetic metadata partitions and `data/tracking/manifests/`.

## Fold construction

The default `K=5` requires at least ten synthetic families: at least five positive and five negative. Positive and negative families are sorted independently and assigned deterministically across folds; each per-class fold count differs by at most one even when the global class totals are unequal. Construction never lowers `K`, drops families, or splits a family.

```bash
python scripts/build_folds.py --k 5 --seed 42
```

Every variant in a family is assigned to the same train, validation, or outer-test partition. Each family—and therefore each synthetic source following its family—appears in an outer-test partition exactly once. This prevents near-duplicate variants with shared camera, subject, and action timing from crossing partitions and inflating held-out scores. Train, validation, test, and external-real source partitions are also disjoint. The command reports family and source composition per outer fold and, when the fold-independent `windows.csv` already exists, positive, negative, and ignored window counts. Generated manifests are immutable by default and must be generated once, reviewed, and committed before training on any machine.

## Window construction and labels

At 25 FPS, a window contains 50 consecutive normalized frames. Window starts advance deterministically at 0.5-second intervals, producing the expected alternating 12/13-frame offsets, and exactly 16 strictly increasing frame indices are sampled from each complete window.

For the matching source and track:

- overlap with an annotated spitting interval of at least 50% of that **action's duration** → `spitting`;
- zero overlap → `non_spitting`;
- partial overlap below 50% → `ignore`.

Ignored windows remain counted in logs and the index but must be excluded from training and every reported metric.

```bash
python scripts/build_windows.py
```

This writes `windows.csv` for person tracks and `windows_fullframe.csv` for the scene-level ablation. Full-frame windows are generated once per source/time span rather than duplicated for every detected person. Both indexes are deliberately fold-independent; training joins source IDs to committed manifests at runtime, avoiding stale fold roles.

## ResNet50 Kaggle job

After committed fold manifests and materialized windows exist, one independent job is:

```bash
python scripts/train_kaggle.py \
  --config configs/models/resnet50.yaml \
  --model resnet50 \
  --fold 1
```

The implementation writes resumable `last.pt`, best-validation `best.pt`, validation and untouched outer-test metrics, outer-test predictions, config and fold snapshots, environment details, `pip freeze`, and seed/config/manifest hashes. `--fold all` runs folds sequentially into collision-free directories; separate accounts should normally receive one fold each. `--resume` requires and validates the existing `last.pt` hashes.

## Mandatory first Kaggle smoke gate

The first Kaggle GPU session must run the isolated smoke gate before any scientific job:

```bash
python scripts/kaggle_smoke_test.py --fold 1
```

It verifies the installed Torch, torchvision, Ultralytics, Transformers and OpenCV distributions; CUDA/AMP; a real materialized training batch; one ResNet50 optimizer step; and checkpoint reload. It never opens validation or outer-test data and writes `scientific_result=false` in `smoke_report.json`. On success it copies the captured `pip freeze` to `outputs/requirements-lock.txt`; that smoke-certified file should be downloaded and committed as the experiment lock rather than creating a lock from an unrelated local machine.

## Running the experiment matrix across multiple Kaggle accounts

Create one versioned Kaggle Dataset containing the validated videos, metadata, annotations, tracking manifests and materialized windows, and make the repository available by clone or a second Kaggle Dataset. Only after the smoke gate succeeds should teammates run independent model/fold jobs using the exact same repository commit, Kaggle Dataset version, manifests, config and seed. Assign each account explicit `--model` and `--fold` combinations, download the entire self-describing run directories, merge them under one local `outputs/runs/` tree, and aggregate only when the matrix is complete.

```bash
# Required track-wise models
python scripts/train_kaggle.py --config configs/models/resnet50.yaml --model resnet50 --fold 1
python scripts/train_kaggle.py --config configs/models/resnet_lstm.yaml --model resnet_lstm --fold 1
python scripts/train_kaggle.py --config configs/models/videomae.yaml --model videomae --fold 1

# Central scene-vs-person ablation
python scripts/train_kaggle.py \
  --config configs/models/videomae_fullframe.yaml \
  --model videomae_fullframe \
  --fold 1
```

ResNet50-LSTM trains ImageNet ResNet layer4 plus its two-layer unidirectional LSTM/head. VideoMAE trains its classifier and final four transformer blocks with AMP, warmup, clipping, and gradient accumulation. If a temporal model encounters CUDA OOM at physical batch size greater than one, the run retries from its last completed epoch at physical batch one, increases accumulation to preserve effective batch size, and records the change in `runtime_adjustments.json`. It never changes batch behavior silently.

The downloaded unit is the complete self-describing run directory—not just weights and headline metrics. It includes checkpoints, metrics, config snapshot, fold manifest, environment/package record, seed, hardware information, git commit, and logs. Annotated MP4 and evidence rendering are restricted to inference, selected error-analysis samples, and the staged-real run.

Every completed fold also exports timestamped outer-test predictions, raw-threshold and temporally aggregated event metrics, temporal-IoU matching, timing error, false positives/minute, applicable multi-track attribution, hard-negative subtype counts, events CSV/JSON, and clearly scoped classifier-only throughput/latency. Ignored windows never enter these metrics.

## Cross-validation aggregation and frozen final policy

Merge complete run directories locally, then execute:

```bash
python scripts/evaluate_cross_validation.py --output-root outputs
```

Aggregation refuses to run until all expected model × fold directories contain checkpoints, manifests, environments, logs, window predictions, event metrics, hard-negative results, and both source-video and family counts for every partition. It writes mean ± sample-standard-deviation rows to `cross_validation_summary.csv`, a paper-ready `cross_validation_partition_counts.csv` with entries such as “22 sources / 8 families,” and real-data paper plots under `outputs/plots/`: per-model confusion matrices and PR curves, the event-F1 model comparison, raw-threshold versus temporal-aggregation ablation, and hard-negative breakdown. Variant counts are never presented as independent-scene counts. It also writes `final_model_policy.json`, selecting among the three track-wise candidates by mean event F1 and then lower false positives/minute, and freezes the winner's median CV best epoch and median validation threshold. Whole-frame VideoMAE remains an ablation and is not eligible to become the final person-centric model.

## Final model and staged-real guard

After complete CV aggregation, final training uses all synthetic sources with no final validation set. Its epoch count is the median CV best epoch and its threshold is the median fold validation threshold, both frozen and recorded before training.

```bash
python scripts/train_final_model.py \
  --policy outputs/cross_validation/final_model_policy.json
```

The command verifies that the current complete configuration bundle has the exact hash selected during CV. It writes `final.pt`, `last.pt`, `training_manifest.json`, the frozen policy, environment/package snapshots, training-only losses, and portable relative paths in `final_model_provenance.json`. There is deliberately no `best.pt`, validation loader, threshold search, or final synthetic test metric.

Ordinary inference defaults to CPU and emits an annotated MP4, research-schema `events.csv`/`events.json` with checkpoint and evidence provenance, per-event evidence clips, intermediate tracking/window artifacts, config/model/environment/package snapshots, `run.log`, and an end-to-end performance summary labelled with its device and measurement scope:

```bash
python scripts/run_inference.py \
  --input path/to/video.mp4 \
  --final-model-provenance outputs/final_model/resnet50/final_model_provenance.json \
  --output-dir outputs/inference/example
```

The staged-real command will refuse to proceed without both complete final-model provenance and explicit acknowledgement:

```bash
python scripts/test_real_staged.py \
  --final-model-provenance outputs/final_model/resnet50/final_model_provenance.json \
  --i-understand-this-is-one-time
```

Before opening any real clip, the command validates all staged-real paths and hashes, prints the frozen model/epoch/threshold provenance, and creates the fixed repository receipt `outputs/staged_real_execution_receipt.json` with status `started`. Changing the result directory therefore cannot bypass the guard. Any existing receipt—including one left by an interrupted run—causes refusal, preventing an accidental second evaluation. A successful run writes per-clip descriptive results to `staged_real_stress_test.csv`; it does not report or tune on an aggregate staged-real accuracy.

## Ethics and privacy

Track IDs are transient within-video trajectory identifiers only. The project performs no face recognition or cross-video civil-identity matching. Real clips remain an external staged stress test, are never used for tuning, and should remain access-controlled according to participant consent and filming authorization.

## Troubleshooting

- “Balanced family-grouped K-fold construction is impossible” means the metadata does not contain at least `K` positive and `K` negative synthetic families. The tool reports the minimum `2K` families and will not weaken the protocol.
- A tracking-run mismatch means annotations were produced with different detector/tracker settings. Regenerate manifests and annotations; do not edit the recorded hash.
- A CUDA OOM during temporal training triggers the declared physical-batch-one fallback only when effective batch size can be preserved. Inspect `runtime_adjustments.json`.
- An incomplete-matrix error lists the exact missing model/fold artifacts. Recover those complete run directories rather than aggregating a subset.
- Staged-real refusal after a receipt exists is intentional. Do not delete the receipt merely to rerun the holdout; document any interrupted execution before deciding how it can be reported.
