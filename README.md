# SpitTrack

SpitTrack is a research-first, person-centric pipeline for spitting-event detection in railway-style surveillance video. The scientific unit of isolation is the **source video**, never an extracted frame, track, or window. The submission deadline is **30 August 2026**, so implementation is prioritized as follows:

1. leakage-safe folds and deterministic windows;
2. ResNet50 baseline results;
3. VideoMAE track-crop and whole-frame results;
4. complete-matrix cross-validation aggregation;
5. one-time staged-real stress testing.

No metric, label, checkpoint, or dataset artifact is fabricated. Missing or malformed inputs produce descriptive failures.

## Current milestone

The repository currently implements the empty data contract, strict metadata and annotation validation, balanced immutable fold construction, 25 FPS normalization, configuration-bound YOLOv8s/ByteTrack extraction, deterministic crop materialization, the frozen overlap-label rule, two-stage ResNet50 Kaggle training, a pretrained VideoMAE adapter, pure event aggregation, strict complete-matrix CV aggregation, and the staged-real execution guard. VideoMAE training and full-frame materialization are the next model milestone.

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

## Install for local contract and CPU tests

Python 3.11 is the target research runtime. Create an environment, then install the package in editable mode:

```bash
python -m pip install -e ".[test]"
pytest
```

GPU/model dependencies in `requirements.txt` use compatible minimum versions until the mandatory Kaggle smoke test establishes a working environment. The actual resolved Kaggle environment will then be frozen in `requirements-lock.txt`; dependency versions will not be guessed in advance.

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

Optional metadata columns from the project plan are accepted when present: `location_type`, `permission_reference`, `consent_reference`, `main_actor_count`, and `crowd_count`. They are not required by the loader. Synthetic generator provenance and source hashes are required for populated synthetic rows.

## Tracking manifest contract

Each per-source CSV in `data/tracking/manifests/` must contain:

```text
source_id, tracking_run_id, frame_index, timestamp_s, track_id,
x1, y1, x2, y2, det_conf
```

`tracking_run_id` binds annotations and windows to the exact detector/tracker configuration. A source cannot mix run IDs. Missing track frames cause affected windows to be rejected; the window builder never interpolates boxes or hallucinates frames.

## Fold construction

The default `K=5` requires at least ten synthetic sources: five positive and five negative. Exact fold balance also requires equal global positive and negative source counts. If these conditions are not met, construction fails and reports `N >= 2K` and the per-class minimum; it never lowers `K`, drops sources, or creates unbalanced folds.

```bash
python scripts/build_folds.py --k 5 --seed 42
```

Each source appears in an outer-test partition exactly once. Train, validation, test, and external-real partitions are disjoint. Generated manifests are immutable by default and must be generated once, reviewed, and committed before training on any machine.

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

The resulting index is deliberately fold-independent. Training joins its source IDs to committed fold manifests at runtime, avoiding duplicated window indexes with stale fold roles.

## ResNet50 Kaggle job

After committed fold manifests and materialized windows exist, one independent job is:

```bash
python scripts/train_kaggle.py \
  --config configs/models/resnet50.yaml \
  --model resnet50 \
  --fold 1
```

The implementation writes resumable `last.pt`, best-validation `best.pt`, validation and untouched outer-test metrics, outer-test predictions, config and fold snapshots, environment details, `pip freeze`, and seed/config/manifest hashes. `--fold all` runs folds sequentially into collision-free directories; separate accounts should normally receive one fold each. `--resume` requires and validates the existing `last.pt` hashes.

## Kaggle execution policy

Before the full experiment matrix, run a roughly ten-minute ResNet50 smoke test on one populated fold. It must verify CUDA and AMP, imports, data loading, a tiny batch, at least one optimizer step, `best.pt`/`last.pt`, resume state, config and fold snapshots, environment capture, and metric export. Only after this succeeds should teammates run independent model/fold jobs using the same repository commit, dataset version, manifests, config, and seed.

The downloaded unit is the complete self-describing run directory—not just weights and headline metrics. It includes checkpoints, metrics, config snapshot, fold manifest, environment/package record, seed, hardware information, git commit, and logs. Annotated MP4 and evidence rendering are restricted to inference, selected error-analysis samples, and the staged-real run.

## Final model and staged-real guard

After complete CV aggregation, final training uses all synthetic sources with no final validation set. Its epoch count is the median CV best epoch and its threshold is the median fold validation threshold, both frozen and recorded before training.

The staged-real command will refuse to proceed without both complete final-model provenance and explicit acknowledgement:

```bash
python scripts/test_real_staged.py \
  --final-model-provenance outputs/final_model/provenance.json \
  --i-understand-this-is-one-time
```

The current guard intentionally stops before opening real clips because staged-real inference is not implemented yet.

## Ethics and privacy

Track IDs are transient within-video trajectory identifiers only. The project performs no face recognition or cross-video civil-identity matching. Real clips remain an external staged stress test, are never used for tuning, and should remain access-controlled according to participant consent and filming authorization.
