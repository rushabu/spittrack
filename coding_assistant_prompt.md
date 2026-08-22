# Coding Assistant Prompt — Person-Centric Spitting-Event Detection (Research-First, Kaggle GPU Workflow)

Paste everything below the line into your coding assistant (Kimi Code / Codex), together with the capstone plan document if available.

---

You are implementing a RESEARCH project for a conference paper (ICMLDE 2026, Procedia Computer Science). The goal is defensible experimental results, not a deployable product. Treat the scientific protocol below as the source of truth. Do not silently change it; if a requirement is infeasible because of an API/library difference, document the reason and make the smallest compatible change. NEVER fabricate dataset files, labels, metrics, checkpoints or results. Never write placeholder accuracy values as if experiments were completed. Missing data must produce an informative error.

## PROJECT

A Person-Centric Spatiotemporal Framework for Spitting-Event Detection in Railway Surveillance.

Pipeline: video -> YOLOv8s person detection (frozen, pretrained) -> ByteTrack tracking -> per-track 2.0 s temporal windows (0.5 s stride, 16 uniformly sampled frames, 224x224) -> action classification (ResNet50 frame baseline, ResNet50+LSTM temporal baseline, pretrained VideoMAE fine-tune) -> temporal smoothing + thresholding + event merging -> structured events (track ID, start/end time, bbox, confidence, evidence clip).

## CRITICAL EXECUTION CONTEXT (READ FIRST)

1. NO LOCAL GPU. All training runs on Kaggle notebooks (free T4/P100 GPU). Architect the code so that:
   - The repo is a clean, pip-installable-style Python package that can be zipped/uploaded or cloned into a Kaggle notebook with zero restructuring.
   - A single entry-point script `scripts/train_kaggle.py` (or a thin Kaggle notebook that calls it) runs ONE full training+evaluation job given a `--config` and a `--fold` argument, so the user can run folds/models as separate Kaggle sessions if needed.
   - All datasets are expected as Kaggle Datasets mounted read-only under `/kaggle/input/`; the code must resolve dataset paths from config, never hardcode local paths.
   - Every training run writes its checkpoint(s), `metrics.json`, `metrics.csv`, config snapshot, resolved package versions, seed, and fold manifest into `/kaggle/working/outputs/` so the user can download them.
   - Training MUST be resumable: save `last.pt` (optimizer + epoch state) in addition to `best.pt`, and support `--resume`.
   - After Kaggle training, the user downloads ONLY the .pt checkpoints + metrics back to their local machine. All local code (inference, event aggregation, plots, paper tables) must run on CPU only. Provide a `--device cpu` path everywhere and make CPU the default for inference.
2. DATASET SIZE IS CONFIGURABLE. The number of synthetic sources N (S01..SN) is NOT fixed at 20. Read the source list exclusively from `data/metadata/source_metadata.csv` (one row per source video). All fold construction, counts, and reports must derive from that CSV at runtime — never hardcode 20 anywhere. The user will update the CSV when the final count is known. Fold logic must handle any reasonable N with a roughly balanced positive/negative split (state the assumed label column: `contains_spit` true/false).
3. REAL PLATFORM CLIPS ALREADY EXIST. Files live in `data/raw/real_staged/` (R01.mp4 ... possibly more). They are an external holdout: they must NEVER enter training, validation, threshold selection, early stopping, or model choice. There is no need to schedule filming or permissions — that is done. The final frozen model is run on them exactly once, and results are reported per-clip (detected? correct track? start/end? confidence? false alerts? notes), never as accuracy percentages.
4. RESEARCH-FIRST PRIORITIES. In case of time pressure, cut in this order: Streamlit dashboard (lowest priority, optional stub is fine) -> pose/ST-GCN branch (do NOT implement unless explicitly asked later) -> extra ablations. NEVER cut: source-grouped folds, hard negatives, the three-model comparison, event-level metrics, the whole-frame vs track-crop ablation, the staged-real stress test.
5. MULTI-MACHINE PARALLEL EXECUTION. The experiment matrix will be split across 3-4 separate Kaggle accounts/machines (the user plus teammates) to run folds and models in parallel. Architect for this explicitly:
   - `scripts/train_kaggle.py` MUST accept `--model {resnet50|resnet_lstm|videomae|videomae_fullframe}` AND `--fold {1..K|all}` AND `--resume`, so any single run is one fully independent job: one model x one fold -> one output directory. No cross-run communication, no shared mutable state, no assumptions about execution order.
   - FOLD MANIFESTS ARE GENERATED ONCE AND COMMITTED. `build_folds` writes manifests to `data/splits/`; these files are treated as immutable inputs to training. Training code must NEVER regenerate folds — it reads the committed manifests. Same for `tracking_run_id` artifacts. This guarantees every machine trains on byte-identical partitions.
   - Every run directory must be self-describing: config snapshot, seed, fold manifest copy, resolved `pip freeze`, GPU name, git commit (if available), metrics CSV/JSON, best.pt/last.pt. Two runs on different machines must be mergeable by simply copying their output directories into one tree — no filename collisions (include model+fold in every output path, e.g. `outputs/metrics/videomae/fold_03/`).
   - Aggregation is a SEPARATE local CPU step (`scripts/evaluate_cross_validation.py`): it scans a directory of merged per-run outputs, validates that the expected model x fold matrix is complete (and reports exactly what is missing), and only then computes cross-fold mean +/- std tables and paper plots. It must refuse to aggregate a partial matrix silently.
   - Seeds and configs are identical across machines (seed=42 everywhere); determinism checks: same manifest + same config + same seed -> the code should warn if a run's recorded config hash differs from the committed one.
   - The FINAL model (trained on all synthetic sources after CV selects the winner) and the staged-real stress test are SERIAL steps run on ONE machine after all parallel results are merged and aggregated — never parallelize or distribute those two steps.
   - README must contain a short "Running the experiment matrix across multiple Kaggle accounts" section: same dataset version + same commit on every machine, per-machine job assignment via --model/--fold, download outputs, merge locally, aggregate.

## DATA CONTRACT

Synthetic sources: `data/raw/synthetic/S01.mp4 ... S<NN>.mp4` (count from metadata CSV).
Real staged: `data/raw/real_staged/R01.mp4 ...` (present already).
Metadata CSV `data/metadata/source_metadata.csv` columns: source_id, file_name, domain (synthetic|real_staged), contains_spit, primary_label, negative_subtype, generator_provider, generator_model, generator_version, generation_date, prompt_id, prompt_text, seed, candidate_index, qc_status, qc_rejection_reason, fps_original, fps_normalized, width, height, duration_s, camera_height_m, camera_angle, lighting, occlusion_level, split_role, sha256. (Blank generator fields for real clips.)
Annotation CSV `data/annotations/action_intervals.csv` columns: source_id, tracking_run_id, track_id, actor_role, action_label, action_start_s, action_end_s, negative_subtype, annotation_confidence, notes.

Validate at load: unique source IDs; referenced files exist; action intervals inside duration; labels in known enumeration; real clips can never be assigned to synthetic folds. Raise descriptive errors on violations.

## PREPROCESSING

- Normalize all videos to 25 fps without changing playback duration; keep originals; log dimensions/FPS; compute SHA-256.
- Detection: Ultralytics YOLOv8s (`yolov8s.pt`), COCO person class only, imgsz=640, conf=0.25, iou=0.50. Do NOT train YOLO. Make it configurable.
- Tracking: Ultralytics ByteTrack (`bytetrack.yaml`), persist tracks. Save per-source manifest (source_id, frame_index, timestamp_s, track_id, x1,y1,x2,y2, det_conf). Store a `tracking_run_id` derived from a hash of detector+tracker config; if config changes, tracking_run_id changes and annotations must be regenerated (warn loudly).
- Track filters: min duration 2.0 s; min bbox height 80 px at 720p-equivalent; bbox padding 18% clamped to frame; never write zero-size crops; log every rejected track with reason.
- Windows: 2.0 s (50 frames @25fps), stride 0.5 s, uniformly sample exactly 16 frame indices (deterministic function, unit-tested). Window label = spitting iff it substantially overlaps an annotated spitting interval; ambiguous boundary windows -> label `ignore`. If a track is briefly absent, reject the window; do NOT interpolate or hallucinate frames.

## MODELS (all on identical folds)

1. ResNet50 baseline: torchvision pretrained, center frame only, Dropout(0.3) + Linear(2048->2). Stage 1: backbone frozen, 3 epochs, head lr 1e-3. Stage 2: unfreeze layer4+fc, <=22 epochs, backbone lr 1e-5, head lr 1e-4, AdamW, weight_decay 1e-4, batch 16, early stop val F1 patience 5.
2. ResNet50+LSTM: shared ResNet50 -> 2048-D per frame -> 2-layer unidirectional LSTM (hidden 256, dropout 0.3) -> Dropout(0.3) -> Linear(256->2). batch 8, AdamW, layer4 lr 1e-5, LSTM/head lr 1e-4, <=25 epochs, patience 5.
3. VideoMAE: Hugging Face `MCG-NJU/videomae-base`, 16 frames 224x224, 2 classes. Fine-tune head + final 4 transformer blocks first; optional full unfreeze after epoch 4 only if validation is stable. AdamW, backbone lr 2e-5, head lr 1e-4, weight_decay 0.05, physical batch 2 + grad accum 4 (eff. 8), <=20 epochs, warmup 10%, grad clip 1.0, early stop val F1 patience 4. NEVER train from scratch. MUST fit in Kaggle T4 (16 GB) with AMP — if OOM, reduce physical batch to 1 and increase accumulation, and log the change.
- All: seed=42 (Python/NumPy/PyTorch), AMP on CUDA, identical source folds, same 16-frame window inputs (ResNet50 sees center frame of the same window).
- Augmentation (training only): RandomResizedCrop scale 0.85-1.0, hflip p=0.5, mild brightness/contrast, rotation <=5 deg — apply the SAME sampled spatial transform to all 16 frames of a clip. No temporal reversal.
- Training sampler: source/class-balanced or per-track cap so long tracks cannot dominate. Report source/track/window counts separately; never describe windows as independent videos.

## EVALUATION (this is the paper — treat as sacred)

- Family-grouped K-fold CV (default K=5): sort positive and negative generation families independently and distribute each class deterministically so per-class fold counts differ by at most one; global class totals need not be equal. Every family is outer-test exactly once, every variant of a family stays in one partition, and real clips remain outside all folds. Select validation by complete families for early stopping and threshold selection. Never drop or split a family; log family, source, and window-label composition for every fold.
- Threshold tau: chosen per fold on VALIDATION only to maximize F1. Never touch outer-test or real clips for tuning.
- Metrics (mean +/- std across folds): window-level precision/recall/F1/balanced accuracy/PR-AUC/confusion matrix; event-level precision/recall/F1 with temporal IoU >= 0.5 match + start/end timing error in seconds; false positives per minute on no-spit sources; track-attribution accuracy on multi-person positives; hard-negative false-positive counts by subtype (drink, cough, sneeze, eat, wipe_mouth, face_touch, bend_pickup, mask_adjust, talk_laugh, normal_activity); inference FPS and end-to-end latency (CPU and, if available, GPU — label clearly).
- Event aggregation (pure, unit-tested functions): sort per track by time -> median smooth over 3 windows -> threshold tau -> require >=2 consecutive positives -> merge same-track candidates with gap <=1.0 s -> event start/end = union -> confidence = mean of top-3 smoothed scores -> peak bbox at max-score window.
- Ablations (required): whole-frame VideoMAE vs track-crop VideoMAE (the central claim — report honestly either way); raw thresholding vs temporal aggregation. (Pose/ST-GCN: NOT in scope now.)
- Final model: after CV, pick winner by predeclared criteria (mean event F1, then FP/min), fix hyperparameters from CV (median epoch/threshold policy), train one final checkpoint on ALL synthetic sources, freeze, then run R-clips ONCE. Output per-clip stress-test table (no accuracy percentages).
- Export per run: checkpoints (best.pt, last.pt), per-fold metrics CSV/JSON, cross_validation_summary.csv, staged_real_stress_test.csv, plots (confusion matrices, PR curves, hard-negative breakdown bar chart, model comparison), events CSV+JSON, annotated MP4, per-event evidence clips, full logs.
- Reproducibility: save config snapshot, resolved `pip freeze` lockfile, Python/torch/CUDA versions, GPU name, seed, fold manifest with every run. No scientifically important magic numbers in code — everything via YAML configs.
- Tests (pytest): metadata schema validation; no source leakage across partitions + every synthetic source outer-tested exactly once + real clips never in folds; deterministic 16-frame sampling; bbox padding/clamping; track buffer stride correctness; event aggregator (isolated spike -> no event; 2 consecutive -> event; <=1 s gap merges; >1 s splits; different tracks never merge; confidence math); metrics on hand-computable arrays; CPU end-to-end smoke test on a tiny dummy video with mocked detector/classifier.

## SUGGESTED STRUCTURE (adapt freely, keep modules thin and typed)

configs/ (data, detector, tracker, resnet50, resnet_lstm, videomae, inference YAMLs)
data/{raw/{synthetic,real_staged}, metadata, annotations, tracking, processed, splits}
src/{common, data, models, train, evaluation, inference}/...
scripts/: validate_sources, normalize_sources, extract_tracks, build_folds, build_windows, train_kaggle (Kaggle entry), evaluate_cross_validation, train_final_model, test_real_staged (CPU), run_inference (CPU)
tests/ as listed above. README with: overview, Kaggle training instructions (upload dataset -> run notebook -> download .pt), local CPU inference instructions, how to update N via metadata CSV, reproducibility notes.

## KAGGLE WORKFLOW (document this in README precisely)

1. User creates a Kaggle Dataset containing the synthetic videos + metadata + annotations, and a second one with the repo (or clones from GitHub in the notebook).
2. Notebook: install pinned deps (pin torch/torchvision/ultralytics/transformers/opencv/scikit-learn/pandas to versions that exist on Kaggle — verify availability, prefer minimum versions over exact pins if conflicts), then run `python scripts/train_kaggle.py --config configs/videomae.yaml --fold all` (or per-fold).
3. On completion, everything needed by the paper is inside `/kaggle/working/outputs/`; user downloads checkpoints + metrics.
4. Local CPU: `scripts/test_real_staged.py` runs the frozen checkpoint on R-clips once; `scripts/run_inference.py` produces annotated video + events CSV/JSON + evidence clips.

## ACCEPTANCE CRITERIA

1. pytest passes (CPU). 2. Raw video -> per-track windows reproducibly. 3. All three models train on identical fold manifests on a Kaggle GPU session and produce real metrics. 4. No source leakage is possible through normal APIs (tested). 5. CV summary + plots + staged-real stress test exist and are computed from real predictions. 6. Full-video CPU inference produces stable track IDs, merged events, annotated MP4, CSV/JSON, evidence clips. 7. All runs save configs/env/seeds. 8. No code reports a metric it did not compute. Prioritize a correct, reproducible core over clever optimizations.

Development order: scaffolding+config+metadata validator+tests -> normalization+detection+tracking+manifests -> annotations+folds+windows+leakage tests -> ResNet50 -> ResNet50+LSTM -> VideoMAE (Kaggle) -> event aggregation+inference -> CV runner+tables -> final model+real stress test -> (optional stub dashboard only if everything else is done).
