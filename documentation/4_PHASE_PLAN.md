# Phase 4 Implementation Plan

**Project:** AnimalClassification  
**Phase:** Phase 4 - Pretrained CNN Transfer Learning and Fine-Tuning  
**Estimated Duration:** 7-10 days  
**Prerequisite:** Phase 1, Phase 2, and Phase 3 completed

---

## 1. Phase Objective

Phase 4 introduces the first family of **pretrained end-to-end CNN classifiers**.

Unlike Phase 2, where pretrained networks were used only as fixed embedding extractors, this phase will train full image-classification models that:

- reuse ImageNet-pretrained weights from `torchvision`
- replace the original classification head with a project-specific 3-class head
- fine-tune part of the pretrained backbone
- log classification quality and deployment-related costs under the same project conventions already used in the `30_` scratch-CNN notebooks

The main goal of Phase 4 is to establish a **strong transfer-learning benchmark** that sits between:

- the very strong fixed-feature baselines from Phase 2
- the custom scratch CNNs from Phase 3
- later architecture families such as ViTs

This phase must be designed around the **real current project structure**, not the older original architecture document.

---

## 2. Scope of This Phase

This phase covers **pretrained CNN backbones from `torchvision` only**.

The official scope is:

1. Validate the pretrained-CNN training contract and environment.
2. Implement a shared transfer-learning pipeline.
3. Train and benchmark a fixed set of pretrained CNN architectures.
4. Save standardized artifacts and MLflow logs.
5. Record latency, throughput, model size, and parameter count using the same style as the Phase 3 notebooks.

This phase does **not** include:

- ViTs
- CLIP fine-tuning
- self-supervised encoders
- centralized all-model evaluation notebook
- model export format benchmarking beyond the same ONNX attempt used in Phase 3

Those can come later after the pretrained CNN baselines are complete.

---

## 3. Exact Model Set To Implement

To keep the phase focused and comparable, the following pretrained CNNs will be implemented.

### Small / efficient models

1. `resnet18_pretrained`
2. `mobilenet_v3_large_pretrained`
3. `efficientnet_b0_pretrained`

### Medium / stronger models

4. `resnet50_pretrained`
5. `efficientnet_b2_pretrained`

These five models are the official Phase 4 model roster.

They were chosen because together they cover:

- residual architectures
- mobile-efficient architectures
- EfficientNet scaling behavior
- small vs medium deployment cost
- direct comparability with the Phase 2 ResNet50 embedding baseline

### Models intentionally excluded from this phase

The following should **not** be part of the first implementation pass:

- VGG
- Inception / GoogLeNet
- DenseNet
- ResNeXt
- ConvNeXt
- any ViT family

They may be added later, but they should not expand the initial scope of Phase 4.

---

## 4. Phase 4 Directory and Notebook Contract

At completion of Phase 4, the repository should contain the following new structure.

```text
models/
    cnn_pretrained/
        resnet18_pretrained/
            run_YYYYMMDD_HHMMSS/
                checkpoint.pt
                exported.onnx
                config.json
                metrics.json
                loss_curve.png
                accuracy_curve.png
        mobilenet_v3_large_pretrained/
            run_YYYYMMDD_HHMMSS/
                checkpoint.pt
                exported.onnx
                config.json
                metrics.json
                loss_curve.png
                accuracy_curve.png
        efficientnet_b0_pretrained/
            run_YYYYMMDD_HHMMSS/
                checkpoint.pt
                exported.onnx
                config.json
                metrics.json
                loss_curve.png
                accuracy_curve.png
        resnet50_pretrained/
            run_YYYYMMDD_HHMMSS/
                checkpoint.pt
                exported.onnx
                config.json
                metrics.json
                loss_curve.png
                accuracy_curve.png
        efficientnet_b2_pretrained/
            run_YYYYMMDD_HHMMSS/
                checkpoint.pt
                exported.onnx
                config.json
                metrics.json
                loss_curve.png
                accuracy_curve.png
```

```text
notebooks/
    40_cnn_pretrained/
        40_00_overview.ipynb
        40_01_resnet18_pretrained.ipynb
        40_02_mobilenet_v3_large_pretrained.ipynb
        40_03_efficientnet_b0_pretrained.ipynb
        40_04_resnet50_pretrained.ipynb
        40_05_efficientnet_b2_pretrained.ipynb
```

```text
src/
    models/
        cnn_pretrained/
            __init__.py
            models.py
            utils.py
```

```text
reports/
    metrics/
        phase4_overview_summary.json
        resnet18_pretrained_run_YYYYMMDD_HHMMSS_metrics.json
        mobilenet_v3_large_pretrained_run_YYYYMMDD_HHMMSS_metrics.json
        efficientnet_b0_pretrained_run_YYYYMMDD_HHMMSS_metrics.json
        resnet50_pretrained_run_YYYYMMDD_HHMMSS_metrics.json
        efficientnet_b2_pretrained_run_YYYYMMDD_HHMMSS_metrics.json
```

The naming convention should mirror the Phase 3 structure as closely as possible.

---

## 4.1 New Files To Be Created In This Phase

The following files are expected to be newly added during Phase 4 implementation.

### New notebooks

```text
notebooks/40_cnn_pretrained/
    40_00_overview.ipynb
    40_01_resnet18_pretrained.ipynb
    40_02_mobilenet_v3_large_pretrained.ipynb
    40_03_efficientnet_b0_pretrained.ipynb
    40_04_resnet50_pretrained.ipynb
    40_05_efficientnet_b2_pretrained.ipynb
```

### New source modules

```text
src/models/cnn_pretrained/
    __init__.py
    models.py
    utils.py
```

### New documentation and reporting outputs expected from runs

```text
reports/metrics/
    phase4_overview_summary.json
    resnet18_pretrained_run_YYYYMMDD_HHMMSS_metrics.json
    mobilenet_v3_large_pretrained_run_YYYYMMDD_HHMMSS_metrics.json
    efficientnet_b0_pretrained_run_YYYYMMDD_HHMMSS_metrics.json
    resnet50_pretrained_run_YYYYMMDD_HHMMSS_metrics.json
    efficientnet_b2_pretrained_run_YYYYMMDD_HHMMSS_metrics.json
```

### New model-artifact families expected from runs

```text
models/cnn_pretrained/
    resnet18_pretrained/
    mobilenet_v3_large_pretrained/
    efficientnet_b0_pretrained/
    resnet50_pretrained/
    efficientnet_b2_pretrained/
```

These folders are expected to be created by the notebooks when training runs actually happen.

---

## 4.2 Existing Files And Modules To Reuse

Phase 4 should deliberately build on the existing project instead of re-implementing already-solved pieces.

### Reuse from Phase 1

#### `src/data/dataset_loader.py`

Reuse for:

- CSV-manifest dataset loading
- portable path normalization
- cross-device path recovery
- transform application

This module is the preferred source of truth for image loading.

#### `src/data/transforms.py`

Reuse for:

- loading `configs/transforms_v1.yaml`
- constructing train transforms
- constructing eval transforms

Phase 4 should not invent a second transform system unless there is a truly necessary architectural reason.

#### `src/data/split_generator.py`

Reuse conceptually for:

- understanding split structure
- preserving the `split_v1` contract

It will likely not be used during Phase 4 training runs themselves, but it remains part of the fixed benchmark contract.

#### `configs/transforms_v1.yaml`

Reuse directly for:

- training augmentations
- evaluation preprocessing
- preserving fair comparison with earlier phases

### Reuse from Phase 2

#### Notebook conventions from the `10_` and `20_` ranges

Reuse for:

- robust project-root discovery
- MLflow setup style
- per-run output directories
- artifact naming conventions
- defensive checks for required files
- cross-platform path normalization logic

#### Artifact organization under `models/` and `reports/metrics/`

Reuse for:

- run-directory naming
- stable summary filenames
- run-specific metrics copies

This is especially important because the project is operated across two different machines and artifacts may be migrated manually.

### Reuse from Phase 3

#### `src/models/cnn_scratch/models.py`

Do not reuse the architecture definitions themselves, since Phase 4 uses pretrained `torchvision` models.

Reuse conceptually for:

- factory-style model construction
- model naming patterns

#### `src/models/cnn_scratch/utils.py`

This is the most important reusable implementation reference for Phase 4.

It should be reused or adapted for:

- training loop structure
- evaluation loop structure
- best-checkpoint logic
- metrics payload design
- parameter counting
- model-size estimation
- inference benchmarking
- ONNX export attempt behavior
- curve saving
- atomic JSON/checkpoint writes
- Run All safe artifact generation

The Phase 4 utilities should mirror this behavior closely unless pretrained-model needs force a change.

#### `30_` notebook conventions

Reuse for:

- overview notebook structure
- staged artifact saving
- MLflow logging contract
- benchmark reporting style
- final run summaries

The `30_` notebooks are the closest architectural template for Phase 4.

---

## 4.3 Notebook Architecture Design Principles

The notebook family for Phase 4 must be intentionally designed as a **safe experimental interface**, not just as ad hoc scripts split into cells.

Each notebook should follow these principles:

1. It must be safe to run from top to bottom with `Run All`.
2. It must remain safe if the user interrupts it midway and later reruns the full notebook.
3. It must not silently duplicate identical artifacts from a partially completed run.
4. It must not assume the current machine is the only execution environment.
5. It must keep the repository tidy even after repeated reruns.

This is a hard architectural requirement for the whole `40_` notebook range.

---

## 4.4 Planned Notebook Structure

The `40_` notebooks should follow a highly regular structure so that every model notebook behaves similarly.

### Common cell structure for training notebooks

Each training notebook should ideally contain sections in roughly this order:

1. imports
2. robust project-root detection
3. path definitions
4. required-file existence checks
5. MLflow setup
6. split loading and path normalization
7. transform loading
8. dataset and dataloader creation
9. device and environment summary
10. run configuration and hyperparameters
11. run directory / checkpoint path preparation
12. model creation
13. optimizer / scheduler / criterion setup
14. staged training
15. restore best weights
16. test evaluation
17. inference benchmarking
18. training-curve generation
19. ONNX export attempt
20. final metrics payload creation
21. report-copy saving
22. MLflow artifact and metric logging
23. final run summary cell

### Common structure for the overview notebook

The overview notebook should not train a model.

It should only:

1. verify environment and imports
2. verify required project files
3. verify transforms and dataset loading
4. verify pretrained-model loading
5. verify device detection
6. verify report and model output locations
7. write a Phase 4 readiness summary
8. log a lightweight MLflow validation run

---

## 5. Reused Project Inputs

Phase 4 must reuse the existing project foundation without changing it.

Required inputs:

- `data/prepared/`
- `data/splits/split_v1/`
- `configs/transforms_v1.yaml`
- `src/data/dataset_loader.py`
- `src/data/transforms.py`
- MLflow tracking directory `mlruns/`

Mandatory constraints:

- all Phase 4 models must use `split_v1`
- all Phase 4 models must reuse the shared transform config
- no architecture may create its own dataset split
- no image copies or alternate prepared datasets may be introduced

---

## 6. Shared Training Philosophy

Each Phase 4 model will follow the same two-stage training protocol.

### Stage A - Head training with frozen backbone

Purpose:

- stabilize the new classifier head
- adapt the pretrained encoder to the 3-class problem
- reduce early catastrophic updates to pretrained weights

Procedure:

1. Load official pretrained weights from `torchvision`.
2. Replace the original classification head with a new 3-class head.
3. Freeze the backbone parameters.
4. Train only the new head for a short warmup period.

Suggested default:

- `epochs_head_only = 3 to 5`

### Stage B - Partial fine-tuning

Purpose:

- adapt the higher-level pretrained features to the animal dataset
- improve over pure linear probing

Procedure:

1. Keep the new classifier head trainable.
2. Unfreeze the last major backbone stage.
3. Continue training with a smaller learning rate on backbone parameters.
4. Select the best checkpoint using validation macro F1.

Suggested default:

- `epochs_finetune = 10 to 20`

### Optional Stage C - Full unfreeze

This stage is **not mandatory in the first pass**.

It should be used only if:

- Stage B plateaus too early
- the model underperforms the scratch-CNN baseline unexpectedly
- validation curves suggest full-network fine-tuning is still stable

The first official implementation should focus on Stages A and B only.

---

## 7. Exact Architecture Handling Rules

The implementation must use the official `torchvision` pretrained variants available in the project environment.

The model factory should not hardcode feature dimensions where avoidable. Instead, it should read the classifier input size from the loaded model and then replace the final layer accordingly.

### 7.1 ResNet18

Model ID:

- `resnet18_pretrained`

Backbone source:

- `torchvision.models.resnet18(weights=...)`

Head replacement rule:

- replace `model.fc`

Fine-tuning rule:

- Stage A: train `fc` only
- Stage B: unfreeze `layer4` and `fc`

### 7.2 MobileNetV3 Large

Model ID:

- `mobilenet_v3_large_pretrained`

Backbone source:

- `torchvision.models.mobilenet_v3_large(weights=...)`

Head replacement rule:

- replace the final classification layer inside `model.classifier`

Fine-tuning rule:

- Stage A: train classifier only
- Stage B: unfreeze the last feature block group plus classifier

### 7.3 EfficientNet-B0

Model ID:

- `efficientnet_b0_pretrained`

Backbone source:

- `torchvision.models.efficientnet_b0(weights=...)`

Head replacement rule:

- replace the final classification layer inside `model.classifier`

Fine-tuning rule:

- Stage A: train classifier only
- Stage B: unfreeze the last feature stage plus classifier

### 7.4 ResNet50

Model ID:

- `resnet50_pretrained`

Backbone source:

- `torchvision.models.resnet50(weights=...)`

Head replacement rule:

- replace `model.fc`

Fine-tuning rule:

- Stage A: train `fc` only
- Stage B: unfreeze `layer4` and `fc`

### 7.5 EfficientNet-B2

Model ID:

- `efficientnet_b2_pretrained`

Backbone source:

- `torchvision.models.efficientnet_b2(weights=...)`

Head replacement rule:

- replace the final classification layer inside `model.classifier`

Fine-tuning rule:

- Stage A: train classifier only
- Stage B: unfreeze the last feature stage plus classifier

---

## 8. Classifier Head Design

To keep the comparison fair and implementation manageable, the new classification head should stay simple.

Default head design:

- dropout
- final linear layer to 3 classes

Recommended default:

```text
Dropout(p=0.2 to 0.5) -> Linear(in_features -> 3)
```

Important notes:

- do not introduce architecture-specific custom heads in the first pass
- do not add attention blocks, MLP heads, or metric-learning losses
- keep the head design lightweight so the comparison emphasizes the pretrained backbone itself

---

## 9. Data Loading and Transform Rules

Phase 4 should reuse the same project transform system already validated in earlier phases.

### Training transforms

Use:

- `get_train_transforms()` from `src/data/transforms.py`

This preserves:

- `RandomResizedCrop`
- random horizontal flip
- random rotation
- color jitter
- ImageNet normalization

### Evaluation transforms

Use:

- `get_eval_transforms()` from `src/data/transforms.py`

This preserves:

- resize
- center crop
- tensor conversion
- ImageNet normalization

### Loader rules

The implementation must:

- use the CSV-manifest split files
- support Windows and Linux path normalization
- reuse the same portable path handling approach already present in the `30_` notebooks
- never hardcode device-specific absolute paths

If the shared `ImageDataset` is reused directly, it should be used in its portable mode:

- `project_root=PROJECT_ROOT`
- `normalize_paths=True`
- `drop_missing=True` only when necessary for defensive validation

---

## 9.1 Run All Safety And Idempotent Notebook Behavior

This is one of the most important architectural constraints for Phase 4.

The notebooks are expected to be:

- rerun frequently
- interrupted sometimes
- executed on one machine and later migrated to another

Therefore the implementation must be **Run All safe** and as close to **idempotent** as practical.

### Core rule

Pressing `Run All` multiple times on the same notebook must not cause repository clutter or ambiguous duplicated artifacts for the same logical run.

### Required behavior

The notebook architecture should satisfy all of the following:

1. If no completed run exists yet for the current experiment configuration, the notebook may create one.
2. If a prior run was only partially created, rerunning should either:
   - resume from the checkpoint when resume behavior is explicitly supported, or
   - cleanly create a new run while ignoring the incomplete one, without corrupting previous artifacts.
3. If a logically identical completed run already exists and the notebook is intended to be deterministic for that config, the notebook should prefer reusing existing outputs instead of saving the same thing twice.
4. Small derived files such as report copies should not be redundantly overwritten in conflicting ways.
5. Checkpoints and JSON outputs should be written atomically to avoid half-written files after interruption.

### Important design choice

Phase 4 should distinguish clearly between:

- **experiment identity**
- **run identity**

An experiment identity is the semantic configuration:

- model name
- split id
- transform ids
- pretrained weights
- training hyperparameters
- seed

A run identity is the concrete execution instance on disk.

This distinction is necessary to decide whether a notebook should:

- reuse
- resume
- skip
- or create a fresh run

---

## 9.2 Recommended Run Directory Policy

The safest policy for this project is:

### Completed runs

Completed runs should remain immutable once successfully finalized.

That means:

- do not reopen a finished run directory and append new outputs later
- do not overwrite `metrics.json` of a finished run
- do not replace a completed checkpoint with a different model state

### Incomplete runs

If a run directory exists but is incomplete, the notebook should detect that clearly.

A run should be considered complete only if the required final artifacts exist, at minimum:

- `config.json`
- `metrics.json`
- `checkpoint.pt`

Preferably also:

- curves
- optional ONNX status recorded

If these are missing, the run should be treated as incomplete.

### Safe behavior on rerun

Recommended behavior:

1. Search the model-family folder for a matching completed run for the same experiment signature.
2. If found, print that it already exists and reuse it.
3. Otherwise search for a matching incomplete run with a recoverable checkpoint.
4. If resume is supported, resume from it.
5. If resume is not supported, create a fresh new run directory and leave the incomplete run untouched for later inspection.

This avoids accidental duplication while staying robust to interruptions.

---

## 9.3 Experiment Signature And Duplicate Prevention

To prevent saving the exact same experiment twice, each training notebook should compute a lightweight **experiment signature** from the logical configuration.

The exact implementation can vary, but conceptually the signature should include:

- model name
- split id
- transform ids
- pretrained weights identifier
- head-only epoch count
- fine-tuning epoch count
- optimizer type
- learning rates
- weight decay
- batch size
- seed

This signature should be saved inside `config.json`.

It should also be used to:

- search for previous matching runs
- detect whether a finished run already exists
- decide whether to skip duplicate artifact creation

### Recommended practical rule

For the initial implementation, it is acceptable if the notebook:

- skips training when a matching completed run already exists
- still allows a new run only when the configuration differs

This is simpler and safer than blindly creating a new timestamped run every time.

---

## 9.4 Checkpointing And Resume Expectations

Checkpointing is strongly recommended for Phase 4 because pretrained models may still be expensive to train.

### Minimum checkpoint requirement

The training notebook should save:

- best model weights
- optimizer state
- current epoch
- scheduler state if available
- best validation metrics so far
- experiment signature

### Resume behavior

Resume support is desirable, but it should only be implemented if it is clean and predictable.

If resume is added, it must:

- verify that the checkpoint matches the current experiment signature
- refuse to resume from a checkpoint belonging to a different configuration
- continue training without duplicating metric history incorrectly

If resume is not added in the first pass, that is acceptable, but then duplicate prevention and incomplete-run detection become even more important.

---

## 9.5 Atomic Saves And Partial-Run Robustness

Because notebook execution can be interrupted, all critical outputs must be written in a way that minimizes corruption risk.

This applies especially to:

- `config.json`
- `metrics.json`
- `checkpoint.pt`

Recommended behavior:

- write to a temporary file first
- replace the target file only after the write succeeds

This behavior already exists conceptually in the Phase 3 utility design and should be reused.

The notebook should also save `config.json` early so that incomplete runs are still identifiable.

---

## 9.6 Cross-Device Migration Design

Phase 4 must be written with the assumption that:

- notebooks may be developed on one device
- training may happen on another device
- artifacts may later be copied back manually

This means the implementation must avoid making a completed run depend on machine-local assumptions.

### Config and metadata should record

- project-relative intent
- actual device used
- actual runtime paths where helpful
- experiment signature
- model family and run id

### Paths in logic should remain portable

The code should compute paths dynamically from `PROJECT_ROOT`.

The code should not require that the original training machine still exists.

### Artifacts should be self-describing

A copied run directory should remain understandable by itself because `config.json` and `metrics.json` explain:

- what model it is
- what data split it used
- what transforms it used
- what hyperparameters were used
- whether ONNX export worked

This is especially important for your manual export workflow.

---

## 10. Device-Agnostic and Cross-OS Requirements

This phase must be written to run on both:

- Windows
- Linux

This is a hard requirement.

### Path handling requirements

All notebook and source code must:

- use `pathlib.Path`
- detect `PROJECT_ROOT` dynamically
- tolerate both slash styles in CSV filepaths
- avoid storing hardcoded `/home/...` or `F:\\...` assumptions in logic

### Device handling requirements

The pipeline must:

- auto-detect `cuda` vs `cpu`
- run on CPU if GPU is unavailable
- enable pinned memory only when helpful
- log the actual device used
- avoid CUDA-only assumptions in the training loop

### Mixed precision

Mixed precision may be used only when CUDA is available.

If used, it must:

- be optional
- degrade gracefully on CPU
- be logged in `config.json` and MLflow params

### DataLoader settings

Default logic should be adaptive rather than hardcoded to one machine.

Recommended pattern:

- `NUM_WORKERS = min(8, os.cpu_count() or 2)`
- `PIN_MEMORY = True if DEVICE == "cuda" else False`

Batch sizes may differ by model size, but the chosen value must always be logged.

---

## 11. Training Hyperparameter Contract

To reduce noise between architectures, all models should begin from a shared default recipe.

### Default optimizer

Recommended:

- `AdamW`

Reason:

- stable for transfer learning
- simple to tune across multiple pretrained backbones
- works well with frozen-to-unfrozen training stages

### Default loss

- `CrossEntropyLoss`

### Default scheduler

Reuse the Phase 3 convention:

- `ReduceLROnPlateau`

Suggested default:

- `mode="min"`
- `factor=0.3`
- `patience=2 or 3`

### Default regularization

- weight decay
- dropout in the new head
- existing data augmentation from `transforms_v1.yaml`

### Default epoch budget

Recommended first-pass budget:

- `head_only_epochs = 5`
- `finetune_epochs = 15`

Total:

- about `20` epochs per architecture

### Default learning rates

Recommended first-pass strategy:

- head-only stage: `1e-3`
- fine-tuning stage:
  - backbone: `1e-4`
  - classifier head: `5e-4`

The code should support separate parameter groups so backbone and head can use different learning rates.

### Gradient clipping

Reuse the Phase 3 convention:

- `max_norm = 1.0`

### Random seed

Reuse project default:

- `seed = 42`

All seeds and deterministic settings must be logged.

---

## 12. Model-Specific Batch Size Guidance

To stay device-agnostic while still being practical, the first implementation should follow approximate default batch-size targets.

### Small models

- `resnet18_pretrained`: target `64` on GPU
- `mobilenet_v3_large_pretrained`: target `64` on GPU
- `efficientnet_b0_pretrained`: target `64` on GPU

### Medium models

- `resnet50_pretrained`: target `32` on GPU
- `efficientnet_b2_pretrained`: target `32` on GPU

### CPU fallback guidance

If running on CPU, use a smaller batch size such as:

- `8`
- `16`

The exact batch size should be treated as a configuration value, not hardcoded into shared logic.

---

## 13. Shared Benchmarking and Inference-Cost Contract

Phase 4 must continue the benchmarking style introduced in the `30_` notebooks.

Every model run must report:

- `test_loss`
- `test_accuracy`
- `test_macro_f1`
- `parameter_count`
- `model_size_mb`
- `latency_ms_per_batch`
- `latency_ms_per_image`
- `throughput_img_per_sec`

### Benchmark methodology

The benchmark should follow the same general pattern used in Phase 3:

1. switch model to `eval()`
2. run several warmup batches
3. time a fixed number of forward-only batches
4. synchronize CUDA if using GPU
5. compute latency and throughput

Recommended defaults:

- `warmup_batches = 5`
- `timed_batches = 20`

### Important comparison note

Because runs may occur on different machines, raw latency values are useful but not perfectly comparable unless the device metadata is also logged.

Therefore every run must log:

- device type
- GPU availability
- GPU model if available
- CPU worker count
- batch size used for benchmark

This is especially important for the user's two-device workflow.

---

## 14. MLflow Logging Contract

MLflow integration is mandatory for every pretrained CNN training notebook.

It should follow the same project-level convention already used in Phase 2 and Phase 3.

### Tracking setup

Use:

```python
mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
mlflow.set_experiment("AnimalClassification")
```

### Required MLflow params

Each run must log at minimum:

- `stage = cnn_pretrained_training`
- `model_name`
- `backbone_name`
- `weights_name`
- `split_id`
- `transform_id_train`
- `transform_id_eval`
- `batch_size`
- `optimizer`
- `learning_rate_head`
- `learning_rate_backbone`
- `weight_decay`
- `scheduler`
- `head_only_epochs`
- `finetune_epochs`
- `seed`
- `device`
- `amp_enabled`

### Required MLflow metrics

Each run must log at minimum:

- best validation metrics
- test metrics
- latency metrics
- throughput metrics
- parameter count
- model size

If epoch-level logging is implemented, it should also log:

- `train_loss`
- `train_accuracy`
- `val_loss`
- `val_accuracy`
- `val_macro_f1`

### Required MLflow artifacts

Each run must log:

- `config.json`
- `metrics.json`
- `checkpoint.pt`
- `loss_curve.png`
- `accuracy_curve.png`

If ONNX export succeeds, also log:

- `exported.onnx`

If ONNX export fails due to missing optional dependencies, that failure should be:

- non-fatal
- recorded in config and metrics
- visible in notebook output

This should match the Phase 3 approach.

---

## 15. Artifact and Config Contract

Each run directory must contain a self-sufficient record of what happened.

### `config.json` must include

- model name
- backbone family
- pretrained weights identifier
- split id
- transform ids
- dataset version
- batch size
- optimizer settings
- scheduler settings
- seed
- device information
- stage configuration
- whether mixed precision was enabled
- whether ONNX export succeeded

### `metrics.json` must include

- full training history
- best validation epoch
- best validation metrics
- test metrics
- confusion matrix
- benchmark metrics
- parameter count
- model size

The structure should remain close to the Phase 3 `metrics.json` contract so future aggregation is easy.

---

## 16. Phase 4 Notebook Responsibilities

### 40_00_overview.ipynb

Purpose:

- validate environment for pretrained models
- confirm `torchvision` pretrained weights can be loaded
- validate shared data pipeline
- validate device selection
- record a Phase 4 readiness summary

Outputs:

- `reports/metrics/phase4_overview_summary.json`
- MLflow overview validation run

This notebook should not perform full training.

### 40_01_resnet18_pretrained.ipynb

Purpose:

- implement and benchmark the lightest residual transfer-learning baseline

### 40_02_mobilenet_v3_large_pretrained.ipynb

Purpose:

- implement and benchmark a mobile-efficient pretrained CNN

### 40_03_efficientnet_b0_pretrained.ipynb

Purpose:

- implement and benchmark a compact EfficientNet baseline

### 40_04_resnet50_pretrained.ipynb

Purpose:

- implement and benchmark a stronger residual transfer-learning model
- provide direct comparison to the already excellent ResNet50 fixed-embedding baseline from Phase 2

### 40_05_efficientnet_b2_pretrained.ipynb

Purpose:

- implement and benchmark the strongest medium-sized EfficientNet in this phase

Each training notebook should:

1. validate required files
2. load shared transforms
3. build datasets and dataloaders
4. build model with pretrained weights
5. run Stage A and Stage B training
6. restore the best checkpoint
7. evaluate on test set
8. benchmark inference cost
9. save artifacts
10. log to MLflow

---

## 17. Shared Source Module Responsibilities

The training logic should not live only inside notebooks.

The notebooks should stay thin and reuse shared Python modules, just like the Phase 3 direction.

### `src/models/cnn_pretrained/models.py`

Responsibilities:

- load supported pretrained backbones
- replace classifier heads
- freeze and unfreeze parameter groups
- expose a simple factory like `build_model(model_name, num_classes, pretrained=True, ...)`

### `src/models/cnn_pretrained/utils.py`

Responsibilities:

- training loop
- validation loop
- duplicate-run detection
- incomplete-run detection
- optional resume helpers if implemented
- staged freezing / unfreezing helpers
- checkpoint saving
- metric building
- curve saving
- benchmark inference helpers
- ONNX export helper

Where practical, Phase 3 utilities should be reused or adapted rather than duplicated.

---

## 17.1 Reusable Helper Behaviors Expected In Phase 4 Utilities

The Phase 4 utility layer should explicitly centralize the behaviors that make the notebooks safe to rerun.

Examples of helper behaviors that should live in shared utilities rather than notebook-local ad hoc code:

- build experiment signature
- discover existing completed runs for a given signature
- discover incomplete runs
- decide whether to skip, resume, or create a fresh run
- create run directories safely
- save config atomically
- save checkpoint atomically
- save metrics atomically
- compute benchmark metrics consistently
- build standardized metrics payloads

This will make the whole `40_` notebook family more reliable and easier to migrate across devices.

---

## 18. Expected Performance Interpretation

This phase is expected to produce models that are competitive with or stronger than the scratch CNNs, but interpretation must be careful.

Important comparison anchors already available:

- handcrafted classical baselines are much weaker
- fixed ResNet50 embeddings are extremely strong in this project
- scratch CNNs already achieved strong results according to the latest README

The pretrained end-to-end CNNs in this phase should therefore be interpreted mainly along two axes:

1. **accuracy / macro F1**
2. **accuracy vs deployment cost**

Key questions to answer:

- Can smaller pretrained CNNs match or beat `CustomCNN v2`?
- Does full end-to-end transfer learning improve on fixed ResNet50 embeddings, or are embeddings already near the ceiling?
- Which model gives the best tradeoff between accuracy and inference cost?

---

## 19. Acceptance Criteria

Phase 4 is complete when all of the following are true:

1. `40_00_overview.ipynb` exists and validates the Phase 4 environment.
2. All five official pretrained CNN notebooks exist.
3. All five models can train end to end using the shared split and transform pipeline.
4. Each run produces:
   - checkpoint
   - config
   - metrics
   - training curves
   - MLflow logs
5. Each run logs latency and throughput metrics.
6. Each notebook is Run All safe and does not create clutter from repeated identical executions.
7. Duplicate completed runs for the same experiment configuration are prevented or explicitly skipped.
8. Incomplete runs are detected safely and handled predictably.
9. The training/evaluation code runs on both Windows and Linux without path-specific assumptions.
10. ONNX export is attempted in the same non-fatal style as Phase 3.
11. Each notebook is rerunnable without breaking previous artifacts.

---

## 20. Recommended Implementation Order

To reduce risk, the models should be implemented in the following order:

1. `40_00_overview.ipynb`
2. `40_01_resnet18_pretrained.ipynb`
3. `40_02_mobilenet_v3_large_pretrained.ipynb`
4. `40_03_efficientnet_b0_pretrained.ipynb`
5. `40_04_resnet50_pretrained.ipynb`
6. `40_05_efficientnet_b2_pretrained.ipynb`

Reasoning:

- start with a simple residual baseline
- validate the shared transfer-learning utilities early
- then add efficient/mobile architectures
- only after the pipeline is stable, move to heavier models

---

## 21. Suggested Day-by-Day Breakdown

### Day 1

- write Phase 4 shared source skeleton
- build overview notebook
- validate pretrained weight loading and path portability

### Day 2

- implement `resnet18_pretrained`
- verify full artifact and MLflow pipeline

### Day 3

- implement `mobilenet_v3_large_pretrained`
- implement `efficientnet_b0_pretrained`

### Day 4

- implement `resnet50_pretrained`

### Day 5

- implement `efficientnet_b2_pretrained`

### Day 6

- rerun unstable notebooks
- review metrics consistency
- verify portability assumptions on both devices

### Day 7

- finalize README and summary documentation updates after runs are complete

---

## 22. Final Output Of Phase 4

At the end of this phase, the project should have:

- a stable pretrained CNN training pipeline
- five concrete transfer-learning baselines
- standardized artifacts and MLflow runs
- portable notebook logic that works across the user's Windows and Linux machines
- comparable accuracy and deployment-cost measurements following the same conventions as the `30_` notebooks

This will establish the pretrained CNN benchmark family before the project moves on to later model groups such as ViTs or broader architecture comparisons.

---

# End of Phase 4 Plan
