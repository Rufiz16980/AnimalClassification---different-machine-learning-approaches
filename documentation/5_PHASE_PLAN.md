# Phase 5 Plan: Vision Transformer Benchmark Family

This document defines the transformer-family image classification phase that follows the pretrained CNN benchmark.

The goal of Phase 5 is not to replace the existing `40_` family, but to add a clean transformer benchmark layer that can be compared directly against the strongest pretrained CNN baselines.

---

## 1. Phase Goal

Phase 5 introduces one representative pretrained model for each selected transformer-family architecture:

- plain ViT
- Swin Transformer
- Swin Transformer V2
- MaxViT

The emphasis is on:

- architectural diversity
- reuse of the existing project pipeline
- notebook safety under repeated reruns
- direct comparability against previous phases

---

## 2. Planned Notebook Structure

All transformer notebooks live in:

- `notebooks/50_vit/`

The planned notebook set is:

- `50_00_overview.ipynb`
- `50_01_vit_b_16.ipynb`
- `50_02_swin_t.ipynb`
- `50_03_swin_v2_s.ipynb`
- `50_04_maxvit_t.ipynb`

### Notebook Roles

- `50_00_overview.ipynb`
  - lightweight validation notebook
  - checks dataset availability
  - checks shared pathing assumptions
  - enumerates supported transformer-family models
  - saves a summary JSON to `reports/metrics`

- each training notebook
  - trains one model family only
  - reuses the shared rerun-safe transfer-learning workflow
  - writes artifacts only into its own model-family run directory

---

## 3. Planned Source Code Layout

The transformer-family implementation should live in:

- `src/models/vit/models.py`
- `src/models/vit/utils.py`
- `src/models/vit/__init__.py`

### Responsibilities

- `models.py`
  - model registry
  - pretrained weight resolution
  - classifier-head replacement
  - staged freezing/unfreezing rules
  - parameter-group construction
  - recommended image size metadata

- `utils.py`
  - reuses the same generic training/run-management helpers already used by the pretrained CNN family
  - keeps notebook structure aligned with Phase 4 conventions

---

## 4. Planned Artifact Layout

Transformer run artifacts should be stored under:

- `models/vit/vit_b_16/`
- `models/vit/swin_t/`
- `models/vit/swin_v2_s/`
- `models/vit/maxvit_t/`

Each completed run should save:

- `checkpoint.pt`
- `config.json`
- `metrics.json`
- `loss_curve.png`
- `accuracy_curve.png`
- optional `exported.onnx` if export succeeds

Metrics report copies should also be written to:

- `reports/metrics/`

---

## 5. Transformer Models Included

### 5.1 `vit_b_16`

Purpose:

- represent the plain Vision Transformer family
- provide a direct non-hierarchical transformer baseline

Notes:

- fixed patch-based transformer
- requires image size alignment with the pretrained configuration

### 5.2 `swin_t`

Purpose:

- represent hierarchical windowed transformers
- provide a lighter Swin-family baseline

Notes:

- local window attention
- stage-based hierarchical feature extraction

### 5.3 `swin_v2_s`

Purpose:

- represent the upgraded Swin V2 family
- provide a stronger hierarchical transformer baseline

Notes:

- different recommended runtime image size than the standard 224-crop family
- should use runtime transform size overrides

### 5.4 `maxvit_t`

Purpose:

- represent the MaxViT family
- provide a hybrid attention/convolution benchmark entry

Notes:

- combines convolutional stem logic with transformer-style attention blocks
- should be benchmarked separately from plain ViT assumptions

---

## 6. Notebook Design Contract

The `50_` notebooks must mimic the proven structure of the `40_` notebooks.

Each notebook should keep the same five-part structure:

1. markdown introduction
2. imports
3. project-root/path resolution plus package imports
4. notebook constants, datasets, loaders, signature setup, run resolution
5. train/evaluate/save/log flow

The key requirement is that the transformer family should not invent a second notebook architecture. It should feel like a direct continuation of the existing transfer-learning benchmark system.

---

## 7. Runtime Safety Rules

The Phase 5 notebooks must remain safe when:

- `Run All` is clicked multiple times
- a notebook is interrupted halfway through
- a completed run already exists
- the same repo is used on Windows and Linux

This means:

- experiment signatures must be deterministic
- completed matching runs must be reused instead of duplicated
- incomplete matching runs must be resumed or restarted in place
- config and metrics files must be written atomically
- checkpoints must be written atomically

---

## 8. Training Protocol

Each transformer-family notebook should use the same staged transfer-learning protocol as the `40_` family:

### Stage 1: head-only training

- freeze the backbone
- train only the replaced classifier head

### Stage 2: partial fine-tuning

- unfreeze the classifier head
- unfreeze only the final transformer blocks or final hierarchical stages
- continue training with smaller backbone learning rate

This preserves comparability with the pretrained CNN family while still respecting architectural differences.

---

## 9. Transform Handling

The transformer family should continue to reuse:

- `configs/transforms_v1.yaml`
- `src/data/transforms.py`

But Phase 5 requires runtime size overrides so that models with different recommended input sizes can still share the same base transform definition.

This avoids:

- duplicated YAML files
- manual notebook-local transform graphs
- hidden differences between architectures

---

## 10. MLflow And Reporting

Each Phase 5 training notebook should log:

- model name
- backbone family
- weights name
- split ID
- transform IDs
- image size
- resize size
- optimizer settings
- final validation/test metrics
- latency and throughput
- parameter count
- model size

Artifacts should include:

- config
- metrics
- checkpoint
- training curves
- ONNX file if available

---

## 11. ONNX Policy

ONNX export should remain:

- attempted automatically
- non-fatal on failure
- reported clearly in `metrics.json`

This keeps deployment/export information available when the environment supports it without making the training notebooks brittle.

---

## 12. Completion Criteria

Phase 5 is considered implemented when:

- all five `50_` notebooks exist
- the `src/models/vit` package exists and supports all four model families
- runtime size overrides are used safely where needed
- run directories are rerun-safe
- artifacts follow the same conventions as the `40_` family
- the notebooks are portable across the user's Windows and Linux machines

---

# End of Phase 5 Plan
