# Using This Repository as a Classification Skeleton

This document explains how this repository can be reused as a ready-made
multi-model benchmarking skeleton for **any image classification task** —
not just the original animal classification problem.

The repo was designed from the start to be modular and reproducible. Almost
every component is already domain-agnostic. Adapting it to a new task is
mostly a matter of replacing data and adjusting a handful of configuration
values — no deep code rewrites are needed.

---

## What Makes This Repo Reusable

The core design decisions that make it a good skeleton:

- **CSV-manifest data loading** — the dataset is defined by plain CSV files,
  not by a hardcoded folder structure or class names. Swap the CSVs and
  you swap the dataset.
- **`num_classes` is always a parameter** — no model architecture hardcodes
  the number of output classes. It is always passed in as an argument.
- **YAML-driven transforms** — augmentation and preprocessing are defined in
  a config file, not scattered across notebooks.
- **Shared training contracts** — all models use the same training loop,
  checkpointing, metrics payload, and artifact layout. A new model family
  inherits all of this for free.
- **MLflow tracking from day one** — experiment tracking is baked in, not
  bolted on later.

---

## High-Level Adaptation Steps

```
1. Replace the dataset (CSVs + images)
2. Update classes.json
3. Adjust transforms_v1.yaml
4. Set num_classes in notebook configs
5. Rename MLflow experiment strings (optional but recommended)
6. Run the overview notebook to verify environment readiness
```

---

## File-by-File Change Guide

Below is a precise list of every file that needs attention, organized by
how much work is involved.

---

### Step 1 — Replace the Dataset

#### `data/splits/split_v1/train.csv`
#### `data/splits/split_v1/val.csv`
#### `data/splits/split_v1/test.csv`

**What to do:** Replace with CSVs for your new task.

The required schema is unchanged:

```
filepath,label
path/to/image1.jpg,0
path/to/image2.jpg,1
...
```

- `filepath` — path to the image file, relative to the project root or
  absolute. The loader supports both styles via `normalize_paths=True`.
- `label` — integer class index starting from 0.

The split naming convention `split_v1` is just a string. You can keep it
or introduce a new version identifier such as `split_v1` for your new task
to maintain clear provenance.

**What does NOT need to change:** the loader code in `src/data/dataset_loader.py`.
It reads any CSV that follows the schema above.

---

#### `data/splits/split_v1/classes.json`

**What to do:** Replace the class index-to-name mapping with your own classes.

Current content (animal task):

```json
{
  "0": "cats",
  "1": "dogs",
  "2": "wildlife"
}
```

Example replacement for a plant disease task:

```json
{
  "0": "healthy",
  "1": "rust",
  "2": "powdery_mildew",
  "3": "leaf_spot"
}
```

Example replacement for a binary task (e.g. defect detection):

```json
{
  "0": "ok",
  "1": "defective"
}
```

This file is read by notebooks for display and logging purposes only. No
model architecture reads from it.

---

#### `data/prepared/`

**What to do:** Replace with your own image files, organized in whatever
folder structure suits your task. The loader does not require a specific
subfolder layout — it follows the filepaths in the CSV manifests.

---

### Step 2 — Adjust the Transformation Pipeline

#### `configs/transforms_v1.yaml`

**What to do:** Review and adapt all augmentation and preprocessing
parameters to match your new domain.

This is one of the most task-sensitive steps. The existing config was tuned
for natural animal photographs. Different domains have different needs.

**Parameters to review:**

| Parameter | Original value | What to consider for a new task |
|---|---|---|
| `RandomResizedCrop scale` | `(0.7, 1.0)` | For tasks where the subject must stay fully visible (e.g. documents, X-rays), raise the lower bound toward `1.0` |
| `RandomHorizontalFlip` | `p=0.5` | Safe for most natural image tasks. Disable for tasks where left/right has meaning (e.g. steering, handwriting) |
| `RandomRotation degrees` | `15` | Safe for most tasks. Raise for rotation-invariant tasks (e.g. aerial imagery, microscopy). Lower or remove if orientation matters (e.g. text recognition, medical scans with fixed orientation) |
| `ColorJitter` | brightness/contrast/saturation/hue | Reduce or remove for tasks where color carries diagnostic information (e.g. histopathology staining, satellite band values) |
| `Resize` / `CenterCrop` | 256/224 | Keep 224 if using ImageNet-pretrained backbones. Adjust if your images have a very different native resolution or aspect ratio |
| `Normalize mean/std` | ImageNet stats | Use ImageNet stats when fine-tuning from ImageNet weights (recommended). Compute dataset-specific stats only if training from scratch on very different image domains |

**Rule of thumb:** the more your images differ from natural RGB photographs,
the more carefully you should review the augmentation strategy. For
satellite imagery, medical scans, documents, or industrial inspection images,
at least review every augmentation individually before keeping it.

---

### Step 3 — Update Notebook Configurations

Each training notebook contains a configuration block near the top.

#### Fields to update in every training notebook:

| Field | Where it appears | What to change |
|---|---|---|
| `MODEL_NAME` | notebook config block | Keep the model registry name (e.g. `resnet18_pretrained`) or add new entries to `models.py` |
| `NUM_CLASSES` | notebook config block | Set to your actual class count |
| `MLFLOW_EXPERIMENT_NAME` | notebook config block | Change to a meaningful name for your new task (e.g. `"PlantDisease_PretrainedCNN"`) |
| `SPLIT_ID` | notebook config block | Update if you renamed your split |
| `TRAIN_CSV` / `VAL_CSV` / `TEST_CSV` | path definitions | Point to your new CSV files |

No changes are needed to the training loop, evaluation loop, checkpointing
logic, metrics payload builder, or artifact saving functions.

---

### Step 4 — (Optional) Add New Model Families

If you want to benchmark additional architectures not yet in the registry,
you only need to:

1. Add an entry to `SUPPORTED_MODELS` in `src/models/cnn_pretrained/models.py`
   (or the equivalent file in the relevant model family module).
2. Create a new training notebook following the existing notebook structure.

The shared `utils.py` training utilities require no changes — they work
with any `nn.Module`.

---

### Step 5 — (Optional) Add a New Model Family Module

If you want to introduce an entirely different family of models
(e.g. Vision Transformers, EfficientNet-v2, CLIP fine-tuning), create a new
module under `src/models/`:

```
src/models/
    your_new_family/
        __init__.py      ← export the public API
        models.py        ← model builders and specs
        utils.py         ← training/eval helpers (or import from existing)
```

Follow the same pattern as `cnn_pretrained/` or `cnn_scratch/`. The
notebook series for that family should follow the numbering convention:

```
notebooks/
    NN_your_family/
        NN_00_overview.ipynb
        NN_01_model_a.ipynb
        NN_02_model_b.ipynb
```

---

## What Requires Zero Changes

The following components are **fully domain-agnostic** and should not need
to be touched when adapting to a new task:

| Component | Reason |
|---|---|
| `src/data/dataset_loader.py` | Reads any `filepath,label` CSV |
| `src/data/split_generator.py` | Generates splits from any CSV manifest |
| `src/data/transforms.py` | Just parses `transforms_v1.yaml` |
| `src/models/cnn_pretrained/utils.py` | Pure PyTorch training utilities |
| `src/models/cnn_scratch/utils.py` | Same |
| All checkpointing and atomic save logic | Fully generic |
| `config.json` and `metrics.json` schemas | No class-name assumptions |
| MLflow parameter/metric logging calls | All string-keyed, no domain logic |
| Artifact directory structure | Driven by model name and timestamp |
| ONNX export | Works on any `nn.Module` with a standard forward pass |
| Inference benchmarking | Measures wall-clock time, no class awareness |

---

## Summary: Effort Estimate

| Change | Effort |
|---|---|
| Replace CSVs and images | Depends on dataset size, not on this repo |
| Update `classes.json` | < 5 minutes |
| Review and adjust `transforms_v1.yaml` | 15–30 minutes |
| Update `num_classes` and experiment names in notebooks | 5–10 minutes |
| Add a new model to an existing family | 10–20 minutes |
| Add an entirely new model family module | 1–2 hours (following existing pattern) |

The repo was designed so that the hard infrastructure work — reproducible
splits, shared transforms, MLflow tracking, idempotent runs, atomic saves,
cross-device path normalization — is done once and inherited by every
subsequent experiment automatically.
