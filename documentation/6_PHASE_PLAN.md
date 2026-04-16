# Phase 6 Plan: Custom Vision Transformers From Scratch

**Project:** AnimalClassification  
**Phase:** Phase 6 - Custom ViT Models From Scratch  
**Estimated Duration:** 5-8 days of implementation and first-run validation  
**Prerequisite:** Phase 1 through Phase 5 completed

---

## 1. Phase Objective

Phase 6 introduces **manually implemented Vision Transformers trained from scratch**.

The goal of this phase is not to beat the strongest pretrained CNN or pretrained ViT baselines. Instead, it is to:

- learn the internals of Vision Transformer design
- implement encoder-only ViT models directly in PyTorch
- benchmark scratch-trained transformer models under the same project conventions already used for CNNs and pretrained backbones

This phase is intentionally educational and architecture-focused.

---

## 2. Scope Of This Phase

This phase covers:

- custom patch embedding
- learnable class token
- learnable positional embeddings
- multi-head self-attention
- MLP feed-forward blocks
- LayerNorm
- residual connections
- encoder-only classification models

This phase does **not** include:

- cross-attention
- encoder-decoder transformers
- masked image modeling
- distillation tokens
- hierarchical window attention
- pretrained ViT backbones

Those belong to other directions and are outside the scope of this scratch-ViT phase.

---

## 3. Architectural Clarification

For image classification in this phase, the intended transformer design is **self-attention**, not cross-attention.

### Why

The original ViT-style classifier is an **encoder-only** model:

- patch tokens attend to each other through self-attention
- a class token aggregates sequence information
- the class token is fed to a classification head

Cross-attention is more relevant for:

- encoder-decoder architectures
- multimodal models
- detection/segmentation variants

So the correct learning target for this phase is a clean encoder-only ViT.

---

## 4. Planned Notebook Structure

All scratch ViT notebooks live in:

- `notebooks/60_vit_scratch/`

The planned notebook set is:

- `60_00_overview.ipynb`
- `60_01_customvit_v1.ipynb`
- `60_02_customvit_v2.ipynb`

### Notebook Roles

- `60_00_overview.ipynb`
  - validates path portability
  - checks dataset inputs
  - enumerates supported scratch-ViT architectures
  - writes a family overview summary to `reports/metrics`

- `60_01_customvit_v1.ipynb`
  - trains the first educational ViT-from-scratch baseline

- `60_02_customvit_v2.ipynb`
  - trains a deeper/wider scratch ViT variant under the same benchmark conventions

---

## 5. Planned Source Code Layout

The scratch ViT implementation should live in:

- `src/models/vit_scratch/models.py`
- `src/models/vit_scratch/utils.py`
- `src/models/vit_scratch/__init__.py`

### Responsibilities

#### `models.py`

- model registry
- architecture specs
- patch embedding
- multi-head self-attention
- MLP block
- transformer encoder block
- full model classes
- classifier-head access
- trainable-parameter grouping

#### `utils.py`

- reuse generic safe project helpers where appropriate
- add scratch-ViT-specific fit/resume helpers
- support epoch-level checkpoint saving for real resume behavior

#### `__init__.py`

- stable public import surface for notebooks

---

## 6. Planned Artifact Layout

Scratch ViT artifacts should be stored under:

- `models/vit_scratch/customvit_v1/`
- `models/vit_scratch/customvit_v2/`

Each completed run should save:

- `checkpoint.pt`
- `config.json`
- `metrics.json`
- `loss_curve.png`
- `accuracy_curve.png`
- optional `exported.onnx`

Metrics report copies should also be written to:

- `reports/metrics/`

---

## 7. Planned Models

### 7.1 `customvit_v1`

Purpose:

- smallest educational Vision Transformer baseline
- direct implementation of the core encoder-only ViT idea

Planned architecture:

- image size: `224`
- patch size: `16`
- embedding dimension: `192`
- depth: `6`
- attention heads: `3`
- MLP ratio: `4.0`
- dropout: `0.1`

Expected role:

- teaching model
- first proof that scratch transformer training works end-to-end in the project

### 7.2 `customvit_v2`

Purpose:

- stronger scratch ViT baseline
- same conceptual design as `v1` with more capacity

Planned architecture:

- image size: `224`
- patch size: `16`
- embedding dimension: `256`
- depth: `8`
- attention heads: `8`
- MLP ratio: `4.0`
- dropout: `0.1`

Expected role:

- more competitive scratch transformer benchmark
- direct “v2” counterpart to the `CustomCNN v2` idea used earlier in the project

---

## 8. Design Philosophy

The scratch ViT family should follow the same architectural design philosophy used by the scratch CNN family.

### What this means

The models should **not** be implemented as one giant `nn.Sequential`.

Instead:

- the top-level model should be a custom `nn.Module`
- linear subpaths like the feed-forward MLP can use `nn.Sequential`
- the encoder block should be an explicit module because it needs residual logic
- the encoder stack should use `nn.ModuleList`

This keeps the design readable and faithful to transformer structure.

---

## 9. Core Modules To Implement

The following internal building blocks are required.

### `PatchEmbedding`

Responsibilities:

- convert `[B, C, H, W]` images into patch tokens
- use a strided convolution for patch projection
- enforce the expected runtime image size

### `MultiHeadSelfAttention`

Responsibilities:

- build `qkv` projections
- split heads
- scale attention scores
- apply softmax
- project the merged output back to embedding space

### `FeedForward`

Responsibilities:

- hidden expansion by `mlp_ratio`
- `GELU`
- dropout
- projection back to the embedding dimension

### `TransformerEncoderBlock`

Responsibilities:

- LayerNorm before attention
- residual connection
- LayerNorm before MLP
- residual connection

### `VisionTransformerScratch`

Responsibilities:

- patch embedding
- class token
- positional embedding
- encoder stack
- final normalization
- classification head on the class token

---

## 10. Initialization Strategy

The scratch ViT family should use transformer-appropriate initialization rather than CNN-style Kaiming initialization everywhere.

### Planned rules

- linear layers: truncated normal
- class token: truncated normal
- positional embeddings: truncated normal
- LayerNorm: weight ones, bias zeros
- patch embedding convolution: Xavier-style initialization

This keeps the models closer to standard ViT training conventions.

---

## 11. Training Contract

The scratch ViT family should reuse the same overall project contract as the `40_` and `50_` families:

- same split: `split_v1`
- same transform base config: `transforms_v1.yaml`
- same project-root detection
- same artifact naming conventions
- same metrics copy conventions
- same MLflow experiment name

### Training style

Unlike pretrained transfer-learning families, the scratch ViT family is trained end-to-end from initialization.

So the default training stage is:

- `full_train`

No pretrained-head warmup stage is required.

---

## 12. Resume And Rerun Safety

This phase should preserve the safety expectations established by the mature notebook families.

### Required behavior

- deterministic experiment signatures
- completed-run reuse for identical experiments
- resume support for interrupted runs
- atomic config writes
- atomic metrics writes
- atomic checkpoint writes

### Additional requirement for scratch ViTs

Because these notebooks use a single full-training stage, they should save training-state checkpoints at the **epoch level**, not only after a whole stage finishes.

This allows genuine resume behavior rather than only stage-boundary recovery.

---

## 13. Data And Transform Handling

The scratch ViT family should continue to reuse:

- `src/data/dataset_loader.py`
- `src/data/transforms.py`
- `configs/transforms_v1.yaml`

Default runtime sizes for this phase:

- input crop: `224`
- eval resize: `256`

The notebook should still apply these through the runtime override mechanism so the structure stays aligned with later model families.

---

## 14. Planned Hyperparameters

### `customvit_v1`

- batch size: `64`
- epochs: `40`
- optimizer: `AdamW`
- learning rate: `3e-4`
- weight decay: `1e-4`
- grad clip: `1.0`

### `customvit_v2`

- batch size: `32`
- epochs: `50`
- optimizer: `AdamW`
- learning rate: `2e-4`
- weight decay: `1e-4`
- grad clip: `1.0`

### Shared scheduler

- `ReduceLROnPlateau`
- factor: `0.3`
- patience: `3`

These values are starting defaults and may be adjusted after the first real runs.

---

## 15. MLflow And Reporting

Each scratch ViT notebook should log:

- model name
- split ID
- transform IDs
- batch size
- epoch count
- learning rate
- weight decay
- device
- AMP status
- patch size
- embedding dimension
- depth
- number of heads
- MLP ratio
- experiment signature

Metrics to log:

- best validation metrics
- final test metrics
- latency
- throughput
- parameter count
- trainable parameter count
- model size

Artifacts to log:

- config
- metrics
- checkpoint
- loss curve
- accuracy curve
- ONNX file if export succeeds

---

## 16. ONNX Policy

ONNX export should follow the same project-wide rule:

- attempt automatically
- do not fail the notebook if export fails
- record status clearly in `metrics.json`

Since the scratch ViT models are plain PyTorch modules without exotic external backends, this phase has a better chance of successful ONNX export than some later future families.

---

## 17. Completion Criteria

Phase 6 is considered implemented when:

- `src/models/vit_scratch/` exists
- `60_00_overview.ipynb` exists
- `60_01_customvit_v1.ipynb` exists
- `60_02_customvit_v2.ipynb` exists
- the notebooks are rerun-safe
- interrupted runs can resume from saved training state
- artifacts follow the established project conventions
- the implementation runs on both Windows and Linux without path-specific assumptions

---

# End of Phase 6 Plan
