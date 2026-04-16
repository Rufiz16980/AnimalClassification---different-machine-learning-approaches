# Frozen Future Plan: Non-CNN / Non-ViT Model Families

**Project:** AnimalClassification  
**Status:** Frozen planning document only  
**Implementation status:** Not started  
**Purpose:** Preserve the agreed future roadmap for model families that sit outside the original CNN / ViT roadmap  
**Important exclusion:** This document does **not** include the reserved `60_` custom-ViT-from-scratch family

---

## 1. Why This Document Exists

The project has already established the main benchmark layers for:

- classical ML on engineered image features
- pretrained fixed-feature backbones plus classical ML
- CNNs from scratch
- pretrained CNN transfer learning
- pretrained ViT-family transfer learning

The next ideas discussed by the user are **outside the scope of the original project plan** and should therefore be preserved in a separate frozen roadmap rather than merged into the active numbered phase plans.

This document is that frozen roadmap.

It should be treated as:

- a future expansion reference
- a design snapshot
- a scope boundary marker

It should **not** be interpreted as a request to implement these families now.

---

## 2. High-Level Future Families

The agreed future non-CNN / non-ViT expansion families are:

### `70_` All-MLP Vision Models

Purpose:

- benchmark architectures that remove both convolution and self-attention
- add a clearly different inductive-bias family after CNNs and ViTs

### `80_` MetaFormer Models

Purpose:

- benchmark token-mixer architectures that are neither classical CNNs nor standard self-attention ViTs
- cover the MetaFormer design space with meaningful, non-redundant representatives

### `90_` State-Space Vision Models

Purpose:

- benchmark modern state-space / Mamba-style image architectures
- add a family that is neither CNN nor standard transformer

---

## 3. Future Range Map

### Reserved Notebook Ranges

- `70_00` to `70_09`: All-MLP family
- `80_00` to `80_09`: MetaFormer family
- `90_00` to `90_09`: State-space family

### Reserved Source Packages

- `src/models/all_mlp/`
- `src/models/metaformer/`
- `src/models/state_space_vision/`

### Reserved Artifact Roots

- `models/all_mlp/`
- `models/metaformer/`
- `models/state_space_vision/`

### Reserved Notebook Folders

- `notebooks/70_all_mlp/`
- `notebooks/80_metaformer/`
- `notebooks/90_state_space/`

---

## 4. Shared Design Principles

All future non-CNN / non-ViT families should follow the same project architecture that was already proven in the `40_` and `50_` families.

### They must reuse the existing project structure

- same dataset split contract: `split_v1`
- same path/root detection pattern
- same report-metrics copy pattern
- same artifact layout style
- same MLflow experiment namespace unless intentionally separated later
- same latency / throughput / size / parameter-count reporting

### They must preserve notebook safety

- safe under repeated `Run All`
- safe under interruption
- safe under rerunning the same experiment
- no duplicate completed runs for identical experiment signatures
- no path assumptions tied to only Windows or only Linux

### They must remain PyTorch-based

No family in this roadmap requires abandoning PyTorch.

The expected backends are:

- PyTorch + `timm` for `70_` and `80_`
- PyTorch + `mambavision` for `90_`

### They should reuse the existing generic training pipeline concepts

Even though they require new family packages, they should still reuse the already-established ideas for:

- experiment signatures
- run resolution
- checkpointing
- metrics/history storage
- benchmarking
- MLflow logging
- ONNX attempt policy

---

## 5. Dependency Strategy

These future families are planned around external PyTorch model ecosystems rather than only `torchvision`.

### `70_` All-MLP and `80_` MetaFormer

Primary dependency:

- `timm`

Why:

- these families are broadly available and actively maintained there
- model-specific data configs can be resolved directly from the backend
- pretrained weights are readily accessible

### `90_` State-Space

Primary dependency:

- `mambavision`

Why:

- it provides an official PyTorch package
- it exposes pretrained models and feature outputs
- it is more realistic and portable than attempting to start with more fragile research-code options such as Vim

### Important dependency note

These packages are **not part of the current active implementation contract**. They should be added only when the future families are actually started.

At future implementation time, dependency validation must happen first on the strong execution device and then on the local migration device.

---

## 6. Shared Helper Philosophy For Future Families

The project should avoid inventing a completely new training system for each future family.

### Minimum helper structure expected for every new family

Each new family package should include:

- `models.py`
- `utils.py`
- `deps.py`
- `__init__.py`

### Planned responsibilities

#### `models.py`

- model registry
- exact pretrained backend identifiers
- backend model construction
- classifier-head replacement or reset to the project’s 3-class output
- recommended image size / resize metadata
- partial fine-tuning rules
- trainable-parameter group construction

#### `utils.py`

- re-export or thin-wrap the generic run-management / evaluation / metrics helpers already used in the `40_` and `50_` families
- keep notebook imports short and consistent

#### `deps.py`

- explicit import guards for optional dependencies
- user-friendly error messages if `timm` or `mambavision` is missing
- backend capability checks

#### `__init__.py`

- stable public import surface for notebooks

### General rule

The notebooks should stay thin.

The family packages should absorb:

- backend-specific model creation
- backend-specific head replacement
- backend-specific stage-unfreeze logic
- backend-specific data-config extraction

This keeps future notebooks aligned with the project’s established architecture instead of turning into backend-specific scripts.

---

## 7. Shared Transform Strategy

The project should continue to reuse:

- `configs/transforms_v1.yaml`
- `src/data/transforms.py`

But future external-backbone families must **not** assume that the current `224 / 256` defaults are always correct.

### Planned rule

Runtime transform overrides must be driven by backend model metadata.

For future implementation:

- `timm` families should derive input size and normalization from model data config
- `mambavision` families should derive input size, normalization, and crop behavior from model config

This should then be mapped into the same runtime transform override pattern already established in the current project.

---

## 8. Family 70: All-MLP Vision Models

### Family Purpose

This family adds image classifiers that do **not** rely on:

- convolution
- standard self-attention token mixing

It is the cleanest “what else exists beyond CNNs and transformers?” family to add after the current benchmark layers.

### Planned Notebook Folder

- `notebooks/70_all_mlp/`

### Planned Notebooks

- `70_00_overview.ipynb`
- `70_01_mlp_mixer.ipynb`
- `70_02_resmlp.ipynb`
- `70_03_gmlp.ipynb`

### Planned Models

#### `MLP-Mixer`

Planned backend identifier:

- `mixer_b16_224.goog_in21k_ft_in1k`

Reason for inclusion:

- canonical all-MLP vision architecture
- strongest educational representative of the family

#### `ResMLP`

Planned backend identifier:

- `resmlp_24_224.fb_in1k`

Reason for inclusion:

- residual-style MLP architecture
- gives diversity inside the all-MLP family without just scaling MLP-Mixer

#### `gMLP`

Planned backend identifier:

- `gmlp_s16_224.ra3_in1k`

Reason for inclusion:

- gating-based MLP token-mixing variant
- good third representative that remains computationally manageable

### Planned Source Package

- `src/models/all_mlp/models.py`
- `src/models/all_mlp/utils.py`
- `src/models/all_mlp/deps.py`
- `src/models/all_mlp/__init__.py`

### Planned Artifact Layout

- `models/all_mlp/mlp_mixer/`
- `models/all_mlp/resmlp/`
- `models/all_mlp/gmlp/`

### Planned notebook-to-model mapping

- `70_01_mlp_mixer.ipynb` -> `MLP-Mixer`
- `70_02_resmlp.ipynb` -> `ResMLP`
- `70_03_gmlp.ipynb` -> `gMLP`

### Planned helper behavior

`models.py` for this family should:

- call into `timm.create_model(...)`
- reset the classifier to `num_classes=3`
- expose recommended image size metadata
- define which last blocks count as “partial fine-tune” units

### Difficulty estimate

- low to medium

### Risk level

- low

### Why this family is realistic

- pure PyTorch workflow
- good support through `timm`
- image sizes stay manageable
- all three selected models are meaningful but not absurdly heavy

---

## 9. Family 80: MetaFormer Models

### Family Purpose

This family adds architectures built around the MetaFormer design idea, where the token mixer can vary while the broader scaffold stays similar.

This family is useful because it sits between:

- plain all-MLP models
- standard transformer attention models

and gives a different kind of token-mixing benchmark.

### Planned Notebook Folder

- `notebooks/80_metaformer/`

### Planned Notebooks

- `80_00_overview.ipynb`
- `80_01_poolformer.ipynb`
- `80_02_convformer.ipynb`
- `80_03_caformer.ipynb`

### Planned Models

#### `PoolFormer`

Planned backend identifier:

- `poolformer_s12.sail_in1k`

Reason for inclusion:

- canonical simplest MetaFormer representative
- attention-free token mixing using pooling

#### `ConvFormer`

Planned backend identifier:

- `convformer_s18.sail_in1k`

Reason for inclusion:

- MetaFormer variant using convolutional token mixing
- more meaningful than repeating only PoolFormer scales

#### `CAFormer`

Planned backend identifier:

- `caformer_s18.sail_in1k`

Reason for inclusion:

- stronger and more hybrid MetaFormer baseline
- gives the family a higher-capacity representative without jumping to very large models

### Planned Source Package

- `src/models/metaformer/models.py`
- `src/models/metaformer/utils.py`
- `src/models/metaformer/deps.py`
- `src/models/metaformer/__init__.py`

### Planned Artifact Layout

- `models/metaformer/poolformer/`
- `models/metaformer/convformer/`
- `models/metaformer/caformer/`

### Planned notebook-to-model mapping

- `80_01_poolformer.ipynb` -> `PoolFormer`
- `80_02_convformer.ipynb` -> `ConvFormer`
- `80_03_caformer.ipynb` -> `CAFormer`

### Planned helper behavior

`models.py` for this family should:

- instantiate models from `timm`
- resolve model-specific data config
- adapt classification head to 3 classes
- define partial fine-tuning as the last stage or last MetaFormer blocks

### Difficulty estimate

- medium

### Risk level

- low to medium

### Why this family is realistic

- pure PyTorch workflow
- good `timm` support
- all three planned models fit the “different but still manageable” criterion
- more meaningful than adding older historical CNN families just for completeness

---

## 10. Family 90: State-Space Vision Models

### Family Purpose

This family adds state-space / Mamba-style vision models.

It is the most distinct future family after CNNs and transformers, but it is also the riskiest from a dependency and portability perspective.

### Planned Notebook Folder

- `notebooks/90_state_space/`

### Planned Notebooks

- `90_00_overview.ipynb`
- `90_01_mambavision_t.ipynb`
- `90_02_mambavision_t2.ipynb`
- `90_03_mambavision_s.ipynb`

### Planned Models

#### `MambaVision-T`

Planned backend identifier:

- `mamba_vision_T`

Reason for inclusion:

- smallest practical official MambaVision entry
- best starting point for validating the family

#### `MambaVision-T2`

Planned backend identifier:

- `mamba_vision_T2`

Reason for inclusion:

- stronger small-family variant
- useful for understanding whether the family scales meaningfully inside project constraints

#### `MambaVision-S`

Planned backend identifier:

- `mamba_vision_S`

Reason for inclusion:

- medium-capacity representative still within realistic project bounds
- gives a stronger benchmark point without jumping to much larger B/L variants

### Planned Source Package

- `src/models/state_space_vision/models.py`
- `src/models/state_space_vision/utils.py`
- `src/models/state_space_vision/deps.py`
- `src/models/state_space_vision/__init__.py`

### Planned Artifact Layout

- `models/state_space_vision/mambavision_t/`
- `models/state_space_vision/mambavision_t2/`
- `models/state_space_vision/mambavision_s/`

### Planned notebook-to-model mapping

- `90_01_mambavision_t.ipynb` -> `MambaVision-T`
- `90_02_mambavision_t2.ipynb` -> `MambaVision-T2`
- `90_03_mambavision_s.ipynb` -> `MambaVision-S`

### Planned helper behavior

`models.py` for this family should:

- guard imports for `mambavision`
- construct official pretrained models
- normalize backend output so the project training loop always receives logits in a predictable shape
- extract mean/std/input-size config from the backend
- define partial fine-tuning over the final state-space / hybrid stages

### Difficulty estimate

- medium to high

### Risk level

- highest of the three future families

### Important limitations

- dependency setup is less standard than `timm`
- Windows/Linux portability must be validated explicitly
- ONNX support is uncertain and should be treated as optional from the start
- licensing must be reviewed again at future implementation time before redistribution decisions are made

### Intentional exclusions from `90_`

The frozen plan intentionally does **not** include Vim or other more fragile research-code state-space options.

Reason:

- higher dependency complexity
- weaker portability confidence
- larger risk of breaking the clean project architecture

The frozen plan therefore treats **MambaVision** as the first viable state-space family for this project.

---

## 11. Planned Overview Notebook Behavior

Each of the three future families should start with an overview notebook:

- `70_00_overview.ipynb`
- `80_00_overview.ipynb`
- `90_00_overview.ipynb`

Each overview notebook should:

- detect project root safely
- verify dataset files
- verify backend dependency availability
- enumerate supported models
- record recommended image sizes
- write a family summary JSON to `reports/metrics`
- optionally create a lightweight MLflow validation run

They should **not** train a model.

---

## 12. Planned Training Notebook Contract

Every future training notebook should preserve the same proven five-cell layout already used in the mature transfer-learning notebooks.

### Required shape

1. markdown introduction
2. imports
3. project-root detection and package imports
4. constants, paths, datasets, loaders, signature setup, run resolution
5. train / evaluate / save / benchmark / MLflow logic

### Required behavior

- rerun-safe
- no duplicate completed runs
- safe resume or restart behavior
- no hardcoded machine-specific paths
- metrics saved both in the run folder and in `reports/metrics`

---

## 13. Planned Artifact Contract

Every future run directory should aim to save:

- `checkpoint.pt`
- `config.json`
- `metrics.json`
- `loss_curve.png`
- `accuracy_curve.png`
- optional `exported.onnx`

And each run should also save a report copy to:

- `reports/metrics/`

The metrics payload should continue to include:

- validation best metrics
- final test metrics
- parameter count
- model size
- latency
- throughput
- device info

---

## 14. Planned MLflow Contract

At future implementation time, all three families should keep the current project MLflow style.

The minimum planned logging set is:

- family stage name
- model name
- backend model identifier
- split ID
- transform IDs
- input image size
- resize size
- batch size
- learning rates
- weight decay
- seed
- device
- AMP status
- experiment signature
- final validation/test metrics
- latency / throughput / size metrics

Artifacts to log:

- config
- metrics
- checkpoint
- curves
- ONNX file if available

---

## 15. Planned ONNX Policy

The future families should keep the same general ONNX philosophy already established elsewhere in the project.

### Rule

ONNX export is:

- attempted automatically
- non-fatal on failure
- recorded in `metrics.json`

### Special note for `90_`

For state-space models, ONNX export should be considered especially uncertain and should not block the family.

---

## 16. Planned Later Implementation Order

When the time comes, the safest later implementation order is:

1. `70_` All-MLP
2. `80_` MetaFormer
3. `90_` State-space

Reason:

- `70_` has the cleanest support story
- `80_` still fits the same `timm`-based integration pattern
- `90_` has the highest dependency and backend risk

---

## 17. Planned Future Support Files

When implementation eventually begins, the following extra support files may be needed.

### Future dependency updates

- add `timm` before implementing `70_` and `80_`
- add `mambavision` before implementing `90_`

### Possible future documentation or audit helpers

- `documentation/70_PHASE_PLAN.md` if the family becomes active
- `documentation/80_PHASE_PLAN.md` if the family becomes active
- `documentation/90_PHASE_PLAN.md` if the family becomes active

These are intentionally **not** created now because the current goal is only to freeze the roadmap, not promote it into the active numbered implementation plan.

---

## 18. Deferred Or Rejected Candidates

These ideas were considered but are not part of the frozen plan:

### Not included here

- custom ViT from scratch
  - excluded because `60_` is separately reserved for that later direction

- additional classical CNN families
  - excluded because the project already covers modern pretrained CNN families very well

- Vim and other harder research-code state-space models
  - excluded because the dependency and portability risk is too high for the clean project workflow

- object detection families such as YOLO
  - excluded because they change the task from image classification to detection

---

## 19. Freeze Statement

This document freezes the future roadmap for three additional non-CNN / non-ViT model families:

- `70_` All-MLP
- `80_` MetaFormer
- `90_` State-space

It records:

- the planned notebook ranges
- the planned models
- the planned source packages
- the planned artifact layout
- the helper responsibilities
- the dependency strategy

It does **not** authorize implementation yet.

Its purpose is to let the team return later with a clear, already-agreed map.

---

## 20. Reference Sources

The future model choices in this frozen plan were informed by the following official or primary model sources:

- [MLP-Mixer paper / model family context](https://research.google/pubs/mlp-mixer-an-all-mlp-architecture-for-vision/)
- [timm MLP-Mixer model card](https://huggingface.co/timm/mixer_b16_224.goog_in21k_ft_in1k)
- [timm ResMLP model card](https://huggingface.co/timm/resmlp_24_224.fb_in1k)
- [timm gMLP model card](https://huggingface.co/timm/gmlp_s16_224.ra3_in1k)
- [timm PoolFormer model card](https://huggingface.co/timm/poolformer_s12.sail_in1k)
- [timm ConvFormer model card](https://huggingface.co/timm/convformer_s18.sail_in1k)
- [timm CAFormer model card](https://huggingface.co/timm/caformer_s18.sail_in1k)
- [MambaVision official repository](https://github.com/NVlabs/MambaVision)
- [MambaVision Hugging Face collection](https://huggingface.co/collections/nvidia/mambavision-66943871a6b36c9e78b327d3)

---

# End of Frozen Future Non-CNN / Non-ViT Plan
