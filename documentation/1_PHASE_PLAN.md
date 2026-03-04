# Phase 1 Implementation Plan  
**Project:** AnimalClassification  
**Phase Duration:** 3–4 days  
**Objective:** Build the **dataset foundation and experiment infrastructure** required for all later model training.

This document is written so that **an agent with zero prior knowledge of the project** can implement Phase 1 from scratch.

---

# 1. Phase 1 Objectives

Phase 1 establishes the **data pipeline and experiment framework**.

At the end of Phase 1 the system must provide:

1. A **clean unified dataset**
2. A **fixed train/validation/test split**
3. A **reusable transformation & augmentation pipeline**
4. A **standard dataset loading interface**
5. A **consistent configuration system**
6. Initial **MLflow experiment tracking**
7. A **dataset validation and inspection notebook**

No models are trained in this phase.

---

# 2. Expected Deliverables

At completion of Phase 1 the repository must contain:

### Dataset
```

data/prepared/  
cats/  
dogs/  
wildlife/

```

### Dataset Splits
```

data/splits/split\_v1/  
train.csv  
val.csv  
test.csv  
classes.json

```

### Transformation Configuration
```

configs/transforms\_v1.yaml

```

### Python Utilities
```

src/data/  
dataset\_loader.py  
transforms.py  
split\_generator.py

```

### Notebooks
```

notebooks/  
00\_project\_setup.ipynb  
01\_data\_prep\_and\_splits.ipynb  
02\_transforms\_and\_augmentation.ipynb

```

### Validation Outputs
```

reports/metrics/data\_summary\_split\_v1.json  
reports/figures/class\_distribution.png  
reports/figures/sample\_images.png

```

---

# 3. Dataset Description

The dataset is a merged collection of multiple public datasets.

All images are categorized into **three classes**:
```

cats  
dogs  
wildlife

```

The cleaned dataset is stored in:
```

data/prepared/

```

Example structure:
```

data/prepared/  
cats/  
img\_001.jpg  
img\_002.jpg  
dogs/  
img\_003.jpg  
wildlife/  
img\_004.jpg

```

Images are already:
- deduplicated
- validated
- merged into a single dataset

Approximate dataset size:

| Class | Images |
|------|------|
| Cats | ~23k |
| Dogs | ~23k |
| Wildlife | ~16k |

Total images ≈ **62k**

---

# 4. Dataset Split Strategy

To ensure **fair benchmarking**, dataset splits must be created **once and reused by all models**.

Split ratios:
```

Train: 80%  
Validation: 10%  
Test: 10%

```

Requirements:

- stratified split (class distribution preserved)
- deterministic (fixed random seed)
- stored as CSV manifests

Example CSV format:
```

filepath,label  
data/prepared/cats/img001.jpg,cats  
data/prepared/dogs/img002.jpg,dogs

```

Output location:
```

data/splits/split\_v1/

```

Files:
```

train.csv  
val.csv  
test.csv  
classes.json

```

`classes.json` must contain:
```

{  
"cats": 0,  
"dogs": 1,  
"wildlife": 2  
}

```

---

# 5. Transformation Pipeline

A reusable transformation system must be implemented.

Transforms are defined in a configuration file:
```

configs/transforms\_v1.yaml

````

Example structure:

```yaml
image_size: 224

train_transforms:
  - random_horizontal_flip
  - random_rotation
  - color_jitter

eval_transforms:
  - resize
  - normalize
````

Two pipelines must exist:

### Training Transformations

Applied during training only.

Typical operations:

*   random horizontal flip
*   random rotation
*   random crop
*   color jitter

### Evaluation Transformations

Used for validation and test.

Typical operations:

*   resize
*   center crop
*   normalization

Augmented images **must not be saved to disk**.

All augmentation is applied **on-the-fly** during training.

* * *

6\. Dataset Loader
==================

A reusable dataset loader must be implemented.

File:

```
src/data/dataset_loader.py
```

Responsibilities:

*   read CSV split files
*   load images from disk
*   apply transforms
*   return `(image_tensor, label)` pairs

The loader must support:

```
train
validation
test
```

Example usage:

```
dataset = ImageDataset(
    split_csv="data/splits/split_v1/train.csv",
    transform=train_transforms
)
```

* * *

7\. Transform Implementation
============================

File:

```
src/data/transforms.py
```

Responsibilities:

*   parse `transforms_v1.yaml`
*   construct PyTorch transform pipelines

Two functions required:

```
get_train_transforms(config)
get_eval_transforms(config)
```

* * *

8\. Split Generation Script
===========================

File:

```
src/data/split_generator.py
```

Responsibilities:

1.  scan dataset directory
2.  collect filepaths and labels
3.  generate stratified splits
4.  save CSV manifests

Random seed must be fixed:

```
seed = 42
```

Libraries allowed:

*   sklearn
*   pandas
*   numpy

* * *

9\. MLflow Initialization
=========================

MLflow must be initialized in the project.

Purpose:

*   experiment tracking
*   parameter logging
*   metrics logging

Setup tasks:

```
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("AnimalClassification")
```

During Phase 1 MLflow logs:

*   dataset size
*   split statistics
*   class distribution

* * *

10\. Notebook Responsibilities
==============================

Notebook 00 — Project Setup
---------------------------

```
notebooks/00_project_setup.ipynb
```

Tasks:

*   verify dataset directory exists
*   verify class folders
*   print dataset summary
*   display random images

Outputs:

*   sanity validation

* * *

Notebook 01 — Data Preparation and Splits
-----------------------------------------

```
notebooks/01_data_prep_and_splits.ipynb
```

Tasks:

1.  scan dataset
2.  compute class distribution
3.  generate stratified splits
4.  save CSV manifests
5.  visualize split distribution

Outputs:

```
reports/metrics/data_summary_split_v1.json
reports/figures/class_distribution.png
```

* * *

Notebook 02 — Transformations and Augmentation
----------------------------------------------

```
notebooks/02_transforms_and_augmentation.ipynb
```

Tasks:

*   implement transform pipelines
*   visualize augmentation effects
*   verify tensor shapes
*   test dataset loader

Outputs:

```
reports/figures/sample_augmented_images.png
```

* * *

11\. Dataset Validation
=======================

Before Phase 1 is complete the following checks must pass:

### Class Distribution

Train/validation/test splits must preserve class ratios.

### Image Loading

All images must load without errors.

### Transform Output

Images must convert to tensors with expected shape.

Expected tensor shape:

```
(3, 224, 224)
```

* * *

12\. Acceptance Criteria
========================

Phase 1 is considered complete when:

1.  Dataset splits exist
2.  Dataset loader works
3.  Transform pipeline works
4.  Notebooks run without errors
5.  MLflow logs dataset information
6.  Visualizations are generated
7.  Dataset statistics are saved

* * *

13\. Estimated Timeline
=======================

### Day 1

*   verify dataset
*   implement split generator
*   create dataset splits

### Day 2

*   implement dataset loader
*   implement transform pipeline

### Day 3

*   create notebooks
*   dataset validation

### Day 4 (optional buffer)

*   debugging
*   documentation refinement

* * *

14\. Phase 1 Completion Output
==============================

At the end of Phase 1 the project must have:

*   deterministic dataset splits
*   reusable dataset loader
*   reusable transform system
*   dataset validation notebooks
*   MLflow experiment initialized

This completes the **data infrastructure required for all future model experiments**.

Phase 2 will implement **feature extraction and classical ML baselines**.


