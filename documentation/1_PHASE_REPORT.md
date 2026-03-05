# Phase 1 Completion Report  
**Project:** AnimalClassification  
**Phase:** Phase 1 — Dataset Infrastructure & Experiment Foundation  
**Status:** Completed (with one optional validation step omitted)

---

# 1. Phase 1 Objective

The objective of Phase 1 was to build the **data infrastructure and experiment foundation** required for all future model training and benchmarking.

This phase focused on establishing:

- a **clean unified dataset**
- a **deterministic train/validation/test split**
- a **configurable transformation pipeline**
- a **standard dataset loading interface**
- **experiment tracking initialization**
- **dataset validation and inspection notebooks**

No models were trained during this phase.

The main goal was to ensure that **all future experiments operate on the exact same dataset structure and preprocessing pipeline**, enabling reproducibility and fair model comparison.

---

# 2. Dataset Status

The dataset has been successfully validated and prepared.

Location:
```

data/prepared/

```

Structure:
```

data/prepared/  
cats/  
dogs/  
wildlife/

```

Dataset statistics:

| Class | Images |
|------|------|
| Cats | 23,693 |
| Dogs | 22,894 |
| Wildlife | 16,072 |

Total images:
```

62,659 images

```

The dataset has already undergone earlier cleaning procedures including:

- dataset merging
- duplicate removal
- dataset preparation

The dataset was verified in **Notebook 00** through sampling and image visualization.

---

# 3. Deterministic Dataset Splits

A deterministic stratified split was generated using the following configuration:
```

Train: 80%  
Validation: 10%  
Test: 10%  
Seed: 42

```

This ensures:

- reproducibility
- balanced class distribution across splits
- fair model evaluation

Split artifacts were generated in:
```

data/splits/split\_v1/

```

Files created:
```

train.csv  
val.csv  
test.csv  
classes.json

```

Example CSV format:
```

filepath,label  
data/prepared/cats/img001.jpg,cats  
data/prepared/dogs/img002.jpg,dogs

````

Class mapping:

```json
{
  "cats": 0,
  "dogs": 1,
  "wildlife": 2
}
````

Split sizes:

| Split | Samples |
| --- | --- |
| Train | 50,127 |
| Validation | 6,266 |
| Test | 6,266 |

The class distribution across splits was verified and confirmed to match the original dataset ratios.

* * *

4\. Transformation & Augmentation System
========================================

A configurable transformation pipeline was implemented.

Configuration file:

```
configs/transforms_v1.yaml
```

This file defines preprocessing and augmentation operations applied during training and evaluation.

### Training Transformations

Used only during model training.

Operations:

*   RandomResizedCrop (224)
*   RandomHorizontalFlip
*   RandomRotation
*   ColorJitter
*   ToTensor
*   Normalize (ImageNet statistics)

These augmentations improve model robustness by introducing variation during training.

### Evaluation Transformations

Used during validation and test evaluation.

Operations:

*   Resize (256)
*   CenterCrop (224)
*   ToTensor
*   Normalize

Evaluation transforms are deterministic to ensure consistent benchmarking.

Important design decision:

**Augmented images are not stored on disk.**  
All transformations are applied **dynamically during dataset loading**.

* * *

5\. Dataset Loader Implementation
=================================

A reusable PyTorch dataset loader was implemented.

Location:

```
src/data/dataset_loader.py
```

Class implemented:

```
ImageDataset
```

Responsibilities:

*   read dataset split CSV files
*   load images from disk
*   convert images to RGB
*   apply configured transforms
*   convert labels to numeric indices
*   return `(image_tensor, label)` pairs

Expected output tensor shape:

```
(3, 224, 224)
```

This loader serves as the **standard interface for all future model training pipelines**.

* * *

6\. Transformation System Implementation
========================================

Transformation utilities were implemented in:

```
src/data/transforms.py
```

Key functions:

```
load_transforms_config()
get_train_transforms()
get_eval_transforms()
```

Responsibilities:

*   read transformation configuration from YAML
*   construct PyTorch transform pipelines
*   separate train and evaluation pipelines

This allows augmentation strategies to be modified **without editing Python code**, only by updating configuration files.

* * *

7\. Dataset Split Generator
===========================

Dataset split utilities were implemented in:

```
src/data/split_generator.py
```

Key responsibilities:

*   scan dataset directory
*   collect filepaths and labels
*   generate stratified dataset splits
*   enforce deterministic behavior using seed
*   save split manifests
*   validate class distribution

This module ensures dataset splits are reproducible and consistent across experiments.

* * *

8\. Notebook Infrastructure
===========================

Three validation notebooks were created.

Notebook 00 — Project Setup
---------------------------

```
notebooks/00_project_setup.ipynb
```

Purpose:

*   verify dataset directories
*   confirm class folders
*   count dataset images
*   visualize random images
*   ensure report directories exist

This notebook confirms that the dataset is accessible and structurally valid.

* * *

Notebook 01 — Data Preparation and Splits
-----------------------------------------

```
notebooks/01_data_prep_and_splits.ipynb
```

Purpose:

*   scan dataset
*   compute dataset statistics
*   generate deterministic dataset splits
*   validate class distribution
*   produce dataset summary reports
*   log dataset metadata to MLflow

Generated outputs:

```
reports/metrics/data_summary_split_v1.json
reports/figures/class_distribution.png
```

* * *

Notebook 02 — Transformations and Dataset Loader
------------------------------------------------

```
notebooks/02_transforms_and_augmentation.ipynb
```

Purpose:

*   load transform configuration
*   construct transform pipelines
*   test dataset loader
*   verify tensor shapes
*   visualize augmented images
*   test PyTorch DataLoader batching

Generated output:

```
reports/figures/sample_augmented_images.png
```

This notebook confirms that the **data pipeline operates correctly**.

* * *

9\. Experiment Tracking Initialization
======================================

MLflow experiment tracking was initialized.

Configuration:

```
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("AnimalClassification")
```

Dataset metadata logged includes:

*   dataset size
*   split sizes
*   class distributions
*   split configuration parameters

MLflow tracking directory:

```
mlruns/
```

This prepares the project for future experiment tracking during model training.

* * *

10\. Generated Outputs
======================

The following artifacts were produced during Phase 1:

### Dataset Splits

```
data/splits/split_v1/
    train.csv
    val.csv
    test.csv
    classes.json
```

### Reports

```
reports/metrics/data_summary_split_v1.json
reports/figures/class_distribution.png
reports/figures/sample_augmented_images.png
```

### Configuration

```
configs/transforms_v1.yaml
```

### Source Code

```
src/data/split_generator.py
src/data/transforms.py
src/data/dataset_loader.py
```

### Notebooks

```
notebooks/00_project_setup.ipynb
notebooks/01_data_prep_and_splits.ipynb
notebooks/02_transforms_and_augmentation.ipynb
```

* * *

11\. Remaining Optional Validation Step
=======================================

One optional validation step from the original plan was not performed:

**Full dataset image loading validation**

This step would iterate through all images and attempt to load them to detect corrupted files.

This step was considered optional because:

*   dataset cleaning had already been performed earlier
*   sample image loading succeeded
*   dataset loader successfully processed batches through DataLoader

Therefore the dataset is considered sufficiently validated for Phase 2.

* * *

12\. Current State of the Project
=================================

At the end of Phase 1 the project now has:

*   a **clean unified dataset**
*   **deterministic dataset splits**
*   a **reusable dataset loader**
*   a **configurable transformation pipeline**
*   **dataset validation notebooks**
*   **experiment tracking infrastructure**
*   **dataset statistics and reports**

Most importantly, the project now has a **stable data pipeline that all future models will use**.

This ensures:

*   reproducibility
*   fair benchmarking
*   easier experimentation

* * *

13\. Next Phase
===============

Phase 2 will build on this foundation and implement:

*   feature extraction
*   classical machine learning baselines
*   initial model benchmarking

The data infrastructure created in Phase 1 will serve as the **core pipeline for all future experiments**.


