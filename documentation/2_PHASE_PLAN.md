
# Phase 2 Implementation Plan  
**Project:** AnimalClassification  
**Phase:** Phase 2 — Classical Machine Learning Baselines  
**Estimated Duration:** 4–6 days  
**Prerequisite:** Phase 1 completed

---

# 1. Phase 2 Objective

Phase 2 introduces the **first set of benchmark models** using **classical machine learning approaches**.

The purpose of this phase is to establish **baseline performance levels** before implementing deep learning models in later phases.

Two categories of classical machine learning models will be implemented:

1. **Classical ML with handcrafted computer vision features**
2. **Classical ML with deep features extracted from pretrained neural networks**

These baselines provide:

- a performance reference
- computational cost comparison
- insight into dataset difficulty
- validation of the experimental pipeline

At the end of Phase 2 the system must produce **fully trained baseline models and recorded benchmark metrics**.

---

# 2. Phase 2 Expected Outputs

At completion of Phase 2 the repository must contain:

## Trained Models
```

models/ml\_basic\_features/  
hog\_svm/  
run\_YYYYMMDD/  
model.pkl  
config.json  
metrics.json

```
lbp_svm/
    run_YYYYMMDD/
        model.pkl
        config.json
        metrics.json

colorhist_lr/
    run_YYYYMMDD/
        model.pkl
        config.json
        metrics.json
```

models/ml\_deep\_features/  
resnet50\_embeddings/  
run\_YYYYMMDD/  
classifier.pkl  
config.json  
metrics.json

```

---

## Cached Embeddings
```

data/processed/embeddings/  
split\_v1/  
encoder\_resnet50/

```
        train.npy
        val.npy
        test.npy

        labels_train.npy
        labels_val.npy
        labels_test.npy
```

```

---

## Training Notebooks
```

notebooks/

10\_ml\_basic\_features/  
10\_00\_overview.ipynb  
10\_01\_hog\_svm.ipynb  
10\_02\_lbp\_svm.ipynb  
10\_03\_colorhist\_lr.ipynb

20\_ml\_deep\_features\_fixed\_encoder/  
20\_00\_overview.ipynb  
20\_01\_extract\_embeddings\_resnet50.ipynb  
20\_02\_lr\_on\_embeddings.ipynb  
20\_03\_svm\_on\_embeddings.ipynb

```

---

## Metrics Output
```

reports/metrics/

hog\_svm\_metrics.json  
lbp\_svm\_metrics.json  
colorhist\_lr\_metrics.json  
resnet50\_lr\_metrics.json  
resnet50\_svm\_metrics.json

```

---

# 3. Required Inputs

Phase 2 uses the infrastructure created in Phase 1.

Required files:
```

data/prepared/  
data/splits/split\_v1/  
configs/transforms\_v1.yaml  
src/data/dataset\_loader.py  
src/data/transforms.py

```

The dataset loader and transform pipeline must be reused.

No dataset modifications are allowed.

---

# 4. Classical ML — Handcrafted Features

This section implements traditional computer vision feature extraction methods.

Pipeline:
```

image → feature extractor → classifier

```

The following feature extractors must be implemented.

---

# 5. Feature Extractor 1 — HOG

Histogram of Oriented Gradients captures edge structures and gradients.

Implementation requirements:

- convert image to grayscale
- compute gradient orientation histograms
- normalize histograms

Libraries allowed:
```

skimage.feature.hog

```

Typical parameters:
```

orientations = 9  
pixels\_per\_cell = (8,8)  
cells\_per\_block = (2,2)

```

Feature output:
```

1D feature vector

```

Classifier:
```

SVM (RBF kernel)

```

Notebook:
```

notebooks/10\_ml\_basic\_features/10\_01\_hog\_svm.ipynb

```

---

# 6. Feature Extractor 2 — LBP

Local Binary Patterns capture local texture information.

Libraries allowed:
```

skimage.feature.local\_binary\_pattern

```

Parameters:
```

P = 8  
R = 1  
method = 'uniform'

```

Classifier:
```

SVM

```

Notebook:
```

notebooks/10\_ml\_basic\_features/10\_02\_lbp\_svm.ipynb

```

---

# 7. Feature Extractor 3 — Color Histogram

Color histograms capture distribution of colors.

Implementation:

- convert image to HSV
- compute histogram for H, S, V channels
- concatenate histograms

Example bins:
```

H: 32  
S: 32  
V: 32

```

Feature size:
```

96 dimensions

```

Classifier:
```

Logistic Regression

```

Notebook:
```

notebooks/10\_ml\_basic\_features/10\_03\_colorhist\_lr.ipynb

```

---

# 8. Classical ML Using Deep Features

Instead of handcrafted features, images will be encoded using a pretrained neural network.

Pipeline:
```

image → pretrained encoder → embedding → classifier

```

Advantages:

- stronger baseline
- reusable features
- faster training

---

# 9. Encoder Selection

Encoder used in Phase 2:
```

ResNet50 (ImageNet pretrained)

```

The classifier layer must be removed.

Only the **feature extractor** portion is used.

Embedding size:
```

2048 dimensions

```

---

# 10. Embedding Extraction Pipeline

Embeddings must be extracted once and cached.

Notebook:
```

notebooks/20\_ml\_deep\_features\_fixed\_encoder/20\_01\_extract\_embeddings\_resnet50.ipynb

```

Process:
```

load dataset  
apply evaluation transforms  
pass image through encoder  
store resulting feature vector

```

Embeddings saved to:
```

data/processed/embeddings/split\_v1/encoder\_resnet50/

```

Files generated:
```

train.npy  
val.npy  
test.npy  
labels\_train.npy  
labels\_val.npy  
labels\_test.npy

```

This avoids recomputing embeddings for each classifier.

---

# 11. Classifiers on Embeddings

Two classifiers must be trained.

### Logistic Regression

Notebook:
```

20\_02\_lr\_on\_embeddings.ipynb

```

### SVM

Notebook:
```

20\_03\_svm\_on\_embeddings.ipynb

```

These classifiers operate directly on cached embeddings.

---

# 12. Model Serialization

All trained models must be saved.

Format:
```

pickle (.pkl)

```

Example:
```

models/ml\_basic\_features/hog\_svm/run\_2026\_03\_05/model.pkl

```

Config file must include:
```

model\_name  
feature\_type  
classifier  
split\_id  
transform\_id  
training\_parameters

```

---

# 13. Metrics Calculation

Each trained model must compute the following metrics:
```

accuracy  
macro\_f1  
precision  
recall  
confusion\_matrix

```

Libraries allowed:
```

sklearn.metrics

```

Results saved to:
```

metrics.json

```

Example:
```

{  
"accuracy": 0.72,  
"f1\_macro": 0.70,  
"precision": 0.71,  
"recall": 0.69  
}

```

---

# 14. MLflow Logging

Each experiment must log:
```

mlflow.log\_param("model\_name", ...)  
mlflow.log\_param("feature\_type", ...)  
mlflow.log\_param("classifier", ...)  
mlflow.log\_param("split\_id", ...)

```

Metrics logged:
```

mlflow.log\_metric("accuracy", ...)  
mlflow.log\_metric("f1\_macro", ...)

```

Artifacts logged:
```

model.pkl  
metrics.json

```

---

# 15. Benchmark Verification

After all models are trained:

- verify metrics were produced
- confirm models load correctly
- confirm inference works

Quick test:
```

model.predict(sample\_features)

```

---

# 16. Estimated Timeline

### Day 1

Implement:
```

HOG feature extraction  
HOG + SVM model

```

---

### Day 2

Implement:
```

LBP + SVM  
Color histogram + Logistic Regression

```

---

### Day 3

Implement:
```

ResNet50 embedding extraction  
embedding caching

```

---

### Day 4

Train classifiers on embeddings:
```

Logistic Regression  
SVM

```

---

### Day 5

Finalize:
```

metrics generation  
mlflow logging  
validation tests

```

---

# 17. Acceptance Criteria

Phase 2 is complete when:

- handcrafted feature models are trained
- embedding-based models are trained
- embeddings are cached
- models are serialized
- metrics are computed
- MLflow logs exist
- notebooks execute without errors

---

# 18. Expected Outcomes

At the end of Phase 2 the project will have:

- multiple classical ML baselines
- deep feature baselines
- reusable embedding cache
- fully tracked experiment runs

These models establish the **initial benchmark performance level** for the dataset.

Later phases will introduce deep learning models that will be compared against these baselines.

---

# End of Phase 2 Plan
