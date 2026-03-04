# AnimalClassification — Project Architecture & Experimentation Plan

## 1. Project Goal

This project benchmarks **different machine learning approaches for animal image classification** across three classes:

- **cats**
- **dogs**
- **wildlife**

The primary objective is **comparative evaluation** of multiple model families under the same dataset, preprocessing pipeline, and evaluation framework.

We will compare:

1. Classical ML with **basic handcrafted features**
2. Classical ML with **deep features extracted from pretrained encoders**
3. **CNN architectures trained from scratch**
4. **CNNs fine-tuned from pretrained models**
5. **Vision Transformers (ViT)** both from scratch and pretrained

In addition to classification accuracy, the system evaluates:

- inference latency
- model size
- parameter count
- throughput
- computational cost

The project is designed to maximize **reproducibility, modularity, and reuse of computation**.

---

# 2. Dataset Overview

The unified dataset is built from several public datasets:

| Source | Description |
|------|------|
| Microsoft Cats vs Dogs | internet images of cats and dogs |
| AFHQv2 | high quality cat/dog/wild images |
| AFD (Animal Face Dataset) | multiple wildlife species |
| HuggingFace animal faces dataset | cat/dog face dataset |

All datasets are unified into a **three-class dataset**:
```

cats  
dogs  
wildlife

```

After cleaning and deduplication:
```

Total images ≈ 62,659  
cats ≈ 23.7k  
dogs ≈ 23.8k  
wildlife ≈ 16k

```

The dataset is stored in:
```

data/prepared/  
cats/  
dogs/  
wildlife/

```

---

# 3. Dataset Splits

The benchmark requires **fixed dataset splits** to guarantee fair comparison between models.

Splits are generated once and reused for every experiment.
```

data/splits/split\_v1/

```
train.csv
val.csv
test.csv
classes.json
```

```

CSV format:
```

filepath,label  
data/prepared/cats/img001.jpg,cats  
data/prepared/dogs/img002.jpg,dogs  
...

```

Advantages of CSV manifests instead of copying folders:

- no duplication of files
- reproducible
- faster to load
- easier auditing
- easy to extend

---

# 4. Project Directory Structure
```

AnimalClassification/

data/  
datasets\_raw/ # original downloaded datasets  
prepared/ # merged cleaned dataset  
splits/  
split\_v1/  
processed/ # cached features or embeddings

models/  
ml\_basic\_features/  
ml\_deep\_features/  
cnn\_scratch/  
cnn\_pretrained/  
vit/

reports/  
metrics/  
figures/

notebooks/  
00\_project\_setup.ipynb  
01\_data\_prep\_and\_splits.ipynb  
02\_transforms\_and\_augmentation.ipynb

```
10_ml_basic_features/
20_ml_deep_features_fixed_encoder/
30_cnn_scratch_custom/
40_cnn_pretrained/
50_vit/
```

scripts/  
src/

90\_evaluate\_all\_models.ipynb

```

---

# 5. Model Families

## 5.1 Classical ML — Basic Features

Traditional computer vision features extracted from images.

Examples:

- HOG (Histogram of Oriented Gradients)
- LBP (Local Binary Patterns)
- Color histograms
- Gabor filters

Pipeline:
```

image → feature extractor → classifier

```

Classifiers may include:

- SVM
- Logistic Regression
- Random Forest
- Gradient Boosted Trees

Artifacts stored in:
```

models/ml\_basic\_features/

```

---

## 5.2 Classical ML — Deep Features

Features are extracted using a **fixed pretrained deep encoder**.

Example encoders:

- ResNet50
- EfficientNet
- CLIP encoder

Pipeline:
```

image → pretrained encoder → embedding → classifier

```

Classifier options:

- Logistic Regression
- SVM
- XGBoost

Advantages:

- fast training
- strong baseline
- reusable embeddings

Artifacts stored in:
```

models/ml\_deep\_features/

```

---

## 5.3 CNNs Trained From Scratch

Custom convolutional neural networks trained entirely from the dataset.

Example architectures:

- CustomCNN v1
- CustomCNN v2

Pipeline:
```

image → CNN → classifier

```

Artifacts:
```

models/cnn\_scratch/

```

---

## 5.4 CNNs Using Pretrained Backbones

Transfer learning with pretrained networks.

Two groups:

### Small models
- ResNet18
- MobileNetV3
- EfficientNet-B0

### Medium models
- ResNet50
- EfficientNet-B2

Artifacts:
```

models/cnn\_pretrained/

```

---

## 5.5 Vision Transformers

Two groups:

### From Scratch
Example:
- ViT Tiny

### Pretrained
Examples:
- DeiT Small
- ViT Base

Artifacts:
```

models/vit/

```

---

# 6. Caching Strategy

Efficient caching is essential to avoid repeated expensive computation.

## 6.1 Dataset Splits Cache

Dataset splits are created **once** and reused by every model.
```

data/splits/split\_v1/

```

No model is allowed to generate its own splits.

---

## 6.2 Embedding Cache

Deep feature extraction can be expensive.

Embeddings are cached:
```

data/processed/embeddings/

```
split_v1/
    encoder_resnet50/
        train.npy
        val.npy
        test.npy

        labels_train.npy
        labels_val.npy
        labels_test.npy
```

```

This allows:

- Logistic Regression
- SVM
- XGBoost

to reuse the **same embeddings** without recomputation.

---

## 6.3 Transform Config Cache

Augmentation parameters are stored in a configuration file.

Example:
```

configs/transforms\_v1.yaml

```

Example parameters:
```

image\_size: 224

train\_transforms:  
random\_flip  
random\_rotation  
color\_jitter

eval\_transforms:  
resize  
normalize

```

Every model references the same transform configuration.

---

## 6.4 Model Artifacts

Each experiment produces a **run folder**.

Example:
```

models/cnn\_pretrained/resnet18/

```
run_2026_03_04/

    config.json
    checkpoint.pt
    exported.onnx
    metrics.json
```

```

Run configuration contains:
```

model\_name  
split\_id  
transform\_id  
training\_params  
dataset\_version

```

This ensures **full reproducibility**.

---

# 7. Metrics and Benchmarking

The system evaluates models using multiple metrics.

## Classification Metrics

- accuracy
- macro F1
- precision
- recall
- confusion matrix

## Efficiency Metrics

- inference latency
- throughput (images/sec)
- model size (MB)
- parameter count
- CPU vs GPU inference performance

---

# 8. Global Model Evaluator

The root notebook:
```

90\_evaluate\_all\_models.ipynb

```

acts as the **central benchmarking system**.

Responsibilities:

1. discover all trained models
2. load their configurations
3. run evaluation on the same test dataset
4. measure inference speed
5. record results

Evaluation tools:

- **MLflow**
- PyTorch profiling
- standardized metrics

Outputs:
```

reports/metrics/leaderboard.csv  
reports/figures/accuracy\_vs\_latency.png  
reports/figures/model\_size\_vs\_accuracy.png

```

---

# 9. MLflow Integration

MLflow is used to:

- track experiments
- log parameters
- store metrics
- compare runs

Each model run logs:
```

mlflow.log\_param(...)  
mlflow.log\_metric(...)  
mlflow.log\_artifact(...)

```

Example logged metrics:
```

accuracy  
f1\_macro  
latency\_ms  
model\_size\_mb  
params

```

---

# 10. Reproducibility Guarantees

To guarantee fair comparison:

- all models use **identical dataset splits**
- evaluation uses the **same test set**
- transforms are defined in shared config
- experiment configs are saved with the model
- MLflow records every run

---

# 11. Experiment Workflow

Typical workflow:
```

1.  prepare dataset
2.  generate dataset splits
3.  define transforms
4.  run model training
5.  export trained model
6.  evaluate model
7.  update leaderboard

```

---

# 12. Future Extensions

Possible extensions:

- cross-dataset generalization tests
- out-of-distribution detection
- adversarial robustness
- quantized model benchmarking
- edge deployment benchmarking

---

# 13. Final Benchmark Output

The final system should produce a **model comparison table** such as:

| Model | Accuracy | F1 | Latency | Params | Size |
|------|------|------|------|------|------|
| HOG + SVM | 0.71 | 0.69 | 2ms | - | 5MB |
| ResNet18 | 0.89 | 0.88 | 7ms | 11M | 44MB |
| EfficientNetB0 | 0.91 | 0.90 | 9ms | 5M | 20MB |
| ViT Small | 0.92 | 0.91 | 15ms | 22M | 85MB |

This provides a **clear comparison between model accuracy and deployment cost**.

---

# End of Documentation
