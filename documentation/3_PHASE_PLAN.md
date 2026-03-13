# Phase 3 Implementation Plan

**Project:** AnimalClassification  
**Phase:** Phase 3 — CNN Models Trained From Scratch  
**Estimated Duration:** 5–7 days  
**Prerequisite:** Phase 1 and Phase 2 completed  

---

## 1. Phase 3 Objective

Phase 3 introduces the first deep learning models trained directly on the dataset without relying on pretrained weights.

The primary objective is to evaluate the capacity of custom Convolutional Neural Networks (CNNs) to learn hierarchical spatial features from scratch via the cross-correlation operation. By establishing these baselines, we can:

* Analyze the optimization landscape (navigating local minima and saddle points).

* Evaluate the effectiveness of architectural regularizers (Batch Normalization, Dropout) and optimization regularizers (Weight Decay, Data Augmentation).
* Provide a rigorous comparison point for the classical ML baselines (Phase 2) and future Transfer Learning/Fine-tuning experiments (Phase 4).

---

## 2. Expected Deliverables

At the completion of Phase 3, the repository must contain:

**Trained Models**
```text
models/cnn_scratch/
    customcnn_v1/
        run_YYYYMMDD_HHMMSS/
            checkpoint.pt
            exported.onnx
            config.json
            metrics.json
    customcnn_v2/
        run_YYYYMMDD_HHMMSS/
            checkpoint.pt
            exported.onnx
            config.json
            metrics.json
```

**Training Notebooks & Metrics**
```text
notebooks/30_cnn_scratch_custom/
    30_00_overview.ipynb
    30_01_customcnn_v1.ipynb
    30_02_customcnn_v2.ipynb
```
* Output JSON files in `reports/metrics/`.
* Generated Training Curves (`loss_curve.png`, `accuracy_curve.png`) saved as artifacts.
* Complete experiment tracking in `mlruns/`, logging hyperparameter tuning, model parameters, and optimization metrics.

---

## 3. Required Inputs

Phase 3 relies strictly on the deterministic infrastructure built in Phase 1. All models must use `split_v1` and `transforms_v1`. No dataset modifications are allowed to ensure valid benchmarking.

---

## 4. Expected Accuracy Baselines

To validate that training is behaving correctly, models will be measured against the following expected ranges based on dataset difficulty and model capacity:

| Model | Expected Accuracy |
|---|---|
| Color Hist Baseline (Phase 2) | 0.55 – 0.60 |
| HOG Baseline (Phase 2) | 0.65 – 0.72 |
| ResNet50 Embeddings (Phase 2) | 0.80 – 0.88 |
| CustomCNN v1 (Phase 3) | 0.78 – 0.85 |
| CustomCNN v2 (Phase 3) | 0.82 – 0.88 |

*Note: If Phase 3 models fall significantly outside these ranges, it indicates an optimization or pipeline error.*

---

## 5. Training Pipeline Overview



[Image of Convolutional Neural Network architecture diagram]


The training pipeline implements a full forward and backward pass, optimized for stability.

**Forward Pass:**
* **Input:** RGB Image (Multiple Channels).
* **Feature Extraction:** Spatial hierarchies learned via 2D cross-correlation using $3 \times 3$ kernels (increasing receptive field while managing parameter counts).
* **Non-linearity:** ReLU activations to mitigate the vanishing gradient problem.
* **Downsampling:** Max Pooling for spatial invariance.
* **Global Pooling:** `AdaptiveAvgPool2d(1)` to reduce feature maps to a 1D vector per channel, massively reducing parameters before the classifier.
* **Classification:** Fully connected layers with Dropout.

**Backward Pass:**
* **Loss Computation:** Cross-Entropy Loss.
* **Backpropagation & Optimization:** Updating weights via gradient descent algorithms with Weight Decay.

---

## 6. CNN Architectures Implemented

Two distinct architectures will be implemented using PyTorch's `nn.Sequential` module. Custom weights will be initialized using He Initialization (`nn.init.kaiming_normal_`) to maintain variance across ReLU layers and prevent vanishing/exploding gradients early in training.

---

## 7. Model 1 — CustomCNN v1 (Standard Baseline)

A lightweight spatial feature extractor designed to establish a raw baseline.

**Structure:**
* **Block 1:** Conv2D ($3 \times 3$ kernel, stride 1, padding 1, $3 \to 32$ channels) $\to$ ReLU $\to$ MaxPool ($2 \times 2$)
* **Block 2:** Conv2D ($3 \times 3$ kernel, stride 1, padding 1, $32 \to 64$ channels) $\to$ ReLU $\to$ MaxPool ($2 \times 2$)
* **Block 3:** Conv2D ($3 \times 3$ kernel, stride 1, padding 1, $64 \to 128$ channels) $\to$ ReLU $\to$ MaxPool ($2 \times 2$)
* **Classifier:** `AdaptiveAvgPool2d(1)` $\to$ Flatten $\to$ Linear ($128 \to 256$) $\to$ ReLU $\to$ Dropout ($p=0.5$) $\to$ Linear ($256 \to 3$)

**Notebook:** `30_01_customcnn_v1.ipynb`

---

## 8. Model 2 — CustomCNN v2 (Advanced VGG-Style Architecture)



A deeper, heavily regularized architecture utilizing stacked $3 \times 3$ convolutions and Batch Normalization.

**Structure:**
* **Block 1:** * Conv2D ($3 \to 32$) $\to$ BatchNorm2D $\to$ ReLU
  * Conv2D ($32 \to 32$) $\to$ BatchNorm2D $\to$ ReLU $\to$ MaxPool ($2 \times 2$)
* **Block 2:** * Conv2D ($32 \to 64$) $\to$ BatchNorm2D $\to$ ReLU
  * Conv2D ($64 \to 64$) $\to$ BatchNorm2D $\to$ ReLU $\to$ MaxPool ($2 \times 2$)
* **Block 3:** * Conv2D ($64 \to 128$) $\to$ BatchNorm2D $\to$ ReLU
  * Conv2D ($128 \to 128$) $\to$ BatchNorm2D $\to$ ReLU $\to$ MaxPool ($2 \times 2$)
* **Classifier:** `AdaptiveAvgPool2d(1)` $\to$ Flatten $\to$ Linear ($128 \to 512$) $\to$ ReLU $\to$ Dropout ($p=0.5$) $\to$ Linear ($512 \to 3$)

**Notebook:** `30_02_customcnn_v2.ipynb`

---

## 9. Training & Optimization Configuration

To efficiently navigate local minima and saddle points, the following hyperparameters will be utilized:

* **Epochs:** 30
* **Batch Size:** 64
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam (Adaptive Moment Estimation).
* **Base Learning Rate:** `1e-3`
* **Weight Decay:** `1e-4` ($L_2$ regularization).
* **Gradient Clipping:** Max norm of 1.0.
* **Learning Rate Scheduler:** `ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=3)`. This explicitly decays the learning rate when validation loss stops improving, ensuring stable convergence.

---

## 10. Data Loading & Augmentation

The PyTorch dataset loader (`src/data/dataset_loader.py`) will be heavily utilized. Data Augmentation (defined in `transforms_v1.yaml`) serves as a critical regularizer, forcing the kernels to learn invariant features rather than memorizing the training distribution.

---

## 11. Training Loop Details

The training loop must track moving averages of the loss and enforce strict mode switching (`model.train()` vs `model.eval()`).

Metrics computed per epoch include `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, and `val_macro_f1`. At the end of training, matplotlib will be used to generate `loss_curve.png` and `accuracy_curve.png` to visualize convergence and diagnose potential overfitting.

---

## 12. Model Checkpointing & MLflow Integration

Experiments must log the following:
* **Parameters:** Architecture type, Optimizer type, Learning Rate, Weight Decay, Scheduler params.
* **Metrics:** Epoch-level losses and validation metrics.
* **Artifacts:** `checkpoint.pt`, `exported.onnx`, `metrics.json`, `loss_curve.png`, `accuracy_curve.png`.

---

## 13. Hardware & Execution

Phase 3 training relies heavily on parallel matrix multiplication and must be executed on a GPU. The dataset size (~62k) fits the memory footprint perfectly.

* **CustomCNN v1:** ~30–50 minutes.
* **CustomCNN v2:** ~60–120 minutes.

---

## 14. Next Phase: Advanced Architectures & Transfer Learning

[Image comparing ResNet, DenseNet, and Inception architectures]

Phase 3 establishes the ceiling for models trained entirely from scratch. Phase 4 will transition into testing state-of-the-art architectures originally trained on the ImageNet Dataset.

Phase 4 will implement Transfer Learning and Fine-tuning techniques on architectures including:
* **ResNet and ResNeXt** (evaluating skip/residual connections).
* **DenseNet**.
* **MobileNet / EfficientNet** (evaluating parameter efficiency).
* **VGGNet and Inception (GoogLeNet)** for architectural lineage comparison.