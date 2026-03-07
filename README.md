# Methodology

## 1. Dataset
- This study uses the **Hillel Yaffe Glaucoma Dataset (HYGD) v1.0.0** from [PhysioNet](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0). 
- Each image in the dataset is accompanied by a structured label file (`Labels.csv`) containing four fields per record: image filename, patient identifier, diagnostic class, and photographic quality score on a scale of 1 to 10.
- The diagnostic target is binary classification between two classes:
	- **GON+** — Glaucomatous Optic Neuropathy positive (glaucoma present)
	- **GON−** — Glaucomatous Optic Neuropathy negative (healthy)

## 2. Data Preprocessing

### 2.1 Label Encoding
- The categorical class labels are converted to binary integer values: 
	- GON+ is encoded as `1` 
	- GON− is encoded as `0`
- This encoding is required by the [[nn.BCEWithLogitsLoss|binary cross-entropy loss function]] used during training.

## 3. Train Test Split
### 3.1 Patient-Level Stratified Split
- The dataset is partitioned into three non-overlapping subsets: **training (70%)**, **validation (15%)**, and **test (15%)**. 
- Critically, the split is performed at the **patient level** rather than the image level.
- A single patient in HYGD may contribute multiple images
- If splitting were done by image, the same patient's eyes could appear in both the training and test sets simultaneously causing **data leakage**.  
- The patient-level split procedure is as follows:
	1. Each patient has their inidvidual ID in "Patient" columns
	2. Patients ID are split into train/val/test groups using stratified random sampling (`StratifiedShuffleSplit`), preserving the class ratio at the patient level.
	3. This guarantees zero patient overlap across all three subsets:
$$\text{Train patients} \cap \text{Val patients} \cap \text{Test patients} = \emptyset$$

## 3.2 Class Stratification
- Class distribution visualisation
	- ![[International Data Science Challenge 2026 - HYGD Dataset-1772852777773.png|298]]
- Stratification on class ensures that the approximately 73% GON+ : 27% GON class ratio present in the full dataset is preserved within each split. 
	- ![[International Data Science Challenge 2026 - HYGD Dataset-1772854269001.png|551]]
- Without stratification, a random split could produce a validation or test set with a severely different class balance, making metric comparisons unreliable.

## 4. Model Architecture
### 4.1 Backbone: EfficientNet-B3
- The feature extraction backbone is **EfficientNet-B3**, implemented via the `timm` library with ImageNet pre-trained weights. 
- EfficientNet-B3 was selected for the following reasons:
	- Able to achieve strong performance at moderate computational cost.
	- Its native input resolution of **380 × 380 pixels** aligns well with the resolution of fundus photographs, preserving fine structural detail around the optic disc and cup.
	- Transfer learning from ImageNet provides a strong initialisation
- The backbone is loaded with `num_classes=0` and `global_pool="avg"`
	- `num_classes=0` removes the original classification head
	- `global_pool="avg"` applies global average pooling
		- EfficientNet-B3 produce 3d tensor of 10x10x1536
		- `global_pool="avg"` average out the 10x10 for all 1536 dimension and output 1536 shaped 1d tensor

### 4.2 Classification Head
A custom classification head is appended after the output of EfficientNet-B3:
```
1536-dim feature vector
    → LayerNorm(1536)
    → Dropout(p=0.4)
    → Linear(1536 → 256)
    → GELU activation
    → Dropout(p=0.2)
    → Linear(256 → 1)
    → scalar logit
```
- The head outputs a single scalar logit per image. 
	- A positive logit indicates GON+; a negative logit indicates GON−. 
	- sigmoid function can be used to get probability, it converts the logit to a probability
	- During loss calculation, [[nn.BCEWithLogitsLoss]] is used for peformance
- `LayerNorm` is applied before the first linear layer to stabilise the distribution of backbone features. 
- `GELU` is chosen as the activation function for its smooth gradient properties compared to ReLU. 
- Dropout is applied at two rates (0.4 and 0.2) to reduce overfitting

## 5. Image Transform Pipeline
All image preprocessing and augmentation is implemented using the **MONAI** (Medical Open Network for AI) transform framework
- `TRAINING` **Base Transforms Head** -> **Training Augmentations** -> **Base Transforms Tail**
- `NON-TRAINING` **Base Transforms Head** -> **Base Transforms Tail**

### 5.1 Base Transforms Head

| Transform             | Parameters                                | Purpose                                            |
| --------------------- | ----------------------------------------- | -------------------------------------------------- |
| `LoadImaged`          | —                                         | Load JPEG/PNG from disk into a NumPy array         |
| `EnsureChannelFirstd` | —                                         | Reorder channel dim to first: H×W×C -> C×H×W       |
| `ScaleIntensityd`     | `minv=0.0, maxv=1.0`                      | Normalise pixel values from [0, 255] to [0.0, 1.0] |
| `Resized`             | `spatial_size=(380, 380), mode="bicubic"` | Resize all images to 380×380 pixels                |

### 5.2 Base Transforms Tail

| Transform             | Parameters                                                          | Purpose                               |
| --------------------- | ------------------------------------------------------------------- | ------------------------------------- |
| `NormalizeIntensityd` | `subtrahend=[0.485, 0.456, 0.406]`, `divisor=[0.229, 0.224, 0.225]` | Channel-wise ImageNet normalisation   |
| `ToTensord`           | —                                                                   | Convert NumPy array to `torch.Tensor` |

- [[ImageNet-1K normalization|ImageNet statistics]] are used for normalisation because the backbone was pre-trained on ImageNet. 
- Aligning the input distribution to what the backbone was trained on is important for effective transfer learning.

### 5.3 Training Augmentations
- During training only, a stochastic augmentation stage is inserted between resizing and normalisation. 
- Each augmentation is applied with a specified probability per image per epoch, meaning every training epoch the model sees a freshly randomised version of each image. 

| Transform                | Probability | Parameters                                                            | Rationale                                                                                                                |
| ------------------------ | ----------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `RandFlipd` (vertical)   | 0.5         | `spatial_axis=0`                                                      | Optic disc pathology is symmetric; orientation is not diagnostically meaningful                                          |
| `RandFlipd` (horizontal) | 0.5         | `spatial_axis=1`                                                      | Same rationale as above                                                                                                  |
| `RandRotate90d`          | 0.5         | `max_k=3`                                                             | Rotational invariance in 90° increments                                                                                  |
| `RandRotated`            | 0.4         | `range_x=0.52` (±30°), `range_y=0.0`, `range_z=0.0`, `keep_size=True` | Continuous in-plane rotation; `range_y` and `range_z` are locked to zero to prevent out-of-plane distortion on 2D images |
| `RandZoomd`              | 0.3         | `min_zoom=0.85, max_zoom=1.15`                                        | Simulates variable patient distance from the camera                                                                      |
| `RandAffined`            | 0.3         | `translate_range=(0.05, 0.05)`                                        | Up to 5% spatial shift (≈19 px at 380 px); value is a fraction of image dimensions                                       |
| `RandAdjustContrastd`    | 0.4         | `gamma=(0.7, 1.5)`                                                    | Simulates exposure variation across different imaging devices and settings                                               |
| `RandGaussianNoised`     | 0.3         | `std=0.02`                                                            | Simulates camera sensor noise                                                                                            |
| `RandGaussianSmoothd`    | 0.2         | `sigma_x=(0.5, 1.5)`, `sigma_y=(0.5, 1.5)`                            | Simulates minor optical blur or defocus                                                                                  |

## 6. Class Imbalance Handling
- The HYGD dataset exhibits a class imbalance of approximately **73% GON+ and 27% GON−**. 
- 2 complementary strategies are applied to prevent the model from developing a majority-class bias.

### 6.1 Weighted Random Sampler
A `WeightedRandomSampler` is used during training to rebalance the frequency with which each class is drawn per epoch. Each image is assigned a sampling weight inversely proportional to its class frequency:
$$w_i = \frac{N}{N_c}$$

where $N$ is the total number of training images and $N_c$ is the number of images in class $c$. This ensures that GON− images are sampled more frequently than their raw proportion, resulting in an approximately balanced training stream.

### 6.2 Positive-Weight Loss
In addition to the sampler, `BCEWithLogitsLoss` is configured with a `pos_weight` parameter

$$\text{pos\_weight} = \frac{N_{\text{GON-}}}{N_{\text{GON+}}}$$

- Standard Binary Cross-Entropy (BCE) treats a False Positive (FP) and a False Negative (FN) as equally "bad."
- pos_weight acts as a multiplier for the loss on the positive class. Since the dataset has more GON+ than GON− images, `pos_weight` < 1, which down-weights the loss contribution of each GON+ sample. 


## 7. Training Procedure
### 7.1 Loss Function
Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) is used as the training objective. This loss numerically combines the sigmoid activation and the binary cross-entropy calculation in a single, numerically stable operation:
$$\mathcal{L} = -\left[ y \cdot \log\sigma(\hat{z}) + (1 - y) \cdot \log(1 - \sigma(\hat{z})) \right]$$
where $\hat{z}$ is the raw logit output and $y \in \{0, 1\}$ is the true label.

### 7.2 Optimiser
**AdamW** is used as the optimiser with a learning rate of $3 \times 10^{-4}$ and weight decay of $1 \times 10^{-4}$. AdamW decouples the weight decay from the gradient update, which has been shown to improve generalisation compared to standard Adam with L2 regularisation.

### 7.3 Learning Rate Schedule
A **Cosine Annealing** schedule is applied over the full training duration (`T_max = 40` epochs), smoothly reducing the learning rate from $3 \times 10^{-4}$ to a minimum of $1 \times 10^{-6}$. Cosine annealing avoids abrupt step decays and allows the optimiser to escape shallow local minima during the early phases of training.

### 7.4 Gradient Clipping
Gradients are clipped to a maximum $\ell_2$ norm of 1.0 before each optimiser step. This prevents gradient explosion.

### 7.5 Mixup Regularisation
**Mixup** augmentation is applied at the batch level during training. Given two randomly selected images $x_i$ and $x_j$ with labels $y_i$ and $y_j$, a blended training sample is constructed as:
$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$
where $\lambda \sim \text{Beta}(\alpha, \alpha)$ with $\alpha = 0.2$. The loss is computed as a weighted combination of the two constituent losses:
$$\tilde{\mathcal{L}} = \lambda \cdot \mathcal{L}(\hat{z}, y_i) + (1 - \lambda) \cdot \mathcal{L}(\hat{z}, y_j)$$

Mixup encourages the model to produce smoother, less overconfident probability estimates and has been shown to improve generalisation and calibration across a range of classification tasks.

### 7.6 Early Stopping
Training is monitored via validation AUC-ROC after each epoch. If the validation AUC does not improve by at least $\Delta = 0.0001$ for **8 consecutive epochs**, training is terminated. The model checkpoint corresponding to the highest observed validation AUC is restored at the end of training, ensuring the final model is the best-performing version rather than the last-epoch version.

### 7.7 Model Checkpointing
The model state dictionary is saved to disk whenever a new best validation AUC is achieved. On early stopping or training completion, this best checkpoint is loaded back into memory before proceeding to test evaluation.

## 8. Test-Time Augmentation (TTA)
- At prediction/inference stage, **Test-Time Augmentation(TTA)** is applied to reduce prediction variance. 
- Each test image is independently passed through five deterministic(No random component) transform pipelines, each applying a different geometric transformation.
- Deterministic transforms (`Flipd`, `Rotate90d`) are used rather than their stochastic counterparts (`RandFlipd`, `RandRotate90d`) to ensure fully reproducible TTA predictions regardless of random state.


| Pass | Transform |
|---|---|
| 1 | Original (no transformation) |
| 2 | Vertical flip (`Flipd`, `spatial_axis=0`) |
| 3 | Horizontal flip (`Flipd`, `spatial_axis=1`) |
| 4 | 90° counter-clockwise rotation (`Rotate90d`, `k=1`) |
| 5 | 270° counter-clockwise rotation (`Rotate90d`, `k=3`) |

The final prediction probability for each image is the arithmetic mean across all five passes:
$$\hat{p}_{\text{TTA}} = \frac{1}{5} \sum_{k=1}^{5} \sigma(\hat{z}_k)$$

> **Implementation note:** Rather than building five separate `CacheDataset` pipelines (which would run the full `transform_head` five times per image), the expensive head transforms are cached once in RAM. Each of the five TTA passes then applies only its geometric variant and the normalisation tail to the cached intermediate tensors, reducing preprocessing cost to a single pass regardless of the number of TTA steps.

## 9. Evaluation
### 9.1 Decision Threshold Selection
Rather than applying a fixed threshold of 0.5, the optimal classification threshold is determined using [[Youden's J statistic]], which maximises the sum of sensitivity and specificity simultaneously:

$$J = \text{Sensitivity} + \text{Specificity} - 1$$
$$\theta^* = \arg\max_\theta \left[ \text{TPR}(\theta) - \text{FPR}(\theta) \right]$$
This threshold is computed from the validation or test ROC curve and applied when converting predicted probabilities to binary class predictions.

### 9.2 Evaluation Metrics
Performance is assessed using a comprehensive set of metrics suited to medical binary classification:

| Metric | Formula | Interpretation |
|---|---|---|
| **AUC-ROC** | Area under the ROC curve | Overall ranking ability, independent of threshold |
| **Accuracy** | $(TP + TN) / N$ | Overall correct classification rate |
| **Sensitivity** (Recall) | $TP / (TP + FN)$ | Proportion of actual GON+ cases correctly identified |
| **Specificity** | $TN / (TN + FP)$ | Proportion of actual GON− cases correctly identified |
| **PPV** (Precision) | $TP / (TP + FP)$ | Proportion of predicted GON+ that are truly GON+ |
| **NPV** | $TN / (TN + FN)$ | Proportion of predicted GON− that are truly GON− |
| **F1 (macro)** | $2 \cdot (PPV \cdot \text{Sensitivity}) / (PPV + \text{Sensitivity})$ | Harmonic mean of precision and recall |

In clinical screening contexts, **Sensitivity** (minimising missed cases) and **AUC-ROC** (threshold-independent discrimination) are the primary metrics of interest.

## 10. Explainability: Grad-CAM
Gradient-weighted Class Activation Mapping (Grad-CAM) is applied to all images in the test set. Manual PyTorch forward and backward hooks are registered on `model.backbone.blocks[-1]` — the last convolutional block before global average pooling, which retains spatial dimensions `(C, H', W')`. The CAM is computed as:

$$\text{CAM} = \text{ReLU}\!\left(\sum_c \bar{g}_c \cdot A_c\right), \quad \bar{g}_c = \frac{1}{H'W'} \sum_{i,j} \frac{\partial \hat{z}}{\partial A_c^{ij}}$$

where $A_c$ is the activation map of channel $c$ and $\bar{g}_c$ is its spatially averaged gradient. The resulting saliency map is upsampled to 380×380 and overlaid on the original fundus image.

Both correctly classified and misclassified images are saved. Misclassified images (false positives and false negatives) are of particular diagnostic interest as they reveal which image regions caused the model to fail. Output files follow the naming convention `{original_filename}__{true_label}__{correct/wrong}.png` and are saved to `outputs/gradcam/`.


## 11. Implementation Details

| **Component**           | **Specification**                                  |
| ----------------------- | -------------------------------------------------- |
| Framework               | PyTorch 2.x + MONAI                                |
| Backbone                | EfficientNet-B3 (`timm`)                           |
| Input resolution        | 380 × 380 px                                       |
| Batch size              | 16                                                 |
| Optimiser               | AdamW                                              |
| Learning rate           | $3 \times 10^{-4}$ (cosine annealing to $10^{-6}$) |
| Weight decay            | $1 \times 10^{-4}$                                 |
| Dropout                 | 0.4 (head layer 1), 0.2 (head layer 2)             |
| Max epochs              | 40                                                 |
| Early stopping patience | 8 epochs                                           |
| Mixup $\alpha$          | 0.2                                                |
| TTA passes              | 5                                                  |
| Loss function           | BCEWithLogitsLoss with pos_weight                  |
| Random seed             | 42 (via `monai.utils.set_determinism`)             |
| Hardware                | CUDA GPU / CPU fallback                            |
