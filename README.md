# Waste Classification: Benchmark vs. Real-World Generalization

This project investigates how well convolutional neural networks (CNNs) trained on the **TrashNet** benchmark dataset generalize to ordinary real-world phone photos of waste.

The main research question is:

> **A CNN may perform well on benchmark data, but will it still classify waste correctly in everyday real-world photos?**

We compare four image-classification architectures:

- Custom CNN
- VGG16
- ResNet50
- MobileNetV2

The project evaluates each model first on the TrashNet benchmark dataset and then on a separate set of real-world phone photos collected by the team.

---

## Project Overview

The experiment has two main phases.

### Phase 1: TrashNet Benchmark Evaluation

All four models are trained using TrashNet and evaluated using a fixed train, validation, and test split.

Three team members train the same architectures using different training seeds:

- Seed 0 — Vivian
- Seed 1 — Joshua
- Seed 2 — Devan

The dataset split remains fixed across all seeds so that differences between runs come from training randomness rather than different train/test images.

### Phase 2: Real-World Evaluation

After training is complete, the models are frozen and evaluated on phone photos collected by the team.

These real-world images are **not used for training or fine-tuning**.

This allows us to compare performance under controlled benchmark conditions with performance on more realistic images containing different:

- backgrounds
- lighting conditions
- camera angles
- object sizes
- distances
- clutter
- object orientations

The difference between benchmark performance and real-world performance is used to study the models' **generalization gap**.

---

## Dataset

### TrashNet

The project uses the TrashNet waste-classification dataset containing **2,527 images** across six categories:

| Class | Images |
|---|---:|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |
| **Total** | **2,527** |

A fixed stratified split is used:

- Training: approximately 70%
- Validation: approximately 15%
- Testing: approximately 15%

This produces:

- 1,768 training images
- 379 validation images
- 380 test images

The shared split is stored in:

```text
data_splits/trashnet_split.csv
```

### Real-World Test Set

The team also collected a separate set of phone photos representing the same six waste categories.

Each participant collected real-world images using a participant identifier such as:

```text
glass_P2_001.jpg
cardboard_P3_004.jpg
```

- The real-world dataset is saved in shared Google doc
