# Lightweight Spatiotemporal Deep Learning Model for Sign Language Recognition

This repository contains the implementation and ablation-study materials for a lightweight spatiotemporal Transformer designed for word-level sign language recognition (SLR). The approach uses MediaPipe hand landmarks extracted from RGB video and models their temporal relationships with a compact Transformer architecture.

> **Paper:** Pending submission  
> **Code:** This repository  
> **Tasks:** Isolated word-level Turkish and Arabic sign language recognition

## Overview

The proposed pipeline:

1. accepts prerecorded video or a live RGB camera stream;
2. extracts 21 landmarks per hand with MediaPipe Hands;
3. samples frames across four temporal segments;
4. constructs a spatiotemporal landmark tensor; and
5. applies a lightweight Transformer for gesture classification.

The experiments use:

- **AUTSL:** 40 sampled frames and an input tensor of `(40, 21, 2, 3)`;
- **KArSL-190:** 20 sampled frames and an input tensor of `(20, 21, 2, 3)`; and
- **KArSL-100:** modality ablations using hand, body, and face landmarks.

All reported accuracy, precision, recall, and F1 values below are percentages.

## Main results

| Dataset | Frames | Hidden size | Attention heads | GFLOPs | Test accuracy | Efficiency (accuracy/GFLOPs) |
|---|---:|---:|---:|---:|---:|---:|
| AUTSL | 40 | 510 | 6 | 0.0878 | 77.77 | 885.8 |
| KArSL-190 | 20 | 510 | 6 | 0.0479 | 96.97 | 2024.4 |

## Ablation study

### 1. Architecture refinement on AUTSL

The baseline LSTM was progressively refined by restructuring the input as a spatiotemporal tensor, increasing the number of recurrent layers, and finally replacing the recurrent architecture with a Transformer.

| Model | Input | Validation accuracy | Precision | Recall | F1 | Model size (MB) |
|---|---|---:|---:|---:|---:|---:|
| Base model | `5040` | 62.25 | 64.20 | 62.52 | 62.36 | 86.9 |
| 1-layer LSTM | `(40, 21, 2, 3)` | 73.67 | 75.07 | 73.67 | 73.32 | 5.2 |
| 2-layer LSTM | `(40, 21, 2, 3)` | 76.40 | 77.57 | 76.40 | 76.06 | 13.7 |
| Transformer | `(40, 21, 2, 3)` | **80.38** | **82.32** | **80.38** | **80.27** | 38.4 |

Final AUTSL test performance:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Transformer | **77.77** | **79.73** | **77.77** | **77.53** |

### 2. Temporal-sampling ablation on AUTSL

Three 40-frame sampling strategies were compared. The proposed four-segment approach produced substantially better test generalization than random or conventional segmented sampling.

#### Validation set

| Sampling method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Random temporal sampling | 77.78 | 80.96 | 77.78 | 77.70 |
| Segmented sampling (`13-14-13`) | 73.01 | 76.33 | 73.01 | 72.97 |
| Proposed sampling method | **80.38** | **82.32** | **80.38** | **80.27** |

#### Test set

| Sampling method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Random temporal sampling | 33.55 | 32.59 | 33.55 | 32.36 |
| Segmented sampling (`13-14-13`) | 31.69 | 32.02 | 31.69 | 31.12 |
| Proposed sampling method | **77.77** | **79.73** | **77.77** | **77.53** |

### 3. Temporal-distribution ablation on KArSL-190

Each experiment samples 20 frames from four temporal segments. The percentages indicate the proportion of frames drawn from each consecutive segment.

#### Validation set

| Temporal distribution | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `20-30-30-20` | **98.46** | **98.63** | **98.46** | **98.42** |
| `25-25-25-25` (original) | 97.98 | 98.15 | 97.98 | 97.96 |
| `10-40-40-10` | 98.32 | 98.48 | 98.32 | 98.31 |

#### Test set

| Temporal distribution | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `20-30-30-20` | **96.99** | 97.17 | **96.99** | 96.76 |
| `25-25-25-25` (original) | 96.97 | **97.36** | 96.97 | **96.91** |
| `10-40-40-10` | 65.38 | 65.91 | 65.38 | 65.43 |

The first two distributions perform nearly identically on the test set. Concentrating 80% of the sampled frames in the middle segments (`10-40-40-10`) produces high validation scores but poor test generalization.

### 4. Landmark-modality ablation on KArSL-100

All modality combinations were evaluated with the same Transformer architecture.

| Landmark feature set | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Hand + body + face | **99.75** | **99.76** | **99.75** | **99.75** |
| Hand + body | 99.56 | 99.58 | 99.56 | 99.56 |
| Hand + face | 98.75 | 98.84 | 98.75 | 98.75 |
| Hand only | 98.69 | 98.80 | 98.69 | 98.67 |

The complete hand-body-face representation achieves the highest score, while the hand-only configuration remains competitive and supports lightweight deployment.

### 5. Lightweight Transformer scaling

| Dataset | Frames | Hidden size | Layers | Heads | Classes | GFLOPs | Parameters | CPU time | GPU time | Model size (MB) | Accuracy | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUTSL | 40 | 510 | 2 | 6 | 225 | 0.0878 | 6,408,392 | 0.98 ms | 0.09 ms | 38.4 | 77.77 | 885.8 |
| KArSL-190 | 20 | 510 | 2 | 6 | 190 | 0.0479 | 6,379,796 | 0.58 ms | 0.07 ms | 38.3 | **96.97** | 2024.4 |
| KArSL-190 | 20 | 512 | 2 | 4 | 190 | 0.0482 | 6,412,990 | 0.53 ms | 0.07 ms | 38.5 | 96.77 | 2007.7 |
| KArSL-190 | 20 | 256 | 1 | 4 | 190 | 0.0157 | 1,369,278 | 0.14 ms | 0.04 ms | 5.6 | 96.41 | 6141.4 |
| KArSL-190 | 20 | 128 | 1 | 4 | 190 | 0.0031 | 620,222 | 0.08 ms | 0.04 ms | 2.5 | 96.71 | 31261.3 |
| KArSL-190 | 20 | 64 | 1 | 4 | 190 | 0.0013 | 294,846 | 0.07 ms | 0.04 ms | 1.2 | 96.57 | 74361.5 |
| KArSL-190 | 20 | 32 | 1 | 4 | 190 | 0.0006 | 144,446 | 0.04 ms | 0.04 ms | 0.597 | 95.51 | 159183.3 |
| KArSL-190 | 20 | 16 | 1 | 4 | 190 | 0.0003 | 72,318 | 0.05 ms | 0.04 ms | 0.302 | 88.43 | 294766.7 |
| KArSL-190 | 20 | 8 | 1 | 4 | 190 | 0.0001 | 37,022 | 0.03 ms | 0.05 ms | 0.057 | 57.83 | 578300.0 |

These results expose the accuracy-efficiency trade-off. Configurations with hidden sizes from 32 to 256 retain useful recognition performance while sharply reducing parameters, model size, and computational cost.

## Datasets

- [AUTSL - Ankara University Computer Vision and Machine Learning Laboratory](https://cvml.ankara.edu.tr/datasets/)
- [KArSL - Arabic Sign Language Dataset](https://hamzah-luqman.github.io/KArSL/)

Please follow the respective dataset licenses, access conditions, and citation requirements. The datasets are not redistributed in this repository.

## Citation

The associated manuscript is pending submission. Citation information and a paper link will be added when publicly available.

## Status

This repository supports ongoing research. Results and documentation may be updated during manuscript review and revision.

