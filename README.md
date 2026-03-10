# StitchNet: Efficient Volumetric Context Modeling for Liver Segmentation Using a Two-Dimensional Deep Learning Framework

StitchNet is a computationally efficient deep learning framework for automated liver segmentation from volumetric medical images. The method introduces a **stitching-based input representation** that embeds inter-slice contextual information into a two-dimensional convolutional neural network (CNN) pipeline, enabling volumetric awareness without the computational burden of full 3D models.

The framework combines a **modified EfficientNet-B7 encoder**, a **Two-Way Feature Pyramid Network (FPN)** for multi-scale feature aggregation, and a **semantic segmentation head** to produce accurate liver segmentation masks.

This repository contains the implementation of **StitchNet and its architectural variants (M1–M4)** along with experimental configurations used for evaluation on the **ATLAS23** and **IRCAD** liver segmentation datasets.

---

## Overview

Medical image segmentation from volumetric scans presents a trade-off between:

- **2D slice-wise models**  
  - Efficient  
  - Limited inter-slice contextual awareness

- **3D CNN models**  
  - Capture volumetric context  
  - Computationally expensive and memory intensive

**StitchNet bridges this gap** by reorganizing adjacent slices from a volume into a **stitched 2D mosaic representation**, allowing the network to implicitly learn volumetric relationships while maintaining the efficiency of a 2D CNN.

---

## Key Contributions

- **Stitching-Based Context Modeling**  
  Introduces a novel input representation that stitches consecutive slices into a composite 2D mosaic, enabling implicit volumetric reasoning within a 2D segmentation pipeline.

- **Efficient Multi-Scale Architecture**  
  Combines a modified EfficientNet-B7 encoder with a Two-Way Feature Pyramid Network (FPN) to capture both global liver morphology and fine boundary details.

- **State-of-the-Art Performance with Low Computational Cost**  
  Achieves superior segmentation accuracy while requiring significantly fewer FLOPs compared to 3D CNN architectures.

---

## Model Architecture

The proposed StitchNet framework consists of three primary components:

1. **Encoder**
   - Modified EfficientNet-B7 backbone
   - Extracts hierarchical feature representations
   - Reduced computational complexity compared to the standard EfficientNet-B7

2. **Two-Way Feature Pyramid Network (FPN)**
   - Enables bidirectional information flow
   - Combines top-down semantic abstraction with bottom-up spatial refinement

3. **Semantic Segmentation Head**
   - Integrates multi-scale features
   - Produces pixel-wise liver segmentation masks

---

## Stitching Strategy

Instead of processing slices independently, StitchNet constructs a stitched representation:

Volume slices → Stitched mosaic → StitchNet → Stitched segmentation mask


This design allows the model to capture **inter-slice continuity** while preserving the computational efficiency of a 2D convolutional network.

---

## Architectural Variants

| Model | Configuration |
|------|---------------|
| **StitchNet M1** | Encoder + Semantic Head |
| **StitchNet M2** | Encoder + One-Way FPN + Semantic Head |
| **StitchNet M3** | Encoder + Two-Way FPN + Semantic Head |
| **StitchNet M4** | Encoder + CBAM + Semantic Head |

---

## Datasets

### ATLAS23 Dataset
- Modality: **MRI**
- Total volumes: **60**
- Training volumes: **48**
- Test volumes: **12**
- Total slices: **4744**

### IRCAD Dataset
- Modality: **CT**
- Total volumes: **20**
- Training volumes: **14**
- Test volumes: **6**

---

## Evaluation Metrics

The models are evaluated using standard medical image segmentation metrics:

- **Dice Similarity Coefficient (DSC)**
- **95th Percentile Hausdorff Distance (HD95)**

---

## Experimental Results

| Dataset | Best Dice Score |
|--------|----------------|
| **ATLAS23 (MRI)** | **94.35%** |
| **IRCAD (CT)** | **92.84%** |

StitchNet consistently outperforms conventional slice-wise baselines while maintaining significantly lower computational complexity compared to volumetric CNN architectures.

---

## Model Complexity

| Architecture | Parameters (M) | GFLOPs |
|-------------|---------------|-------|
| UNETR (3D) | 93.07 | 316.45 |
| UNet++ | 9.16 | 34.66 |
| Modified EfficientNet-B7 | 54.37 | 6.73 |
| StitchNet M3 | 57.36 | 7.85 |

StitchNet achieves **over 40× lower computational cost compared to UNETR (3D)** while delivering higher segmentation accuracy.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Manideep-Nanepalli/StitchNet.git
cd SFR/code
```
