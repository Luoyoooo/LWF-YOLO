# LWF-YOLO: A Lightweight Framework Based on YOLO for Blood Cell Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Base Model](https://img.shields.io/badge/Base-YOLOv11-green)](https://github.com/ultralytics/ultralytics)

> **Abstract:** Accurate detection of blood cells (RBCs, WBCs, and platelets) is essential for diagnosing hematological disorders. However, traditional approaches are labor-intensive, and many deep learning models are too computationally heavy for resource-constrained devices. **LWF-YOLO** is a lightweight, high-precision object detection framework built upon YOLOv11. It introduces novel edge-aware and dynamic channel-mixing mechanisms to achieve state-of-the-art performance with minimal computational overhead.

---

## 📖 Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Datasets](#datasets)
- [Citation](#citation)

---

## 🚀 Introduction

LWF-YOLO is designed to solve the trade-off between accuracy and efficiency in medical imaging. [cite_start]It specifically addresses the challenge of resolving overlapping cells and subtle morphological variations in dense cellular scenes[cite: 70, 72].

[cite_start]By prioritizing efficiency, LWF-YOLO enables real-time diagnostics on portable, resource-constrained devices suitable for point-of-care testing in underserved regions[cite: 140, 149].

---

## ✨ Key Features

1.  **Multi-Scale Edge-Aware Feature Enhancement (MSEFE):**
    * Leverages Sobel operators and multi-scale fusion to sharpen cell boundary delineation.
    * [cite_start]Directly addresses the separation of overlapping cells[cite: 73].
2.  **Dynamic Channel-Mixing Gated Former (DCGFormer):**
    * Replaces standard C3K2 modules in the backbone.
    * [cite_start]Employs identity mapping, dynamic channel splitting, and adaptive gating to preserve spatial details while reducing computational redundancy[cite: 73, 225].
3.  **Dynamic Deformable Convolution Network (DyDCN):**
    * Integrates DCNv4 adaptive sampling with multi-dimensional attention (scale, spatial, and task-aware).
    * [cite_start]Improves robust localization across diverse cell scales (from platelets to WBCs)[cite: 138, 226].

---

## 🧠 Architecture

The LWF-YOLO architecture consists of three main stages:

* [cite_start]**Backbone:** Integrates the **MSEFE** module for edge enhancement and **DCGFormer** blocks for efficient feature extraction[cite: 555].
* [cite_start]**Neck:** Uses a Path Aggregation Network (PANet) for multi-scale feature fusion[cite: 663].
* [cite_start]**Head:** A **DyDCN** detection head applied at each scale for precise object localization[cite: 664].

> *Note: Please refer to `Fig. 1` in the paper for the complete architectural diagram.*

---

## 📊 Performance

LWF-YOLO achieves state-of-the-art (SOTA) performance on standard blood cell detection benchmarks while maintaining a lightweight footprint.

### [cite_start]Comparison on BCCD Dataset [cite: 1426]

| Model | Parameters (M) | GFLOPs | mAP@50 (%) | mAP@50:95 (%) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| YOLOv8 | 3.01 | 8.10 | 89.00 | 60.80 | 129.39 |
| YOLOv9t | 1.97 | 7.60 | 91.30 | 62.10 | 60.29 |
| YOLOv11n (Baseline) | 2.58 | 6.30 | 90.20 | 61.00 | 119.45 |
| RT-DETR-R18 | 19.88 | 56.90 | 87.70 | 56.32 | 54.61 |
| **LWF-YOLO (Ours)** | **2.89** | **9.80** | **92.50** | **62.90** | **108.47** |

**Highlights:**
* **+2.3% mAP@50** improvement over the YOLOv11n baseline.
* [cite_start]Outperforms the larger RT-DETR-R18 while being nearly **7x smaller**[cite: 1534].
* [cite_start]Superior **RBC Recall (91%)** compared to baseline (85%), proving the effectiveness of the edge-aware design[cite: 1613].

### [cite_start]Generalization [cite: 1804, 1806]
* **CBC Dataset:** 95.70% mAP@50.
* **LISC Dataset:** 99.20% mAP@50 (Fine-grained WBC classification).
* **Br35H (Brain Tumor):** Validated cross-domain robustness on MRI data.

---

## 🛠️ Installation

**System Requirements:**
* [cite_start]OS: Linux (Tested on Ubuntu 22.04 LTS) [cite: 1296]
* GPU: NVIDIA (Tested on RTX A5000, 24GB VRAM)
* Python: 3.10.16

**Dependencies:**
```bash
pip install torch==2.5.1 torchvision
pip install ultralytics  # Based on YOLOv11
pip install opencv-python matplotlib****
📂 Datasets
This project utilizes the following public datasets. Please organize them in standard YOLO format (images/labels):


BCCD (Blood Cell Count and Detection): 364 images, 4,888 instances.


CBC (Complete Blood Count): 420 images, 5,672 instances.


LISC: 241 images, 5 classes of WBCs.

🧪 Experiments
Training
To train LWF-YOLO on the BCCD dataset (example configuration):

Bash
python train.py \
  --model lwf_yolo.yaml \
  --data bccd.yaml \
  --epochs 500 \
  --batch 32 \
  --img 640

Note: We use SGD optimizer with momentum 0.937 and weight decay 0.0005.

Ablation Studies
We conducted extensive ablation studies to validate our modules:

Baseline (YOLOv11): 90.20% mAP

+ MSEFE: 91.00% mAP (Improved edge detection)

+ DCGFormer: 91.10% mAP (Efficient feature extraction)

+ DyDCN: 91.10% mAP (Robust localization)

Full LWF-YOLO: 92.50% mAP

📜 Citation
If you use this code or findings in your research, please cite our paper:

代码段
@article{Mao2026LWFYOLO,
  title={LWF-YOLO: A Lightweight framework based YOLO for blood cell detection},
  author={Rui Mao and Dazhi Huang and Yuanyuan Wu and Biao Cai},
  journal={Author Submitted Manuscript},
  year={2026}
}
🤝 Acknowledgements
This work was supported by the Sichuan Science and Technology Program (No.2024ZYD0263).


---
