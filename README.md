# LWF-YOLO

LWF-YOLO is a lightweight object detector for blood cell detection built on top of the Ultralytics YOLO11n codebase.
This repository reorganizes the research code into a clean, standalone project that can be published as an academic
open-source release while preserving runtime compatibility with the original Ultralytics package structure.

## Highlights

- **MSEFE**: multi-scale edge-aware feature enhancement for crowded cell boundaries.
- **DCGFormer**: a dynamic channel-mixing gated module that replaces the standard `C3k2` blocks.
- **DyDCN**: a deformable-convolution detection head that combines DCNv3 offset/mask generation with DyHead-style
  multi-dimensional attention.
- **YOLO11n-based**: the custom architecture is defined on top of the YOLO11n design.
- **GitHub-ready layout**: packaging, licensing, configs, weights, and docs have been separated into a standard
  repository structure.

## Repository Layout

```text
LWF-YOLO/
|-- ultralytics/                 # packaged runtime code
|-- configs/
|   |-- models/lwf-yolo.yaml     # easy-to-discover model config mirror
|   `-- datasets/bccd.yaml.example
|-- docs/
|   `-- custom_modules.md        # mapping from paper modules to source files
|-- scripts/
|   `-- build_dcnv3.sh           # helper for compiling the DCNv3 extension
|-- weights/
|   `-- README.md   # weights are not tracked; train to reproduce
|-- pyproject.toml
`-- LICENSE
```

## Custom Modules

| Paper module | Main implementation | Notes |
| --- | --- | --- |
| `MSEFE` | `ultralytics/nn/modules/lwf_modules.py` | Implemented by `SobelConv`, `ScaleEdge`, `EdgeFusion`, and `GetIndexOutput`. |
| `DCGFormer` | `ultralytics/nn/modules/lwf_modules.py` | `DCGFormerBlock` provides the gated channel-mixing core and is wrapped into the YOLO backbone blocks. |
| `DyDCN` | `ultralytics/nn/modules/lwf_modules.py` | `DyDCNBlock` is used inside `Detect_DyDCN` and relies on the bundled DCNv3 extension code. |

The architecture entrypoint is:

- `ultralytics/cfg/models/11/lwf-yolo.yaml`

A mirrored copy is also provided at:

- `configs/models/lwf-yolo.yaml`

## Installation

This project is distributed as a standalone repository, but the runtime package name remains `ultralytics` for code
compatibility with the original YOLO training and inference pipeline.

```bash
git clone <your-github-url>/LWF-YOLO.git
cd LWF-YOLO
python -m pip install -e .
```

### Build the DCNv3 extension

`DyDCN` depends on the custom DCNv3 operators bundled under `ultralytics/nn/modules/ops_dcnv3`.

```bash
bash scripts/build_dcnv3.sh
```

If you only need to inspect the model code or run non-DyDCN components, you can postpone this step.

## Quick Start

### Train

Use the provided template dataset file and adapt the dataset path to your machine:

```bash
lwf-yolo detect train \
  model=lwf-yolo.yaml \
  data=configs/datasets/bccd.yaml.example \
  epochs=300 \
  imgsz=640 \
  batch=16
```

### Python inference

```python
from ultralytics import YOLO

model = YOLO("weights/BCCD/best.pt")
results = model("path/to/image.jpg")
```

### Load the model definition from scratch

```python
from ultralytics import YOLO

model = YOLO("lwf-yolo.yaml")
```

## Pretrained Weights

Pretrained weights are not stored in this repository
(excluded via `.gitignore`). To reproduce the results,
train from scratch using the provided configs and the
public datasets below.

## Datasets

All datasets used in this paper are publicly available:

| Dataset | Task | Download |
|---------|------|----------|
| BCCD | RBC / WBC / Platelet detection | https://github.com/Shenggan/BCCD_Dataset |
| CBC | RBC / WBC / Platelet detection | https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset |
| LISC | WBC subtype classification | https://universe.roboflow.com/wbcs/wbc-lisc |
| Br35H | Brain tumor detection (cross-domain) | https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection |

After downloading, place each dataset under `data/<DATASET>/`
and update the corresponding yaml config.
A template is provided at `configs/datasets/bccd.yaml.example`.

## Results on BCCD

| Model | Params (M) | GFLOPs | mAP@50 (%) | FPS |
|-------|-----------|--------|------------|-----|
| YOLOv5 | 2.50 | 7.10 | 90.70 | 120.93 |
| YOLOv8 | 3.01 | 8.10 | 89.00 | 129.39 |
| YOLOv11n | 2.58 | 6.30 | 90.20 | 119.45 |
| RT-DETR-R18 | 19.88 | 56.90 | 87.70 | 56.32 |
| TE-YOLOF | 16.76 | 6.60 | 91.90 | — |
| YOLO-FMS | 15.06 | — | 92.50 | — |
| **LWF-YOLO (Ours)** | **2.89** | **9.80** | **92.50** | **108.47** |

> Full results on CBC, LISC, and Br35H datasets are reported
> in the paper.

## License and Upstream Base

This repository includes both original LWF-YOLO research code and the upstream Ultralytics code required to run it.
Because it is a derivative work of the Ultralytics AGPL codebase, the repository is released under **AGPL-3.0**.

The clean extraction notes are documented in `docs/custom_modules.md`.

## Citation

If you use this repository in academic work, please cite both:

- the LWF-YOLO paper
- the upstream Ultralytics project that provides the base training and inference framework

An initial `CITATION.cff` is included and can be updated with the final paper metadata before public release.

```bibtex
@article{mao2025lwfyolo,
  title   = {LWF-YOLO: A Lightweight Framework Based YOLO
             for Blood Cell Detection},
  author  = {Mao, Rui and Huang, Dazhi and
             Wu, Yuanyuan and Cai, Biao},
  journal = {Biomedical Physics \& Engineering Express},
  year    = {2025},
  note    = {Under review. DOI to be updated upon acceptance.}
}
```
