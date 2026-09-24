<div align="center">

# GeoPeak3D

### Geometry-Guided Multi-Peak Disambiguation for 3D Acoustic Source Localization

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-AV16.3-6f42c1.svg)](http://www.glat.info/ma/av16.3/)

[[Demo](#qualitative-demo)] · [[Data](#data-preparation)] · [[Training](#training)] · [[Evaluation](#evaluation)]

</div>

---

GeoPeak3D is an audio-only framework for single-source 3D localization in reverberant environments. Instead of committing to the strongest stGCF response, it preserves competing peaks, represents their weighted geometry on a calibrated view frustum, and learns to correct the acoustic response distribution. Inference uses microphone signals and camera calibration parameters—no RGB image is required.

## Highlights

- **Multi-peak reasoning:** layer-wise NMS retains four hypotheses at every depth.
- **Geometry-aware fusion:** a learned candidate field is fused with the complete stGCF volume.
- **Frustum-native encoder:** separate image-plane and ray-wise operators respect calibrated viewing geometry.
- **Continuous 3D output:** residual response correction and offset-aware expectation avoid voxel-center quantization.

```mermaid
flowchart LR
    A[stGCF volume<br/>9 × 96 × 120] --> B[Layer-wise NMS<br/>Top-4 per depth]
    B --> C[11-D candidate tokens<br/>global weighting]
    C --> D[Gaussian geometry field]
    A --> E[Dense response features]
    D --> F[Anisotropic frustum fusion]
    E --> F
    A --> G[Physics-referenced residual]
    F --> G
    G --> H[Offset-aware expectation]
    H --> I[Continuous 3D position]
```

## Qualitative demo

<div align="center">
  <a href="assets/demo.mp4">
    <img src="assets/demo-cover.svg" width="88%" alt="GeoPeak3D qualitative demo placeholder">
  </a>
  <br>
  <sub>Reserved for a qualitative demo. Add <code>assets/demo.mp4</code> or replace this link with a GitHub-hosted video URL.</sub>
</div>

## Repository layout

```text
GeoPeak3D/
├── geopeak3d/
│   ├── model.py        # complete GeoPeak3D architecture
│   ├── data.py         # AV16.3 stGCF dataset interface
│   └── geometry.py     # calibrated 3D back-projection
├── tools/
│   ├── train.py        # model training entry point
│   ├── evaluate.py     # sequence-camera 3D/2D evaluation
│   └── smoke_test.py   # lightweight model/API check
├── assets/             # README figures and demo-video slot
├── requirements.txt
└── LICENSE
```

This repository contains the core implementation only. Raw datasets, generated stGCF volumes, experiment logs, and checkpoints are intentionally excluded.

## Installation

```bash
git clone https://github.com/langlibai66/GeoPeak3D.git
cd GeoPeak3D

conda create -n geopeak3d python=3.10 -y
conda activate geopeak3d
pip install -e .
```

Verify the installation on CPU:

```bash
python tools/smoke_test.py
```

## Data preparation

1. Download AV16.3 sequences `01`, `02`, `03`, `08`, `11`, and `12`, together with `cam.mat` and `rigid010203.mat` calibration files.
2. Use the official [STNet repository](https://github.com/liyidi/STNet) to perform audio framing and generate the stGCF response volumes. Follow its `tools/prepareAudio.py`, `tools/prepare_gccphat.py`, and `GCF/stGCF.py` pipeline.
3. Arrange the generated arrays in the layout below. This repository does **not** redistribute AV16.3 or generated data.

```text
data/
├── seq01-1p-0000/
│   ├── gcf/
│   │   ├── cam1.npy          # [frames, 9, 96, 120]
│   │   ├── cam2.npy
│   │   └── cam3.npy
│   └── labels/
│       ├── gt2d_mouth_xy.npy # [frames, 3, 2], zero-based pixels
│       └── gt3d_xyz.npy      # [frames, 3], metres
├── seq02-1p-0000/
├── seq03-1p-0000/
├── seq08-1p-0100/
├── seq11-1p-0100/
├── seq12-1p-0100/
└── depth_evaluation/
    └── <sequence>/
        └── gt_camera_depths_m.npy  # [frames, 3]
```

The default split uses `seq01+seq02` for training, `seq03` for validation, and `seq08+seq11+seq12` for testing. Each camera view is treated as an independent sample.

## Training

The default configuration uses AdamW, learning rate `1e-3`, weight decay `1e-2`, batch size `32`, at most `15` epochs, and early-stopping patience `5`.

```bash
python tools/train.py \
  --data-root /path/to/data \
  --output-dir runs/geopeak3d \
  --batch-size 32 \
  --epochs 15 \
  --patience 5 \
  --seed 7
```

## Evaluation

`--calibration-root` must contain the AV16.3 `session08`, `session09`, and `session10` directories, each with `cam.mat` and `rigid010203.mat`.

```bash
python tools/evaluate.py \
  --data-root /path/to/data \
  --calibration-root /path/to/av16.3 \
  --checkpoint runs/geopeak3d/best.pt \
  --output results/test_metrics.csv
```

The evaluator writes per-sequence/per-camera 3D MAE (cm) and 2D MAE (px), then prints the nine-domain averages.

## Acknowledgements

Data preparation and stGCF construction build on [STNet](https://github.com/liyidi/STNet). The repository layout and concise train/test presentation are inspired by [MSMD-AVT](https://github.com/moyitech/MSMD-AVT).

## License

Released under the [MIT License](LICENSE).
