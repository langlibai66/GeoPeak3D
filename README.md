<div align="center">

# GeoPeak3D

### Geometry-Guided Multi-Peak Disambiguation for 3D Acoustic Source Localization

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-AV16.3-6f42c1.svg)](http://www.glat.info/ma/av16.3/)

[[Architecture](#architecture)] · [[Installation](#installation)] · [[Data](#data-preparation)] · [[Training](#training)] · [[Evaluation](#evaluation)] · [[Demo](#qualitative-demo)]

</div>

---

GeoPeak3D is an audio-only model for single-source 3D localization in reverberant environments. Given a calibrated stGCF volume, it retains competing response peaks, models their geometry on the view frustum, and predicts a continuous 3D source position. Inference requires microphone signals and camera calibration parameters, but no RGB images.

## Highlights

- **Multi-peak reasoning:** layer-wise NMS retains four hypotheses at every depth.
- **Geometry-aware fusion:** a learned candidate field is fused with the complete stGCF volume.
- **Frustum-native encoder:** separate image-plane and ray-wise operators respect calibrated viewing geometry.
- **Continuous 3D output:** residual response correction and offset-aware expectation avoid voxel-center quantization.

## Architecture

<div align="center">
  <a href="assets/network-architecture.pdf">
    <img src="assets/network-architecture.png" width="100%" alt="GeoPeak3D network architecture">
  </a>
  <br>
  <sub>GeoPeak3D network architecture. Click the figure to open the vector PDF.</sub>
</div>

## Qualitative demo

<div align="center">
  <img src="assets/demo-cover.svg" width="88%" alt="GeoPeak3D qualitative demo placeholder">
  <br>
  <sub>Qualitative results will be added here.</sub>
</div>

## Installation

```bash
git clone https://github.com/langlibai66/GeoPeak3D.git
cd GeoPeak3D

conda create -n geopeak3d python=3.10 -y
conda activate geopeak3d
pip install -e .
```

Optional sanity check:

```bash
python -m unittest discover -s tests -v
```

## Data preparation

This repository starts from precomputed stGCF volumes; raw-audio preprocessing is not included.

1. Download AV16.3 sequences `01`, `02`, `03`, `08`, `11`, and `12`, together with the `cam.mat` and `rigid010203.mat` calibration files.
2. Use [STNet](https://github.com/liyidi/STNet) to perform audio framing and generate the stGCF response volumes. The relevant upstream scripts are `tools/prepareAudio.py`, `tools/prepare_gccphat.py`, and `GCF/stGCF.py`.
3. Convert the generated outputs to NumPy arrays and arrange them as shown below. AV16.3 and generated data are not redistributed in this repository.

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

The default split is `seq01+seq02` for training, `seq03` for validation, and `seq08+seq11+seq12` for testing. Camera views are treated as independent samples.

## Training

The default configuration uses AdamW, learning rate `1e-3`, weight decay `1e-2`, batch size `32`, at most `15` epochs, and early-stopping patience `5`.

```bash
python tools/train.py \
  --data-root data \
  --output-dir runs/geopeak3d \
  --batch-size 32 \
  --epochs 15 \
  --patience 5 \
  --seed 7
```

Training writes `best.pt`, `latest.pt`, and `history.csv` to the selected output directory.

## Evaluation

`--calibration-root` must contain the AV16.3 `session08`, `session09`, and `session10` directories, each with `cam.mat` and `rigid010203.mat`.

```bash
python tools/evaluate.py \
  --data-root data \
  --calibration-root av16.3 \
  --checkpoint runs/geopeak3d/best.pt \
  --output results/test_metrics.csv
```

The evaluator writes per-sequence/per-camera 3D MAE (cm) and 2D MAE (px), then prints the nine-domain averages.

## Repository layout

```text
GeoPeak3D/
├── geopeak3d/
│   ├── model.py        # GeoPeak3D architecture
│   ├── data.py         # stGCF dataset interface
│   └── geometry.py     # calibrated 3D back-projection
├── tools/
│   ├── train.py        # training entry point
│   └── evaluate.py     # evaluation entry point
├── tests/
│   └── test_model.py   # lightweight model test
├── assets/             # architecture figure and demo assets
├── pyproject.toml
└── LICENSE
```

The repository contains the core implementation only. Datasets, generated stGCF volumes, experiment logs, and checkpoints are intentionally excluded.

## Acknowledgements

Data preparation and stGCF construction build on [STNet](https://github.com/liyidi/STNet).

## License

Released under the [MIT License](LICENSE).
