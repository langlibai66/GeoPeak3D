#!/usr/bin/env python3
"""Evaluate GeoPeak3D on AV16.3 sequences 08, 11, and 12."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from geopeak3d import GeoPeak3D
from geopeak3d.data import CAMERAS, FrustumDataset, TEST_SEQUENCES
from geopeak3d.geometry import SEQUENCE_SESSIONS, backproject_to_world, load_camera_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/test_metrics.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GeoPeak3D()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    loader = DataLoader(
        FrustumDataset(args.data_root, TEST_SEQUENCES), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    groups = defaultdict(lambda: {"pixels": [], "depths": [], "gt_xy": [], "gt_xyz": []})
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            volume = batch["gcf"].to(device, non_blocking=True)
            context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with context:
                logits, offsets = model(volume)
                coordinate = model.continuous_coordinates(logits, offsets).float().cpu().numpy()
            pixels = np.column_stack((coordinate[:, 2] * 359.0 / 119.0, coordinate[:, 1] * 287.0 / 95.0))
            depths = batch["depth_base"].numpy() + batch["depth_step"].numpy() * coordinate[:, 0]
            for index, sequence in enumerate(batch["sequence"]):
                group = groups[(sequence, int(batch["camera_index"][index]))]
                group["pixels"].append(pixels[index])
                group["depths"].append(float(depths[index]))
                group["gt_xy"].append(batch["xy_image"][index].numpy())
                group["gt_xyz"].append(batch["gt3d_xyz"][index].numpy())

    rows = []
    for sequence in TEST_SEQUENCES:
        bundle = load_camera_bundle(args.calibration_root, SEQUENCE_SESSIONS[sequence])
        for camera_index, camera in enumerate(CAMERAS):
            group = groups[(sequence, camera_index)]
            pixels = np.asarray(group["pixels"])
            depths = np.asarray(group["depths"])
            gt_xy = np.asarray(group["gt_xy"])
            gt_xyz = np.asarray(group["gt_xyz"])
            xyz = backproject_to_world(pixels, depths, bundle, camera_index)
            error_3d = np.linalg.norm(xyz - gt_xyz, axis=1)
            error_2d = np.linalg.norm(pixels - gt_xy, axis=1)
            rows.append(
                {
                    "sequence": sequence,
                    "camera": camera,
                    "samples": len(error_3d),
                    "3d_mae_cm": float(error_3d.mean() * 100.0),
                    "2d_mae_px": float(error_2d.mean()),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"3D MAE: {np.mean([row['3d_mae_cm'] for row in rows]):.1f} cm")
    print(f"2D MAE: {np.mean([row['2d_mae_px'] for row in rows]):.1f} px")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
