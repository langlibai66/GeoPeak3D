"""Dataset interface for STNet-generated stGCF volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


CAMERAS = ("cam1", "cam2", "cam3")
TRAIN_SEQUENCES = ("seq01-1p-0000", "seq02-1p-0000")
VAL_SEQUENCES = ("seq03-1p-0000",)
TEST_SEQUENCES = ("seq08-1p-0100", "seq11-1p-0100", "seq12-1p-0100")
IMAGE_SIZE = (360, 288)
MAP_SIZE = (120, 96)

# Per-sequence-camera optical-axis depth: z_d = base + step * d.
DEPTH_CONFIG = {
    "seq01-1p-0000": ((1.5, 2.75, 2.3), (0.35, 0.25, 0.35)),
    "seq02-1p-0000": ((1.5, 2.75, 2.3), (0.35, 0.25, 0.35)),
    "seq03-1p-0000": ((1.5, 2.75, 2.3), (0.35, 0.25, 0.35)),
    "seq08-1p-0100": ((1.5, 2.5, 2.1), (0.35, 0.25, 0.3)),
    "seq11-1p-0100": ((1.5, 2.5, 2.0), (0.35, 0.2, 0.3)),
    "seq12-1p-0100": ((1.5, 2.5, 2.0), (0.35, 0.2, 0.3)),
}


def image_to_map(xy: np.ndarray) -> np.ndarray:
    result = np.asarray(xy, dtype=np.float32).copy()
    result[..., 0] *= (MAP_SIZE[0] - 1) / (IMAGE_SIZE[0] - 1)
    result[..., 1] *= (MAP_SIZE[1] - 1) / (IMAGE_SIZE[1] - 1)
    return result


class FrustumDataset(Dataset):
    """Treat every sequence-camera frame as an independent training sample."""

    def __init__(self, root: str | Path, sequences: Iterable[str]) -> None:
        self.root = Path(root)
        self.sequences = tuple(sequences)
        if not self.sequences:
            raise ValueError("At least one sequence is required")
        self.gcf: list[list[np.ndarray]] = []
        self.xy: list[np.ndarray] = []
        self.xyz: list[np.ndarray] = []
        self.depths: list[np.ndarray] = []
        self.records: list[tuple[int, int, int]] = []

        for sequence_index, sequence in enumerate(self.sequences):
            sequence_dir = self.root / sequence
            gcfs = [
                np.load(sequence_dir / "gcf" / f"{camera}.npy", mmap_mode="r")
                for camera in CAMERAS
            ]
            xy = np.load(sequence_dir / "labels" / "gt2d_mouth_xy.npy", mmap_mode="r")
            xyz = np.load(sequence_dir / "labels" / "gt3d_xyz.npy", mmap_mode="r")
            depths = np.load(
                self.root / "depth_evaluation" / sequence / "gt_camera_depths_m.npy",
                mmap_mode="r",
            )
            expected = (len(xy), 9, MAP_SIZE[1], MAP_SIZE[0])
            if any(array.shape != expected for array in gcfs):
                raise ValueError(f"Unexpected stGCF shape in {sequence}; expected {expected}")
            self.gcf.append(gcfs)
            self.xy.append(xy)
            self.xyz.append(xyz)
            self.depths.append(depths)
            self.records.extend(
                (sequence_index, frame_index, camera_index)
                for frame_index in range(len(xy))
                for camera_index in range(len(CAMERAS))
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sequence_index, frame_index, camera_index = self.records[index]
        sequence = self.sequences[sequence_index]
        gcf = np.array(self.gcf[sequence_index][camera_index][frame_index], dtype=np.float32, copy=True)
        minimum, maximum = float(gcf.min()), float(gcf.max())
        gcf = (gcf - minimum) / max(maximum - minimum, 1e-8)
        xy_image = np.array(self.xy[sequence_index][frame_index, camera_index], dtype=np.float32, copy=True)
        xy_map = image_to_map(xy_image)
        depth_m = float(self.depths[sequence_index][frame_index, camera_index])
        bases, steps = DEPTH_CONFIG[sequence]
        depth_index = np.clip((depth_m - bases[camera_index]) / steps[camera_index], 0.0, 8.0)
        coordinate = np.array((depth_index, xy_map[1], xy_map[0]), dtype=np.float32)
        return {
            "gcf": torch.from_numpy(gcf).unsqueeze(0),
            "coordinate_dvu": torch.from_numpy(coordinate),
            "xy_image": torch.from_numpy(xy_image),
            "gt3d_xyz": torch.from_numpy(np.array(self.xyz[sequence_index][frame_index], dtype=np.float32, copy=True)),
            "depth_m": torch.tensor(depth_m, dtype=torch.float32),
            "depth_base": torch.tensor(bases[camera_index], dtype=torch.float32),
            "depth_step": torch.tensor(steps[camera_index], dtype=torch.float32),
            "sequence": sequence,
            "camera_index": camera_index,
            "frame_index": frame_index,
        }


def gaussian_volume_target(
    coordinates: torch.Tensor,
    shape: tuple[int, int, int] = (9, 96, 120),
    sigma: tuple[float, float, float] = (0.75, 2.0, 2.0),
) -> torch.Tensor:
    axes = [
        torch.arange(size, device=coordinates.device, dtype=coordinates.dtype)
        for size in shape
    ]
    grid_d, grid_v, grid_u = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack((grid_d, grid_v, grid_u), dim=0).unsqueeze(0)
    scale = coordinates.new_tensor(sigma).view(1, 3, 1, 1, 1)
    target = torch.exp(-0.5 * ((grid - coordinates[:, :, None, None, None]) / scale).square().sum(dim=1))
    return target / target.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)
