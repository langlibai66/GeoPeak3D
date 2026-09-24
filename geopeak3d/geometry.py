"""AV16.3 calibration helpers used by the evaluation script."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


SEQUENCE_SESSIONS = {
    "seq01-1p-0000": "session08",
    "seq02-1p-0000": "session09",
    "seq03-1p-0000": "session08",
    "seq08-1p-0100": "session10",
    "seq11-1p-0100": "session09",
    "seq12-1p-0100": "session10",
}


def load_camera_bundle(calibration_root: str | Path, session: str) -> dict[str, np.ndarray]:
    session_dir = Path(calibration_root) / session
    camera = loadmat(session_dir / "cam.mat")["cam"][0]
    rigid = loadmat(session_dir / "rigid010203.mat")["rigid"][0]
    return {
        "Pmat": np.stack([camera[index][0] for index in range(3)]),
        "K": np.stack([camera[index][1] for index in range(3)]),
        "alpha_c": np.asarray([camera[index][2].item() for index in range(3)]),
        "kc": np.stack([camera[index][3].reshape(-1) for index in range(3)]),
        "align_mat": rigid[0][1],
    }


def undistort_pixels(pixels_xy: np.ndarray, intrinsic: np.ndarray, distortion: np.ndarray, alpha: float) -> np.ndarray:
    center = intrinsic[[0, 1], 2]
    focal = intrinsic[[0, 1], [0, 1]]
    distorted = ((pixels_xy - center) / focal).T
    distorted[0] -= alpha * distorted[1]
    k1, k2, p1, p2, k3 = np.pad(distortion.reshape(-1), (0, max(0, 5 - distortion.size)))[:5]
    points = distorted.copy()
    for _ in range(20):
        radius2 = np.sum(points * points, axis=0)
        radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        tangential = np.vstack(
            (
                2.0 * p1 * points[0] * points[1] + p2 * (radius2 + 2.0 * points[0] ** 2),
                p1 * (radius2 + 2.0 * points[1] ** 2) + 2.0 * p2 * points[0] * points[1],
            )
        )
        points = (distorted - tangential) / radial
    return intrinsic @ np.vstack((points, np.ones((1, len(pixels_xy)))))


def backproject_to_world(pixels_xy: np.ndarray, depths: np.ndarray, bundle: dict, camera_index: int) -> np.ndarray:
    rays = undistort_pixels(
        pixels_xy,
        bundle["K"][camera_index],
        bundle["kc"][camera_index],
        float(bundle["alpha_c"][camera_index]),
    )
    projected_xyz = rays * depths.reshape(1, -1)
    projection = bundle["Pmat"][camera_index]
    camera_xyz = np.linalg.inv(projection[:, :3]) @ (projected_xyz - projection[:, 3:4])
    world = bundle["align_mat"] @ np.vstack((camera_xyz, np.ones((1, len(depths)))))
    return world[:3].T
