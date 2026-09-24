"""Build matched Raw/GeoPeak3D demo coordinates from the same input volumes.

The fields NPZ contains raw (depth-collapsed normalized input), final (D,V,U
model predictions), and gt (D,V,U labels) for the selected continuous window.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from geopeak3d.data import DEPTH_CONFIG
from geopeak3d.geometry import SEQUENCE_SESSIONS, backproject_to_world, load_camera_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--calibration-root', type=Path, required=True)
    parser.add_argument('--fields', type=Path, required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--camera', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fields = np.load(args.fields)
    final = fields['final'].astype(np.float64)
    n = len(final)
    if n == 0 or args.start < 0:
        raise ValueError('Expected nonempty predictions and a nonnegative start')
    window = slice(args.start, args.start + n)
    seq_dir = args.data_root / args.sequence
    volume = np.array(np.load(seq_dir / 'gcf' / f'cam{args.camera}.npy', mmap_mode='r')[window])
    assert len(volume) == n
    lo, hi = volume.min((1, 2, 3), keepdims=True), volume.max((1, 2, 3), keepdims=True)
    normalized = (volume - lo) / np.maximum(hi - lo, 1e-8)
    np.testing.assert_allclose(normalized.max(1), fields['raw'], atol=1e-6, rtol=1e-6,
                               err_msg='Model input heatmap and baseline data differ')
    raw_dvu = np.stack(np.unravel_index(volume.reshape(n, -1).argmax(1), volume.shape[1:]), axis=1)
    scale = np.array([359 / 119, 287 / 95])
    raw_xy, ours_xy = raw_dvu[:, [2, 1]] * scale, final[:, [2, 1]] * scale
    gt_xy = np.load(seq_dir / 'labels/gt2d_mouth_xy.npy')[window, args.camera - 1]
    np.testing.assert_allclose(fields['gt'][:, [2, 1]] * scale, gt_xy, atol=1e-4)
    gt = np.load(seq_dir / 'labels/gt3d_xyz.npy')[window]
    bases, steps = DEPTH_CONFIG[args.sequence]
    base, step = bases[args.camera-1], steps[args.camera-1]
    bundle = load_camera_bundle(args.calibration_root, SEQUENCE_SESSIONS[args.sequence])
    raw = backproject_to_world(raw_xy, base + step * raw_dvu[:, 0], bundle, args.camera - 1)
    ours = backproject_to_world(ours_xy, base + step * final[:, 0], bundle, args.camera - 1)
    errors = {f'{key}_{dim}': np.linalg.norm(pred - target, axis=1)
              for key, pred3, pred2 in [('raw', raw, raw_xy), ('ours', ours, ours_xy)]
              for dim, pred, target in [('3d_m', pred3, gt), ('2d_px', pred2, gt_xy)]}
    rows = []
    for j in range(n):
        row = {'sample': args.start + j}
        for name, xyz in [('gt', gt), ('raw', raw), ('ours', ours)]:
            row.update({f'{name}_{axis}': float(xyz[j, k]) for k, axis in enumerate('xyz')})
        row.update({f'raw_{axis}': int(raw_dvu[j, k]) for k, axis in enumerate('dvu')})
        row.update({name: float(values[j]) for name, values in errors.items()})
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {'sequence': args.sequence, 'camera': args.camera, 'start': args.start,
               'end': args.start + n - 1, 'frames': n, 'input_dataset': args.data_root.name,
               'model_input_heatmap_verified': True,
               'means': {key: float(value.mean()) for key, value in errors.items()}}
    for dim in ['2d_px', '3d_m']:
        diff = errors[f'raw_{dim}'] - errors[f'ours_{dim}']
        summary[dim] = {'ours_better': int((diff > 1e-8).sum()),
                        'raw_better': int((diff < -1e-8).sum()),
                        'ties': int((abs(diff) <= 1e-8).sum())}
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
