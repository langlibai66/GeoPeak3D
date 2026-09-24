"""Render a continuous 3D trajectory comparison from per-frame world coordinates.

Input CSV: sample, gt_x/y/z, raw_x/y/z, ours_x/y/z (coordinates in metres).
Requires matplotlib and FFmpeg in addition to the project dependencies.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--subtitle", default="Continuous 3D localization")
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or args.fps <= 0:
        raise ValueError("Nonempty input and positive FPS required")
    samples = np.array([int(r["sample"]) for r in rows])
    if not np.all(np.diff(samples) == 1):
        raise ValueError("Samples must be consecutive and ordered")
    points = {k: np.array([[float(r[f"{k}_{a}"]) for a in "xyz"] for r in rows])
              for k in ("gt", "raw", "ours")}
    all_points = np.concatenate(list(points.values()))
    if not np.isfinite(all_points).all():
        raise ValueError("Coordinates must be finite")
    colors = {"gt": "#16984e", "raw": "#ca5261", "ours": "#276bc0"}
    markers = {"gt": "s", "raw": "^", "ours": "o"}
    labels = {"gt": "Ground truth", "raw": "Raw stGCF", "ours": "GeoPeak3D"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelcolor": "#394553", "text.color": "#243442"})
    matplotlib.rcParams["animation.ffmpeg_path"] = args.ffmpeg
    fig = plt.figure(figsize=(12, 9), dpi=120, facecolor="white")
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.76], projection="3d")
    fig.text(0.5, 0.956, "3D Localization Trajectories", ha="center", fontsize=23,
             fontfamily="DejaVu Serif", weight="bold")
    fig.text(0.5, 0.916, args.subtitle, ha="center", fontsize=12, color="#687482")
    handles = [Line2D([], [], color=colors[k], marker=markers[k], markerfacecolor="white",
                      markersize=7, linewidth=1.6, label=labels[k]) for k in points]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065),
               ncol=3, frameon=False, columnspacing=3)
    footer = fig.text(0.5, 0.034, "", ha="center", fontsize=12)
    lo, hi = all_points.min(0), all_points.max(0)
    span = np.maximum(hi - lo, 0.5)
    center = (lo + hi) / 2
    limits = np.stack([center - span * 0.56, center + span * 0.56], axis=1)
    for axis, name, lim in zip((ax.xaxis, ax.yaxis, ax.zaxis), "XYZ", limits):
        axis.set_label_text(f"{name} (m)")
        axis.labelpad = 12
        axis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_pane_color((0.985, 0.989, 0.993, 1))
        axis._axinfo["grid"].update(color=(0.83, 0.86, 0.89, 0.6), linewidth=0.6)
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
    ax.set_box_aspect(span)
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(labelsize=10, colors="#637180")
    artists = {}
    for k in ("raw", "gt", "ours"):
        line, = ax.plot([], [], [], color=colors[k], lw=1.1, alpha=0.5 if k == "raw" else 0.85,
                        linestyle="None" if k == "raw" else "-",
                        marker=markers[k], markersize=4, markerfacecolor="white", markevery=1)
        head, = ax.plot([], [], [], color=colors[k], marker=markers[k], markersize=8,
                        markeredgecolor="white", markeredgewidth=1)
        artists[k] = line, head
    errors = {k: np.linalg.norm(points[k] - points["gt"], axis=1) * 100 for k in ("raw", "ours")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, codec="libopenh264", bitrate=5000,
                         extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(args.output), dpi=120):
        for i in range(len(rows)):
            for k, (line, head) in artists.items():
                line.set_data_3d(*points[k][:i+1].T)
                head.set_data_3d(*points[k][i:i+1].T)
            footer.set_text(f"Frame {i+1:03d}/{len(rows)}     |     3D error: "
                            f"Raw {errors['raw'][i]:.1f} cm    ·    GeoPeak3D {errors['ours'][i]:.1f} cm")
            writer.grab_frame()
            if i == len(rows) - 1:
                fig.savefig(args.output.with_suffix(".png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
