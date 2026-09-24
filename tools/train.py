#!/usr/bin/env python3
"""Train GeoPeak3D on STNet-generated stGCF volumes."""

from __future__ import annotations

import argparse
import csv
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from geopeak3d import GeoPeak3D
from geopeak3d.data import FrustumDataset, TRAIN_SEQUENCES, VAL_SEQUENCES, gaussian_volume_target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None, scaler=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "volume": 0.0, "coordinate": 0.0, "candidate": 0.0, "pixel": 0.0}
    seen = 0
    for batch in tqdm(loader, leave=False, desc="train" if training else "valid"):
        volume = batch["gcf"].to(device, non_blocking=True)
        target = batch["coordinate_dvu"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
            volume = (volume + 0.01 * torch.randn_like(volume)).clamp_(0.0, 1.0)
        context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with torch.set_grad_enabled(training), context:
            logits, offsets, candidates, candidate_logits = model(volume, return_aux=True)
            target_volume = gaussian_volume_target(target)
            log_probability = F.log_softmax(logits.flatten(1), dim=1).reshape_as(logits)[:, 0]
            volume_loss = -(target_volume * log_probability).sum(dim=(-3, -2, -1)).mean()
            prediction = model.continuous_coordinates(logits, offsets)
            scale = target.new_tensor((8.0, 95.0, 119.0))
            coordinate_loss = F.smooth_l1_loss(prediction / scale, target / scale, beta=0.03)
            target_xy = torch.stack((target[:, 2], target[:, 1]), dim=1)
            candidate_loss = model.candidate_loss(candidates, candidate_logits, target_xy)
            loss = volume_loss + 5.0 * coordinate_loss + 0.2 * candidate_loss
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        pixel_u = (prediction[:, 2] - target[:, 2]) * 359.0 / 119.0
        pixel_v = (prediction[:, 1] - target[:, 1]) * 287.0 / 95.0
        batch_size = len(volume)
        seen += batch_size
        for key, value in (("loss", loss), ("volume", volume_loss), ("coordinate", coordinate_loss), ("candidate", candidate_loss)):
            totals[key] += float(value.detach()) * batch_size
        totals["pixel"] += float(torch.sqrt(pixel_u.square() + pixel_v.square()).sum())
    return {key: value / seen for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/main"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader = DataLoader(
        FrustumDataset(args.data_root, TRAIN_SEQUENCES), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        FrustumDataset(args.data_root, VAL_SEQUENCES), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    model = GeoPeak3D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        checkpoint = {"epoch": epoch, "model_state": model.state_dict(), "args": vars(args), "best_val_pixel": min(best, val_metrics["pixel"])}
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if val_metrics["pixel"] < best:
            best, stale = val_metrics["pixel"], 0
            torch.save(checkpoint, args.output_dir / "best.pt")
        else:
            stale += 1
        print(f"epoch {epoch:02d} | val 2D MAE {val_metrics['pixel']:.2f} px")
        if stale >= args.patience:
            break
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    main()
