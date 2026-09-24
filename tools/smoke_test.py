#!/usr/bin/env python3
"""Minimal CPU check for the public model API."""

import torch

from geopeak3d import GeoPeak3D


def main() -> None:
    model = GeoPeak3D().eval()
    inputs = torch.rand(1, 1, 9, 96, 120)
    with torch.no_grad():
        logits, offsets = model(inputs)
        coordinates = model.continuous_coordinates(logits, offsets)
    assert logits.shape == (1, 1, 9, 96, 120)
    assert offsets.shape == (1, 3, 9, 96, 120)
    assert coordinates.shape == (1, 3)
    print(f"GeoPeak3D smoke test passed; predicted (d,v,u)={coordinates[0].tolist()}")


if __name__ == "__main__":
    main()
