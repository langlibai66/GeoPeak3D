"""Lightweight tests for the public GeoPeak3D API."""

import unittest

import torch

from geopeak3d import GeoPeak3D


class GeoPeak3DTest(unittest.TestCase):
    def test_forward_and_decoding(self) -> None:
        model = GeoPeak3D().eval()
        inputs = torch.rand(1, 1, 9, 24, 30)
        with torch.no_grad():
            logits, offsets = model(inputs)
            coordinates = model.continuous_coordinates(logits, offsets)

        self.assertEqual(logits.shape, (1, 1, 9, 24, 30))
        self.assertEqual(offsets.shape, (1, 3, 9, 24, 30))
        self.assertEqual(coordinates.shape, (1, 3))
        self.assertTrue(torch.isfinite(coordinates).all())


if __name__ == "__main__":
    unittest.main()
