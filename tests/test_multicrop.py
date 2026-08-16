from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from multicrop import (
    CROP_BOXES, collapse_crop_results, crop_image, decode_crop_id, encode_crop_id,
    load_multicrop_index,
)


class MultiCropTests(unittest.TestCase):
    def test_partial_manifest_is_ignored_until_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps({
                "model": "siglip", "frames": 10,
                "crop_count": len(CROP_BOXES), "completed": 3,
                "index_vectors": 0, "dimension": 8,
            }), encoding="utf-8")
            self.assertIsNone(load_multicrop_index(path, 10, "siglip"))

    def test_manifest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps({
                "model": "other", "frames": 10,
                "crop_count": len(CROP_BOXES), "completed": 0,
                "index_vectors": 0, "dimension": 8,
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_multicrop_index(path, 10, "siglip")

    def test_crop_layout_magnifies_overlapping_regions(self) -> None:
        image = Image.new("RGB", (100, 80))
        try:
            self.assertEqual(len(CROP_BOXES), 5)
            self.assertEqual(crop_image(image, 0).size, (60, 48))
            self.assertEqual(crop_image(image, 4).size, (60, 48))
        finally:
            image.close()

    def test_crop_id_round_trip(self) -> None:
        for global_id in (0, 7, 177320):
            for crop_index in range(len(CROP_BOXES)):
                self.assertEqual(
                    decode_crop_id(encode_crop_id(global_id, crop_index)),
                    (global_id, crop_index),
                )

    def test_collapse_keeps_best_crop_per_frame(self) -> None:
        ids = np.asarray([
            encode_crop_id(3, 2), encode_crop_id(3, 1),
            encode_crop_id(9, 4), encode_crop_id(1, 0),
        ])
        scores = np.asarray([.9, .8, .7, .6], dtype=np.float32)
        frames, frame_scores, crops = collapse_crop_results(ids, scores, 2)
        self.assertEqual(frames, [3, 9])
        np.testing.assert_allclose(frame_scores, [.9, .7])
        self.assertEqual(crops, {3: 2, 9: 4})


if __name__ == "__main__":
    unittest.main()
