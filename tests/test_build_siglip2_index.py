from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from build_siglip2_index import (
    FrameRecord, validate_contiguous_global_ids, write_faiss_index,
)


class Siglip2IndexBuilderTests(unittest.TestCase):
    def test_rejects_non_contiguous_global_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 1, got 2"):
            validate_contiguous_global_ids([
                FrameRecord(0, "L21_V001", 1),
                FrameRecord(2, "L21_V001", 2),
            ])

    def test_partial_index_preserves_global_ids(self) -> None:
        vectors = np.asarray([
            [1.0, 0.0],
            [0.0, 1.0],
            [.8, .2],
            [-1.0, 0.0],
        ], dtype=np.float16)
        completed = np.asarray([True, False, True, True])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keyframes.faiss"
            count = write_faiss_index(vectors, completed, path, chunk_size=2)
            index = faiss.read_index(str(path))
            _, ids = index.search(
                np.asarray([[1.0, 0.0]], dtype=np.float32), 3,
            )
        self.assertEqual(count, 3)
        self.assertEqual(index.ntotal, 3)
        self.assertEqual(ids[0].tolist(), [0, 2, 3])


if __name__ == "__main__":
    unittest.main()
