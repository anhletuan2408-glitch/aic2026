from __future__ import annotations

import sqlite3
import io
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from PIL import Image

from qna_search import (
    clean_answer, compose_qa_hypothesis_candidates, context_images,
    expand_qa_context_rows,
    fuse_qa_candidate_rows,
    rank_qa_answers,
)


class QnaSearchTests(unittest.TestCase):
    def test_context_images_adds_selected_region_after_full_frame(self) -> None:
        source = Image.new("RGB", (100, 80), "red")
        payload = io.BytesIO()
        source.save(payload, format="JPEG")
        source.close()

        class Store:
            def get_bytes(self, video_id: str, keyframe_no: int) -> bytes:
                if keyframe_no != 2:
                    raise KeyError(keyframe_no)
                return payload.getvalue()

        images = context_images(Store(), "V", 2, radius=1, crop_index=4)
        try:
            self.assertEqual(
                [image.size for image in images], [(100, 80), (60, 48)]
            )
        finally:
            for image in images:
                image.close()

    def test_hypothesis_composition_protects_visual_prefix_and_heads(self) -> None:
        def row(video: str, frame: int) -> dict[str, object]:
            return {"video_id": video, "frame_idx": frame}
        primary = [row("P", frame) for frame in range(1, 6)]
        hypotheses = [[row("H1", 1), row("H1", 2)], [row("H2", 1)]]
        reranked = [row("R", 1), row("P", 4), row("H1", 2)]
        output = compose_qa_hypothesis_candidates(primary, hypotheses, reranked, limit=8)
        self.assertEqual(
            [(item["video_id"], item["frame_idx"]) for item in output],
            [("P", 1), ("P", 2), ("P", 3), ("H1", 1),
             ("H2", 1), ("H1", 2), ("R", 1), ("P", 4)],
        )
        self.assertEqual([item["rank"] for item in output], list(range(1, 9)))

    def test_hypothesis_composition_round_robins_answer_depth(self) -> None:
        def row(video: str, frame: int) -> dict[str, object]:
            return {"video_id": video, "frame_idx": frame}
        hypotheses = [
            [row("one", index) for index in range(20)],
            [row("five", index) for index in range(20)],
        ]
        output = compose_qa_hypothesis_candidates(
            [row("scene", 1)], hypotheses, [], limit=15,
            primary_prefix=1, hypothesis_depth=6,
        )
        keys = [(item["video_id"], item["frame_idx"]) for item in output]
        self.assertEqual(keys[:5], [
            ("scene", 1), ("one", 0), ("five", 0),
            ("one", 1), ("five", 1),
        ])
        self.assertIn(("five", 5), keys)

    def test_context_expansion_preserves_prefix_then_adds_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.sqlite3"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "CREATE TABLE keyframes (video_id TEXT,keyframe_no INTEGER,"
                    "frame_idx INTEGER,pts_time REAL)"
                )
                connection.executemany(
                    "INSERT INTO keyframes VALUES (?,?,?,?)",
                    [("V", 1, 100, 1.0), ("V", 2, 200, 2.0),
                     ("V", 3, 300, 3.0), ("W", 1, 400, 4.0)],
                )
                connection.commit()
            rows = [
                {"video_id":"V", "keyframe_no":2, "frame_idx":200, "score":.9},
                {"video_id":"W", "keyframe_no":1, "frame_idx":400, "score":.8},
            ]
            expanded = expand_qa_context_rows(rows, path, selected_count=1)
            self.assertEqual(
                [(row["video_id"], row["frame_idx"]) for row in expanded],
                [("V", 200), ("V", 100), ("V", 300), ("W", 400)],
            )
            self.assertEqual([row["rank"] for row in expanded], [1, 2, 3, 4])
    def test_clean_answer_removes_wrapper(self) -> None:
        self.assertEqual(clean_answer('Answer: "Five people."'), "Five people")

    def test_candidate_fusion_adds_original_question_recall(self) -> None:
        primary = [
            {"video_id": "A", "frame_idx": 10},
            {"video_id": "B", "frame_idx": 20},
        ]
        secondary = [
            {"video_id": "C", "frame_idx": 30},
            {"video_id": "A", "frame_idx": 10},
        ]
        fused = fuse_qa_candidate_rows(primary, secondary)
        self.assertEqual((fused[0]["video_id"], fused[0]["frame_idx"]), ("A", 10))
        self.assertIn(
            ("C", 30),
            [(row["video_id"], row["frame_idx"]) for row in fused],
        )
        self.assertEqual([row["rank"] for row in fused], [1, 2, 3])

    def test_candidate_fusion_protects_quality_top_ten(self) -> None:
        primary = [
            {"video_id": f"P{index}", "frame_idx": index}
            for index in range(12)
        ]
        secondary = [
            {"video_id": f"S{index}", "frame_idx": index}
            for index in range(100)
        ]
        fused = fuse_qa_candidate_rows(primary, secondary, limit=20)
        self.assertEqual(
            [(row["video_id"], row["frame_idx"]) for row in fused[:10]],
            [(row["video_id"], row["frame_idx"]) for row in primary[:10]],
        )
    def test_consensus_answers_fill_ranked_frames_without_header(self) -> None:
        rows = [
            {"video_id": "L21_V001", "frame_idx": 100},
            {"video_id": "L21_V002", "frame_idx": 200},
            {"video_id": "L21_V003", "frame_idx": 300},
        ]
        answers = rank_qa_answers(
            rows, [(0, "Five"), (1, "Five"), (2, "Four")]
        )
        self.assertEqual(
            (answers[0].video_id, answers[0].frame_id, answers[0].answer),
            ("L21_V001", 100, "Five"),
        )
        self.assertIn(
            ("L21_V003", 300, "Five"),
            [(item.video_id, item.frame_id, item.answer) for item in answers],
        )

    def test_context_frame_inherits_its_source_answer_before_consensus(self) -> None:
        rows = [
            {"video_id":"V", "frame_idx":100},
            {"video_id":"W", "frame_idx":200},
            {"video_id":"V", "frame_idx":110, "context_of_rank":1},
        ]
        answers = rank_qa_answers(rows, [(0, "red"), (1, "blue")])
        triples = [
            (item.video_id, item.frame_id, item.answer) for item in answers
        ]
        self.assertLess(
            triples.index(("V", 110, "red")),
            triples.index(("V", 110, "blue")),
        )

if __name__ == "__main__":
    unittest.main()
