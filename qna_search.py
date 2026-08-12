from __future__ import annotations

import argparse
import gc
import io
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from submission import QAAnswer, write_qa_submission
from web_app import KeyframeStore
if TYPE_CHECKING:
    from web_app_vi import MultilingualFaissEngine
else:
    MultilingualFaissEngine = Any


QWEN_VL_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def clean_answer(value: str) -> str:
    value = value.strip().splitlines()[0].strip()
    value = re.sub(r"^(?:answer|trả lời)\s*:\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().strip('"').strip()
    if value.endswith((".", "!", "?")):
        value = value[:-1].rstrip()
    if not value:
        raise ValueError("VLM returned an empty answer")
    return value[:100]


def rank_qa_answers(
    rows: list[dict[str, object]],
    predictions: list[tuple[int, str]],
    max_answers: int = 100,
) -> list[QAAnswer]:
    cleaned = [(index, clean_answer(answer)) for index, answer in predictions]
    frequencies = Counter(answer for _, answer in cleaned)
    first_seen = {answer: pos for pos, (_, answer) in enumerate(cleaned)}
    answers = sorted(
        frequencies,
        key=lambda answer: (-frequencies[answer], first_seen[answer]),
    )
    ranked: list[QAAnswer] = []
    seen: set[tuple[str, int, str]] = set()

    def add(row: dict[str, object], answer: str) -> None:
        key = (str(row["video_id"]), int(row["frame_idx"]), answer)
        if key not in seen and len(ranked) < max_answers:
            seen.add(key)
            ranked.append(QAAnswer(*key))

    for index, answer in cleaned:
        add(rows[index], answer)
    for row in rows:
        for answer in answers:
            add(row, answer)
    return ranked


class QwenVLAnswerer:
    def __init__(self, device: str = "cuda", model_name: str = QWEN_VL_MODEL) -> None:
        self.device = device
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=192 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        if device == "cuda":
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization,
                device_map="cuda",
                dtype=torch.float16,
            ).eval()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, dtype=torch.float32
            ).eval()

    def answer(self, question: str, images: list[Image.Image]) -> str:
        content = [{"type": "image", "image": image} for image in images]
        content.append({
            "type": "text",
            "text": (
                "Answer the visual question using only the images. Return only "
                "the shortest exact answer, preferably 1-5 words, without an "
                f"explanation. Answer in the same language as the question. "
                f"Question: {question}"
            ),
        })
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=24
            )
        prompt_length = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )[0]
        return clean_answer(text)


def context_images(
    store: KeyframeStore, video_id: str, keyframe_no: int, radius: int = 1
) -> list[Image.Image]:
    images: list[Image.Image] = []
    for number in range(max(1, keyframe_no - radius), keyframe_no + radius + 1):
        try:
            payload = store.get_bytes(video_id, number)
        except (KeyError, FileNotFoundError):
            continue
        images.append(Image.open(io.BytesIO(payload)).convert("RGB"))
    return images


def search_qna(
    engine: MultilingualFaissEngine,
    store: KeyframeStore,
    answerer: QwenVLAnswerer,
    question: str,
    question_for_vlm: str | None = None,
    vlm_candidates: int = 12,
) -> list[QAAnswer]:
    rows = engine.search(
        question, 100, candidate_k=5000,
        per_video=3, min_time_gap=2.0, quality=False,
    )
    predictions: list[tuple[int, str]] = []
    for index, row in enumerate(rows[:vlm_candidates]):
        images = context_images(
            store, str(row["video_id"]), int(row["keyframe_no"])
        )
        if not images:
            continue
        try:
            predictions.append((
                index, answerer.answer(question_for_vlm or question, images)
            ))
        finally:
            for image in images:
                image.close()
    if not predictions:
        raise RuntimeError("VLM produced no Q&A predictions")
    return rank_qa_answers(rows, predictions)


def unload_retrieval_engine(engine: MultilingualFaissEngine) -> None:
    """Free retrieval model allocations before loading the 4-bit VLM."""
    if hasattr(engine, "model"):
        engine.model = None
    if getattr(engine, "hybrid", None) is not None and hasattr(engine.hybrid, "model"):
        engine.hybrid.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def answer_rows(
    rows: list[dict[str, object]],
    store: KeyframeStore,
    answerer: QwenVLAnswerer,
    question: str,
    vlm_candidates: int,
) -> list[QAAnswer]:
    predictions: list[tuple[int, str]] = []
    for index, row in enumerate(rows[:vlm_candidates]):
        images = context_images(store, str(row["video_id"]), int(row["keyframe_no"]))
        if not images:
            continue
        try:
            predictions.append((index, answerer.answer(question, images)))
        finally:
            for image in images:
                image.close()
    if not predictions:
        raise RuntimeError("Qwen-VL produced no Q&A predictions")
    return rank_qa_answers(rows, predictions)

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve frames and answer visual questions")
    parser.add_argument("question", help="Question text or path to a UTF-8 query file")
    parser.add_argument("--question-en", help="Optional English translation for the VLM")
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--zip-dir", type=Path, default=Path(".."))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--vlm-candidates", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = Path(args.question)
    question = path.read_text(encoding="utf-8-sig").strip() if path.is_file() else args.question
    from web_app_vi import MultilingualFaissEngine as Engine
    engine = Engine(args.index_dir, args.device)
    store = KeyframeStore(args.zip_dir)
    rows = engine.search(
        question, 100, candidate_k=5000,
        per_video=3, min_time_gap=2.0, quality=False,
    )
    unload_retrieval_engine(engine)
    del engine
    answerer = QwenVLAnswerer(args.device)
    answers = answer_rows(
        rows, store, answerer, args.question_en or question, args.vlm_candidates
    )
    write_qa_submission(args.output, answers)
    print(f"Wrote {len(answers)} Q&A rows to {args.output}")


if __name__ == "__main__":
    main()
