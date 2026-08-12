# Vietnamese retrieval UI

The UI accepts one Vietnamese query: either a short keyword or a long natural-language
description. It embeds the complete text with multilingual CLIP, searches the organizer
CLIP image vectors with exact FAISS inner-product search, then diversifies results.

## Run

```powershell
cd E:\AIC2026
powershell -ExecutionPolicy Bypass -File .\run_ui_vi.ps1
```

Open http://127.0.0.1:7860. Use `-Mode fast` for exploration and `quality` for final ranked results.
The launcher reuses the model cache inside `.venv\model-cache`.

- `fast`: hybrid retrieval only, lowest latency.
- `balanced`: rerank 120 candidates with SigLIP2 Base-224.
- `quality` (default): rerank 300 candidates with SigLIP2 Large-384; measured peak was about 2.70 GB VRAM on this machine.
For live competition, launch the server in `quality` mode so the reranker is
loaded, then choose per query in the web UI:

- `Nhanh` (UI default): interactive exploration without SigLIP2 latency.
- `Chất lượng`: rerank 300 candidates before selecting/exporting answers.


## Three preliminary tasks

- Textual KIS: hybrid FAISS retrieval with optional SigLIP2 reranking.
- Visual Q&A has two flows: **Selected frame** asks Qwen directly without retrieval; **Auto search** retrieves 100, reranks, runs Qwen on the top 5/8/10 frames, and returns up to 100 consensus-ranked `<video_id>, <frame_idx>, <answer>` rows.
- TRAKE: splits ordered events and uses k-best dynamic programming to align increasing frames within one video.

The task tabs control the CSV schema automatically. In free-query mode, run KIS and click **Hỏi frame này** for direct Q&A. Imported QA queries use automatic search. Model swapping is serialized and stays within the 4 GB GPU target.
## Two operating modes

- Free query: keeps the original workflow for ad-hoc KIS, Q&A, and TRAKE searches and individual CSV downloads.
- Imported data: accepts a ZIP containing root-level UTF-8 `.txt` queries. Select KIS, Q&A, or TRAKE for each query, save its results, then export one validated `submission.zip` containing `submission/<query-name>.csv`. A `-kis`, `-qa`, or `-trake` suffix is only an initial suggestion and can be changed.

## Ground-truth benchmark

Create an organizer-style UTF-8 JSONL following `ground_truth/README.md`, place no-header predictions in one directory as `<query_id>.csv`, then run:

```powershell
.\.venv\Scripts\python.exe evaluate_suite.py ground_truth\preliminary.jsonl outputs\predictions --output outputs\benchmark_suite.json
```

The report contains R@1/5/20/50/100 and Final Score overall, per task, and per query. The 500-query metadata-title report is only a proxy throughput/ablation benchmark, not official accuracy.
## OCR fusion

Run `run_ocr_index.ps1` to build or resume the CPU RapidOCR FTS5 index while the CUDA web app remains available. See `OCR.md`. The System status panel reports the number of committed OCR frames. Exact/strong text matches are fused with CLIP, object, metadata, and SigLIP rankings.

For a real accuracy loop, select a visually verified result and click **Lưu làm Ground Truth**, then run `benchmark_live.py` as documented in `ground_truth/README.md`.