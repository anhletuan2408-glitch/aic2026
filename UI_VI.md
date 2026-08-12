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
- Visual Q&A: one click internally retrieves candidate frames, then Qwen2.5-VL-3B-Instruct NF4 answers every candidate and returns final `<video_id>, <frame_idx>, <answer>` rows.
- TRAKE: splits ordered events and aligns increasing frames within one video.

The task tabs control the CSV schema automatically. Q&A has no separate image-selection stage. Model swapping is serialized and may take roughly 1-2 minutes on a 4 GB RTX 3050 Ti.
## Two operating modes

- Free query: keeps the original workflow for ad-hoc KIS, Q&A, and TRAKE searches and individual CSV downloads.
- Imported data: accepts a ZIP containing root-level UTF-8 `.txt` queries. Select KIS, Q&A, or TRAKE for each query, save its results, then export one validated `submission.zip` containing `submission/<query-name>.csv`. A `-kis`, `-qa`, or `-trake` suffix is only an initial suggestion and can be changed.
