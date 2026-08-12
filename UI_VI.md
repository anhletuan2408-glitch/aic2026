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
- Visual Q&A: retrieves candidate frames first. The user selects 1-20 frames, then Qwen2.5-VL-3B-Instruct NF4 answers each selected frame separately before CSV export.
- TRAKE: splits ordered events and aligns increasing frames within one video.

The task tabs control the CSV schema automatically. Q&A CSV rows are created only after the user selects frames and runs Qwen. Model swapping is serialized and may take roughly 1-2 minutes on a 4 GB RTX 3050 Ti.