# Vietnamese retrieval UI

The UI accepts one Vietnamese query: either a short keyword or a long natural-language
description. It embeds the complete text with multilingual CLIP, searches the organizer
CLIP image vectors with exact FAISS inner-product search, then diversifies results.

## Run

```powershell
cd E:\AIC2026
powershell -ExecutionPolicy Bypass -File .\run_ui_vi.ps1
```

For the recommended one-command startup (quality Web UI plus resumable CPU OCR):

```powershell
cd E:\AIC2026
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
.\status.ps1
```

`run_all.ps1` starts only missing services, writes timestamped logs under `outputs/`, and resumes `index/ocr.sqlite3` instead of rebuilding completed frames. When invoked from a source checkout without `.venv`, it automatically uses `E:\AIC2026` as the deployed runtime.

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
## Measured smoke benchmark (not organizer accuracy)

The local ground truth currently contains only 17 visually inspected smoke queries and several KIS/QA ranges are a single keyframe, so these numbers must not be presented as qualification probability or official accuracy.

- KIS: Final Score improved from `0.32` to `0.46` on 10 local queries; R@1 improved from `0.20` to `0.30` and R@5 from `0.30` to `0.40`.
- TRAKE: one three-event smoke query remains `0.20` (one of three moments matched in the top 20); the conditioned-video dynamic-programming pass did not regress this query.
- Automatic QA: use a visual-only candidate profile (multilingual CLIP + SigLIP2) because KIS object/metadata/OCR fusion polluted visual-question candidates. On the QR diagnostic, the correct frame moved from outside the submitted top 100 to candidate rank 6. Qwen answered that frame as `Đỏ`, while the conservative local label currently accepts only pink variants, so the official local QA score remains zero until the label is visually adjudicated.
- `evaluate_suite.py` reports `diagnostics.qa_frame` separately from the official QA score. This diagnostic does not alter organizer scoring; it only separates frame-retrieval failure from answer mismatch.

For the 4 GB GPU, the recommended live QA workflow remains: retrieve with KIS, manually choose a verified frame, then use Selected-frame Q&A. Automatic QA analyzes 10 visual candidates and is useful as a fallback, but currently takes about 2.5-3 minutes per query on this machine.