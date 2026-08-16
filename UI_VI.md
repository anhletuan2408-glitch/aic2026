# Vietnamese retrieval UI

## Start

```powershell
cd E:\AIC2026
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
.\status.ps1
```

Open http://127.0.0.1:7860.

- `Nhanh`: organizer CLIP retrieval for rapid exploration.
- `Chat luong`: global CLIP + SigLIP2 fusion, bounded OCR for KIS, and task-specific ranking.
- If the complete global SigLIP2 index is unavailable, quality mode falls back to on-demand SigLIP2 image reranking.
- The multi-crop index is loaded only when its manifest is complete; partial builds are safely ignored.

The status panel should show 177,321 CLIP vectors and 177,321 SigLIP2 vectors. OCR is resumable and may show a smaller number until indexing completes.

## Two workflows

### Free query

Use KIS for short keywords or long Vietnamese descriptions. For QA, select a retrieved image and ask Qwen directly, or use automatic QA to retrieve candidates and answer them sequentially. TRAKE accepts ordered events separated by numbered lines or semicolons.

### Imported data

Import a ZIP containing root-level UTF-8 `.txt` query files and correct the type (`KIS`, `QA`, or `TRAKE`) before starting. Press **Chạy tự động toàn bộ** to process every unfinished query sequentially and save up to 100 ranked rows per query.

Automatic results receive a task-specific confidence score. KIS uses the Top-1 margin and video support, QA uses answer consensus across distinct candidate frames, and TRAKE uses target-video/path consistency. Only low-confidence queries enter **Duyệt câu thiếu chắc chắn**, ordered from riskiest to safest. Running and saving a reviewed query marks it as human-verified; high-confidence queries remain auto-accepted. Export is allowed after every query has a valid result.

Suggested three-hour operating window:

1. Before 18:30: import data, verify task types, and start the automatic run.
2. 18:30-21:15: open the review queue and inspect Top-1/Top-5, QA answer, or temporal path only for flagged queries.
3. 21:15-21:30: ensure there are no automatic errors, export `submission.zip`, and inspect filenames/row counts.

The exported files have no header and use organizer frame IDs:

```text
KIS:   video_id,frame_idx
QA:    video_id,frame_idx,answer
TRAKE: video_id,frame_idx_1,...,frame_idx_N
```

## QA behavior

Automatic QA first retrieves 100 candidate frames using a scene-only rewrite, the original question, fair answer-type hypotheses, and best-region multi-crop search. It adds neighboring frames and the selected high-resolution crop around the candidates Qwen actually inspects, unloads retrieval models, and loads `Qwen/Qwen2.5-VL-3B-Instruct` in NF4 4-bit. Answers stay in the question language and are capped at 100 characters.

Selected-frame QA performs no hidden image search: Qwen receives only the image(s) selected by the user.

## Ground-truth benchmark

Store organizer-style UTF-8 JSONL in `E:\AIC2026\ground_truth\local.jsonl`. Use:

```powershell
.\.venv\Scripts\python.exe benchmark_kis_live.py E:\AIC2026\ground_truth\local.jsonl --base-url http://127.0.0.1:7860 --quality --hybrid --ocr --report E:\AIC2026\outputs\kis.json
.\.venv\Scripts\python.exe benchmark_candidates.py E:\AIC2026\ground_truth\local.jsonl --base-url http://127.0.0.1:7860 --force --report E:\AIC2026\outputs\qa-candidates.json
```

Current smoke evidence: KIS `0.42`, QA frame diagnostic `0.2333`, exact QA `0.0`, and TRAKE `0.40`. These numbers come from only 17 local queries and are not a qualification estimate.