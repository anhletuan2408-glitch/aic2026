# AIC 2026 multimedia retrieval assistant

Vietnamese-first retrieval and submission tooling for the three preliminary tasks: Textual KIS, visual Q&A, and TRAKE. The runtime is designed for a GPU with less than 4 GB VRAM and keeps the organizer data in `E:\AIC2026`.

See [PRELIMINARY.md](PRELIMINARY.md) for the complete pipeline, [UI_VI.md](UI_VI.md) for the Web UI, and [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) for Codabench packaging.

## Verified dataset

- 873 videos and 177,321 keyframes
- Organizer CLIP ViT-B/32 vectors: 512-dimensional `float16`
- Global SigLIP2 Large-384 index: 177,321 vectors, 1024 dimensions
- `npy[n - 1]` maps to image `NNN.jpg` and CSV keyframe number `n`
- Submissions use CSV `frame_idx`, not the JPG/keyframe number

## Current pipeline

```text
Vietnamese query
  -> task-aware rewrite / ordered-event split
  -> exact FAISS retrieval over organizer CLIP vectors
  + exact FAISS retrieval over global SigLIP2 vectors
  + exact FAISS retrieval over five overlapping SigLIP2 region crops
  + one bounded OCR wildcard (KIS only)
  -> Top-k-aware fusion and soft video diversity
  -> KIS: ranked <video_id>,<frame_idx>
  -> QA: full-frame/region candidates -> Qwen2.5-VL-3B 4-bit on context + crop
  -> TRAKE: per-event fusion -> same-video k-best dynamic programming
  -> validated headerless CSV files -> submission.zip
```

FAISS is the exact in-memory vector search engine. Replacing it with Qdrant, pgvector, or Chroma does not improve embedding quality or ranking accuracy by itself.

## Run

```powershell
cd E:\AIC2026
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
.\status.ps1
```

Open http://127.0.0.1:7860. The UI supports free queries and imported organizer-style query packages.

## Build and benchmark

```powershell
.\.venv\Scripts\python.exe build_index.py --data-dir data --output-dir index
.\.venv\Scripts\python.exe evaluate_suite.py ground_truth\local.jsonl outputs\predictions --output outputs\benchmark.json
```

The repository also contains `benchmark_kis_live.py`, `benchmark_candidates.py`, and `tune_dense_fusion.py` for ablation and ground-truth calibration.

## Current local evidence

The local set is tiny (10 KIS, 6 QA, 1 TRAKE), several ranges contain one frame, and it is not a qualification guarantee.

- KIS: `0.42` Final Score after CLIP/SigLIP2 calibration plus one OCR wildcard.
- QA candidate/frame diagnostic: `0.2333`; two of six exact local frames are in the top 100. Exact QA remains `0.0` until both retrieval and answer matching improve.
- TRAKE: `0.40` after global SigLIP2 event fusion and same-video dynamic programming.
- Full unit suite: 104 tests.

Object and metadata signals remain available, but their default fusion weights are zero because the current ablation showed that noisy unconditional fusion reduced the score.