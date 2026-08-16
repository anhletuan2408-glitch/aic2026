# Preliminary-round pipeline

The system handles all three AIC preliminary tasks while staying below the 4 GB VRAM target by never keeping SigLIP2 retrieval and Qwen-VL answer generation active on the GPU at the same time.

## Shared data and retrieval

```text
Organizer videos/keyframes/features
  -> metadata.sqlite3 (global_id, video_id, keyframe_no, frame_idx, time)
  -> organizer CLIP vectors -> exact FAISS IndexFlatIP
  -> SigLIP2 Large-384 keyframe encoding -> exact FAISS IndexIDMap2
  -> five overlapping SigLIP2 region crops per keyframe -> resumable FAISS index
  -> resumable OCR FTS5 index
```

At query time, Vietnamese text is embedded directly. CLIP and SigLIP2 rankings are fused in rank space. The calibrated KIS SigLIP2 weight is `1.5`. A soft diversity prefix covers multiple videos without deleting deep same-video recall.

### Why FAISS

FAISS is not the weak part of the system: both current indexes perform exact inner-product search. Qdrant, pgvector, and Chroma are useful for distributed serving, filtering, and persistence, but moving the same vectors into them will return the same neighbors. Accuracy work belongs in embeddings, query rewriting, signal calibration, reranking, and task-specific alignment.

## 1. Textual KIS

```text
full Vietnamese query + lightweight variants
  -> CLIP top 10,000
  + global SigLIP2 top 10,000
  -> calibrated rank fusion
  -> keep visual winner at rank 1
  -> optionally insert only the strongest OCR hit at rank 2
  -> soft per-video/time diversity for the first five rows
  -> preserve remaining dense order up to 100 answers
```

Object-label and metadata retrieval are retained for diagnostics but disabled by default. On the local ablation, dense CLIP/SigLIP2 scored `0.34`, unconditional OCR RRF scored `0.30`, and bounded OCR scored `0.42`.

## 2. Visual Q&A

QA is not a text-search result disguised as an answer. It has separate retrieval and answering phases:

```text
Vietnamese question
  -> remove the unknown answer slot to obtain a scene query
  + retain the original question
  + generate answer-type hypotheses (colors, counts, sports, headwear, names)
  -> retrieve 100 visual candidates
  -> fuse full-frame and best-region crop rankings
  -> fairly round-robin hypothesis rankings; do not bias the first answer class
  -> add temporal neighbor frames around the candidates Qwen will inspect
  -> unload retrieval models
  -> Qwen2.5-VL-3B-Instruct NF4 sees temporal context plus the selected crop
  -> clean answers to <=100 characters
  -> consensus-rank <video_id>,<frame_idx>,<answer> rows
```

The UI also supports selected-frame QA: the user chooses one or more retrieved images and Qwen answers only those images. This is the safest live workflow when automatic frame recall is uncertain.

The local benchmark reports frame recall separately from exact answer accuracy. This prevents a wrong frame from being mistaken for a Qwen reasoning failure.

Build or resume the crop representation offline with:

```powershell
.\run_multicrop_index.ps1 -Device cuda -BatchSize 6
```

Partial crop state is deliberately ignored by the Web UI. It becomes active only
after all frames and all five crops are indexed, preventing optimistic benchmarks
against an incomplete distractor set.

## 3. TRAKE

```text
ordered event description
  -> split into N events
  -> CLIP + global SigLIP2 retrieval for every event
  -> identify videos jointly supported by all events
  -> conditioned search over all indexed frames in each candidate video
  -> k-best dynamic programming with strictly increasing frame IDs
  -> export up to 100 same-video event paths
```

Every output row contains exactly N frame IDs in event order. The submitted IDs are `frame_idx` values.

## Submission and scoring

Each query file is headerless and contains at most 100 rows:

```text
KIS:   <video_id>,<frame_idx>
QA:    <video_id>,<frame_idx>,<answer>
TRAKE: <video_id>,<frame_idx_1>,...,<frame_idx_N>
```

The UI imports a ZIP of UTF-8 `.txt` queries, lets the user choose/correct each task type, saves ranked results, and exports `submission.zip` with `submission/<query-name>.csv`.

The scorer averages the best R-Score at ranks 1, 5, 20, 50, and 100. Therefore the pipeline explicitly protects rank 1 and rank 5 instead of maximizing recall alone.

## Reproduce local benchmarks

```powershell
# KIS live ablation
.\.venv\Scripts\python.exe benchmark_kis_live.py E:\AIC2026\ground_truth\local.jsonl --base-url http://127.0.0.1:7860 --quality --hybrid --ocr --report E:\AIC2026\outputs\kis.json

# QA frame-candidate recall without loading Qwen
.\.venv\Scripts\python.exe benchmark_candidates.py E:\AIC2026\ground_truth\local.jsonl --base-url http://127.0.0.1:7860 --force --report E:\AIC2026\outputs\qa-candidates.json

# Official-format aggregate
.\.venv\Scripts\python.exe evaluate_suite.py E:\AIC2026\ground_truth\local.jsonl E:\AIC2026\outputs\predictions --output E:\AIC2026\outputs\benchmark.json
```

Current smoke results are KIS `0.42`, QA frame diagnostic `0.2333`, exact QA `0.0`, and TRAKE `0.40`. The set is too small to estimate qualification probability; expand it before further weight tuning.