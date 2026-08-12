# AIC 2026 Batch 1 - FAISS baseline

For the current competitive Vietnamese hybrid pipeline and its three runtime
profiles, see [HYBRID.md](HYBRID.md) and [UI_VI.md](UI_VI.md).

This project builds an exact cosine-similarity index from the organizer-provided
OpenAI CLIP ViT-B/32 keyframe vectors. It does not re-encode videos or keyframes.
The text encoder uses OpenCLIP's `ViT-B-32-quickgelu` model identifier because
the original OpenAI checkpoint uses QuickGELU.

## Verified dataset

- 873 videos
- 177,321 keyframe vectors
- Every feature file has shape `(N, 512)` and dtype `float16`
- Every feature row matches one `map-keyframes` CSV row
- `npy[n - 1]` maps to image `NNN.jpg` and CSV keyframe number `n`
- The submission frame is CSV `frame_idx`, not the JPG number

## Data layout

```text
data/
  clip-features-32/*.npy
  map-keyframes/*.csv
  media-info/*.json
```

The keyframe and video ZIP files can remain compressed while building the index.

## Pipeline

```text
Vietnamese query + English/query variants
  -> OpenAI CLIP ViT-B/32 QuickGELU text encoder
  -> normalized 512-dimensional query vector
  -> exact FAISS IndexFlatIP search
  -> SQLite global_id mapping
  -> limit near-duplicate results per video
  -> diagnostic ranked CSV + KIS submission CSV
```

## Environment

Create a dedicated virtual environment and install the CPU dependencies. A CUDA
PyTorch build can be substituted later if a compatible NVIDIA GPU is available.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`open_clip` downloads the OpenAI ViT-B/32 weights on the first query.

## Build the index

Run from this directory:

```powershell
.\.venv\Scripts\python.exe build_index.py --data-dir data --output-dir index
```

Expected raw vector memory in FAISS float32 format is approximately 0.338 GiB.

## Search

Include an English translation as a query variant because the original OpenAI
CLIP text encoder generally retrieves English wording more reliably.

```powershell
.\.venv\Scripts\python.exe search.py `
  "Một người đàn ông mặc áo xanh đang phát biểu trước đám đông" `
  --variant "A man in a blue shirt speaking in front of a crowd" `
  --top-k 100 `
  --output outputs\query_001.csv
```

Outputs:

- `outputs/query_001.csv`: scores and diagnostic metadata
- `outputs/query_001_submission.csv`: `<video_id>,<frame_idx>` rows for KIS

## Next improvements

1. Add keyframe previews without extracting every image.
2. Add temporal-neighbor diversification and video-level aggregation.
3. Rerank candidates with OCR, objects, metadata, or a vision-language model.
4. For Q&A, inspect a dense clip around each candidate timestamp.
5. For TRAKE, retrieve each event and enforce increasing time with dynamic
   programming before refining against dense frames from the source video.
