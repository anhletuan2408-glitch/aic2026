# Batch KIS benchmark

`batch_kis.py` loads the CLIP text model once, encodes query groups in batches,
runs all query vectors through the exact FAISS index, maps results to source
frames, and applies the same KIS diversification logic as `search_kis.py`.

## Input format

One JSON object per line:

```json
{"query_id":"q001","query":"Vietnamese query","variants":["English query"]}
```

## Command

```powershell
.\.venv\Scripts\python.exe batch_kis.py queries.jsonl `
  --index-dir index `
  --output-dir outputs\batch_500 `
  --limit 500 `
  --encode-batch-size 64 `
  --candidate-k 5000 `
  --top-k 100 `
  --device cpu
```

The combined batch submission file includes `query_id` and `rank` columns for
testing. A competition adapter can split these rows into the exact submission
transport expected by the organizer.

## Measured local result

Tested with 500 distinct metadata-title queries against 177,321 supplied
keyframe vectors:

| Metric | Result |
|---|---:|
| Queries | 500 |
| Answers per query | 100 |
| Total answers | 50,000 |
| CLIP text encoding | 13.123 s |
| Exact FAISS search | 0.347 s |
| End-to-end total | 17.164 s |
| Throughput | 29.130 queries/s |

Validation confirmed that every query had ranks 1 through 100 and every output
row contained a valid `Lxx_Vxxx` video ID plus a non-negative source
`frame_idx`.

This is a throughput and format test, not an accuracy benchmark. Accuracy
requires organizer-style natural-language queries with ground-truth video and
frame ranges.
