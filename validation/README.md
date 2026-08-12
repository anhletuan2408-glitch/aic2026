# Organizer-style KIS validation

Create one JSON object per line. Ground truth must be annotated independently from
the retrieval models; do not generate it from CLIP, Objects, metadata, or SigLIP2.

```json
{"query_id":"manual_001","query":"Mô tả sự kiện bằng tiếng Việt","video_id":"L21_V001","start":500,"end":510,"round":"practice-b1"}
```

Run all three modes and retain a change only when it improves organizer-style
metrics without unacceptable latency regression:

```powershell
python evaluate_retrieval.py validation\queries.jsonl --mode baseline
python evaluate_retrieval.py validation\queries.jsonl --mode hybrid
python evaluate_retrieval.py validation\queries.jsonl --mode quality
```

Keep the `round` field so batch 1, batch 2, and batch 3 can be reported separately.
