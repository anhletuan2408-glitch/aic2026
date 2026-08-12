# Ground-truth benchmark

Use one UTF-8 JSON object per line. Prediction files are named `<query_id>.csv` and use the exact no-header competition format.

```json
{"query_id":"kis-001","task":"kis","query":"mô tả khoảnh khắc","video_id":"L21_V001","start":100,"end":200}
{"query_id":"qa-001","task":"qa","query":"Có bao nhiêu người?","video_id":"L21_V001","start":100,"end":200,"answers":["3","Ba người"]}
{"query_id":"trake-001","task":"trake","query":"sự kiện 1; sự kiện 2","video_id":"L21_V001","moments":[[100,200],[300,400]]}
```

Run:

```powershell
.\.venv\Scripts\python.exe evaluate_suite.py ground_truth\preliminary.jsonl outputs\predictions --output outputs\benchmark_suite.json
```

A credible tuning set should contain organizer-style natural-language queries manually labelled against the videos, with all accepted QA answer strings. Metadata-title and model-generated labels are useful smoke tests but must not be reported as retrieval accuracy.