# AIC 2026 preliminary-round pipeline

This implementation follows the three preliminary-round query types and keeps
FAISS retrieval as the first-stage candidate generator.

## Shared retrieval pipeline

```text
Vietnamese query + English/query variants
  -> OpenAI CLIP ViT-B/32 QuickGELU text embedding
  -> L2-normalized 512-dimensional vector
  -> FAISS IndexFlatIP over 177,321 supplied keyframe vectors
  -> global_id lookup in SQLite
  -> map-keyframes frame_idx conversion
  -> task-specific reranking and output
```

The organizer-provided keyframe feature at array position `n - 1` maps to CSV
row `n`. The submitted frame is `frame_idx`, not the keyframe JPG number.

## Textual KIS

Run exact FAISS retrieval and export at most 100 ranked answers:

```powershell
.\.venv\Scripts\python.exe search_kis.py `
  "Một người đàn ông mặc áo xanh đang phát biểu trước đám đông" `
  --variant "A man in a blue shirt speaking in front of a crowd" `
  --index-dir index `
  --candidate-k 5000 `
  --top-k 100 `
  --per-video 3 `
  --min-time-gap 2.0 `
  --output outputs\query_001.csv
```

Outputs:

- `outputs/query_001.csv`: diagnostic ranking with similarity and metadata.
- `outputs/query_001_submission.csv`: `<video_id>,<frame_id>` rows.

Selection keeps the best candidate from different videos near the top, then
adds second and third candidates in later rounds. Frames closer than
`--min-time-gap` within the same video are treated as near-duplicates.

## Q&A

Required output:

```text
<video_id>,<frame_id>,<answer>
```

Planned task-specific stage:

1. Use the same FAISS retrieval to find candidate videos and timestamps.
2. Decode a dense clip around each candidate timestamp.
3. Run a vision-language model over the clip and question.
4. Normalize Vietnamese/English answers and rerank by retrieval plus answer
   confidence.
5. Export at most 100 ranked rows.

## TRAKE

Required output:

```text
<video_id>,<frame_id_1>,...,<frame_id_n>
```

Planned task-specific stage:

1. Split the query into ordered semantic events.
2. Encode every event and retrieve candidates with the same FAISS index.
3. Aggregate event evidence to rank a single video.
4. Enforce increasing timestamps with dynamic programming.
5. Decode dense frames around every chosen keyframe for sub-10-frame
   alignment.
6. Export at most 100 ranked event sequences.

## Local scoring

`submission.py` implements KIS, Q&A, and TRAKE R-Scores plus the official
average of `R@1`, `R@5`, `R@20`, `R@50`, and `R@100`.

Ground-truth JSON examples:

```json
{"video_id": "L01_V001", "start": 500, "end": 510}
```

```json
{
  "video_id": "L05_V005",
  "start": 800,
  "end": 900,
  "answers": ["màu xanh", "blue"]
}
```

```json
{
  "video_id": "L10_V010",
  "moments": [[95, 105], [145, 155], [195, 205], [245, 255]]
}
```

Score one ranked CSV:

```powershell
.\.venv\Scripts\python.exe evaluate.py kis `
  outputs\query_001_submission.csv ground_truth\query_001.json
```

The official-mode local Q&A scorer uses exact answer matching against the accepted
ground-truth strings. See [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) and run
`package_submission.py` before using a Codabench attempt.
