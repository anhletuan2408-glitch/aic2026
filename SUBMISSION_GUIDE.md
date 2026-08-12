# Codabench preliminary submission

This repository follows the previous preliminary-round contract supplied by the
organizer. Confirm the 2026 contract when it is published before using a limited
submission attempt.

## Query files

Every query has one UTF-8 CSV file with no header and at most 100 rows:

```text
query-1-kis.csv     video_id,frame_idx
query-2-qa.csv      video_id,frame_idx,answer
query-3-trake.csv   video_id,frame_1,...,frame_N
```

Use integer `frame_idx`, never the keyframe JPG number, and omit `.mp4` from the
video ID. Q&A answers contain 1-100 characters. Every TRAKE row must contain the
requested number of events in strictly increasing temporal order.

## Scoring implications

For each query, the score is the mean of the best R-Score at ranks 1, 5, 20, 50,
and 100. A first correct row scores 1.0; a first hit at ranks 2-5 scores at most
0.8; ranks 6-20 score at most 0.6; ranks 21-50 score at most 0.4; and ranks
51-100 score at most 0.2.

There is no stated penalty for additional valid rows. Use the 100-row budget for
diverse, plausible candidates, while placing the strongest candidates first.

- KIS: diversify videos and timestamps after the strongest results.
- Q&A: the row must simultaneously match video, frame range, and exact answer.
  A few concise answer variants can be paired with strong frames, but variants
  consume ranking positions.
- TRAKE: partial R-Score is available for correctly aligned events in the correct
  video. Generate several strictly increasing sequences around strong moments.

## Validate and package

Put every result CSV in a working directory, then run:

```powershell
cd E:\AIC2026
.\.venv\Scripts\python.exe package_submission.py outputs\round1_csv outputs\team_round1.zip
```

The command rejects common format errors and creates the required archive shape:

```text
team_round1.zip
└── submission/
    ├── query-1-kis.csv
    ├── query-2-qa.csv
    └── query-3-trake.csv
```

Before uploading, run the packager again from the exact final CSV directory and
inspect its `OK` report. The previous rules allowed only three submissions per
query package and ranked the final attempt, so a format error is costly.
