# Hybrid retrieval profiles

The competitive KIS path is deliberately multi-stage:

1. multilingual CLIP searches all organizer `clip-ViT-B-32` vectors;
2. multilingual E5 matches Vietnamese queries to OpenImages object labels and video metadata;
3. weighted reciprocal-rank fusion combines independent rankings;
4. SigLIP2 reranks real keyframe images read directly from ZIP archives;
5. temporal and per-video diversification produces at most 100 answers.

## Build one data batch

```powershell
python build_hybrid_index.py --metadata E:\AIC2026\index\metadata.sqlite3 --objects-zip E:\objects-aic25-b1.zip --output-dir E:\AIC2026\index\hybrid
```

Keep a manifest and an index directory for every preliminary batch. Rebuild the
combined FAISS/hybrid index after each organizer release, then run validation by
round before changing the production profile.

## Runtime profiles

```powershell
.\run_ui_vi.ps1 -Device cuda -Mode fast
.\run_ui_vi.ps1 -Device cuda -Mode balanced
.\run_ui_vi.ps1 -Device cuda -Mode quality
```

`quality` is the submission profile. `fast` is for query exploration.

## Measured diagnostics on batch 1

These are proxy diagnostics, not organizer accuracy:

| Proxy | Metric | CLIP baseline | Hybrid |
|---|---:|---:|---:|
| 500 metadata titles | video Hit@100 | 26.4% | 47.4% |
| 500 metadata titles | video Hit@50 | 20.2% | 26.8% |
| 150 object labels | frame Hit@1 | 34.7% | 64.7% |
| 150 object labels | frame Hit@100 | 81.3% | 100.0% |

SigLIP2 Large-384 plus multilingual CLIP peaked at about 2.25 GB VRAM on an
RTX 3050 Ti Laptop GPU. Batch 32 peaked at about 2.70 GB; quality mode reranks 300 candidates.

## Three-round discipline

- Freeze code, model names, weights, and manifests for each submission.
- Report validation metrics separately for each round/batch and combined.
- Never tune on metadata-title/object proxies alone.
- Maintain manually annotated organizer-style KIS queries with independent
  video IDs and frame ranges under `validation/`.
- Keep the previous passing profile so a regression can be rolled back quickly.
