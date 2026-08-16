# Resumable OCR index

The OCR layer uses RapidOCR 3.x with ONNX Runtime on CPU. It does not consume the GPU memory reserved for CLIP, SigLIP2, and Qwen-VL.

Start or resume:

```powershell
cd E:\AIC2026
powershell -ExecutionPolicy Bypass -File .\run_ocr_index.ps1
```

Progress is committed every 25 frames to `index/ocr.sqlite3`. Restarting skips completed `global_id` values. The first pass prioritizes every fifth keyframe so the corpus gets broad partial coverage before the remaining frames are filled.

Monitor:

```powershell
Get-Content .\outputs\ocr-index.stdout.log -Tail 10
Invoke-RestMethod http://127.0.0.1:7860/api/health
```

Measured on this machine: two workers process about 2.0 frames/s; four workers are slower because ONNX Runtime already uses CPU threads internally. Expected time is roughly 5 hours for one-fifth coverage and 24-25 hours for all 177,321 frames.

The web search loads OCR FTS5 automatically when `index/ocr.sqlite3` exists. OCR candidates must satisfy token-coverage filtering before receiving an RRF weight, which prevents common logos or one-word overlaps from dominating visual queries.