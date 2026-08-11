# Vietnamese retrieval UI

The UI accepts one Vietnamese query: either a short keyword or a long natural-language
description. It embeds the complete text with multilingual CLIP, searches the organizer
CLIP image vectors with exact FAISS inner-product search, then diversifies results.

## Run

```powershell
cd E:\AIC2026
powershell -ExecutionPolicy Bypass -File .\run_ui_vi.ps1
```

Open http://127.0.0.1:7860. Pass `-Device cuda` when CUDA is available.
The launcher reuses the model cache inside `.venv\model-cache`.

