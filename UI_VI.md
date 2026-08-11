# Vietnamese retrieval UI

The UI accepts one Vietnamese query: either a short keyword or a long natural-language
description. It embeds the complete text with multilingual CLIP, searches the organizer
CLIP image vectors with exact FAISS inner-product search, then diversifies results.

## Run

```powershell
cd E:\AIC2026
.\.venv\Scripts\python.exe web_app_vi.py --index-dir E:\AIC2026\index --zip-dir E:\
```

Open http://127.0.0.1:7860. The model downloads once on first use.

