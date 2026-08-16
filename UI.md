# Local keyframe review UI

The browser UI keeps the same CLIP-to-FAISS retrieval pipeline and adds human
verification before exporting a KIS answer file. Keyframes are read directly
from the supplied ZIP archives; they are not extracted in bulk.

## Install the small UI dependency

```powershell
cd E:\AIC2026
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
```

## Start the UI

```powershell
.\.venv\Scripts\python.exe web_app.py `
  --index-dir E:\AIC2026\index `
  --zip-dir E:\ `
  --device cpu `
  --host 127.0.0.1 `
  --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

Enter the Vietnamese query and, when possible, an English translation. Review
the keyframe grid, select likely answers, and click **Tải CSV đã chọn**. The
downloaded file contains headerless `<video_id>,<frame_id>` KIS rows.

The server binds to localhost by default. Do not use `--host 0.0.0.0` unless
you intentionally want other devices on the network to access it.

## Current scope

- exact FAISS `IndexFlatIP` search
- one model/index load per server process
- top 20/50/100 controls
- per-video/time diversification
- keyframe images streamed from ZIP
- manual selection and KIS CSV export

Opening and seeking the source video around `pts_time` is the next stage.
