param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [int]$Port = 7860
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:HF_HOME = Join-Path $ProjectDir ".venv\model-cache\huggingface"
$env:SENTENCE_TRANSFORMERS_HOME = Join-Path $ProjectDir ".venv\model-cache\sentence-transformers"
$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"

& $PythonExe (Join-Path $ProjectDir "web_app_vi.py") `
    --index-dir (Join-Path $ProjectDir "index") `
    --zip-dir "E:\" `
    --device $Device `
    --port $Port
