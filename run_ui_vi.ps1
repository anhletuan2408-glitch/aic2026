param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [ValidateSet("fast", "balanced", "quality")]
    [string]$Mode = "quality",
    [int]$Port = 7860
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:HF_HOME = Join-Path $ProjectDir ".venv\model-cache\huggingface"
$env:SENTENCE_TRANSFORMERS_HOME = Join-Path $ProjectDir ".venv\model-cache\sentence-transformers"
$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$ModeArgs = @()
if ($Mode -eq "fast") {
    $ModeArgs = @("--no-rerank")
} elseif ($Mode -eq "balanced") {
    $ModeArgs = @("--reranker-model", "google/siglip2-base-patch16-224")
}
$ModeArgs += @("--query-ensemble")

& $PythonExe (Join-Path $ProjectDir "web_app_vi.py") `
    --index-dir (Join-Path $ProjectDir "index") `
    --zip-dir "E:\" `
    --device $Device `
    --port $Port `
    @ModeArgs
