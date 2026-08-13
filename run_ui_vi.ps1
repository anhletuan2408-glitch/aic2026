param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [ValidateSet("fast", "balanced", "quality")]
    [string]$Mode = "quality",
    [int]$Port = 7860,
    [int]$MaxRestarts = 5,
    [int]$RestartDelay = 5
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

$restart = 0
do {
    & $PythonExe (Join-Path $ProjectDir "web_app_vi.py") `
        --index-dir (Join-Path $ProjectDir "index") `
        --zip-dir "E:\" `
        --device $Device `
        --port $Port `
        @ModeArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        exit 0
    }
    $restart += 1
    if ($restart -gt $MaxRestarts) {
        Write-Error "Web UI stopped after $MaxRestarts restart attempts (exit $exitCode)."
        exit $exitCode
    }
    Write-Warning "Web UI exited with code $exitCode; restart $restart/$MaxRestarts in $RestartDelay seconds."
    Start-Sleep -Seconds $RestartDelay
} while ($true)
