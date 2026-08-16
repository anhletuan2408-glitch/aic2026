param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [int]$BatchSize = 6,
    [int]$FlushEvery = 600
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeDir = if (Test-Path (Join-Path $ProjectDir ".venv\Scripts\python.exe")) {
    $ProjectDir
} else {
    "E:\AIC2026"
}
$PythonExe = Join-Path $RuntimeDir ".venv\Scripts\python.exe"
& $PythonExe (Join-Path $ProjectDir "build_multicrop_siglip2_index.py") `
    --metadata (Join-Path $RuntimeDir "index\metadata.sqlite3") `
    --zip-dir "E:\" `
    --output-dir (Join-Path $RuntimeDir "index\siglip2-crops") `
    --device $Device `
    --batch-size $BatchSize `
    --flush-every $FlushEvery
