param(
    [int]$PollSeconds = 60,
    [int]$Workers = 2,
    [int]$PriorityStride = 5,
    [int]$MaxRestarts = 6
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manifest = Join-Path $ProjectDir "index\siglip2-crops\manifest.json"
$CropIndex = Join-Path $ProjectDir "index\siglip2-crops\crops.faiss"
$Outputs = Join-Path $ProjectDir "outputs"
$BuilderName = "build_multicrop_siglip2_index.py"
$OcrName = "ocr_index.py"
$restarts = 0

function Find-PythonProcess([string]$ScriptName) {
    return @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -like "*$ScriptName*"
    })
}

Write-Host "Waiting for complete multi-crop manifest: $Manifest"
while ($true) {
    if (Test-Path -LiteralPath $Manifest) {
        try {
            $state = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
            $frames = [int]$state.frames
            $completed = [int]$state.completed
            $cropCount = [int]$state.crop_count
            $indexed = [int]$state.index_vectors
            $expectedVectors = $frames * $cropCount
            if (
                $frames -gt 0 -and $cropCount -gt 0 -and
                $completed -eq $frames -and $indexed -eq $expectedVectors -and
                (Test-Path -LiteralPath $CropIndex)
            ) {
                Write-Host "Multi-crop complete: $completed frames, $indexed crop vectors."
                break
            }
        } catch {
            Write-Warning "Manifest exists but is not complete/readable yet."
        }
    }
    if ((Find-PythonProcess $BuilderName).Count -eq 0) {
        if ($restarts -ge $MaxRestarts) {
            throw "Multi-crop failed $restarts times; OCR was not started. Check outputs logs."
        }
        $restarts += 1
        $batchSize = if ($restarts -le 2) { 2 } else { 1 }
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $stdout = Join-Path $Outputs "multicrop-recovery-$stamp.stdout.log"
        $stderr = Join-Path $Outputs "multicrop-recovery-$stamp.stderr.log"
        Write-Warning (
            "Multi-crop stopped; recovery $restarts/$MaxRestarts " +
            "with batch $batchSize."
        )
        Start-Process -FilePath "powershell.exe" -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            (Join-Path $ProjectDir "run_multicrop_index.ps1"),
            "-Device", "cuda", "-BatchSize", "$batchSize", "-FlushEvery", "200"
        ) -WindowStyle Hidden -RedirectStandardOutput $stdout `
          -RedirectStandardError $stderr | Out-Null
        Start-Sleep -Seconds 10
    }
    Start-Sleep -Seconds $PollSeconds
}

if ((Find-PythonProcess $OcrName).Count -gt 0) {
    Write-Host "OCR is already running; no duplicate process was started."
    exit 0
}

Write-Host "Resuming OCR with $Workers workers and priority stride $PriorityStride."
& (Join-Path $ProjectDir "run_ocr_index.ps1") `
    -Workers $Workers `
    -PriorityStride $PriorityStride
