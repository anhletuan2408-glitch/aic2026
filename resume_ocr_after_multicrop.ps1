param(
    [int]$PollSeconds = 60,
    [int]$Workers = 2,
    [int]$PriorityStride = 5
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manifest = Join-Path $ProjectDir "index\siglip2-crops\manifest.json"
$BuilderName = "build_multicrop_siglip2_index.py"
$OcrName = "ocr_index.py"

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
            $indexed = [int]$state.index_frames
            if ($frames -gt 0 -and $completed -eq $frames -and $indexed -eq $frames) {
                Write-Host "Multi-crop complete: $completed/$frames frames."
                break
            }
        } catch {
            Write-Warning "Manifest exists but is not complete/readable yet."
        }
    }
    if ((Find-PythonProcess $BuilderName).Count -eq 0) {
        throw "Multi-crop builder stopped before a complete manifest was written; OCR was not started."
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

