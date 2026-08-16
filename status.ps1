param(
    [int]$Port = 7860,
    [int]$LogLines = 8,
    [string]$RuntimeDir = ""
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $RuntimeDir) {
    if (Test-Path (Join-Path $ProjectDir ".venv\Scripts\python.exe")) {
        $RuntimeDir = $ProjectDir
    } else {
        $RuntimeDir = "E:\AIC2026"
    }
}
$RuntimeDir = (Resolve-Path $RuntimeDir).Path

function Show-AicProcess([string]$Label, [string]$ScriptName) {
    $escaped = [regex]::Escape((Join-Path $RuntimeDir $ScriptName))
    $processes = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -match $escaped
    })
    if ($processes.Count -eq 0) {
        Write-Host "$($Label): DUNG"
    } else {
        $pids = ($processes | ForEach-Object { $_.ProcessId }) -join ", "
        Write-Host "$($Label): DANG CHAY (PID $pids)"
    }
}

Show-AicProcess "Web UI" "web_app_vi.py"
Show-AicProcess "OCR index" "ocr_index.py"

try {
    $health = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 5
    Write-Host ("Health: OK | frames={0} | OCR={1} | device={2}" -f `
        $health.vectors, $health.ocr_frames, $health.device)
} catch {
    Write-Host "Health: CHUA SAN SANG ($($_.Exception.Message))"
}

$latest = Get-ChildItem (Join-Path $RuntimeDir "outputs") -Filter "*.stdout.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 2
foreach ($log in $latest) {
    Write-Host ""
    Write-Host "--- $($log.Name) ---"
    Get-Content $log.FullName -Tail $LogLines -ErrorAction SilentlyContinue
}
