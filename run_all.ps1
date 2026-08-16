param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [ValidateSet("fast", "balanced", "quality")]
    [string]$Mode = "quality",
    [int]$Port = 7860,
    [int]$OcrWorkers = 2,
    [int]$OcrPriorityStride = 5,
    [string]$RuntimeDir = ""
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $RuntimeDir) {
    if (Test-Path (Join-Path $ProjectDir ".venv\Scripts\python.exe")) {
        $RuntimeDir = $ProjectDir
    } else {
        $RuntimeDir = "E:\AIC2026"
    }
}
$RuntimeDir = (Resolve-Path $RuntimeDir).Path
if (-not (Test-Path (Join-Path $RuntimeDir ".venv\Scripts\python.exe"))) {
    throw "Khong tim thay Python runtime tai $RuntimeDir"
}
$OutputDir = Join-Path $RuntimeDir "outputs"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Find-AicProcess([string]$ScriptName) {
    $escaped = [regex]::Escape((Join-Path $RuntimeDir $ScriptName))
    return @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -match $escaped
    })
}

function Start-AicService(
    [string]$Name,
    [string]$Launcher,
    [string[]]$LauncherArgs,
    [string]$PythonScript
) {
    $running = Find-AicProcess $PythonScript
    if ($running.Count -gt 0) {
        Write-Host "$Name dang chay (PID $($running[0].ProcessId)); bo qua."
        return
    }

    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdout = Join-Path $OutputDir "$Name-$stamp.stdout.log"
    $stderr = Join-Path $OutputDir "$Name-$stamp.stderr.log"
    $processArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $RuntimeDir $Launcher)
    ) + $LauncherArgs
    $process = Start-Process powershell.exe `
        -ArgumentList $processArgs `
        -WorkingDirectory $RuntimeDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
    Set-Content -Path (Join-Path $OutputDir "$Name.pid") -Value $process.Id
    Write-Host "$Name da khoi dong (launcher PID $($process.Id))."
    Write-Host "  stdout: $stdout"
    Write-Host "  stderr: $stderr"
}

Start-AicService `
    -Name "ocr-index" `
    -Launcher "run_ocr_index.ps1" `
    -LauncherArgs @("-Workers", "$OcrWorkers", "-PriorityStride", "$OcrPriorityStride") `
    -PythonScript "ocr_index.py"

Start-AicService `
    -Name "web-ui" `
    -Launcher "run_ui_vi.ps1" `
    -LauncherArgs @("-Device", $Device, "-Mode", $Mode, "-Port", "$Port") `
    -PythonScript "web_app_vi.py"

Write-Host ""
Write-Host "Runtime: $RuntimeDir"
Write-Host "Mo UI: http://127.0.0.1:$Port/"
Write-Host "Kiem tra: .\status.ps1 -Port $Port"
