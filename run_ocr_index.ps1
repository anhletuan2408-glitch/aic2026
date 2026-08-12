param(
    [int]$Workers = 2,
    [int]$PriorityStride = 5
)
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectDir '.venv\Scripts\python.exe'
& $PythonExe (Join-Path $ProjectDir 'ocr_index.py') `
    --metadata (Join-Path $ProjectDir 'index\metadata.sqlite3') `
    --zip-dir 'E:\' `
    --output (Join-Path $ProjectDir 'index\ocr.sqlite3') `
    --workers $Workers `
    --priority-stride $PriorityStride `
    --commit-every 25