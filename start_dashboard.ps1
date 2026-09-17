param(
    [string]$PythonPath = 'C:\Users\skana\dev\trading-AI\.venv\Scripts\python.exe',
    [int]$Port = 8504
)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskLogs = Join-Path $taskRoot 'work\live'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'PythonPath에 기존 Python 가상환경 경로를 지정하세요.' }
$env:TRADING_AI_DATA_DIR = Join-Path $taskRoot 'data\operational\spot'
$taskWorkerPidPath = Join-Path $taskLogs 'worker.pid'
$taskWorker = $null
if (Test-Path -LiteralPath $taskWorkerPidPath) {
    $taskWorker = Get-Process -Id ([int](Get-Content -LiteralPath $taskWorkerPidPath)) -ErrorAction SilentlyContinue
    if ($taskWorker -and $taskWorker.Path -ne $PythonPath) { throw '저장된 worker PID가 다른 프로세스를 가리킵니다.' }
}
if (-not $taskWorker) {
    $taskWorker = Start-Process -FilePath $PythonPath -ArgumentList @('-u', 'refresh_live_data.py', '--watch') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs 'worker.log') -RedirectStandardError (Join-Path $taskLogs 'worker-error.log')
    $taskWorker.Id | Set-Content -LiteralPath $taskWorkerPidPath
}
try { Invoke-WebRequest -Uri "http://localhost:$Port/_stcore/health" -TimeoutSec 2 | Out-Null }
catch {
    $taskServer = Start-Process -FilePath $PythonPath -ArgumentList @('-m', 'streamlit', 'run', 'dashboard/app.py', '--server.headless', 'true', '--server.address', '127.0.0.1', '--server.port', "$Port") -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs 'dashboard.log') -RedirectStandardError (Join-Path $taskLogs 'dashboard-error.log')
    $taskServer.Id | Set-Content -LiteralPath (Join-Path $taskLogs 'dashboard.pid')
}
Write-Output "대시보드: http://localhost:$Port / 수집 로그: work/live/worker.log"
