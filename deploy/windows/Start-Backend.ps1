<#
.SYNOPSIS
    Starts the VisionEdge Lab backend on a Windows host (no Docker).

.DESCRIPTION
    Binds to 127.0.0.1 only: Caddy is the sole public entry point, so the API is
    never reachable except through the proxy. That is stricter than the container
    deployment, where the backend at least sat on an internal network.

    Fails loudly on missing prerequisites rather than starting a server that would
    serve errors  -  a backend running without a model answers every request with a
    failure, which is worse than not starting.
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$SiteAddress = $(if ($env:SITE_ADDRESS) { $env:SITE_ADDRESS } else { 'visionedge.c-maron.space' }),
    [int]$RateLimitPerMin = 60
)

$ErrorActionPreference = 'Stop'

$python = Join-Path $RepoRoot 'backend\.venv\Scripts\python.exe'
$model  = Join-Path $RepoRoot 'models\yolov8n.onnx'

if (-not (Test-Path $python)) {
    throw "Python venv not found at $python. Create it first: py -3.12 -m venv backend\.venv"
}
if (-not (Test-Path $model)) {
    throw "Model not found at $model. Fetch it first: $python scripts\download_models.py --install yolov8n-onnx"
}

# Same-origin deployment: the browser never issues a cross-origin request. Set the
# origin anyway as defence in depth in case the port is ever exposed directly.
$env:VE_HOST                 = '127.0.0.1'
$env:VE_PORT                 = '8000'
$env:VE_LOG_JSON             = 'true'
$env:VE_CORS_ORIGINS         = "https://$SiteAddress"
$env:VE_RATE_LIMIT_PER_MIN   = "$RateLimitPerMin"
# Frames are never sent to a remote VLM unless this is deliberately enabled.
$env:VE_ALLOW_FRAME_TRANSMISSION = 'false'
$env:PYTHONPATH              = Join-Path $RepoRoot 'backend'

Set-Location (Join-Path $RepoRoot 'backend')

Write-Host "Starting backend on 127.0.0.1:8000 (rate limit ${RateLimitPerMin}/min, origin https://$SiteAddress)"
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
