<#
.SYNOPSIS
    Starts Caddy on a Windows host, serving the built SPA and proxying the API.

.DESCRIPTION
    Caddy obtains and renews the TLS certificate automatically, which requires
    port 80 to be reachable from the internet for the ACME challenge and a DNS
    record already pointing at this host.

    Certificates are stored under the *service account's* profile. Running this
    under a different account than the one that first issued them means a fresh
    issuance, so keep the account stable to avoid hitting Let's Encrypt rate limits.
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$SiteAddress = $(if ($env:SITE_ADDRESS) { $env:SITE_ADDRESS } else { 'visionedge.c-maron.space' }),
    [string]$CaddyExe = 'caddy.exe'
)

$ErrorActionPreference = 'Stop'

$dist   = Join-Path $RepoRoot 'frontend\dist'
$config = Join-Path $RepoRoot 'deploy\Caddyfile.windows'
$logDir = Join-Path $RepoRoot 'logs'

if (-not (Get-Command $CaddyExe -ErrorAction SilentlyContinue)) {
    throw "caddy.exe not found on PATH. Download it from https://caddyserver.com/download and place it on PATH."
}
if (-not (Test-Path (Join-Path $dist 'index.html'))) {
    throw "Frontend not built at $dist. Run: cd frontend; npm ci; npm run build  (do NOT create frontend\.env  -  an unset VITE_API_BASE is what makes the app use its own origin)"
}
if (-not (Test-Path $config)) {
    throw "Caddyfile not found at $config"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Caddy accepts forward slashes on Windows; backslashes are escapes in Caddyfile tokens.
$env:SITE_ADDRESS = $SiteAddress
$env:SITE_ROOT    = $dist.Replace('\', '/')
$env:SITE_LOG     = (Join-Path $logDir 'caddy.log').Replace('\', '/')

Write-Host "Serving $SiteAddress from $env:SITE_ROOT"
& $CaddyExe run --config $config
