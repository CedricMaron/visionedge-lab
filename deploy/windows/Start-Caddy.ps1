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
    [string]$PortfolioAddress = 'c-maron.space',
    # C:\inetpub\c-maron is where c-maron's scripts\deploy.ps1 publishes. NOT
    # C:\inetpub\wwwroot - that is the IIS default and holds a stale build of a
    # different site. This value is exported as PORTFOLIO_ROOT and therefore
    # overrides the Caddyfile's own default, so it has to be right here too.
    [string]$PortfolioRoot = 'C:\inetpub\c-maron',
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

if (-not (Test-Path $PortfolioRoot)) {
    throw "Portfolio root not found at $PortfolioRoot. Confirm it in IIS Manager (Default Web Site > Basic Settings > Physical path) and pass -PortfolioRoot."
}

# Caddy needs 80 and 443. If IIS still holds them, ACME fails and the portfolio
# would go down without this site coming up - check before starting, not after.
foreach ($port in 80, 443) {
    $holder = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
              Select-Object -First 1
    if ($holder) {
        $proc = Get-Process -Id $holder.OwningProcess -ErrorAction SilentlyContinue
        $name = if ($proc) { $proc.ProcessName } else { "PID $($holder.OwningProcess)" }
        if ($name -notmatch 'caddy') {
            throw "Port $port is already held by '$name'. Release it first (see docs/DEPLOY_WINDOWS.md): Stop-Service W3SVC"
        }
    }
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Caddy accepts forward slashes on Windows; backslashes are escapes in Caddyfile tokens.
$env:SITE_ADDRESS      = $SiteAddress
$env:SITE_ROOT         = $dist.Replace('\', '/')
$env:SITE_LOG          = (Join-Path $logDir 'caddy.log').Replace('\', '/')
$env:PORTFOLIO_ADDRESS = $PortfolioAddress
$env:PORTFOLIO_ROOT    = $PortfolioRoot.Replace('\', '/')
$env:PORTFOLIO_LOG     = (Join-Path $logDir 'portfolio.log').Replace('\', '/')

Write-Host "Serving $SiteAddress from $env:SITE_ROOT"
Write-Host "Serving $PortfolioAddress (+www) from $env:PORTFOLIO_ROOT"

# Caddy writes all of its logging to stderr. Under ErrorActionPreference='Stop',
# PowerShell can promote native stderr to a terminating error and kill the server
# on its first ordinary log line, so relax it for the long-running process.
$ErrorActionPreference = 'Continue'
& $CaddyExe run --config $config
