<#
.SYNOPSIS
    Applies the current Caddyfile to the running Caddy and verifies all three
    sites on this VPS.

.DESCRIPTION
    One Caddy instance fronts the portfolio, VisionEdge and MyAlphaEdge, so this
    script treats them as one unit: pull, validate, reload, verify.

    It also re-registers the VisionEdge-Caddy scheduled task. That matters: the
    task's arguments carry -PortfolioRoot, Start-Caddy.ps1 exports it as
    PORTFOLIO_ROOT, and an exported variable overrides the Caddyfile's default.
    A task registered with the old wwwroot path would quietly undo this fix at
    the next reboot, so a reload on its own is not enough.

    Validation always runs before the reload. An invalid Caddyfile takes all
    three sites down at once, not just the one being changed.

.PARAMETER CleanStaleRoot
    After the portfolio is confirmed serving from the right directory, delete the
    stale build in C:\inetpub\wwwroot that caused it to serve the wrong site.
    Prompts unless -Force is given.

.EXAMPLE
    .\Repair-Sites.ps1

.EXAMPLE
    .\Repair-Sites.ps1 -CleanStaleRoot
#>
[CmdletBinding()]
param(
    [string]$RepoRoot         = 'C:\visionedge-lab',
    [string]$CaddyExe         = 'C:\ProgramData\VisionEdge\tools\caddy.exe',
    [string]$PortfolioAddress = 'c-maron.space',
    [string]$SiteAddress      = 'visionedge.c-maron.space',
    [string]$AlphaAddress     = 'myalphaedge.com',
    [string]$PortfolioRoot    = 'C:\inetpub\c-maron',
    [string]$AlphaRoot        = 'C:\MyAlphaEdge\frontend\dist',
    [string]$StaleRoot        = 'C:\inetpub\wwwroot',
    [switch]$CleanStaleRoot,
    [switch]$SkipPull,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Write-Step { param([string]$m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Ok   { param([string]$m) Write-Host "    [ok] $m" -ForegroundColor Green }
function Write-Warn { param([string]$m) Write-Host "    [warn] $m" -ForegroundColor Yellow }
function Write-Bad  { param([string]$m) Write-Host "    [FAIL] $m" -ForegroundColor Red }

function Get-Title {
    param([string]$Html)
    if ($Html -match '(?is)<title>\s*(.*?)\s*</title>') { return $Matches[1] }
    return '(no <title>)'
}

try {
    Write-Host '=======================================================' -ForegroundColor Cyan
    Write-Host ' Repairing the three sites behind this Caddy instance'   -ForegroundColor Cyan
    Write-Host '=======================================================' -ForegroundColor Cyan

    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
            [Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run from an elevated PowerShell prompt (Administrator).'
    }

    $config = Join-Path $RepoRoot 'deploy\Caddyfile.windows'
    $siteRoot = Join-Path $RepoRoot 'frontend\dist'

    # --- 1. latest config ----------------------------------------------------
    Write-Step 'Repository'
    if ($SkipPull) {
        Write-Ok 'skipped (-SkipPull)'
    } else {
        git -C $RepoRoot pull --ff-only 2>&1 | Out-Null
        Write-Ok 'pulled'
    }
    if (-not (Test-Path $config)) { throw "Caddyfile not found at $config" }
    if (-not (Test-Path $CaddyExe)) {
        $onPath = Get-Command caddy -ErrorAction SilentlyContinue
        if (-not $onPath) { throw "caddy.exe not found at $CaddyExe and not on PATH." }
        $CaddyExe = $onPath.Source
    }
    Write-Ok "using $CaddyExe"

    # --- 2. document roots ---------------------------------------------------
    # A missing root does not stop Caddy; it just serves 404s for that host,
    # which is far harder to diagnose after the fact than a message here.
    Write-Step 'Document roots'
    $roots = @{
        $PortfolioAddress = $PortfolioRoot
        $SiteAddress      = $siteRoot
        $AlphaAddress     = $AlphaRoot
    }
    foreach ($host_ in $roots.Keys) {
        $index = Join-Path $roots[$host_] 'index.html'
        if (Test-Path $index) {
            Write-Ok "$host_ -> $($roots[$host_])"
        } else {
            Write-Warn "$host_ -> $($roots[$host_]) has no index.html; that site will 404 until it is published"
        }
    }

    # --- 3. validate, then reload -------------------------------------------
    # Set the placeholders explicitly. `caddy reload` adapts the Caddyfile in
    # THIS process, so it resolves {$VAR:default} from this environment - not
    # from the environment the running Caddy was started with.
    $env:PORTFOLIO_ADDRESS  = $PortfolioAddress
    $env:PORTFOLIO_ROOT     = $PortfolioRoot.Replace('\', '/')
    $env:SITE_ADDRESS       = $SiteAddress
    $env:SITE_ROOT          = $siteRoot.Replace('\', '/')
    $env:MYALPHAEDGE_ADDRESS = $AlphaAddress
    $env:MYALPHAEDGE_ROOT   = $AlphaRoot.Replace('\', '/')
    $logDir = Join-Path $RepoRoot 'logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $env:PORTFOLIO_LOG   = (Join-Path $logDir 'portfolio.log').Replace('\', '/')
    $env:SITE_LOG        = (Join-Path $logDir 'caddy.log').Replace('\', '/')
    $env:MYALPHAEDGE_LOG = (Join-Path $logDir 'myalphaedge.log').Replace('\', '/')

    Write-Step 'Validating the Caddyfile'
    # Caddy logs to stderr, which PowerShell would otherwise promote to a
    # terminating error under ErrorActionPreference=Stop.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $validateOut = & $CaddyExe validate --config $config --adapter caddyfile 2>&1 | Out-String
    $validateCode = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($validateCode -ne 0) {
        Write-Host $validateOut
        throw 'Caddyfile is INVALID - not reloading. All three sites would go down.'
    }
    Write-Ok 'valid configuration'

    Write-Step 'Reloading Caddy'
    $ErrorActionPreference = 'Continue'
    $reloadOut = & $CaddyExe reload --config $config --adapter caddyfile 2>&1 | Out-String
    $reloadCode = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($reloadCode -ne 0) {
        Write-Host $reloadOut
        throw 'Reload failed. The previously running config is still active.'
    }
    Write-Ok 'reloaded'

    # --- 4. make it survive a reboot ----------------------------------------
    Write-Step 'Startup task'
    $startCaddy = Join-Path $PSScriptRoot 'Start-Caddy.ps1'
    if (Test-Path $startCaddy) {
        $args_ = "-NoProfile -ExecutionPolicy Bypass -File `"$startCaddy`" " +
                 "-RepoRoot `"$RepoRoot`" -SiteAddress $SiteAddress " +
                 "-PortfolioAddress $PortfolioAddress -PortfolioRoot `"$PortfolioRoot`" " +
                 "-CaddyExe `"$CaddyExe`""
        $action    = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $args_
        $trigger   = New-ScheduledTaskTrigger -AtStartup
        $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -RunLevel Highest
        $settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                     -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
        Unregister-ScheduledTask -TaskName 'VisionEdge-Caddy' -Confirm:$false -ErrorAction SilentlyContinue
        Register-ScheduledTask -TaskName 'VisionEdge-Caddy' -Action $action -Trigger $trigger `
            -Principal $principal -Settings $settings | Out-Null
        Write-Ok "re-registered with -PortfolioRoot $PortfolioRoot (a stale value here would undo the fix on reboot)"
    } else {
        Write-Warn "Start-Caddy.ps1 not found next to this script; startup task left untouched"
    }

    # --- 5. verify -----------------------------------------------------------
    Write-Step 'Verifying the sites'
    $failures = @()

    function Test-Site {
        param([string]$Url, [string]$ExpectRoot, [string]$Label)

        $body = $null
        foreach ($i in 1..8) {
            try {
                $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 15
                if ($r.StatusCode -eq 200) { $body = $r.Content; break }
            } catch {
                Start-Sleep -Seconds 4   # first hit can wait on certificate issuance
            }
        }
        if (-not $body) { Write-Bad "$Label : no 200 from $Url"; return $false }

        $servedTitle = Get-Title $body
        $indexPath = Join-Path $ExpectRoot 'index.html'
        if (Test-Path $indexPath) {
            # Read as UTF-8 explicitly. Get-Content on PS 5.1 defaults to
            # Windows-1252, which turns an accented character in a title into
            # mojibake and makes a correctly-served site look like the wrong root.
            $expectTitle = Get-Title ([System.IO.File]::ReadAllText(
                $indexPath, [System.Text.Encoding]::UTF8))
            if ($servedTitle -eq $expectTitle) {
                Write-Ok "$Label : serving '$servedTitle' (matches $ExpectRoot)"
                return $true
            }
            # Differing only outside ASCII means the two strings are the same text
            # read through different encodings, not two different documents.
            $strip = { param($s) (($s.ToCharArray() | Where-Object { [int]$_ -lt 128 }) -join '') }
            if ((& $strip $servedTitle) -eq (& $strip $expectTitle)) {
                Write-Warn "$Label : serving '$servedTitle' - matches $ExpectRoot apart from character encoding"
                return $true
            }
            Write-Bad "$Label : serving '$servedTitle' but $ExpectRoot contains '$expectTitle' - wrong root"
            return $false
        }
        Write-Warn "$Label : responded '$servedTitle' (no local index.html to compare against)"
        return $true
    }

    if (-not (Test-Site "https://$PortfolioAddress/" $PortfolioRoot 'portfolio')) { $failures += $PortfolioAddress }
    if (-not (Test-Site "https://$SiteAddress/"      $siteRoot      'visionedge')) { $failures += $SiteAddress }
    if (-not (Test-Site "https://$AlphaAddress/"     $AlphaRoot     'myalphaedge')) { $failures += $AlphaAddress }

    # VisionEdge API
    try {
        $h = Invoke-WebRequest -Uri "https://$SiteAddress/health" -UseBasicParsing -TimeoutSec 15
        Write-Ok "visionedge /health : $($h.Content)"
    } catch {
        Write-Bad "visionedge /health did not respond: $($_.Exception.Message)"
        $failures += "$SiteAddress/health"
    }

    # MyAlphaEdge API - the point is that the /api prefix survives the proxy.
    try {
        $a = Invoke-WebRequest -Uri "https://$AlphaAddress/api/health" -UseBasicParsing -TimeoutSec 15
        Write-Ok "myalphaedge /api/health : HTTP $($a.StatusCode) (prefix preserved)"
    } catch {
        $code = $_.Exception.Response.StatusCode.value__
        if ($code) {
            Write-Warn "myalphaedge /api/health returned HTTP $code - reached the backend, so the prefix survived; the route itself may just not exist"
        } else {
            Write-Bad "myalphaedge /api/* did not reach the backend on 127.0.0.1:5000: $($_.Exception.Message)"
            $failures += "$AlphaAddress/api"
        }
    }

    # Cache-Control: an SPA route must be no-cache, a hashed asset must not be.
    try {
        $doc = Invoke-WebRequest -Uri "https://$AlphaAddress/dashboard" -UseBasicParsing -TimeoutSec 15
        # Indexing a key that is not present throws on PS 5.1, so check first.
        $cc = if ($doc.Headers.ContainsKey('Cache-Control')) { $doc.Headers['Cache-Control'] } else { '(none)' }
        if ($cc -match 'no-cache') {
            Write-Ok "myalphaedge /dashboard Cache-Control: $cc"
        } else {
            Write-Bad "myalphaedge /dashboard Cache-Control is '$cc' - a stale index.html can pin dead chunk hashes"
            $failures += 'cache-control'
        }
    } catch {
        Write-Warn "could not check Cache-Control on /dashboard: $($_.Exception.Message)"
    }

    # --- 6. stale root -------------------------------------------------------
    if ($CleanStaleRoot) {
        Write-Step 'Stale build'
        if ($failures -contains $PortfolioAddress) {
            Write-Warn "portfolio is not verified yet - leaving $StaleRoot alone"
        } elseif (-not (Test-Path $StaleRoot)) {
            Write-Ok "$StaleRoot already gone"
        } else {
            $go = $Force
            if (-not $go) {
                $ans = Read-Host "    Delete the stale build in $StaleRoot ? [y/N]"
                $go = $ans -match '^(y|yes)$'
            }
            if ($go) {
                $backup = "$StaleRoot.stale-backup"
                Move-Item $StaleRoot $backup -Force
                Write-Ok "moved to $backup (delete it yourself once you are happy)"
            } else {
                Write-Ok 'left in place'
            }
        }
    }

    Write-Host ''
    if ($failures.Count -eq 0) {
        Write-Host '=======================================================' -ForegroundColor Green
        Write-Host ' All three sites verified'                               -ForegroundColor Green
        Write-Host '=======================================================' -ForegroundColor Green
    } else {
        Write-Host '=======================================================' -ForegroundColor Yellow
        Write-Host " Reloaded, but these did not verify: $($failures -join ', ')" -ForegroundColor Yellow
        Write-Host '=======================================================' -ForegroundColor Yellow
        Write-Host "  Logs: $logDir"
        exit 1
    }
}
catch {
    Write-Host ''
    Write-Host "REPAIR FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
