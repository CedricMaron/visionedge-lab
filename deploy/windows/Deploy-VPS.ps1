<#
.SYNOPSIS
    One-shot deployment of VisionEdge Lab (and the existing portfolio) onto a
    Windows VPS, behind Caddy with automatic HTTPS.

.DESCRIPTION
    Runs the whole sequence: prerequisites, repository, Python venv, model
    download, frontend build, IIS handover, scheduled tasks, firewall, and a
    verification pass against the live URLs.

    Safe to re-run. Every step checks whether it is already done, so a second run
    after fixing one problem does not redo the rest.

    THE RISKY STEP is stopping IIS: between that and Caddy obtaining
    certificates, the portfolio is offline. If anything fails from that point on,
    the script puts IIS back automatically before exiting.

.PARAMETER RepoRoot
    Where to clone/find the repository. Created if missing.

.PARAMETER PortfolioRoot
    Directory the existing portfolio is served from. Defaults to
    C:\inetpub\c-maron (where c-maron's deploy.ps1 publishes), consulting IIS only
    if that is absent.

.PARAMETER SkipPrereqs
    Do not attempt to install Python/Node/Caddy/Git via winget; only check.

.PARAMETER Force
    Do not prompt before taking the portfolio offline.

.EXAMPLE
    .\Deploy-VPS.ps1

.EXAMPLE
    .\Deploy-VPS.ps1 -SiteAddress visionedge.c-maron.space -RateLimitPerMin 30
#>
[CmdletBinding()]
param(
    [string]$RepoRoot         = 'C:\visionedge-lab',
    [string]$RepoUrl          = 'https://github.com/CedricMaron/visionedge-lab.git',
    [string]$SiteAddress      = 'visionedge.c-maron.space',
    [string]$PortfolioAddress = 'c-maron.space',
    [string]$PortfolioRoot    = '',
    [int]$RateLimitPerMin     = 60,
    [switch]$SkipPrereqs,
    [switch]$Force,
    [switch]$NoSelfUpdate
)

$ErrorActionPreference = 'Stop'
$script:IisWasRunning = $false
$script:OurServicesStopped = $false
$script:CaddyExe = 'caddy'

# PowerShell 5.1 on Windows Server still defaults to TLS 1.0/1.1, which modern
# download endpoints refuse. Without this, every Invoke-WebRequest below fails
# with an unhelpful "connection closed" error.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Write-Step { param([string]$Message) Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Message) Write-Host "    [ok] $Message" -ForegroundColor Green }
function Write-Warn { param([string]$Message) Write-Host "    [warn] $Message" -ForegroundColor Yellow }

# Stops the services this script owns, and waits for them to actually exit.
#
# Two separate bugs need this:
#   1. Start-ScheduledTask is a no-op against a task already Running, so without
#      an explicit stop the old backend survives a deploy and keeps serving stale
#      code while the health probe happily returns 200 from it.
#   2. The port 80/443 availability check cannot tell our own Caddy from a foreign
#      squatter, so on a RE-deploy our own Caddy fails the check. Stopping first
#      means anything still holding the ports afterwards is genuinely foreign.
#
# Stop-ScheduledTask returns before the process has exited, hence the wait.
function Stop-OurServices {
    foreach ($taskName in @('VisionEdge-Backend', 'VisionEdge-Caddy')) {
        $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if (-not $task -or $task.State -ne 'Running') { continue }

        Stop-ScheduledTask -TaskName $taskName
        foreach ($i in 1..20) {
            Start-Sleep -Milliseconds 500
            if ((Get-ScheduledTask -TaskName $taskName).State -ne 'Running') { break }
        }
        if ((Get-ScheduledTask -TaskName $taskName).State -eq 'Running') {
            throw "$taskName would not stop. Stop it manually (Stop-ScheduledTask $taskName) and re-run."
        }
        $script:OurServicesStopped = $true
        Write-Ok "$taskName stopped"
    }
}

# Bring back whatever this script took down. On a re-deploy IIS was never running
# (Caddy held the ports), so restoring IIS alone would leave the site dark - the
# services we stopped are the ones that need to come back.
function Restore-OurServices {
    if (-not $script:OurServicesStopped) { return }
    Write-Warn 'Deployment failed after the services were stopped - restarting them.'
    foreach ($taskName in @('VisionEdge-Backend', 'VisionEdge-Caddy')) {
        if (-not (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue)) { continue }
        try {
            Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
            Write-Ok "$taskName restarted"
        } catch {
            Write-Host "    [error] Could not restart ${taskName}: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "    Run manually: Start-ScheduledTask $taskName" -ForegroundColor Red
        }
    }
}

function Restore-Iis {
    if ($script:IisWasRunning) {
        Write-Warn 'Deployment failed after IIS was stopped - restoring the portfolio.'
        try {
            Start-Service W3SVC -ErrorAction Stop
            Set-Service W3SVC -StartupType Automatic
            Write-Ok 'IIS restarted. The portfolio is back to its previous state.'
        } catch {
            Write-Host "    [error] Could not restart IIS: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "    Run manually: Start-Service W3SVC" -ForegroundColor Red
        }
    }
}

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-RealCommand {
    <#
        Get-Command alone is not enough for python: Windows ships an App Execution
        Alias at python.exe that only opens the Microsoft Store. It resolves, then
        every later call fails confusingly. Require the command to actually run.
    #>
    param([string]$Command, [string[]]$VersionArgs = @('--version'))

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) { return $false }
    try {
        $out = & $Command @VersionArgs 2>&1 | Out-String
        return -not [string]::IsNullOrWhiteSpace($out)
    } catch {
        return $false
    }
}

function Install-Caddy {
    <#
        winget is absent on most Windows Server installs, so fall back to Caddy's
        official download endpoint, which serves a bare .exe (no archive). The
        GitHub release zip is the second choice if that endpoint is unreachable.
        Installed outside the repository so a later git clone cannot collide with it.
    #>
    $toolDir = Join-Path $env:ProgramData 'VisionEdge\tools'
    $exe     = Join-Path $toolDir 'caddy.exe'

    if (Test-RealCommand -Command $exe -VersionArgs @('version')) {
        Write-Ok "Caddy already installed at $exe"
        return $exe
    }

    New-Item -ItemType Directory -Force -Path $toolDir | Out-Null
    $direct = 'https://caddyserver.com/api/download?os=windows&arch=amd64'
    Write-Host "    downloading Caddy from caddyserver.com..."
    try {
        Invoke-WebRequest -Uri $direct -OutFile $exe -UseBasicParsing -TimeoutSec 300
    } catch {
        Write-Warn "direct download failed ($($_.Exception.Message)); trying the GitHub release"
        $zip = Join-Path $toolDir 'caddy.zip'
        $rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/caddyserver/caddy/releases/latest' `
                                 -UseBasicParsing -TimeoutSec 120
        $asset = $rel.assets | Where-Object { $_.name -like '*windows_amd64.zip' } | Select-Object -First 1
        if (-not $asset) { throw 'No windows_amd64 asset in the latest Caddy release.' }
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -UseBasicParsing -TimeoutSec 300
        Expand-Archive -Path $zip -DestinationPath $toolDir -Force
        Remove-Item $zip -Force
    }

    if (-not (Test-RealCommand -Command $exe -VersionArgs @('version'))) {
        throw "Caddy downloaded to $exe but will not run. Install it manually from https://caddyserver.com/download."
    }
    Write-Ok "Caddy installed at $exe"
    return $exe
}

function Install-Prereq {
    param([string]$Command, [string]$WingetId, [string]$Url, [string]$Label)

    if (Test-RealCommand -Command $Command) {
        Write-Ok "$Label present"
        return $true
    }
    if ($SkipPrereqs -or -not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "    [error] $Label missing. Install it from $Url and re-run." -ForegroundColor Red
        return $false
    }
    Write-Host "    installing $Label via winget..."
    winget install --id $WingetId --silent --accept-source-agreements --accept-package-agreements | Out-Null
    # winget updates PATH for new processes only; refresh this session's copy.
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User')
    if (Test-RealCommand -Command $Command) {
        Write-Ok "$Label installed"
        return $true
    }
    Write-Host "    [error] $Label still not on PATH. Open a new shell and re-run, or install from $Url." -ForegroundColor Red
    return $false
}

try {
    Write-Host '=======================================================' -ForegroundColor Cyan
    Write-Host ' VisionEdge Lab - Windows VPS deployment'               -ForegroundColor Cyan
    Write-Host '=======================================================' -ForegroundColor Cyan

    # --- 0. self-update ------------------------------------------------------
    # This has to happen BEFORE preflight. The repository update used to live in
    # step 2, so running an outdated copy failed in preflight and never reached
    # the pull that would have fixed it.
    if (-not $NoSelfUpdate) {
        $selfRepo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        if (Test-Path (Join-Path $selfRepo '.git')) {
            Write-Step 'Checking for a newer version of this script'
            $before = (Get-FileHash $PSCommandPath -Algorithm SHA256).Hash
            try {
                git -C $selfRepo pull --ff-only 2>&1 | Out-Null
            } catch {
                Write-Warn "could not pull: $($_.Exception.Message). Continuing with the local copy."
            }
            $after = (Get-FileHash $PSCommandPath -Algorithm SHA256).Hash
            if ($before -ne $after) {
                Write-Host ''
                Write-Host '    This script was updated by the pull. PowerShell is still running the' -ForegroundColor Yellow
                Write-Host '    old copy, so nothing further was changed. Re-run the same command to' -ForegroundColor Yellow
                Write-Host '    continue with the new version.' -ForegroundColor Yellow
                exit 0
            }
            Write-Ok 'up to date'
        }
    }

    # --- 1. preflight --------------------------------------------------------
    Write-Step 'Preflight checks'
    if (-not (Test-Admin)) {
        throw 'Run this from an elevated PowerShell prompt (right-click > Run as Administrator).'
    }
    Write-Ok 'running as Administrator'

    $ok = $true
    $ok = (Install-Prereq -Command 'git'    -WingetId 'Git.Git'            -Url 'https://git-scm.com/download/win'  -Label 'Git') -and $ok
    $ok = (Install-Prereq -Command 'python' -WingetId 'Python.Python.3.12' -Url 'https://www.python.org/downloads/' -Label 'Python 3.12') -and $ok
    $ok = (Install-Prereq -Command 'node'   -WingetId 'OpenJS.NodeJS.LTS'  -Url 'https://nodejs.org/'               -Label 'Node.js') -and $ok
    if (-not $ok) { throw 'Missing prerequisites (see above). Install them and re-run.' }

    # Caddy is handled separately: it has an official direct download, so it needs
    # neither winget nor a manual step even on a bare Windows Server.
    if (Test-RealCommand -Command 'caddy' -VersionArgs @('version')) {
        $script:CaddyExe = 'caddy'
        Write-Ok 'Caddy present'
    } elseif ($SkipPrereqs) {
        throw 'Caddy missing and -SkipPrereqs was given. Install from https://caddyserver.com/download.'
    } else {
        $script:CaddyExe = Install-Caddy
    }

    # DNS must resolve before Caddy asks Let's Encrypt for a certificate.
    foreach ($name in @($SiteAddress, $PortfolioAddress)) {
        try {
            $a = Resolve-DnsName -Name $name -Type A -ErrorAction Stop |
                 Where-Object { $_.IPAddress } | Select-Object -First 1
            Write-Ok "$name resolves to $($a.IPAddress)"
        } catch {
            throw "$name does not resolve. Add the DNS A record first (see docs/DEPLOY_WINDOWS.md) and allow it to propagate."
        }
    }

    # --- 2. repository -------------------------------------------------------
    Write-Step "Repository at $RepoRoot"
    if (Test-Path (Join-Path $RepoRoot '.git')) {
        # Already pulled in step 0 when this script is running from that clone.
        git -C $RepoRoot pull --ff-only 2>&1 | Out-Null
        Write-Ok 'existing clone up to date'
    } else {
        git clone $RepoUrl $RepoRoot 2>&1 | Out-Null
        Write-Ok 'cloned'
    }

    # --- 3. backend ----------------------------------------------------------
    Write-Step 'Backend (Python venv + dependencies)'
    $venvPython = Join-Path $RepoRoot 'backend\.venv\Scripts\python.exe'
    if (-not (Test-Path $venvPython)) {
        # Prefer the py launcher: it picks a real 3.12 rather than whatever
        # "python" happens to resolve to on this machine.
        if (Test-RealCommand -Command 'py' -VersionArgs @('-3.12', '--version')) {
            py -3.12 -m venv (Join-Path $RepoRoot 'backend\.venv')
        } else {
            python -m venv (Join-Path $RepoRoot 'backend\.venv')
        }
        if (-not (Test-Path $venvPython)) { throw 'venv creation failed - no python.exe in backend\.venv\Scripts.' }
        Write-Ok 'venv created'
    } else {
        Write-Ok 'venv already present'
    }
    & $venvPython -m pip install --upgrade pip --quiet
    & $venvPython -m pip install -r (Join-Path $RepoRoot 'backend\requirements\base.txt') --quiet
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed.' }
    Write-Ok 'dependencies installed'

    # --- 4. models -----------------------------------------------------------
    # InferenceLab demonstrates three modalities, so all three models are installed.
    # Only the detector is required: the platform reports an absent model as
    # "not installed" with an install hint rather than failing, so a download that
    # fails leaves a working site with one fewer modality instead of no site.
    Write-Step 'Models'
    $models = @(
        @{ Id = 'yolov8n-onnx';                Path = 'models\yolov8n.onnx';                                  Required = $true  },
        @{ Id = 'mobilenetv4-conv-small-onnx'; Path = 'models\classification\mobilenetv4_conv_small.onnx';    Required = $false },
        @{ Id = 'all-minilm-l6-v2-onnx';       Path = 'models\embedding\all-MiniLM-L6-v2.onnx';              Required = $false }
    )
    foreach ($m in $models) {
        $target = Join-Path $RepoRoot $m.Path
        if (Test-Path $target) {
            Write-Ok ("{0}: already installed" -f $m.Id)
            continue
        }
        Push-Location $RepoRoot
        & $venvPython 'scripts\download_models.py' --install $m.Id -y
        Pop-Location

        if (Test-Path $target) {
            Write-Ok ("{0}: downloaded and checksum-verified" -f $m.Id)
        } elseif ($m.Required) {
            throw ("Required model '{0}' failed to install." -f $m.Id)
        } else {
            Write-Warn ("{0}: install failed - the site will report it as not installed" -f $m.Id)
        }
    }

    # --- 5. frontend ---------------------------------------------------------
    Write-Step 'Frontend build'
    $frontend = Join-Path $RepoRoot 'frontend'
    # An unset VITE_API_BASE is what makes the built app address its own origin.
    $dotenv = Join-Path $frontend '.env'
    if (Test-Path $dotenv) {
        Move-Item $dotenv "$dotenv.disabled-by-deploy" -Force
        Write-Warn 'moved frontend\.env aside - it would have pointed the deployed site at localhost'
    }
    Push-Location $frontend
    Remove-Item Env:VITE_API_BASE -ErrorAction SilentlyContinue
    npm ci --silent
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'npm ci failed.' }
    npm run build --silent
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'npm run build failed.' }
    Pop-Location
    if (-not (Test-Path (Join-Path $frontend 'dist\index.html'))) { throw 'Build produced no dist\index.html.' }
    Write-Ok 'SPA built'

    # --- 6. portfolio location ----------------------------------------------
    Write-Step 'Portfolio location'
    if (-not $PortfolioRoot) {
        # Where c-maron's scripts\deploy.ps1 publishes. Checked FIRST and trusted
        # over anything IIS reports: IIS's Default Web Site points at
        # C:\inetpub\wwwroot, which holds a stale build of a different site, and
        # believing it is what made c-maron.space serve the wrong content.
        $published = 'C:\inetpub\c-maron'
        if (Test-Path (Join-Path $published 'index.html')) {
            $PortfolioRoot = $published
        } else {
            try {
                Import-Module WebAdministration -ErrorAction Stop
                $site = Get-Website -ErrorAction Stop | Select-Object -First 1
                if ($site -and $site.physicalPath) {
                    $PortfolioRoot = [Environment]::ExpandEnvironmentVariables($site.physicalPath)
                    Write-Warn "using the IIS path $PortfolioRoot - confirm it is the portfolio and not a stale build"
                }
            } catch {
                Write-Warn 'could not query IIS; falling back to the published path'
            }
            if (-not $PortfolioRoot) { $PortfolioRoot = $published }
        }
    }
    if (-not (Test-Path $PortfolioRoot)) {
        throw "Portfolio root '$PortfolioRoot' not found. Pass -PortfolioRoot with the correct path."
    }
    Write-Ok "serving the portfolio from $PortfolioRoot"

    # --- 7. IIS handover (the risky part) -----------------------------------
    Write-Step 'Handing ports 80/443 over from IIS'

    # Release our own ports first. On a re-deploy Caddy is already running and
    # holding 80/443, and the check below would otherwise report our own process
    # as the squatter and abort after IIS had been stopped.
    Stop-OurServices

    $iis = Get-Service W3SVC -ErrorAction SilentlyContinue
    if ($iis -and $iis.Status -eq 'Running') {
        if (-not $Force) {
            Write-Host ''
            Write-Host "    This stops IIS. $PortfolioAddress goes offline until Caddy has" -ForegroundColor Yellow
            Write-Host '    its certificates (usually under a minute). It is put back' -ForegroundColor Yellow
            Write-Host '    automatically if the rest of this script fails.' -ForegroundColor Yellow
            $answer = Read-Host '    Continue? [y/N]'
            if ($answer -notmatch '^(y|yes)$') { throw 'Cancelled before stopping IIS. Nothing was changed.' }
        }
        Stop-Service W3SVC -Force
        Set-Service W3SVC -StartupType Manual
        $script:IisWasRunning = $true
        Write-Ok 'IIS stopped and set to Manual so it does not reclaim the ports on reboot'
    } else {
        Write-Ok 'IIS not running'
    }

    $held = Get-NetTCPConnection -LocalPort 80, 443 -State Listen -ErrorAction SilentlyContinue
    if ($held) {
        $names = ($held | ForEach-Object {
            (Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName
        } | Sort-Object -Unique) -join ', '
        $detail = ($held | ForEach-Object {
            $proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
            "$($proc.ProcessName) (PID $($_.OwningProcess)) on port $($_.LocalPort)"
        } | Sort-Object -Unique) -join '; '
        throw ("Ports 80/443 are still held by: $names. This is not a service this " +
               "script manages - ours were already stopped. Holder(s): $detail. " +
               'Stop that process and re-run.')
    }
    Write-Ok 'ports 80 and 443 are free'

    # --- 8. services + firewall ---------------------------------------------
    Write-Step 'Registering startup tasks and firewall rules'
    & (Join-Path $PSScriptRoot 'Install-Services.ps1') `
        -RepoRoot $RepoRoot -SiteAddress $SiteAddress -RateLimitPerMin $RateLimitPerMin

    # Install-Services registers Caddy without the portfolio arguments; re-register
    # that one task so it serves both sites.
    $caddyArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $PSScriptRoot 'Start-Caddy.ps1')`" " +
                 "-RepoRoot `"$RepoRoot`" -SiteAddress $SiteAddress " +
                 "-PortfolioAddress $PortfolioAddress -PortfolioRoot `"$PortfolioRoot`" " +
                 "-CaddyExe `"$script:CaddyExe`""
    $action    = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $caddyArgs
    $trigger   = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -RunLevel Highest
    $settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                 -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
    Unregister-ScheduledTask -TaskName 'VisionEdge-Caddy' -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName 'VisionEdge-Caddy' -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings | Out-Null
    Write-Ok 'tasks registered'

    # --- 9. restart ----------------------------------------------------------
    # Stop BEFORE starting. Start-ScheduledTask on a task that is already Running is
    # a no-op, so without this the previous backend survives the deploy and keeps
    # serving the old code - while the health probe below happily returns 200 from
    # it and the deploy declares success. That is exactly how a rebuilt frontend
    # ended up talking to a backend that had never heard of /api/lab.
    Write-Step 'Restarting services'
    $deployedCommit = (& git -C $RepoRoot rev-parse HEAD).Trim()
    Write-Host "    deploying commit $($deployedCommit.Substring(0,12))"

    # Already stopped in step 7, before the port check. Re-running is harmless and
    # covers a task that was (re)started by its trigger in between.
    Stop-OurServices

    Start-ScheduledTask -TaskName 'VisionEdge-Backend'
    Start-ScheduledTask -TaskName 'VisionEdge-Caddy'
    Write-Host '    waiting for the backend to load the model...'

    $health = $null
    foreach ($i in 1..30) {
        Start-Sleep -Seconds 2
        try {
            $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -eq 200) { $health = $r.Content | ConvertFrom-Json; break }
        } catch { }
    }
    if (-not $health) {
        throw 'Backend did not become healthy within 60s. Check: Get-ScheduledTask VisionEdge-Backend; and run .\deploy\windows\Start-Backend.ps1 in the foreground to see the error.'
    }

    # A 200 only proves SOME backend is up. Verify it is the one just deployed.
    if ($health.git_commit -and $deployedCommit -and $health.git_commit -ne $deployedCommit) {
        throw ("Backend is healthy but running commit $($health.git_commit.Substring(0,12)), " +
               "not the deployed $($deployedCommit.Substring(0,12)). A stale process survived " +
               'the restart - stop it with: Stop-ScheduledTask VisionEdge-Backend')
    }
    if (-not $health.git_commit) {
        Write-Warn 'backend did not report a commit; cannot verify it is running the deployed code'
    }
    # Services are up and verified; a later failure must not claim to restart them.
    $script:OurServicesStopped = $false
    Write-Ok "backend healthy on 127.0.0.1:8000 (version $($health.version), commit $($deployedCommit.Substring(0,12)))"

    if ($health.warnings -and $health.warnings.Count -gt 0) {
        foreach ($w in $health.warnings) { Write-Warn $w }
    }

    Write-Host '    waiting for Caddy to obtain certificates (first run can take ~30s)...'
    $siteUp = $false
    foreach ($i in 1..30) {
        Start-Sleep -Seconds 3
        try {
            $r = Invoke-WebRequest -Uri "https://$SiteAddress/health" -UseBasicParsing -TimeoutSec 8
            if ($r.StatusCode -eq 200) { $siteUp = $true; break }
        } catch { }
    }

    # --- 10. verify ----------------------------------------------------------
    Write-Step 'Verification'
    if ($siteUp) {
        Write-Ok "https://$SiteAddress/health responded 200"
    } else {
        Write-Warn "https://$SiteAddress/health did not respond yet."
        Write-Warn 'Certificates can lag; check the log before assuming failure:'
        Write-Warn "  Get-Content '$RepoRoot\logs\caddy.log' -Tail 40"
    }
    try {
        $p = Invoke-WebRequest -Uri "https://$PortfolioAddress/" -UseBasicParsing -TimeoutSec 10
        Write-Ok "https://$PortfolioAddress responded $($p.StatusCode) - the portfolio now has working HTTPS"
    } catch {
        Write-Warn "https://$PortfolioAddress did not respond yet: $($_.Exception.Message)"
    }

    Write-Host ''
    Write-Host '=======================================================' -ForegroundColor Green
    Write-Host ' Deployment complete'                                    -ForegroundColor Green
    Write-Host '=======================================================' -ForegroundColor Green
    Write-Host "  App:        https://$SiteAddress"
    Write-Host "  Portfolio:  https://$PortfolioAddress"
    Write-Host "  Logs:       $RepoRoot\logs\"
    Write-Host ''
    Write-Host '  Open the app and check the Live Inference page - the WebSocket'
    Write-Host '  and camera are the parts a proxy misconfiguration breaks first.'
    Write-Host ''
    Write-Host '  Roll back to IIS:  Stop-ScheduledTask VisionEdge-Caddy; Start-Service W3SVC'
}
catch {
    Write-Host ''
    Write-Host "DEPLOYMENT FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Restore-OurServices
    Restore-Iis
    exit 1
}
