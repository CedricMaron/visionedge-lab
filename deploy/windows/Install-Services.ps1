<#
.SYNOPSIS
    Registers VisionEdge Lab to start at boot, and opens the firewall for HTTP/HTTPS.

.DESCRIPTION
    Uses Scheduled Tasks rather than real Windows services so that nothing beyond
    Windows itself is required  -  a Python process is not a service binary, so the
    usual alternative (NSSM, WinSW) means installing third-party software. If you
    already use NSSM, register the two Start-*.ps1 scripts with it instead; this
    script is only a dependency-free default.

    Run from an elevated PowerShell prompt.

.EXAMPLE
    .\Install-Services.ps1 -SiteAddress visionedge.c-maron.space
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$SiteAddress = 'visionedge.c-maron.space',
    [int]$RateLimitPerMin = 60,
    [string]$RunAsUser = 'SYSTEM'
)

$ErrorActionPreference = 'Stop'

if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this from an elevated PowerShell prompt (Administrator).'
}

$backendScript = Join-Path $PSScriptRoot 'Start-Backend.ps1'
$caddyScript   = Join-Path $PSScriptRoot 'Start-Caddy.ps1'
foreach ($s in @($backendScript, $caddyScript)) {
    if (-not (Test-Path $s)) { throw "Missing $s" }
}

function Register-VisionEdgeTask {
    param([string]$Name, [string]$Script, [string]$Arguments)

    $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Script`" $Arguments"
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId $RunAsUser -RunLevel Highest
    # Restart on failure: an inference server that dies should come back without a human.
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero)

    if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    }
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings | Out-Null
    Write-Host "registered scheduled task: $Name"
}

Register-VisionEdgeTask -Name 'VisionEdge-Backend' -Script $backendScript `
    -Arguments "-RepoRoot `"$RepoRoot`" -SiteAddress $SiteAddress -RateLimitPerMin $RateLimitPerMin"
Register-VisionEdgeTask -Name 'VisionEdge-Caddy' -Script $caddyScript `
    -Arguments "-RepoRoot `"$RepoRoot`" -SiteAddress $SiteAddress"

# Caddy needs 80 reachable for the ACME challenge, not just 443.
foreach ($port in 80, 443) {
    $rule = "VisionEdge HTTP$port"
    if (-not (Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $rule -Direction Inbound -Protocol TCP `
            -LocalPort $port -Action Allow | Out-Null
        Write-Host "firewall rule added: $rule"
    } else {
        Write-Host "firewall rule already present: $rule"
    }
}

Write-Host ''
Write-Host 'Installed. Start now without rebooting:'
Write-Host '  Start-ScheduledTask -TaskName VisionEdge-Backend'
Write-Host '  Start-ScheduledTask -TaskName VisionEdge-Caddy'
Write-Host ''
Write-Host 'Your VPS provider may also have its own firewall  -  80 and 443 must be open there too.'
