# Deploying on a Windows VPS (no Docker)

The Docker path (`docker-compose.prod.yml`) uses Linux containers and needs WSL2 or
Hyper-V, which many Windows VPS plans cannot provide because nested virtualization
is disabled. This is the native alternative: the same single-origin design, with
Caddy and the backend running as ordinary Windows processes.

> **Verification status.** The Caddyfile is validated (`caddy validate` against
> Caddy 2.8), and the backend is the same code covered by 108 passing tests.
> `Start-Backend.ps1`, `Start-Caddy.ps1` and `Install-Services.ps1` were checked
> with the PowerShell 5.1 parser; `Deploy-VPS.ps1` was written after that route
> stopped working and has only had a structural check (balanced blocks, ASCII-only).
> **No script has been executed** — they were written on Linux and cannot run
> there. Treat the first deployment as a test and use the "Verify it worked"
> section rather than assuming.

## Quick start: one script

`Deploy-VPS.ps1` runs the whole sequence below — prerequisites, clone, venv, model
download, frontend build, IIS handover, scheduled tasks, firewall, and a
verification pass. From an **elevated** PowerShell prompt:

```powershell
# Add the DNS record first (see the DNS section), then:
git clone https://github.com/CedricMaron/visionedge-lab.git C:\visionedge-lab
cd C:\visionedge-lab
.\deploy\windows\Deploy-VPS.ps1
```

It is safe to re-run: each step checks whether it is already done, so fixing one
problem and running again does not redo the rest. It prompts once before stopping
IIS (pass `-Force` to skip), and if any later step fails it restarts IIS
automatically so the portfolio comes back.

Useful switches: `-SiteAddress`, `-PortfolioRoot`, `-RateLimitPerMin`,
`-SkipPrereqs` (check tools but never install them), `-Force`.

The rest of this document is the manual equivalent, and the reference for
diagnosing anything the script reports.

## Architecture

Caddy is the only public listener on the VPS and serves **two** sites:

| Hostname | Content |
|---|---|
| `c-maron.space`, `www.c-maron.space` | the existing static portfolio (moved off IIS) |
| `visionedge.c-maron.space` | this project's SPA, with `/api/*` and `/health` proxied to `127.0.0.1:8000` |

Same origin for the app means no CORS, and `wss://` derives from `https://`
automatically. The backend binds to loopback only, so it is unreachable except
through Caddy.

Both sites get automatic Let's Encrypt certificates, which is also the fix for the
portfolio currently redirecting to an HTTPS endpoint that presents no certificate.

> The two sites deliberately send **different** `Permissions-Policy` headers. The
> portfolio keeps `camera=()` as IIS was sending; the app needs `camera=(self)`
> because Live Inference calls `getUserMedia`. Copying the portfolio's header to
> the app would break the camera with no actionable error.

## Handing ports 80/443 over from IIS

IIS currently owns both ports, so Caddy cannot start until it releases them. Do
this in one sitting: between stopping IIS and Caddy obtaining certificates, the
portfolio is offline.

```powershell
# 1. Note where the portfolio files live - you need this path for Caddy.
#    IIS Manager > Sites > Default Web Site > Basic Settings > Physical path
#    (default: C:\inetpub\wwwroot)

# 2. Stop IIS and prevent it grabbing the ports again after a reboot.
Stop-Service W3SVC
Set-Service W3SVC -StartupType Manual

# 3. Confirm both ports are now free.
Get-NetTCPConnection -LocalPort 80,443 -State Listen -ErrorAction SilentlyContinue
#    No output = free. Start-Caddy.ps1 also refuses to start if they are taken.
```

**Rollback**, if anything goes wrong and you need the portfolio back immediately:

```powershell
Stop-ScheduledTask -TaskName VisionEdge-Caddy   # or Ctrl+C the foreground Caddy
Start-Service W3SVC
Set-Service W3SVC -StartupType Automatic
```

That restores the previous state exactly, including the broken HTTPS.

## Prerequisites

- Windows Server 2019+ or Windows 10/11, with Administrator access
- **Python 3.12** — <https://www.python.org/downloads/windows/> ("Add to PATH")
- **Node.js 20+** — <https://nodejs.org/> (build only; not needed at runtime)
- **caddy.exe** — <https://caddyserver.com/download>, placed on `PATH`
- A DNS `A`/`AAAA` record for your hostname pointing at the VPS
- Ports **80 and 443** open in Windows Firewall *and* your provider's firewall.
  Port 80 is not optional: Caddy uses it for the ACME challenge.

## Install

```powershell
git clone https://github.com/CedricMaron/visionedge-lab.git C:\visionedge-lab
cd C:\visionedge-lab

# --- backend ---
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install --upgrade pip
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements\base.txt

# --- model (a fresh clone has none; .onnx files are gitignored) ---
backend\.venv\Scripts\python.exe scripts\download_models.py --install yolov8n-onnx

# --- frontend ---
cd frontend
npm ci
npm run build
cd ..
```

> **Do not create `frontend\.env`.** Leaving `VITE_API_BASE` unset is what makes the
> built app address its own origin. Setting it to `http://localhost:8000` — as
> `.env.example` does for local development — would make the deployed site try to
> reach the visitor's own machine.

## Run

Foreground, to watch the first start and catch certificate problems:

```powershell
# terminal 1
.\deploy\windows\Start-Backend.ps1 -SiteAddress visionedge.c-maron.space

# terminal 2 - pass the portfolio path you noted from IIS Manager
.\deploy\windows\Start-Caddy.ps1 `
    -SiteAddress visionedge.c-maron.space `
    -PortfolioAddress c-maron.space `
    -PortfolioRoot 'C:\inetpub\wwwroot'
```

Watch the first Caddy start closely: it issues three certificates
(`c-maron.space`, `www.c-maron.space`, `visionedge.c-maron.space`). Let's Encrypt
allows only **5 failed attempts per hostname per hour**, so if it fails, read the
error before retrying — repeated blind retries will lock you out for an hour.

Once that works, register both to start at boot (elevated prompt):

```powershell
.\deploy\windows\Install-Services.ps1 -SiteAddress visionedge.c-maron.space
Start-ScheduledTask -TaskName VisionEdge-Backend
Start-ScheduledTask -TaskName VisionEdge-Caddy
```

If your portfolio root is not `C:\inetpub\wwwroot`, edit the `VisionEdge-Caddy`
task's arguments to add `-PortfolioRoot '<your path>'`, or the portfolio site will
serve the wrong directory.

`Install-Services.ps1` uses Scheduled Tasks so that nothing beyond Windows is
required — a Python process is not a service binary, so real services would mean
installing NSSM or WinSW. If you already use one of those, point it at the two
`Start-*.ps1` scripts instead.

## Verify it worked

Run these on the VPS, then from your own machine:

```powershell
# backend directly (should be reachable ONLY from the VPS itself)
curl.exe -s http://127.0.0.1:8000/health

# through Caddy
curl.exe -s https://visionedge.c-maron.space/health
curl.exe -s https://visionedge.c-maron.space/api/capabilities

# HTTP must redirect to HTTPS
curl.exe -s -o NUL -w "%{http_code} %{redirect_url}`n" http://visionedge.c-maron.space/

# the portfolio must still work - and now over working HTTPS
curl.exe -s -o NUL -w "%{http_code}`n" https://c-maron.space/
curl.exe -s -o NUL -w "%{http_code}`n" https://www.c-maron.space/
```

Expected: `{"status":"ok","detection_health":"ready","warnings":[]}` from both health
checks, `308` with an `https://` redirect target, and `200` for both portfolio URLs.
Before this change `https://c-maron.space/` failed the TLS handshake entirely, so a
`200` there is the signal that the handover worked.

Then open the site and check the **Live Inference** page — the WebSocket path
(`wss://.../api/ws/detect`) is the part a misconfigured proxy breaks first. Camera
capture requires HTTPS, which is why the certificate has to work before it will
function at all.

## What the numbers will say

The Device Capabilities page and every benchmark report the hardware they actually
run on. On a VPS that is the VPS's CPU, not the RTX 2060 the README's framing
describes. Two further Windows-specific notes:

- `_cpu_model()` reads `/proc/cpuinfo` and falls back to `platform.processor()` off
  Linux, so the CPU string will be coarser than on the development machine. It is
  still a real probe, not a placeholder.
- There is no GPU on a typical VPS, so `nvidia_gpu_present` will be `false` and
  CUDA will correctly report as unavailable.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Caddy hangs or fails at startup on certificates | Port 80 unreachable from the internet, or DNS not yet propagated. Both are required for ACME. |
| Site loads but every API call fails | The app was built with `VITE_API_BASE` set. Delete `frontend\.env`, rebuild, restart Caddy. |
| `detection_health` is not `ready` | The model is missing. Re-run `download_models.py --install yolov8n-onnx`; it verifies the checksum from the registry. |
| Live Inference connects then immediately drops | The WebSocket is not being proxied. Confirm `deploy\Caddyfile.windows` is the config actually in use. |
| Repeated certificate issuance / rate-limit errors | The scheduled task is running as a different account than the first issuance. Certificates live in the service account's profile — keep `-RunAsUser` stable. |
| 429 responses under normal use | `-RateLimitPerMin` is too low for your traffic. It defaults to 60 per IP per minute. |

## DNS (Namecheap)

The domain already resolves to the VPS, so only the subdomain is new. In
**Domain List > c-maron.space > Advanced DNS > Host Records > Add New Record**:

| Field | Value |
|---|---|
| Type | `A Record` |
| Host | `visionedge` (just the label; Namecheap appends the domain) |
| Value | `167.86.79.227` |
| TTL | `Automatic`, or 5 min while testing |

Leave the existing `@` and `www` A records alone - they point at the same VPS, and
Caddy serves all three hostnames. The domain has no CAA records, so Let's Encrypt
is free to issue.

Confirm it resolves before starting Caddy, or the ACME challenge fails:

```powershell
Resolve-DnsName visionedge.c-maron.space -Type A
```

| Portfolio 404s after the handover | `-PortfolioRoot` points at the wrong directory. Check IIS Manager > Default Web Site > Basic Settings > Physical path. |
| Camera blocked on Live Inference | A restrictive `Permissions-Policy` reached the app's site block. The app needs `camera=(self)`; only the portfolio should send `camera=()`. |
| Caddy exits immediately with a port error | IIS still holds 80/443. `Stop-Service W3SVC`, then confirm with `Get-NetTCPConnection -LocalPort 80,443 -State Listen`. |
