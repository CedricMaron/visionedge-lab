# Deploying on a Windows VPS (no Docker)

The Docker path (`docker-compose.prod.yml`) uses Linux containers and needs WSL2 or
Hyper-V, which many Windows VPS plans cannot provide because nested virtualization
is disabled. This is the native alternative: the same single-origin design, with
Caddy and the backend running as ordinary Windows processes.

> **Verification status.** The Caddyfile is validated (`caddy validate` against
> Caddy 2.8), and the backend is the same code covered by 108 passing tests. The
> PowerShell scripts have **not** been executed — they were written on Linux and
> cannot be run there. Treat the first deployment as a test, and check the
> "Verify it worked" section rather than assuming.

## Architecture

Identical to the Linux deployment. Caddy is the only public listener; it serves the
built SPA and reverse-proxies `/api/*` and `/health` to the backend on
`127.0.0.1:8000`. Same origin means no CORS, and `wss://` derives from `https://`
automatically. The backend binds to loopback only, so it is unreachable except
through Caddy.

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
$env:SITE_ADDRESS = "visionedge.c-maron.space"

# terminal 1
.\deploy\windows\Start-Backend.ps1

# terminal 2
.\deploy\windows\Start-Caddy.ps1
```

Once that works, register both to start at boot (elevated prompt):

```powershell
.\deploy\windows\Install-Services.ps1 -SiteAddress visionedge.c-maron.space
Start-ScheduledTask -TaskName VisionEdge-Backend
Start-ScheduledTask -TaskName VisionEdge-Caddy
```

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
```

Expected: `{"status":"ok","detection_health":"ready","warnings":[]}` from both health
checks, and `308` with an `https://` redirect target.

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
