# VisionEdge Lab — Platform Upgrade Design

**Date:** 2026-07-26
**Status:** Approved, ready for implementation planning

Four independent workstreams: device-class detection, live latency attribution,
a SOTA model catalog with RT-DETR made genuinely runnable, and a light theme
built on semantic design tokens.

Each workstream can be built, tested and shipped on its own. They share no state
beyond the existing type contracts.

---

## Governing constraint: honesty first

The README's standing promise is that this project "never fabricates capabilities,
benchmark numbers, or model quality". Every workstream below is bounded by it:

- Device class is **derived from probed browser APIs**, never from parsing a
  user-agent string, and the evidence is shown to the user.
- Every latency bar is **one real measurement**. The one quantity that cannot be
  honestly decomposed (uplink vs. downlink) is shown merged, with the reason stated.
- Catalog models carry **author-reported** metrics, labelled as such with a source
  link, kept strictly separate from the Benchmarks page which stays measured-only.
  Any figure that cannot be sourced is omitted, not estimated.
- A model that cannot run here says so, and says why.

---

## Workstream 1 — Device class detection (phone vs. PC)

### Problem

`DeviceCapabilitiesPage` reports browser feature probes but never answers the
first question a visitor has: *what kind of device am I on?* Separately,
`backend/app/api/detection.py:202` hardcodes `execution_location=PC_LOCAL` into
every session record and stores the raw user-agent as `client_device`, so session
history is wrong for any phone visitor.

### Classifier

New `frontend/src/utils/deviceClass.ts`. Weighted evidence vote over probed signals:

| Signal | API | Weight | Reads as phone when |
|---|---|---|---|
| Mobile hint | `navigator.userAgentData.mobile` | strong (3) | `true` |
| Coarse pointer | `matchMedia('(pointer: coarse)').matches` | strong (3) | `true` |
| Fine pointer absent | `matchMedia('(any-pointer: fine)').matches` | strong (3) | `false` |
| Touch points | `navigator.maxTouchPoints` | medium (2) | `> 0` |
| Screen min side | `min(screen.width, screen.height)` | medium (2) | `< 500` CSS px |
| Logical cores | `navigator.hardwareConcurrency` | weak (1) | `<= 8` |
| Device memory | `navigator.deviceMemory` | weak (1) | `<= 4` GB |

`navigator.userAgentData.mobile` is a structured boolean exposed by the User-Agent
Client Hints API. It is a browser-provided value, not a parsed string, so it does
not violate the no-UA-guessing rule. It is Chromium-only; absence contributes no vote.

**Output type:**

```ts
type DeviceClass = 'phone' | 'tablet' | 'desktop' | 'unknown';

interface DeviceEvidence {
  signal: string;        // human label, e.g. "Coarse pointer"
  value: string;         // observed value rendered for display, or "unavailable"
  available: boolean;    // whether the API answered at all
  vote: 'phone' | 'desktop' | 'none';
  weight: number;
}

interface DeviceClassification {
  deviceClass: DeviceClass;
  confidence: number;    // 0..1 = |phoneScore - desktopScore| / totalAvailableWeight
  evidence: DeviceEvidence[];
}
```

**Decision rules**, applied in order:

1. If total available weight is 0 → `unknown`, confidence 0.
2. Tablet: coarse pointer **and** min screen side >= 500 CSS px → `tablet`.
3. Otherwise the higher of `phoneScore` / `desktopScore` wins; a tie → `unknown`.

Screen size uses CSS pixels (`screen.width`), not physical pixels, so a
high-DPR phone is not misread as a desktop.

### Surfaces

- **Device Capabilities page** — a new card at the top: the verdict as a large
  badge (PHONE / TABLET / PC), the confidence, and the full evidence table with one
  row per signal showing value, availability and vote. Unavailable signals are shown
  greyed with "unavailable", never silently dropped.
- **TopBar** — a compact badge with the verdict, visible on every page.
- **Model Selector** — the `execution_location` default is derived: a phone or
  tablet client defaults to `remote_server`, a desktop client to `local_server`.
  `phone_local` stays selectable but renders with an existing-style "planned" marker,
  because browser-local inference is not yet wired (README lists it as planned).

### Backend session fix

The WebSocket client sends a JSON text message immediately after connect:

```json
{ "type": "hello", "device_class": "phone", "confidence": 0.86 }
```

`ws_detect` currently ignores text messages (`if data is None: continue`). It will
parse a `hello` message, validate `device_class` against the four literals, and use
it for `db.upsert_session(client_device=..., execution_location=...)`:

- `phone` / `tablet` → `ExecutionLocation.REMOTE_SERVER`
- `desktop` / `unknown` / no hello received → `ExecutionLocation.LOCAL_SERVER`

Any malformed or absent hello falls back to the same default as today's behaviour,
so an old client cannot break the session write. The session upsert moves to after
the first message (or a short grace period) so the hello is available; if none
arrives the fallback is written.

### Tests

`frontend/src/utils/deviceClass.test.ts` — classifier over injected signal sets:
phone-like, desktop-like, tablet-like, all-signals-unavailable, exact tie.
The classifier takes its raw signals as an injectable argument so tests never
have to monkey-patch `navigator`.

Backend: extend `backend/tests/test_api.py` with a WS test asserting the session
record reflects a `hello`-declared phone, and another asserting the fallback when
no hello is sent.

---

## Workstream 2 — Latency attribution in Live Inference

### Problem

The live view reports one number: `inference_ms`. The WS worker calls
`state.detection.predict` (untimed) rather than `predict_timed`, so even the
per-stage split the backend already computes for `/api/infer` is thrown away on
the live path. Nothing attributes time to image encoding, transport, or drawing.

### What is measurable, and what is not

One-way network delay cannot be measured between two clocks with an unknown offset.
A round trip can. So the design measures every stage that sits on one machine, and
reports transport as a single derived round-trip bucket.

| Stage | Measured where | How |
|---|---|---|
| capture + encode | client | `performance.now()` around `drawImage` + `toBlob` |
| network + queueing | derived | `rtt - server_total_ms` |
| queue (backpressure) | server | frame receive timestamp → worker pickup |
| decode | server | around `decode_image_bytes` |
| preprocess | server | `predict_timed` |
| inference | server | `predict_timed` |
| postprocess | server | `predict_timed` |
| draw overlay | client | `performance.now()` around `drawDetections` |

`rtt` is measured entirely on the client clock: `now - sendTs[client_frame_id]`.
`server_total_ms` is measured entirely on the server clock: receive → just before
`send_json`. Their difference is a duration, not a timestamp comparison, so no
clock sync is required and the result is exact up to timer resolution.

The `network + queueing` bucket contains uplink, downlink, TLS/WS framing and any
socket-level queueing. The UI states this explicitly and states why it is not split.

### Wire protocol

Frames stay binary. The client prefixes an 8-byte header:

```
bytes 0..3   ASCII magic "VE01"
bytes 4..7   uint32 little-endian client_frame_id
bytes 8..    JPEG payload
```

Header parsing is **optional** on the server: if the payload does not start with
`VE01`, the whole payload is the JPEG and `client_frame_id` is `null`. Existing
tests and any older client keep working unchanged.

The reply gains fields (all additive — existing consumers ignore them):

```json
{
  "frame_id": 412,
  "client_frame_id": 987,
  "detections": [],
  "timings": {
    "inference_ms": 41.7,
    "queue_ms": 2.1,
    "decode_ms": 4.2,
    "preprocess_ms": 3.8,
    "postprocess_ms": 1.9,
    "server_total_ms": 53.7
  },
  "dropped": 3,
  "backend": "onnxruntime-cpu"
}
```

`server_total_ms` is measured independently, not summed from the stages; the
difference between it and the sum is real scheduler/threadpool overhead and the UI
shows it as an "other (server)" remainder rather than hiding it.

### Client bucket math

```
network_ms = max(0, rtt_ms - server_total_ms)
other_server_ms = max(0, server_total_ms - (queue + decode + preprocess + inference + postprocess))
total_ms = capture_ms + rtt_ms + draw_ms
```

If `rtt_ms - server_total_ms` is negative — possible when both are near timer
resolution — the bucket clamps to 0 and the UI renders "below measurement
resolution" for that bar instead of a number. It never renders a negative or an
invented value.

### UI

New `frontend/src/components/LatencyBreakdown.tsx`, placed in the Live Inference
left column under the existing stat cards.

- **Pipeline diagram** — the stages as a left-to-right row of labelled boxes
  (Camera → Encode → Network → Decode → Preprocess → Model → Postprocess → Network → Draw),
  each box's width proportional to its share of total. Client stages, transport and
  server stages are visually grouped so the machine boundary is legible. Built with
  flexbox and CSS, no charting dependency.
- **Per-stage table** — stage, instantaneous ms, rolling p50 ms, % of total.
  Single-frame numbers jitter heavily, so p50 over the last 30 frames is the
  headline figure and the instantaneous value sits beside it.
- **Footnote** — states that transport is round-trip and why it is not split.

Rolling state lives in a new `frontend/src/hooks/useLatencyStats.ts` keeping a
30-sample ring buffer per stage. `useDetectionSocket` gains the per-frame timing
payload and the RTT; the send path records `sendTs` in a `Map` keyed by
`client_frame_id`, pruned to the last 120 entries so a dropped frame cannot leak.

### Tests

- `frontend/src/hooks/useLatencyStats.test.ts` — ring buffer, p50 across an even
  and an odd sample count, clamping of negative network, "other (server)" remainder.
- `frontend/src/hooks/useDetectionSocket.test.ts` (extend) — header is written on
  send; timings and `client_frame_id` are read back; RTT is computed against the
  right send timestamp; the sendTs map is pruned.
- `backend/tests/test_api.py` (extend) — WS accepts a `VE01`-prefixed frame and
  echoes `client_frame_id`; WS accepts a bare JPEG and returns `client_frame_id: null`;
  the reply carries all six timing fields and `server_total_ms > 0`.

---

## Workstream 3 — SOTA model catalog, with RT-DETR made runnable

### 3a. Registry schema

`DetectionModelEntry` gains optional fields. All default to `None`/absent, so
`models/registry.json` stays valid and no existing entry needs editing beyond
opting into the new metadata.

| Field | Type | Purpose |
|---|---|---|
| `task_family` | str | `detection`, `open_vocabulary`, `segmentation`, `tracking`, `vision_language`, `predictive` |
| `decoder` | str | `yolov8`, `rtdetr`, `none` — selects the postprocessor |
| `nms_required` | bool | drives the NMS-free badge |
| `paper_url` | str \| None | verified before commit |
| `repo_url` | str \| None | verified before commit |
| `parameters_m` | float \| None | author-reported |
| `reported_coco_map` | float \| None | author-reported, omitted when unsourced |
| `metrics_source` | str \| None | URL backing the two fields above |
| `install_hint` | str \| None | concrete command or steps |
| `not_installed_reason` | str \| None | e.g. "requires GPU — this server is CPU-only" |

`decoder` defaults to `yolov8` so existing entries behave exactly as today.

**Validation rules**, enforced by a registry test rather than by Pydantic (so the
data can be fixed without a schema migration):

- every entry with `deployment_status != "installed"` has a non-empty
  `not_installed_reason` **and** an `install_hint`;
- `reported_coco_map` and `parameters_m` are present only together with
  `metrics_source`;
- `repo_url` and `paper_url`, when present, are `https://` URLs;
- `decoder` is one of the three literals, and `decoder != "none"` implies the
  entry declares at least one supported runtime.

### 3b. Catalog contents

Roughly 22 entries across six task families. Existing YOLOv8 entries are retained
and enriched, and explicitly framed as the **baseline** rather than the highlight.

- **Detection** — RF-DETR (nano/small/base), RT-DETR (l, x), D-FINE (s), RTMDet (s),
  YOLO11 (n), YOLOv8 (n/s/m, existing), PicoDet (s)
- **Open vocabulary** — Grounding DINO (tiny), Grounding DINO 1.5, OWLv2 (base),
  Florence-2 (base)
- **Segmentation** — SAM2 (hiera-tiny), EfficientSAM
- **Tracking** — ByteTrack, BoT-SORT
- **Vision-language** — Qwen2.5-VL (3B), InternVL3 (2B), MiniCPM-V
- **Predictive** — V-JEPA 2

`deployment_status` is not a hand-maintained claim — `refresh_deployment_status`
already derives it from whether the file exists on disk. `rt-detr-l` ships with a
`download_url` and is added to `scripts/download_models.py`, so a fresh deploy
fetches it and the status resolves to `installed`; on a machine where it has not
been fetched the UI honestly shows `missing`. Every other catalog entry is
`not_installed` with a specific reason and no download path. Repository URLs, paper URLs,
licences and author-reported metrics are **verified against the actual sources
during implementation**; anything that cannot be verified is omitted rather than
guessed. The user-supplied arXiv id for RF-DETR (2511.09554) is treated as a lead
to verify, not as a fact.

### 3c. RT-DETR made genuinely runnable

**New `backend/app/inference/postprocess_detr.py`** — a from-scratch NumPy decoder
for the DETR-family head, mirroring how `postprocess.py` decodes YOLOv8 without the
ultralytics runtime:

- output is `[1, num_queries, 4 + num_classes]` (typically `[1, 300, 84]`);
- the 4 box channels are normalized `cx, cy, w, h` in `[0, 1]` — multiply by the
  model input size to reach model-space pixels, matching what `decode_yolov8` returns
  so the existing letterbox-unmapping path is reused unchanged;
- class channels are **logits**; apply sigmoid, take per-query argmax and max;
- filter by confidence and by `allowed_class_ids` before selection, same ordering
  guarantee as the YOLOv8 path;
- **no NMS and no suppression of any kind** — the surviving queries are simply
  sorted by score, descending. RT-DETR's one-to-one bipartite matching means each
  object is already claimed by a single query, which is the architectural point of
  the model; the code comments say so.

The exact exported tensor layout is verified against a real export before the
decoder is finalized; the transpose-tolerant normalization helper from
`postprocess.py` is reused so a `[1, 84, 300]` layout is also handled.

**Decoder selection.** `OnnxRuntimeBackend` (and the OpenVINO/TensorRT backends,
which share the decode step) currently call `decode_yolov8` directly. They gain a
`decoder` constructor argument, passed by `build_backend` from the registry entry,
dispatching to `decode_yolov8` or `decode_rtdetr`. Unknown decoder → `ConfigInvalidError`.

**Export script** — `scripts/export_rtdetr_onnx.py`, following the shape of the
existing `scripts/export_onnx.py`: export `rtdetr-l` to ONNX, record file size and
SHA-256, and print the registry fields to paste. The exported artifact is not
committed to git (consistent with the current `models/` handling); the registry
`download_url` points at the release asset and `deployment_status` reflects whether
the file is present, which `refresh_deployment_status` already handles.

**Validation** — `scripts/validate_onnx.py` is extended (or a sibling added) to
confirm agreement between the RT-DETR ONNX path and a PyTorch reference on a fixed
image, the same way the YOLOv8 NumPy decoder was validated against Ultralytics.

### 3d. Model Selector redesign

- **Grouped by task family**, one section per family, with a count per section.
- **Filter chips** — runnable on this server / real-time capable / architecture
  (CNN · DETR · ViT · VLM · SSL) / permissive licence.
- **Card** — display name, architecture badge, NMS-free badge where applicable,
  licence, author-reported params and mAP with a "reported by authors" label linking
  to `metrics_source`, repo and paper links, and a status pill. Not-installed cards
  show `not_installed_reason` and `install_hint`.
- **Capability matrices.** The Runtime row (CUDA · TensorRT · ONNX Runtime ·
  OpenVINO · CoreML · CPU) and the Task row (Detection · Tracking · Segmentation ·
  Pose · OCR · Captioning · VQA · JEPA) are rendered from the intersection of the
  model's declared support and the **live `/api/capabilities` probe**. A runtime the
  server does not have renders disabled with "not available on this server". This
  ties the platform UI to real measurement instead of being decorative.

The page currently holds list, filter, config-panel and job-polling concerns in one
file. The catalog grid, the filter bar and the capability matrix move into
`frontend/src/components/` as separate components so `ModelSelectorPage` stays a
composition rather than growing past readable size.

### Tests

- `backend/tests/test_postprocess_detr.py` — normalized box scaling; sigmoid
  applied to logits; transpose-tolerant layout; class filtering before selection;
  score ordering; empty result on an all-low-confidence tensor; **no suppression of
  overlapping same-class boxes** (the property that distinguishes it from the YOLO path).
- `backend/tests/test_registry_and_config.py` (extend) — the four validation rules
  above hold for every catalog entry; existing entries still parse; `decoder`
  defaults to `yolov8`.
- Factory test — a registry entry with `decoder: "rtdetr"` builds a backend whose
  decode path is the DETR one; an unknown decoder raises `ConfigInvalidError`.
- `frontend/src/pages/ModelSelectorPage.test.tsx` (extend) — grouping by family,
  filter chips narrow the grid, a runtime absent from the capability probe renders
  disabled.

---

## Workstream 4 — Light theme via semantic tokens

### Problem

165 hard-coded dark palette classes (`surface-*`, `slate-*`) across 19 files. A
find-and-replace to lighter greys would leave the codebase with a `surface-950`
that means near-white, and would have to be repeated for any future change.

### Tokens

`frontend/src/index.css` defines CSS variables on `:root` as space-separated RGB
channels; `tailwind.config.js` maps colour names to `rgb(var(--token) / <alpha-value>)`
so opacity modifiers (`bg-accent/10`, used widely today) keep working.

| Token | Role |
|---|---|
| `--canvas` | page background |
| `--panel` | card background |
| `--elevated` | inputs, inset wells, hover surfaces |
| `--border-subtle` / `--border-strong` | hairlines / emphasized borders |
| `--text-primary` / `--text-secondary` / `--text-muted` | body / labels / hints |
| `--accent` / `--accent-hover` / `--accent-contrast` | primary action + its foreground |
| `--good` / `--warn` / `--bad` | status |

Class names become semantic: `bg-canvas`, `bg-panel`, `border-subtle`,
`text-secondary`, and so on. The sweep across 19 files is mechanical, and afterwards
the palette lives in one place.

The `surface-*` and dark `slate-*` scales are **removed** from the Tailwind config
once the sweep is complete, so a stray old class fails the build rather than
silently rendering a dark patch on a light page.

### Palette

Light is the default. `#38bdf8` reads toy-ish on white, so the accent moves to a
deeper professional blue; the exact value is chosen during implementation subject to
the contrast rule below. Status colours are darkened from their dark-theme values so
they remain legible on a light background.

**Contrast rule:** every text-on-background pair used in the app meets WCAG AA
(4.5:1 for body text, 3:1 for large text and for UI borders). This is checked with a
computed-contrast unit test over the token table, not by eye, so the theme cannot
silently regress.

The video viewport keeps its black background — correct for video content.

### Dark toggle

Because the palette is now a variable set, dark mode is a second block of variable
values under `:root[data-theme="dark"]`. A persisted toggle lives in Settings,
stored in the existing `settingsStore`, applied by stamping `data-theme` on
`document.documentElement`. **Default is light**; the toggle is opt-in and is not
surfaced in the TopBar.

### Tests

- `frontend/src/theme/contrast.test.ts` — parses the token table and asserts the
  AA ratios for each documented text/background pair, in both light and dark values.
- `frontend/src/stores/settingsStore` (extend its existing coverage) — theme
  persists and rehydrates; default is `light`.
- A grep-based check in CI (or a lint rule) that no `surface-[0-9]` class remains.

---

## Sequencing

The workstreams are independent and can land as separate commits in any order.
The suggested order minimizes conflicts:

1. **Workstream 4 (theme)** first — it touches the most files, and landing it early
   means the components added later are written against semantic tokens directly
   rather than being rewritten.
2. **Workstream 1 (device class)** — small, self-contained, adds one component.
3. **Workstream 2 (latency)** — backend WS changes plus one new component.
4. **Workstream 3 (models)** — the largest, and the only one requiring a real
   model export and external source verification.

## Out of scope

- Making RF-DETR, Grounding DINO, Florence-2, SAM2 or any VLM actually runnable.
  They are catalog entries with honest status and install instructions.
- Browser-local (`phone_local`) inference. Interfaces exist; wiring them is a
  separate project.
- Splitting uplink from downlink latency. Requires a clock-sync handshake and would
  make two of the bars estimates.
- Tracking, pose, OCR and captioning task implementations. The task matrix reports
  what each catalogued model supports; it does not run them.
