# Vision-Language Model (VLM) Architecture

This document explains how the vision-language slice of VisionEdge Lab is designed: how a
VLM differs from the object detector, the backend interface, how detector output *grounds*
the VLM, structured output, and the honest status of each backend.

> Honesty note up front: the always-on default VLM in this build is a **deterministic
> mock** that writes sentences from detector context — it does not look at pixels and is
> not a neural model. The real local model (SmolVLM) and the remote server model
> (Qwen2.5-VL class) are **opt-in**: they require installing `transformers` (local) or
> configuring a remote endpoint. Nothing here fabricates a model that is not loaded.

## 1. Detector vs. VLM: two different jobs

The YOLOv8 detector answers a **closed-vocabulary, geometric** question: *where are the 80
COCO classes in this frame?* It returns boxes, class ids and confidences. It cannot say
"the person on the left is reaching for the door", because "reaching for a door" is neither
a box nor a COCO class.

A **vision-language model** answers **open-ended language** questions about an image:
describe the scene, answer a question, extract a structured summary. It maps pixels (and a
text prompt) to text. The two are complementary:

| | Object detector (YOLOv8) | Vision-language model |
| --- | --- | --- |
| Output | boxes + class id + score | free text (and optional JSON) |
| Vocabulary | fixed 80 COCO classes | open |
| Question it answers | *where / how many of class X* | *what is happening / why / describe* |
| Latency | ~tens of ms (CPU) | ~hundreds of ms to seconds |
| Failure mode | misses novel objects, no semantics | hallucination, ungrounded claims |

VisionEdge Lab uses the detector as the fast, reliable spatial layer and the VLM as an
opt-in semantic layer that is *grounded on* — but never blindly trusting — the detector.

## 2. Two families of vision-language models

There are two broad architectural families, and the platform is aware of both:

- **Embedding / dual-encoder models (CLIP-style).** An image encoder and a text encoder
  are trained to place matching image–text pairs near each other in a shared space
  (Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*,
  2021, <https://arxiv.org/abs/2103.00020>). These are great for retrieval, zero-shot
  classification and similarity, but they do **not generate** language.
- **Generative VLMs (LLaVA / SmolVLM / Qwen-VL style).** A vision encoder produces visual
  tokens that are projected into the embedding space of a language model, which then
  *generates* text autoregressively. This is the family used for description and VQA here.
  - LLaVA established the "vision encoder → projector → LLM, tuned on visual instructions"
    recipe: Liu et al., *Visual Instruction Tuning*, NeurIPS 2023,
    <https://arxiv.org/abs/2304.08485>.
  - **SmolVLM** is a small, efficient generative VLM designed to run on modest hardware:
    Marafioti et al., *SmolVLM: Redefining small and efficient multimodal models*, 2025,
    <https://arxiv.org/abs/2504.05299>. The `smolvlm-256m` / `smolvlm-500m` entries use
    `HuggingFaceTB/SmolVLM-256M-Instruct` and `-500M-Instruct`.
  - **Qwen2.5-VL** is a larger, more capable generative VLM (grounding, video, long
    context): Bai et al., *Qwen2.5-VL Technical Report*, 2025,
    <https://arxiv.org/abs/2502.13923>. The `qwen2.5-vl-3b` entry targets a GPU server.

This build integrates the **generative** family (mock, SmolVLM, remote Qwen-class). The
detector's own embeddings are used for the JEPA/representation experiments (see
`JEPA_ARCHITECTURE.md`), not for VLM retrieval.

## 3. The `VisionLanguageBackend` interface

The contract lives in `backend/app/vlm/base.py` as a `typing.Protocol`
(`@runtime_checkable`), with a `BaseVLMBackend` providing shared defaults. Every backend
exposes the same surface:

```python
class VisionLanguageBackend(Protocol):
    model_id: str
    runtime: str               # "python" | "transformers" | "remote-openai-compatible"
    execution_location: str    # "pc_local" | "remote_server"

    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def describe_image(self, image, prompt=None, grounding=None) -> VLMResponse: ...
    def answer_question(self, image, question, grounding=None) -> VLMResponse: ...
    def analyze_video(self, frames, prompt, grounding=None) -> VLMResponse: ...
    def health(self) -> HealthState: ...
    def unload(self) -> None: ...
```

- `image` is a BGR `uint8` H×W×3 NumPy array — the **same** format the detector consumes,
  so a single decoded frame feeds both.
- `grounding` is an optional dict of detector context (see §5).
- Every call returns a `VLMResponse` (pydantic, `backend/app/core/types.py`) carrying not
  just `text` and optional `structured_output`, but the honesty/telemetry fields the
  evaluation layer needs: `prompt_tokens`, `generated_tokens`, `time_to_first_token_ms`,
  `generation_latency_ms`, `total_latency_ms`, `memory_usage_mb`, and a `warnings` list.

There is deliberately **no request object** — inputs are passed positionally, keeping the
mock and the real backends trivially interchangeable.

### The four backends

| Backend | Class (`app/vlm/…`) | runtime / location | Status in this build |
| --- | --- | --- | --- |
| Mock | `MockVLMBackend` | `python` / `pc_local` | **Always on, default.** Deterministic. |
| Local | `LocalTransformersVLM` | `transformers` / `pc_local` | Opt-in; needs `transformers` + weights. |
| Remote | `RemoteVLMBackend` | `remote-openai-compatible` / `remote_server` | Opt-in; needs a configured endpoint. |
| Base | `BaseVLMBackend` | — | Shared scaffolding, not used directly. |

The **`VLMManager`** (`app/vlm/manager.py`) owns selection. `mock-vlm` → mock;
`smolvlm-256m` / `smolvlm-500m` → local; anything else, when `VE_VLM_REMOTE_URL` is set →
remote; otherwise a `ModelNotFoundError`. `switch()` implements **last-known-good
rollback**: it builds, loads and warms the new backend and, on any failure, keeps the
previous one and reports `{ok: False, rolled_back: True}` — a bad switch never takes the
VLM offline.

## 4. The mock is deterministic — and says so

`MockVLMBackend` reads `grounding["detected_objects"]`, counts class names, and emits a
sentence such as *"The scene appears to contain 2 persons and 1 chair."* It answers "how
many" questions from the detector counts and **refuses** questions it cannot ground (e.g.
indoor/outdoor). Every response carries the warning *"mock VLM: deterministic output
derived from detector context, not a real model."* Token counts are word counts; latencies
are real wall-clock measurements. This makes the whole pipeline runnable and testable on a
CPU-only box with zero downloads, without ever pretending to be a real model.

## 5. Detector grounding (and why it is a *hint*, not truth)

`app/vlm/prompting.py` turns detections into grounding and injects them into the prompt:

- `detections_to_grounding(dets, w, h)` → `{"detected_objects": [{"class", "confidence",
  "box":[normalized x1,y1,x2,y2]}]}`.
- `format_grounding_context(...)` renders that as a hint block prefixed with: *"An object
  detector reported the following (it may be incomplete or wrong; use it only as a hint and
  trust the image over it)."*

This framing is intentional. Grounding **narrows hallucination** (the VLM is nudged toward
objects that are actually present) without turning detector mistakes into asserted facts.
The API attaches a `detector_agreement` block and a disclaimer that VLM output is
model-generated interpretation, not verified truth. Neither model is assumed correct.

## 6. Structured output

`app/vlm/structured_output.py` defines a pydantic `SceneUnderstanding` schema (`summary`,
`environment`, `people[]`, `important_objects[]`, `actions[]`, `possible_risks[]`,
`uncertainties[]`). When structured mode is requested the manager appends
`STRUCTURED_INSTRUCTION` ("Respond ONLY with a JSON object matching this schema: …") and
then validates with `parse_structured(text) -> (SceneUnderstanding | None, warnings, raw)`:
it extracts the first balanced `{…}`, `json.loads` it, and `model_validate`s it. On missing
JSON, a decode error, or a validation error it returns `None` plus a warning and **always
preserves the raw text** — it never invents a valid-looking object to satisfy the schema.
The presence of an `uncertainties` field is deliberate: the model is asked to declare what
it is unsure about rather than to sound confident.

## 7. Temporal / video analysis — honest scope

`analyze_video(frames, …)` exists on every backend, but the current implementations are
**single-frame**: the mock and local backends analyze the **middle frame** and append a
limitation warning; there is no cross-frame temporal *reasoning* inside the VLM. Genuine
temporal understanding in this platform lives in the JEPA/temporal slice (future-embedding
prediction and anomaly-from-surprise; see `WORLD_MODEL_EXPERIMENT.md`), which can *select*
informative frames to hand to the VLM. Multi-frame temporal VLM reasoning is
**interface-defined; implementation planned (later phase)**.

## 8. Privacy: remote is off by default

`RemoteVLMBackend` talks to an OpenAI-compatible `/chat/completions` endpoint, sending the
frame as a base64 JPEG. It is gated by `allow_frame_transmission` (env
`VE_ALLOW_FRAME_TRANSMISSION`). When that flag is false the backend returns immediately with
empty text and the warning *"frame transmission … disabled … No frame was sent."* — no
bytes leave the machine. Retries use bounded backoff and, on exhaustion, raise rather than
returning a fabricated answer.

## 9. Evaluation pointer

VLM quality is not scored by exact-match. `app/vlm/evaluation.py` provides lexical,
non-fabricating checks — `keyword_coverage`, `object_count_consistency`,
`detector_agreement` — while latency/TTFT/token metrics ride on `VLMResponse`. See
`MULTIMODAL_EVALUATION.md` for the full rationale.

## References

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*
  (CLIP), 2021. <https://arxiv.org/abs/2103.00020>
- Liu et al., *Visual Instruction Tuning* (LLaVA), NeurIPS 2023.
  <https://arxiv.org/abs/2304.08485>
- Marafioti et al., *SmolVLM: Redefining small and efficient multimodal models*, 2025.
  <https://arxiv.org/abs/2504.05299>
- Bai et al., *Qwen2.5-VL Technical Report*, 2025. <https://arxiv.org/abs/2502.13923>
