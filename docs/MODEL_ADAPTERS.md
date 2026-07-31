# Model adapters

A model adapter owns everything model-specific and nothing runtime-specific. That
split is what makes the (model × runtime) matrix real rather than a claim: the same
YOLOv8 adapter runs on ONNX Runtime CPU or CUDA without knowing which, and the same
ONNX runtime adapter serves detection, classification and embedding models without
knowing what their tensors mean.

## Contract

`app/adapters/base.py::ModelAdapter`

```python
class ModelAdapter(Protocol):
    metadata: ModelMetadata
    preprocess_phase: Phase          # PREPROCESSING, or TOKENIZATION for text

    def load(self, config: LoadConfig) -> LoadResult: ...
    def preprocess(self, request: InferenceRequest) -> PreparedInput: ...
    def infer(self, prepared: PreparedInput) -> RawOutput: ...
    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> InferenceOutput: ...
    def evaluate(self, predictions, references) -> QualityMetrics: ...
    def synthetic_request(self, batch_size: int = 1) -> InferenceRequest: ...
    def unload(self) -> None: ...
```

### Rules

**Adapters never time themselves.** The benchmark engine wraps each call in its own
span. An adapter that also timed would double-count and, worse, would make two
adapters' numbers incomparable.

**Adapters never create sessions.** They receive a `RuntimeAdapter` and call it.
No adapter imports `onnxruntime`.

**A load that did not honour its request must fail.** ONNX Runtime silently falls back
to CPU when a CUDA session cannot be created. Every adapter checks `handle.honored`
and raises rather than adopting the session, because otherwise every subsequent
latency and energy figure would be attributed to hardware that did no work. This is
not hypothetical — it is the actual behaviour on the reference machine, where
`libcublasLt.so.12` is missing.

**`preprocess_phase` declares what preprocessing means for this modality.** Text
adapters return `Phase.TOKENIZATION`, so the most informative phase of a text pipeline
is not hidden inside a generic bucket.

**`synthetic_request` must be deterministic.** The YOLOv8 adapter returns flat mid-grey
rather than noise: noise produces thousands of spurious low-confidence boxes, so NMS
cost would dominate and the measurement would reflect the postprocessor rather than
the model.

## Metadata

`ModelMetadata` requires both `model_license` and `weights_license`. They genuinely
differ — a model can be Apache-2.0 code with non-commercial weights — and reporting
only the first would mislead someone deciding whether they may ship it.
`commercial_use_permitted = None` means "not reviewed" and renders as *unreviewed*
rather than as permission.

`is_test_adapter` marks the mock. It is filtered out of production listings by
`ModelRegistry.production_models()`, at the registry rather than the presentation
layer, so a route that forgot to filter cannot leak a fabricated result.

## Implemented adapters

| Adapter | Task | Model | Licence | Notes |
|---|---|---|---|---|
| `detection/yolov8.py` | object detection | YOLOv8n ONNX | AGPL-3.0 | From-scratch NumPy decode + NMS, validated against Ultralytics |
| `classification/mobilenet.py` | image classification | MobileNetV4 Conv Small | Apache-2.0 | timm preprocessing read from `config.json` |
| `embedding/minilm.py` | text embedding | all-MiniLM-L6-v2 | Apache-2.0 | Attention-masked mean pooling + L2 normalization |
| `mock.py` | — | none | MIT | **Test only.** Fabricates output; three independent guards |

## Correctness traps encountered

These are the mistakes that fail *silently* rather than raising, which is why each has
a test:

- **Classification preprocessing must match training.** Resize by `1/crop_pct`,
  centre-crop, scale to [0,1], normalize by ImageNet statistics. Any deviation
  degrades accuracy without any error. The parameters are read from the model's own
  `config.json` rather than hardcoded.
- **Class labels must be ordered by numeric index.** String-sorting the keys of an
  `id2label` map puts class 10 before class 2.
- **Logits are not probabilities.** A max-shifted softmax is applied; reporting raw
  logits as confidences is wrong by a per-image factor.
- **Mean pooling must be attention-masked.** Averaging over padding still yields
  unit-norm vectors, so the failure is invisible to a shape or norm check — the test
  asserts semantic ordering (`cos(dog, puppy) > cos(dog, finance)`) instead.
- **Batched requests must be refused when the export is static.** Quietly dropping
  images would report throughput for work that never happened.

## Adding an adapter

1. Implement the protocol in `app/adapters/<task>/<name>.py`.
2. Declare complete `ModelMetadata`, including both licences and `known_limitations`.
3. Set `preprocess_phase`.
4. Add a `models/registry.json` entry with checksums and any `companion_files`.
5. Register it in `app/cli.py::_MODELS`.
6. Add a scenario under `benchmarks/scenarios/` if the task is new.
7. Test: an equivalence or correctness test that would fail on a silent-degradation
   bug, not only on a crash.
