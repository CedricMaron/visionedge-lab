"""Sentence embedding over an ONNX MiniLM graph.

This is the third modality and the first text one, so it is also where the
tokenization phase of the timeline becomes real rather than theoretical.

Two postprocessing steps are not optional and are the usual source of embeddings
that "work" but retrieve badly:

**Mean pooling must be attention-masked.** The graph emits a per-token hidden
state; sentence-transformers averages over *real* tokens only. Averaging over
padding too pulls every short sentence toward the padding vector, which quietly
destroys similarity ranking without any error.

**Vectors must be L2-normalized.** all-MiniLM-L6-v2 is trained with cosine
similarity, so downstream code is entitled to assume unit norm and use a plain dot
product.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.adapters.base import (
    HardwareRequirement,
    InferenceOutput,
    InferenceRequest,
    LoadConfig,
    LoadResult,
    ModelMetadata,
    PreparedInput,
    RawOutput,
    ReferenceOutput,
)
from app.core.errors import ConfigInvalidError, ModelLoadError
from app.core.logging import get_logger
from app.runtimes.base import RuntimeAdapter, SessionConfig, SessionHandle
from app.schemas.enums import DeviceKind, Modality, Phase, Precision, Task
from app.schemas.measurement import Measurement
from app.schemas.quality import EmbeddingQuality, QualityMetrics

log = get_logger("adapters.minilm")

DEFAULT_MAX_LENGTH = 256


def mean_pool(hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Average token vectors over real tokens only.

    ``hidden_state`` is (batch, seq, dim); ``attention_mask`` is (batch, seq) with 1
    for real tokens. The clip guards a degenerate all-padding row from dividing by
    zero and producing NaNs that would propagate into a stored vector.
    """
    mask = attention_mask[..., None].astype(np.float32)
    summed = (hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class MiniLmEmbeddingAdapter:
    """Text embedding. Tokenization is a first-class, separately-timed phase."""

    #: Text preprocessing IS tokenization; naming it 'preprocessing' would hide the
    #: most informative phase of a text pipeline.
    preprocess_phase = Phase.TOKENIZATION

    def __init__(
        self,
        model_path: str | Path,
        runtime: RuntimeAdapter,
        tokenizer_path: str | Path | None = None,
        model_id: str = "all-minilm-l6-v2-onnx",
        display_name: str = "all-MiniLM-L6-v2 (ONNX)",
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_path = Path(model_path)
        self.runtime = runtime
        self.tokenizer_path = (
            Path(tokenizer_path) if tokenizer_path else self.model_path.parent / "tokenizer.json"
        )
        self.max_length = max_length
        self._handle: SessionHandle | None = None
        self._tokenizer = None
        self.embedding_dim = 384

        self.metadata = ModelMetadata(
            model_id=model_id,
            display_name=display_name,
            family="minilm",
            task=Task.TEXT_EMBEDDING,
            modality=Modality.TEXT,
            source_repository="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
            model_license="Apache-2.0",
            weights_license="Apache-2.0",
            commercial_use_permitted=True,
            auto_download_permitted=True,
            parameters_millions=22.7,
            model_size_bytes=self.model_path.stat().st_size if self.model_path.exists() else None,
            supported_precisions=[Precision.FP32, Precision.INT8],
            supported_devices=[DeviceKind.CPU, DeviceKind.CUDA],
            supported_runtimes=["onnxruntime"],
            input_format="UTF-8 text, one string per item",
            output_format="L2-normalized float32 vectors of dimension 384",
            dynamic_input_supported=True,
            streaming_supported=False,
            batch_supported=True,
            supported_quantizations=["int8-dynamic"],
            hardware_requirements=HardwareRequirement(min_ram_mb=512, min_disk_mb=100),
            known_limitations=[
                f"Input is truncated at {max_length} tokens; longer text loses its tail.",
                "Trained primarily on English.",
                "Retrieval quality metrics require a labelled corpus and are unavailable "
                "without one.",
            ],
        )

    # --- lifecycle -------------------------------------------------------

    def load(self, config: LoadConfig) -> LoadResult:
        if not self.model_path.exists():
            raise ModelLoadError(f"model file missing: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise ModelLoadError(
                f"tokenizer not found at {self.tokenizer_path}; text cannot be tokenized "
                "and no substitute would be faithful to the trained model"
            )

        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ModelLoadError(
                "the 'tokenizers' package is required for text embedding but is not installed"
            ) from exc

        t0 = time.perf_counter()
        self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self._tokenizer.enable_truncation(max_length=self.max_length)
        self._tokenizer.enable_padding()

        handle = self.runtime.create_session(
            SessionConfig(
                model_path=str(self.model_path),
                device=config.device,
                device_index=config.device_index,
                precision=config.precision,
                intra_op_threads=config.thread_config.get("intra_op"),
                inter_op_threads=config.thread_config.get("inter_op"),
            )
        )
        load_ms = (time.perf_counter() - t0) * 1000.0

        if not handle.honored:
            self.runtime.release(handle)
            raise ModelLoadError(handle.mismatch_message())

        self._handle = handle
        self.embedding_dim = self._infer_embedding_dim(handle)
        return LoadResult(
            ok=True,
            effective_device=handle.effective_device,
            effective_precision=handle.effective_precision,
            execution_provider=handle.execution_provider,
            runtime_version=handle.runtime_version,
            load_ms=load_ms,
            weights_bytes=self.model_path.stat().st_size,
            message=f"loaded on {handle.execution_provider}",
        )

    @staticmethod
    def _infer_embedding_dim(handle: SessionHandle, default: int = 384) -> int:
        try:
            shape = handle.session.get_outputs()[0].shape
            last = shape[-1]
            return int(last) if isinstance(last, int) else default
        except Exception:  # noqa: BLE001
            return default

    def unload(self) -> None:
        if self._handle is not None:
            self.runtime.release(self._handle)
            self._handle = None
        self._tokenizer = None

    # --- execution -------------------------------------------------------

    def preprocess(self, request: InferenceRequest) -> PreparedInput:
        """Tokenize. This is the phase the timeline records as TOKENIZATION."""
        if not request.text:
            raise ConfigInvalidError("MiniLmEmbeddingAdapter requires at least one text input")
        if self._tokenizer is None:
            raise ModelLoadError("adapter is not loaded")

        encodings = self._tokenizer.encode_batch(list(request.text))
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        return PreparedInput(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
            context={"attention_mask": attention_mask, "batch": len(encodings)},
            token_count=int(attention_mask.sum()),
        )

    def infer(self, prepared: PreparedInput) -> RawOutput:
        if self._handle is None:
            raise ModelLoadError("adapter is not loaded")
        # Feed only the inputs this graph declares: some exports omit token_type_ids,
        # and passing an undeclared name is an ORT error rather than a no-op.
        inputs = {
            name: prepared.tensors[name]
            for name in self._handle.input_names
            if name in prepared.tensors
        }
        missing = set(self._handle.input_names) - set(inputs)
        if missing:
            raise ConfigInvalidError(f"graph expects inputs this adapter did not prepare: {missing}")
        return RawOutput(tensors=self.runtime.run(self._handle, inputs),
                         names=list(self._handle.output_names))

    def synchronize(self) -> None:
        if self._handle is not None:
            self.runtime.synchronize(self._handle)

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> InferenceOutput:
        hidden_state = raw.tensors[0]
        # Some exports already pool; only pool when a token axis is present.
        if hidden_state.ndim == 3:
            pooled = mean_pool(hidden_state, prepared.context["attention_mask"])
        else:
            pooled = hidden_state
        embeddings = l2_normalize(pooled.astype(np.float32))
        return InferenceOutput(
            embeddings=embeddings,
            extra={
                "dimension": int(embeddings.shape[-1]),
                "batch_size": int(embeddings.shape[0]),
                "tokens": prepared.token_count,
            },
        )

    # --- benchmarking ----------------------------------------------------

    def synthetic_request(self, batch_size: int = 1) -> InferenceRequest:
        """Fixed-length deterministic text, so tokenization cost is stable across runs."""
        sentence = (
            "InferenceLab measures inference latency, throughput, memory and energy "
            "across runtimes and devices with reproducible methodology."
        )
        return InferenceRequest(text=[sentence] * batch_size)

    def evaluate(
        self,
        predictions: list[InferenceOutput],
        references: list[ReferenceOutput],
    ) -> QualityMetrics:
        """Embedding properties that are measurable without a corpus, and honesty about the rest.

        Dimensionality and storage cost are facts about the output and are reported.
        Recall, MRR and NDCG require a labelled retrieval corpus; without one they
        are unavailable rather than estimated.
        """
        no_corpus = (
            "retrieval metrics require a labelled query/document corpus, which was not "
            "supplied for this run"
        )
        dimension = None
        for prediction in predictions:
            if prediction.embeddings is not None:
                dimension = int(prediction.embeddings.shape[-1])
                break

        return QualityMetrics(
            embedding=EmbeddingQuality(
                mrr=Measurement[float].unavailable(no_corpus),
                ndcg=Measurement[float].unavailable(no_corpus),
                dimensionality=(
                    Measurement[int].of(dimension, "dimensions", "model output shape")
                    if dimension
                    else Measurement[int].unavailable("no embeddings were produced")
                ),
                bytes_per_vector=(
                    Measurement[int].derived(
                        dimension * 4, "bytes", "dimension x 4 bytes (float32)",
                        note="uncompressed float32 storage; quantized indexes are smaller",
                    )
                    if dimension
                    else Measurement[int].unavailable("no embeddings were produced")
                ),
            ),
            reference_dataset=None,
            sample_count=len(references),
        )
