"""Playground API: one generic, instrumented inference call for every adapter.

The detection slice already had ``/api/infer``, but it is YOLO-shaped: it talks to
the singleton :class:`DetectionManager` and returns detections. The Playground needs
the same thing for *any* adapter in the registry — detection, classification, text
embedding — and it needs the intermediate tensors, not just the final answer, so the
Pipeline Inspector can show how an input became an output.

Rather than reimplement inference, this module drives the existing
:class:`~app.adapters.base.ModelAdapter` contract directly, timing each of its four
phases and describing the tensors that cross between them. Everything reported here
was measured on this call; steps the adapter performs internally without separate
instrumentation are listed with a ``null`` duration and said so, never with an
invented number.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.adapters.base import InferenceRequest, LoadConfig
from app.api.imaging import decode_image_bytes
from app.core.errors import InferenceLabError
from app.core.logging import get_logger
from app.core.state import get_state
from app.schemas.enums import DeviceKind, Precision

log = get_logger("api.playground")
router = APIRouter(prefix="/api/playground", tags=["playground"])

#: Loading an ONNX session costs far more than one inference, so adapters are kept
#: alive between calls. Keyed by the full configuration: a different device or
#: precision is a different session, never a reused one.
_ADAPTERS: dict[tuple, Any] = {}
_ADAPTER_LOCK = threading.Lock()
_MAX_CACHED = 4


def _adapter_key(model_id: str, runtime_id: str, device: str, precision: str, size: int | None):
    return (model_id, runtime_id, device, precision, size)


def _acquire_adapter(
    model_id: str, runtime_id: str, device: str, precision: str, input_size: int | None
) -> tuple[Any, float | None]:
    """Return a loaded adapter plus its load time when this call did the loading."""
    from app.cli import _build_adapter  # shared resolution with the CLI and lab API

    key = _adapter_key(model_id, runtime_id, device, precision, input_size)
    with _ADAPTER_LOCK:
        cached = _ADAPTERS.get(key)
        if cached is not None:
            return cached, None

        adapter = _build_adapter(model_id, runtime_id, input_size)
        t0 = time.perf_counter()
        adapter.load(
            LoadConfig(
                runtime_id=runtime_id,
                device=DeviceKind(device),
                precision=Precision(precision),
                input_size=input_size,
            )
        )
        load_ms = (time.perf_counter() - t0) * 1000.0

        # Bound the cache: sessions hold real memory, and this endpoint is not a
        # model server.
        while len(_ADAPTERS) >= _MAX_CACHED:
            _, evicted = _ADAPTERS.popitem()
            try:
                evicted.unload()
            except Exception:  # noqa: BLE001 - eviction must not fail a request
                pass
        _ADAPTERS[key] = adapter
        return adapter, load_ms


def _tensor_info(
    name: str,
    array: np.ndarray,
    *,
    layout: str | None = None,
    device: str = "cpu",
    role: str = "tensor",
) -> dict:
    """Metadata and summary statistics for one tensor. Never the tensor itself."""
    values = array.astype(np.float64, copy=False) if array.size else None
    return {
        "name": name,
        "role": role,
        "shape": [int(d) for d in array.shape],
        "dtype": str(array.dtype),
        "layout": layout,
        "device": device,
        "bytes": int(array.nbytes),
        "elements": int(array.size),
        "min": float(values.min()) if values is not None else None,
        "max": float(values.max()) if values is not None else None,
        "mean": float(values.mean()) if values is not None else None,
        "std": float(values.std()) if values is not None else None,
    }


def _stage(
    stage_id: str,
    name: str,
    *,
    duration_ms: float | None = None,
    detail: str | None = None,
    tensors: list[dict] | None = None,
    substeps: list[dict] | None = None,
    device: str | None = None,
    runtime: str | None = None,
    note: str | None = None,
) -> dict:
    return {
        "id": stage_id,
        "name": name,
        "duration_ms": None if duration_ms is None else round(duration_ms, 4),
        "detail": detail,
        "tensors": tensors or [],
        "substeps": substeps or [],
        "device": device,
        "runtime": runtime,
        "note": note,
    }


_UNTIMED = "performed inside the adapter phase above; not separately instrumented"


def _substep(name: str, detail: str | None = None) -> dict:
    return {"name": name, "detail": detail, "duration_ms": None, "note": _UNTIMED}


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return None


@router.post("/infer")
async def playground_infer(
    request: Request,
    model_id: str = Form(...),
    runtime_id: str = Form("onnxruntime"),
    device: str = Form("cpu"),
    precision: str = Form("fp32"),
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    input_size: int | None = Form(None),
    confidence: float | None = Form(None),
    iou: float | None = Form(None),
    classes: str | None = Form(None),
    top_k: int = Form(5),
):
    """Run one inference through any registered adapter and return the whole trace.

    The response carries the typed output, the four measured phase durations, and a
    tensor-level description of each boundary — which is exactly what the Pipeline
    page renders. It is deliberately synchronous and single-item: this is an
    inspection tool, not a serving path.
    """
    state = get_state(request)
    request_id = uuid.uuid4().hex[:12]
    t_received = time.perf_counter()

    try:
        adapter, load_ms = _acquire_adapter(model_id, runtime_id, device, precision, input_size)
    except InferenceLabError as exc:
        raise HTTPException(400, exc.user_message) from exc
    except (NotImplementedError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:  # bad device/precision literal
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.warning("playground_load_failed", model=model_id, error=str(exc))
        raise HTTPException(400, f"{model_id} could not be loaded: {exc}") from exc

    metadata = adapter.metadata
    task = metadata.task.value
    modality = metadata.modality.value

    # --- input ---------------------------------------------------------------
    stages: list[dict] = []
    image = None
    decode_ms = None
    source_bytes = None

    if file is not None:
        data = await file.read()
        source_bytes = len(data)
        if source_bytes > state.settings.max_upload_bytes:
            raise HTTPException(413, "file too large")
        t0 = time.perf_counter()
        try:
            image = decode_image_bytes(data)
        except InferenceLabError as exc:
            raise HTTPException(400, exc.user_message) from exc
        decode_ms = (time.perf_counter() - t0) * 1000.0

    if modality.startswith("image") and image is None:
        raise HTTPException(400, f"{model_id} requires an image input")
    if modality == "text" and not (text and text.strip()):
        raise HTTPException(400, f"{model_id} requires text input")

    if image is not None:
        stages.append(
            _stage(
                "input",
                "Input",
                detail=(
                    f"{image.shape[1]} × {image.shape[0]} "
                    f"{(file.content_type or 'image').split('/')[-1].upper()}"
                    + (f" · {source_bytes / 1024:.0f} KB encoded" if source_bytes else "")
                ),
            )
        )
        stages.append(
            _stage(
                "decode",
                "Decode",
                duration_ms=decode_ms,
                detail=f"{image.shape[1]} × {image.shape[0]} × {image.shape[2]} · uint8 BGR",
                tensors=[_tensor_info("decoded_image", image, layout="HWC", role="output")],
            )
        )
    else:
        stages.append(_stage("input", "Input", detail=f"UTF-8 text · {len(text or '')} characters"))

    allowed = {int(c) for c in classes.split(",") if c.strip().isdigit()} if classes else None

    inference_request = InferenceRequest(
        images=[image] if image is not None else None,
        text=[text] if text else None,
        confidence=confidence,
        iou=iou,
        allowed_class_ids=allowed,
        request_id=request_id,
    )

    # --- execute -------------------------------------------------------------
    try:
        t0 = time.perf_counter()
        prepared = adapter.preprocess(inference_request)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        raw = adapter.infer(prepared)
        sync = getattr(adapter, "synchronize", None)
        if callable(sync):
            sync()
        inference_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if task == "image_classification":
            output = adapter.postprocess(raw, prepared, top_k=top_k)
        else:
            output = adapter.postprocess(raw, prepared)
        postprocess_ms = (time.perf_counter() - t0) * 1000.0
    except InferenceLabError as exc:
        raise HTTPException(400, exc.user_message) from exc
    except Exception as exc:  # noqa: BLE001
        log.warning("playground_infer_failed", model=model_id, error=str(exc))
        raise HTTPException(400, f"inference failed: {exc}") from exc

    handle = getattr(adapter, "_handle", None)
    provider = getattr(handle, "execution_provider", None)
    runtime_version = getattr(handle, "runtime_version", None)

    # --- preprocessing stage -------------------------------------------------
    prepared_tensors = [
        _tensor_info(
            name,
            tensor,
            layout=("NCHW" if tensor.ndim == 4 else None),
            role="output",
        )
        for name, tensor in prepared.tensors.items()
    ]

    if task == "object_detection":
        letterbox = prepared.context.get("letterbox")
        size = getattr(adapter, "input_size", None)
        substeps = [
            _substep(
                "Letterbox resize",
                f"aspect-preserving resize to {size} × {size} with padding"
                + (f" (scale {letterbox.scale:.3f})" if letterbox is not None else ""),
            ),
            _substep("BGR → RGB", "channel order expected by the exported graph"),
            _substep("Scale to [0,1] · HWC → NCHW", "float32 tensor layout"),
        ]
        preprocess_name = "Resize / letterbox / normalize"
    elif task == "image_classification":
        substeps = [
            _substep("Resize short side", f"1 / crop_pct = {getattr(adapter, 'crop_pct', '?')}"),
            _substep("Centre crop", f"{getattr(adapter, 'input_size', '?')} px square"),
            _substep("ImageNet normalize · HWC → NCHW", "per-channel mean/std the weights were trained with"),
        ]
        preprocess_name = "Resize / crop / normalize"
    elif task == "text_embedding":
        substeps = [
            _substep("WordPiece tokenization", f"truncated at {getattr(adapter, 'max_length', '?')} tokens"),
            _substep("Padding + attention mask", "batch padded to the longest sequence"),
        ]
        preprocess_name = "Tokenization"
    else:
        substeps = []
        preprocess_name = "Preprocess"

    tokens_preview: list[str] | None = None
    if task == "text_embedding":
        tokenizer = getattr(adapter, "_tokenizer", None)
        if tokenizer is not None and text:
            try:
                encoding = tokenizer.encode(text)
                tokens_preview = list(encoding.tokens)[:128]
            except Exception:  # noqa: BLE001 - a preview is never worth failing a run
                tokens_preview = None

    stages.append(
        _stage(
            "preprocess",
            preprocess_name,
            duration_ms=preprocess_ms,
            detail=" · ".join(
                f"{name} {list(tensor.shape)} {tensor.dtype}"
                for name, tensor in prepared.tensors.items()
            ),
            tensors=prepared_tensors,
            substeps=substeps,
            device="cpu",
            note="the phase the timeline records as tokenization for text models",
        )
    )

    # --- inference stage -----------------------------------------------------
    raw_tensors = [
        _tensor_info(
            raw.names[i] if i < len(raw.names) else f"output_{i}",
            tensor,
            role="output",
        )
        for i, tensor in enumerate(raw.tensors)
    ]
    stages.append(
        _stage(
            "inference",
            "Model inference",
            duration_ms=inference_ms,
            detail=metadata.display_name,
            tensors=raw_tensors,
            device=device,
            runtime=f"{runtime_id}{f' {runtime_version}' if runtime_version else ''}"
            + (f" · {provider}" if provider else ""),
        )
    )

    # --- postprocessing stage + typed output ---------------------------------
    result: dict[str, Any] = {}
    if task == "object_detection":
        detections = output.detections or []
        raw_shape = list(raw.tensors[0].shape)
        candidates = raw_shape[-1] if len(raw_shape) == 3 else None
        post_substeps = [
            _substep(
                "Decode boxes",
                f"{candidates} anchor predictions → xyxy" if candidates else "anchor decode",
            ),
            _substep(
                "Confidence filter",
                f"threshold {prepared.context.get('confidence')}",
            ),
            _substep("Non-maximum suppression", f"IoU {prepared.context.get('iou')}"),
            _substep("Scale boxes to source pixels", "undo letterbox padding and scale"),
        ]
        result["detections"] = [d.model_dump() for d in detections]
        result["count"] = len(detections)
        post_detail = f"{len(detections)} detections"
    elif task == "image_classification":
        ranked = output.classifications or []
        post_substeps = [
            _substep("Softmax over logits", "numerically stable; raw logits are not probabilities"),
            _substep("Top-k ranking", f"k = {top_k}"),
        ]
        result["classifications"] = [
            {"class_id": cid, "label": label, "probability": prob} for cid, label, prob in ranked
        ]
        post_detail = ranked[0][1] if ranked else "no class above threshold"
    elif task == "text_embedding":
        embeddings = output.embeddings
        vector = embeddings[0] if embeddings is not None and len(embeddings) else None
        post_substeps = [
            _substep("Attention-masked mean pooling", "padding tokens excluded from the average"),
            _substep("L2 normalization", "unit-norm vectors, so cosine similarity is a dot product"),
        ]
        result["embedding"] = {
            "dimension": int(vector.shape[0]) if vector is not None else None,
            "preview": [float(v) for v in vector[:16]] if vector is not None else [],
            "norm": float(np.linalg.norm(vector)) if vector is not None else None,
            "tokens": prepared.token_count,
            "token_preview": tokens_preview,
        }
        post_detail = f"[1, {int(vector.shape[0])}] embedding" if vector is not None else "no vector"
    else:
        post_substeps = []
        result["extra"] = {k: str(v) for k, v in (output.extra or {}).items()}
        post_detail = None

    stages.append(
        _stage(
            "postprocess",
            "Post-process",
            duration_ms=postprocess_ms,
            detail=post_detail,
            substeps=post_substeps,
            device="cpu",
        )
    )
    stages.append(_stage("output", "Output", detail=post_detail))

    server_total_ms = (time.perf_counter() - t_received) * 1000.0

    return {
        "request_id": request_id,
        "execution": "server",
        "task": task,
        "modality": modality,
        "model": {
            "model_id": metadata.model_id,
            "display_name": metadata.display_name,
            "family": metadata.family,
            "parameters_millions": metadata.parameters_millions,
            "input_format": metadata.input_format,
            "output_format": metadata.output_format,
        },
        "runtime": {
            "runtime_id": runtime_id,
            "runtime_version": runtime_version,
            "execution_provider": provider,
            "device": device,
            "precision": precision,
            "input_size": getattr(adapter, "input_size", None),
        },
        "timings": {
            "model_load_ms": None if load_ms is None else round(load_ms, 3),
            "decode_ms": None if decode_ms is None else round(decode_ms, 3),
            "preprocess_ms": round(preprocess_ms, 3),
            "inference_ms": round(inference_ms, 3),
            "postprocess_ms": round(postprocess_ms, 3),
            "server_total_ms": round(server_total_ms, 3),
        },
        "memory": {
            "process_rss_mb": _rss_mb(),
            "input_tensor_bytes": int(sum(t.nbytes for t in prepared.tensors.values())),
            "output_tensor_bytes": int(sum(t.nbytes for t in raw.tensors)),
        },
        "stages": stages,
        "result": result,
    }


@router.post("/unload")
async def unload_all():
    """Release every cached session. Useful before measuring a cold start."""
    with _ADAPTER_LOCK:
        count = len(_ADAPTERS)
        for adapter in _ADAPTERS.values():
            try:
                adapter.unload()
            except Exception:  # noqa: BLE001
                pass
        _ADAPTERS.clear()
    return {"unloaded": count}
