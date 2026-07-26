"""Closed vocabularies shared by every adapter, runtime and benchmark result.

These are the words the whole platform agrees on. A string that is not in one of
these enums is rejected at the schema boundary rather than propagating into a
result row where nobody can interpret it later.
"""
from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """The kind of data a model consumes or produces."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    MULTIMODAL = "multimodal"


class Task(str, Enum):
    """What a model does. Drives which quality metrics and phases apply."""

    OBJECT_DETECTION = "object_detection"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_SEGMENTATION = "image_segmentation"
    IMAGE_GENERATION = "image_generation"
    IMAGE_CAPTIONING = "image_captioning"
    VISION_LANGUAGE = "vision_language"
    VIDEO_UNDERSTANDING = "video_understanding"
    VIDEO_GENERATION = "video_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO_CLASSIFICATION = "audio_classification"
    TEXT_EMBEDDING = "text_embedding"
    RERANKING = "reranking"
    TEXT_GENERATION = "text_generation"
    MULTIMODAL_GENERATION = "multimodal_generation"


class MetricKind(str, Enum):
    """How a number came to exist.

    The distinction is load-bearing: the UI renders derived and estimated values
    differently from measured ones, and ``ESTIMATED`` requires a documented
    methodology before it may be used at all.
    """

    MEASURED = "measured"       # read directly from an instrumentation source
    DERIVED = "derived"         # computed from other measured values, exactly
    ESTIMATED = "estimated"     # modelled under stated assumptions


class Phase(str, Enum):
    """Timeline phases of one inference.

    No workload uses every phase. A phase that does not apply is omitted from the
    timeline entirely rather than recorded as zero, because zero and
    not-applicable are different claims.
    """

    # --- local pipeline ---
    REQUEST_PREPARATION = "request_preparation"
    INPUT_LOADING = "input_loading"
    INPUT_DECODING = "input_decoding"
    INPUT_VALIDATION = "input_validation"
    PREPROCESSING = "preprocessing"
    TOKENIZATION = "tokenization"
    HOST_MEMORY_ALLOCATION = "host_memory_allocation"
    HOST_TO_DEVICE = "host_to_device"
    QUEUE_WAIT = "queue_wait"
    MODEL_EXECUTION = "model_execution"
    DEVICE_SYNCHRONIZATION = "device_synchronization"
    DEVICE_TO_HOST = "device_to_host"
    POSTPROCESSING = "postprocessing"
    OUTPUT_SERIALIZATION = "output_serialization"
    CLIENT_RENDERING = "client_rendering"

    # --- remote-only phases (see docs/BENCHMARK_METHODOLOGY.md) ---
    REQUEST_SERIALIZATION = "request_serialization"
    DNS_RESOLUTION = "dns_resolution"
    CONNECTION_ESTABLISHMENT = "connection_establishment"
    TLS_HANDSHAKE = "tls_handshake"
    UPLOAD = "upload"
    SERVER_QUEUE = "server_queue"
    SERVER_PREPROCESSING = "server_preprocessing"
    SERVER_MODEL_EXECUTION = "server_model_execution"
    SERVER_POSTPROCESSING = "server_postprocessing"
    RESPONSE_SERIALIZATION = "response_serialization"
    DOWNLOAD = "download"
    CLIENT_PARSING = "client_parsing"

    # --- explicit remainder ---
    # Never a measurement. The unattributed difference between a measured total and
    # the sum of its measured parts. Section 18 of the brief forbids calling this
    # "network"; it is labelled residual overhead wherever it is shown.
    RESIDUAL_OVERHEAD = "residual_overhead"


class ExecutionLocation(str, Enum):
    """Where the model actually executed."""

    IN_PROCESS = "in_process"
    LOCAL_WORKER = "local_worker"
    LOCAL_CONTAINER = "local_container"
    LAN_SERVER = "lan_server"
    REMOTE_SERVER = "remote_server"
    BROWSER = "browser"
    EDGE_DEVICE = "edge_device"


class DeviceKind(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    INTEL_GPU = "intel_gpu"
    NPU = "npu"
    APPLE_SILICON = "apple_silicon"
    WEBGPU = "webgpu"
    WASM = "wasm"


class Precision(str, Enum):
    FP32 = "fp32"
    TF32 = "tf32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    Q8 = "q8"
    Q4 = "q4"
    MIXED = "mixed"


class BenchmarkMode(str, Enum):
    """Instrumentation depth. Results from different modes are NOT comparable.

    Every result records its mode, and the comparison UI refuses to place two
    different modes side by side without an explicit warning.
    """

    STANDARD = "standard"   # minimal overhead, aggregate metrics
    DETAILED = "detailed"   # high-frequency hardware sampling + full phase timing
    PROFILER = "profiler"   # framework profilers on; materially perturbs timing


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"     # finished, but some iterations failed
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class IterationPhaseGroup(str, Enum):
    """Why an iteration ran. Warm-up samples are stored but excluded from statistics."""

    WARMUP = "warmup"
    MEASURED = "measured"
    COLD_START = "cold_start"
