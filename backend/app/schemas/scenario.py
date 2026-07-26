"""Benchmark scenario definitions — the versionable, human-readable run recipe.

A scenario says *what* to measure and *how carefully*; it deliberately does not
name a model or a runtime, so the same scenario can be run across many
configurations and the results remain comparable. Model and runtime are supplied
at execution time and recorded in the result.

Scenarios are loaded from YAML under ``benchmarks/scenarios/`` and validated here.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.schemas.enums import BenchmarkMode, Task


class GenerationSettings(BaseModel):
    """Applies to generative workloads only; ignored (and not recorded) elsewhere."""

    max_new_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    denoising_steps: int | None = Field(default=None, ge=1)
    guidance_scale: float | None = None


class ScenarioSpec(BaseModel):
    """One reproducible measurement recipe."""

    schema_version: int = 1
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    task: Task
    description: str = ""

    input_dataset: str | None = Field(
        default=None,
        description="Dataset id under benchmarks/datasets/. None means the adapter's "
                    "deterministic synthetic input is used, which is recorded in the result.",
    )

    # --- run shape ---
    warmup_iterations: int = Field(default=3, ge=0)
    measured_iterations: int = Field(default=20, ge=1)
    cooldown_seconds: float = Field(default=0.0, ge=0.0)
    batch_size: int = Field(default=1, ge=1)
    concurrency: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=300.0, gt=0.0)

    # --- input shape ---
    input_size: int | None = Field(default=None, ge=1)
    sequence_length: int | None = Field(default=None, ge=1)
    output_length: int | None = Field(default=None, ge=1)

    # --- determinism ---
    random_seed: int | None = 42
    deterministic: bool = False
    reuse_input: bool = Field(
        default=True,
        description="True feeds the identical input every iteration, isolating runtime "
                    "variance from input variance. False cycles through the dataset.",
    )
    clear_cache_between_iterations: bool = False

    # --- instrumentation ---
    mode: BenchmarkMode = BenchmarkMode.STANDARD
    streaming: bool = False
    measure_cold_start: bool = Field(
        default=False,
        description="When true the model is loaded inside the measured window so load, "
                    "compile and first-inference costs are captured as a cold start.",
    )

    generation: GenerationSettings | None = None

    @model_validator(mode="after")
    def _streaming_requires_generative_task(self) -> ScenarioSpec:
        generative = {
            Task.TEXT_GENERATION,
            Task.TEXT_TO_SPEECH,
            Task.IMAGE_GENERATION,
            Task.VIDEO_GENERATION,
            Task.MULTIMODAL_GENERATION,
            Task.VISION_LANGUAGE,
            Task.IMAGE_CAPTIONING,
        }
        if self.streaming and self.task not in generative:
            raise ValueError(
                f"streaming=true is meaningless for task '{self.task.value}' — "
                "streaming metrics (TTFT, inter-chunk latency) require a generative task"
            )
        return self

    @model_validator(mode="after")
    def _warn_thin_sample(self) -> ScenarioSpec:
        # Not an error: a 1-iteration smoke run is legitimate. But the result carries
        # an insufficient-sample warning, raised by the engine from this same threshold.
        return self

    @property
    def has_sufficient_samples(self) -> bool:
        """Below this, percentiles are not meaningful and the run is flagged."""
        return self.measured_iterations >= 10
