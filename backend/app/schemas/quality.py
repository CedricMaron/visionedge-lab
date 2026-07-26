"""Quality and accuracy metrics, separated by task.

Two separations matter here and are structural, not stylistic:

* **Objective vs. subjective.** Computed metrics live in the per-task models.
  Anything requiring a human sits in :class:`SubjectiveEvaluation`, which holds
  *hooks* — it never contains a machine-generated score pretending to be a human
  judgement. An empty MOS field means nobody has listened yet, and the UI says so.
* **Quality vs. performance.** A model that is fast and wrong should be visibly
  fast and wrong, so quality never mixes into the throughput or latency structures.

Every field is a ``Measurement``, so an unevaluated metric carries the reason it is
missing (usually "no reference dataset supplied for this run").
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.measurement import FloatMeasurement, IntMeasurement


class DetectionQuality(BaseModel):
    map_50_95: FloatMeasurement
    map_50: FloatMeasurement
    map_75: FloatMeasurement
    precision: FloatMeasurement
    recall: FloatMeasurement
    per_class_ap: dict[str, float] = Field(default_factory=dict)


class ClassificationQuality(BaseModel):
    top1_accuracy: FloatMeasurement
    top5_accuracy: FloatMeasurement
    f1_macro: FloatMeasurement
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    class_labels: list[str] = Field(default_factory=list)


class SegmentationQuality(BaseModel):
    mean_iou: FloatMeasurement
    dice: FloatMeasurement
    boundary_f1: FloatMeasurement
    per_class_iou: dict[str, float] = Field(default_factory=dict)


class TextGenerationQuality(BaseModel):
    exact_match: FloatMeasurement
    perplexity: FloatMeasurement
    output_validity_rate: FloatMeasurement = Field(
        description="Fraction of outputs that parsed as the requested format."
    )
    structured_output_compliance: FloatMeasurement = Field(
        description="Fraction of outputs that validated against the requested schema."
    )
    benchmark_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Named task benchmark scores, e.g. {'gsm8k': 0.41}. Only populated from a real eval run.",
    )


class EmbeddingQuality(BaseModel):
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: FloatMeasurement
    ndcg: FloatMeasurement
    dimensionality: IntMeasurement
    bytes_per_vector: IntMeasurement


class RerankingQuality(BaseModel):
    mrr: FloatMeasurement
    ndcg: FloatMeasurement
    recall_at_k: dict[int, float] = Field(default_factory=dict)


class SpeechToTextQuality(BaseModel):
    word_error_rate: FloatMeasurement
    character_error_rate: FloatMeasurement


class TextToSpeechQuality(BaseModel):
    audio_duration_s: FloatMeasurement
    sample_rate_hz: IntMeasurement
    clipping_ratio: FloatMeasurement = Field(
        description="Fraction of samples at full scale. Measurable without a reference."
    )
    silence_ratio: FloatMeasurement
    speaker_similarity: FloatMeasurement


class ImageGenerationQuality(BaseModel):
    width: IntMeasurement
    height: IntMeasurement
    denoising_steps: IntMeasurement
    clip_similarity: FloatMeasurement
    seed_reproducible: bool | None = Field(
        default=None,
        description="Whether re-running with the same seed produced a bit-identical image. "
                    "None when not verified in this run.",
    )


class VideoQuality(BaseModel):
    frame_count: IntMeasurement
    temporal_consistency: FloatMeasurement
    encode_settings: dict[str, str] = Field(default_factory=dict)


class SubjectiveEvaluation(BaseModel):
    """Human evaluation hooks. Never auto-populated."""

    mos_score: float | None = Field(
        default=None, ge=1.0, le=5.0,
        description="Mean opinion score, 1-5. Set only by a human rater through the API.",
    )
    rater_count: int = 0
    notes: str | None = None

    @property
    def evaluated(self) -> bool:
        return self.rater_count > 0


class QualityMetrics(BaseModel):
    """Container. Exactly the sub-model matching the run's task is populated."""

    detection: DetectionQuality | None = None
    classification: ClassificationQuality | None = None
    segmentation: SegmentationQuality | None = None
    text_generation: TextGenerationQuality | None = None
    embedding: EmbeddingQuality | None = None
    reranking: RerankingQuality | None = None
    speech_to_text: SpeechToTextQuality | None = None
    text_to_speech: TextToSpeechQuality | None = None
    image_generation: ImageGenerationQuality | None = None
    video: VideoQuality | None = None

    subjective: SubjectiveEvaluation = Field(default_factory=SubjectiveEvaluation)

    reference_dataset: str | None = Field(
        default=None,
        description="Dataset the objective metrics were computed against. None means no "
                    "quality evaluation was performed and every metric above is unavailable.",
    )
    sample_count: int = Field(
        default=0, description="Number of examples evaluated. Reported alongside every score."
    )
