"""Classification and embedding adapters — the second and third working modalities.

These exercise real weights, so they skip cleanly when the models are not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.adapters.base import InferenceRequest, LoadConfig, ReferenceOutput
from app.adapters.classification.mobilenet import (
    IMAGENET_MEAN,
    MobileNetClassifierAdapter,
    softmax,
)
from app.adapters.embedding.minilm import (
    MiniLmEmbeddingAdapter,
    l2_normalize,
    mean_pool,
)
from app.core.config import REPO_ROOT
from app.core.errors import ConfigInvalidError, ModelLoadError
from app.models.registry import load_registry, refresh_deployment_status
from app.runtimes.onnxruntime_adapter import OnnxRuntimeAdapter
from app.schemas.enums import DeviceKind, Modality, Phase, Task

CLASSIFIER = REPO_ROOT / "models" / "classification" / "mobilenetv4_conv_small.onnx"
EMBEDDER = REPO_ROOT / "models" / "embedding" / "all-MiniLM-L6-v2.onnx"

needs_classifier = pytest.mark.skipif(
    not CLASSIFIER.exists(),
    reason="classifier not installed (scripts/download_models.py --install mobilenetv4-conv-small-onnx)",
)
needs_embedder = pytest.mark.skipif(
    not EMBEDDER.exists(),
    reason="embedder not installed (scripts/download_models.py --install all-minilm-l6-v2-onnx)",
)


class TestSoftmax:
    def test_produces_a_probability_distribution(self):
        probs = softmax(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        assert probs.sum() == pytest.approx(1.0)
        assert (probs > 0).all()

    def test_is_numerically_stable_for_large_logits(self):
        # Without subtracting the max, exp(1000) overflows to inf and yields NaN.
        probs = softmax(np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32))
        assert np.isfinite(probs).all()
        assert probs.sum() == pytest.approx(1.0)

    def test_preserves_ordering(self):
        probs = softmax(np.array([[0.5, 3.0, 1.0]], dtype=np.float32))[0]
        assert probs.argmax() == 1


class TestMeanPooling:
    def test_ignores_padding_tokens(self):
        # Averaging over padding drags every short sentence toward the pad vector,
        # silently destroying similarity ranking. Only real tokens may contribute.
        hidden = np.array([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]], dtype=np.float32)
        mask = np.array([[1, 1, 0]], dtype=np.int64)
        pooled = mean_pool(hidden, mask)
        assert pooled[0].tolist() == pytest.approx([2.0, 2.0])

    def test_all_padding_does_not_divide_by_zero(self):
        hidden = np.ones((1, 3, 2), dtype=np.float32)
        pooled = mean_pool(hidden, np.zeros((1, 3), dtype=np.int64))
        assert np.isfinite(pooled).all()

    def test_handles_a_batch(self):
        hidden = np.ones((4, 5, 8), dtype=np.float32)
        mask = np.ones((4, 5), dtype=np.int64)
        assert mean_pool(hidden, mask).shape == (4, 8)


class TestL2Normalize:
    def test_produces_unit_vectors(self):
        vectors = l2_normalize(np.array([[3.0, 4.0]], dtype=np.float32))
        assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)

    def test_zero_vector_does_not_produce_nan(self):
        assert np.isfinite(l2_normalize(np.zeros((1, 4), dtype=np.float32))).all()


@needs_classifier
class TestClassificationAdapter:
    @pytest.fixture
    def adapter(self):
        a = MobileNetClassifierAdapter(CLASSIFIER, OnnxRuntimeAdapter())
        a.load(LoadConfig(runtime_id="onnxruntime", device=DeviceKind.CPU))
        yield a
        a.unload()

    def test_reads_preprocessing_parameters_from_the_model_config(self, adapter):
        # Hardcoding these would silently degrade accuracy for any other checkpoint.
        assert adapter.input_size == 224
        assert 0.5 < adapter.crop_pct <= 1.0
        assert len(adapter.labels) == 1000

    def test_labels_are_ordered_by_numeric_index(self, adapter):
        # String-sorted keys would put class 10 before class 2.
        assert adapter.labels[0].startswith("tench")
        assert "goldfish" in adapter.labels[1]

    def test_produces_a_valid_ranked_distribution(self, adapter):
        image = np.full((300, 400, 3), 128, dtype=np.uint8)
        prepared = adapter.preprocess(InferenceRequest(images=[image]))
        output = adapter.postprocess(adapter.infer(prepared), prepared)

        assert len(output.classifications) == 5
        scores = [c[2] for c in output.classifications]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert output.extra["num_classes"] == 1000

    def test_preprocessing_normalizes_to_imagenet_statistics(self, adapter):
        white = np.full((224, 224, 3), 255, dtype=np.uint8)
        tensor = adapter.preprocess(InferenceRequest(images=[white])).tensors["pixel_values"]
        # White maps to (1 - mean) / std per channel, in CHW order.
        expected_r = (1.0 - IMAGENET_MEAN[0]) / 0.229
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor[0, 0].mean() == pytest.approx(expected_r, abs=0.05)

    def test_supports_batching(self, adapter):
        images = [np.full((256, 256, 3), v, dtype=np.uint8) for v in (50, 128, 200)]
        prepared = adapter.preprocess(InferenceRequest(images=images))
        assert prepared.tensors["pixel_values"].shape[0] == 3
        assert adapter.postprocess(adapter.infer(prepared), prepared).extra["batch_size"] == 3

    def test_declares_the_preprocessing_phase(self, adapter):
        assert adapter.preprocess_phase is Phase.PREPROCESSING

    def test_metadata_is_permissively_licensed(self, adapter):
        assert adapter.metadata.commercial_use_permitted is True
        assert adapter.metadata.task is Task.IMAGE_CLASSIFICATION
        assert adapter.metadata.modality is Modality.IMAGE

    def test_missing_config_is_a_load_error_not_a_guess(self, tmp_path):
        weights = tmp_path / "model.onnx"
        weights.write_bytes(b"not really onnx")
        with pytest.raises(ModelLoadError, match="config not found"):
            MobileNetClassifierAdapter(weights, OnnxRuntimeAdapter())

    def test_empty_request_is_rejected(self, adapter):
        with pytest.raises(ConfigInvalidError, match="at least one image"):
            adapter.preprocess(InferenceRequest(images=[]))

    def test_accuracy_is_unavailable_without_references(self, adapter):
        quality = adapter.evaluate([], [])
        assert not quality.classification.top1_accuracy.available
        assert "no labelled reference" in quality.classification.top1_accuracy.unavailable_reason

    def test_accuracy_is_computed_when_references_are_supplied(self, adapter):
        from app.adapters.base import InferenceOutput

        predictions = [
            InferenceOutput(classifications=[(7, "a", 0.9), (2, "b", 0.05)]),
            InferenceOutput(classifications=[(1, "c", 0.8), (9, "d", 0.1)]),
        ]
        references = [ReferenceOutput(class_id=7), ReferenceOutput(class_id=9)]

        quality = adapter.evaluate(predictions, references)
        assert quality.classification.top1_accuracy.value == pytest.approx(0.5)
        assert quality.classification.top5_accuracy.value == pytest.approx(1.0)
        assert quality.sample_count == 2

    def test_mismatched_prediction_and_reference_counts_raise(self, adapter):
        with pytest.raises(ConfigInvalidError, match="references"):
            adapter.evaluate([], [ReferenceOutput(class_id=1)])


@needs_embedder
class TestEmbeddingAdapter:
    @pytest.fixture
    def adapter(self):
        a = MiniLmEmbeddingAdapter(EMBEDDER, OnnxRuntimeAdapter())
        a.load(LoadConfig(runtime_id="onnxruntime", device=DeviceKind.CPU))
        yield a
        a.unload()

    def test_produces_unit_norm_vectors_of_the_declared_dimension(self, adapter):
        prepared = adapter.preprocess(InferenceRequest(text=["hello world"]))
        output = adapter.postprocess(adapter.infer(prepared), prepared)

        assert output.embeddings.shape == (1, 384)
        assert np.linalg.norm(output.embeddings[0]) == pytest.approx(1.0, abs=1e-5)

    def test_embeddings_are_semantically_ordered(self, adapter):
        """The functional test that matters: wrong pooling still yields unit vectors."""
        texts = ["a dog runs in the park", "a puppy is running outside",
                 "quarterly financial report"]
        prepared = adapter.preprocess(InferenceRequest(text=texts))
        vectors = adapter.postprocess(adapter.infer(prepared), prepared).embeddings

        similarity = vectors @ vectors.T
        assert similarity[0, 1] > similarity[0, 2]
        assert similarity[0, 1] > 0.4

    def test_tokenization_is_its_own_phase(self, adapter):
        # Calling this 'preprocessing' would hide the most informative phase of a
        # text pipeline behind a generic label.
        assert adapter.preprocess_phase is Phase.TOKENIZATION

    def test_token_count_is_reported(self, adapter):
        prepared = adapter.preprocess(InferenceRequest(text=["one two three four"]))
        assert prepared.token_count and prepared.token_count > 3

    def test_batching_pads_to_a_common_length(self, adapter):
        prepared = adapter.preprocess(
            InferenceRequest(text=["short", "a considerably longer sentence than the first one"])
        )
        ids = prepared.tensors["input_ids"]
        assert ids.shape[0] == 2
        # Padding is masked, so the short input keeps fewer real tokens.
        mask = prepared.tensors["attention_mask"]
        assert mask[0].sum() < mask[1].sum()

    def test_missing_tokenizer_is_a_load_error(self, tmp_path):
        weights = tmp_path / "model.onnx"
        weights.write_bytes(b"stub")
        adapter = MiniLmEmbeddingAdapter(weights, OnnxRuntimeAdapter())
        with pytest.raises(ModelLoadError, match="tokenizer not found"):
            adapter.load(LoadConfig(runtime_id="onnxruntime"))

    def test_empty_request_is_rejected(self, adapter):
        with pytest.raises(ConfigInvalidError, match="at least one text"):
            adapter.preprocess(InferenceRequest(text=[]))

    def test_retrieval_metrics_need_a_corpus(self, adapter):
        prepared = adapter.preprocess(InferenceRequest(text=["x"]))
        output = adapter.postprocess(adapter.infer(prepared), prepared)
        quality = adapter.evaluate([output], [])

        assert not quality.embedding.mrr.available
        assert "corpus" in quality.embedding.mrr.unavailable_reason
        # Dimensionality and storage cost are facts about the output, so they ARE reported.
        assert quality.embedding.dimensionality.value == 384
        assert quality.embedding.bytes_per_vector.value == 384 * 4

    def test_truncation_limit_is_disclosed(self, adapter):
        assert any("truncated" in limit for limit in adapter.metadata.known_limitations)


class TestAdapterRegistryEntries:
    def test_all_three_modalities_are_registered(self):
        registry = refresh_deployment_status(load_registry())
        tasks = {m.task for m in registry.models}
        assert {"object_detection", "image_classification", "text_embedding"} <= tasks

    def test_status_is_derived_from_disk_not_declared(self):
        registry = refresh_deployment_status(load_registry())
        for entry in registry.models:
            on_disk = (REPO_ROOT / entry.local_path).exists()
            if not on_disk:
                assert entry.deployment_status != "installed"
                assert entry.not_installed_reason

    def test_missing_companion_makes_a_model_incomplete_not_installed(self, tmp_path):
        # A model whose tokenizer is absent would load and produce wrong output, so
        # it must not be reported as installed.
        from app.models.registry import AdapterModelEntry, CompanionFile, ModelRegistry

        weights = tmp_path / "w.onnx"
        weights.write_bytes(b"stub")
        entry = AdapterModelEntry(
            model_id="x", display_name="X", family="f", task="text_embedding",
            modality="text", adapter="minilm", model_license="MIT", weights_license="MIT",
            file_name="w.onnx",
            local_path=str(weights.relative_to(REPO_ROOT)) if REPO_ROOT in weights.parents
            else str(weights),
            companion_files=[CompanionFile(file_name="tokenizer.json", download_url="https://x")],
        )
        registry = refresh_deployment_status(ModelRegistry(models=[entry]))
        assert registry.models[0].deployment_status == "incomplete"
        assert "tokenizer.json" in registry.models[0].not_installed_reason

    def test_every_entry_declares_both_licences(self):
        registry = load_registry()
        for entry in registry.models:
            assert entry.model_license and entry.weights_license

    def test_test_adapters_are_excluded_from_production_listings(self):
        registry = load_registry()
        assert all(not m.is_test_adapter for m in registry.production_models())
