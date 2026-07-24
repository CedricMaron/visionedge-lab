// Honest stub pages. Each renders a real component shell with an explicit
// "Planned — Phase N" banner and a description of intended capability.
// No fake data, no fake charts.

import { PlannedPage } from '@/components/NotImplementedBanner';

export function TemporalSceneAnalysisPage() {
  return (
    <PlannedPage
      title="Temporal Scene Analysis"
      subtitle="Track objects and events across time, not just single frames."
      phase="Phase 5"
      description="Will consume the detection stream to build per-track histories, event timelines, dwell-time and trajectory analytics, and scene-change summaries over a rolling window."
      plannedFeatures={[
        'Multi-object tracking with stable track IDs across frames',
        'Event timeline (enter / exit / linger / interaction)',
        'Trajectory heatmaps and dwell-time aggregation',
        'Temporal windowing controls tied to the live session',
      ]}
    />
  );
}

export function WorldModelLabPage() {
  return (
    <PlannedPage
      title="World Model Lab"
      subtitle="Predictive scene representation and rollout visualization."
      phase="Phase 6"
      description="An experimental surface for learned world models: encode the current scene into a latent state and visualize predicted future states / rollouts for planning research."
      plannedFeatures={[
        'Latent scene-state encoding from the live feed',
        'Forward rollout / prediction visualization',
        'Counterfactual "what-if" perturbation controls',
      ]}
    />
  );
}

export function JepaTrainingPage() {
  return (
    <PlannedPage
      title="JEPA Training"
      subtitle="Self-supervised joint-embedding predictive architecture training."
      phase="Phase 4"
      description="Configure and monitor JEPA training runs: dataset selection, masking ratio, batch/epoch settings, and live loss curves streamed from the training backend."
      plannedFeatures={[
        'Run configuration (dataset, epochs, batch size, mask ratio)',
        'Live loss / learning-rate curves from the training service',
        'Checkpoint management and export to the model registry',
      ]}
    />
  );
}

export function EmbeddingExplorerPage() {
  return (
    <PlannedPage
      title="Embedding Explorer"
      subtitle="Interactive 2D/3D projection of learned feature embeddings."
      phase="Phase 4"
      description="Explore the model's embedding space with UMAP/t-SNE projections, nearest-neighbour search by image or text, and cluster inspection."
      plannedFeatures={[
        'UMAP / t-SNE projection of the embedding space',
        'Nearest-neighbour retrieval by image or text query',
        'Cluster labeling and outlier inspection',
      ]}
    />
  );
}

export function AnomalyDetectionPage() {
  return (
    <PlannedPage
      title="Anomaly Detection"
      subtitle="Flag out-of-distribution frames and unusual scene events."
      phase="Phase 5"
      description="Score frames against a learned normalcy model and surface anomalies with severity, timestamp, and a review queue."
      plannedFeatures={[
        'Per-frame anomaly scoring against a normalcy baseline',
        'Severity-ranked event queue with review workflow',
        'Configurable thresholds and alerting hooks',
      ]}
    />
  );
}

export function CrossModalSearchPage() {
  return (
    <PlannedPage
      title="Cross-Modal Search"
      subtitle="Search visual history with natural language and vice versa."
      phase="Phase 6"
      description="Query recorded sessions with text ('a person carrying a backpack') or an example image, powered by shared vision-language embeddings."
      plannedFeatures={[
        'Text-to-image and image-to-image retrieval',
        'Session-scoped and global search indexes',
        'Relevance ranking with confidence and timestamps',
      ]}
    />
  );
}

export function MultimodalBenchmarksPage() {
  return (
    <PlannedPage
      title="Multimodal Benchmarks"
      subtitle="Standardized VLM and cross-modal task benchmarks on this device."
      phase="Phase 4"
      description="Measured VLM throughput, latency, and task accuracy across prompts and grounding modes — clearly labeled as measured on the local hardware."
      plannedFeatures={[
        'VQA / captioning / grounding latency and token throughput',
        'Comparison across local vs server execution',
        'Exportable, device-labeled benchmark reports',
      ]}
    />
  );
}

export function ServerConnectionsPage() {
  return (
    <PlannedPage
      title="Server Connections"
      subtitle="Manage multiple inference backends and route workloads."
      phase="Phase 5"
      description="Register additional inference servers, monitor their health, and route detection / VLM workloads across them with failover."
      plannedFeatures={[
        'Register and health-check multiple backends',
        'Per-workload routing and failover policy',
        'Latency-aware load balancing',
      ]}
    />
  );
}

export function LogsPage() {
  return (
    <PlannedPage
      title="Logs"
      subtitle="Structured, streamed logs from the inference services."
      phase="Phase 3"
      description="A live, filterable log console backed by a server log stream. Until the streaming endpoint ships, per-page event lists (e.g. detection status events) are the source of truth."
      plannedFeatures={[
        'Live server log streaming with level filters',
        'Full-text search and time-range scoping',
        'Correlation with session and request IDs',
      ]}
    />
  );
}

export function ModelComparisonPage() {
  return (
    <PlannedPage
      title="Model Comparison"
      subtitle="Side-by-side quality and speed comparison across models."
      phase="Phase 4"
      description="Run the same inputs through multiple models and compare detections, agreement, latency, and quality metrics side by side. The Benchmarks page already provides measured single-model results today."
      plannedFeatures={[
        'Synchronized multi-model inference on identical inputs',
        'Detection agreement / disagreement visualization',
        'Speed vs quality trade-off scatter plots',
      ]}
    />
  );
}

export function OptimizationAdvisorPage() {
  return (
    <PlannedPage
      title="Optimization Advisor"
      subtitle="Recommend the best model/runtime/precision for your device."
      phase="Phase 4"
      description="Combine device capabilities and measured benchmarks to recommend an optimal configuration for a target (max FPS, best quality, lowest memory). The Live page presets offer a lightweight version of this today."
      plannedFeatures={[
        'Capability- and benchmark-aware configuration recommendations',
        'Target-driven optimization (speed / quality / memory / battery)',
        'One-click apply via the detection switch endpoint',
      ]}
    />
  );
}
