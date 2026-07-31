import { PageHeader } from '@/components/ui';

function Note({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card card-pad">
      <h2 className="mb-2 text-sm font-semibold text-primary">{title}</h2>
      <div className="space-y-2 text-sm leading-relaxed text-secondary">{children}</div>
    </div>
  );
}

export default function ResearchNotesPage() {
  return (
    <div>
      <PageHeader
        title="Research Notes"
        subtitle="Background on the models and techniques this platform is built around. Static reference material — not live data."
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <Note title="Object detection (COCO)">
          <p>
            The detection slice reports the 80 COCO classes. Models are single-stage detectors
            producing axis-aligned boxes with a class id, confidence, and label. Confidence
            thresholds the detections shown; IoU controls non-maximum suppression (how aggressively
            overlapping boxes are merged).
          </p>
        </Note>

        <Note title="Runtimes & precision">
          <p>
            The same model weights can run under different runtimes (ONNX Runtime, PyTorch,
            OpenVINO, TensorRT) and execution providers (CPU, CUDA). Lower precision (FP16, INT8)
            trades a little accuracy for large speed and memory wins — the Benchmarks page measures
            the real trade-off on your hardware rather than quoting datasheet numbers.
          </p>
        </Note>

        <Note title="JEPA — joint-embedding predictive architecture">
          <p>
            JEPA is a self-supervised approach that predicts representations of masked regions in a
            latent space, rather than reconstructing pixels. It learns semantic features useful for
            downstream perception without labels. The JEPA Training page (Phase 4) will drive and
            monitor these runs.
          </p>
        </Note>

        <Note title="Vision-language models & grounding">
          <p>
            VLMs answer natural-language questions about images. "Detector-grounding" injects the
            object detector's outputs into the prompt so the language model reasons over verified
            detections — reducing hallucination and enabling structured, checkable answers.
          </p>
        </Note>

        <Note title="World models">
          <p>
            A world model learns to predict how a scene evolves. Given the current latent state it
            can roll forward plausible futures, supporting planning and counterfactual analysis. The
            World Model Lab (Phase 6) is the experimental surface for this.
          </p>
        </Note>

        <Note title="On-device vs server inference">
          <p>
            Running inference in the browser (onnxruntime-web / WebGPU) keeps imagery on the device
            — a strong privacy property — at the cost of raw throughput. Server inference is faster
            and supports larger models. VisionEdge Lab is designed to route per-workload between the
            two; the privacy indicator throughout the UI reflects where a given inference ran.
          </p>
        </Note>
      </div>
    </div>
  );
}
