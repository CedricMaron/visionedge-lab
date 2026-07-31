# Development

## Setup

```bash
make venv && make install          # backend, CPU-first
make frontend-install              # frontend
make model                         # export the YOLOv8n baseline
python scripts/download_models.py --list
python scripts/download_models.py --install mobilenetv4-conv-small-onnx
python scripts/download_models.py --install all-minilm-l6-v2-onnx
```

Requires Python 3.12 and Node 20+.

## Running

```bash
make backend       # http://localhost:8000  (docs at /docs)
make frontend      # http://localhost:5173
```

## Benchmarking from the CLI

The CLI and the web API share one engine, so a benchmark means the same thing either way.

```bash
inference-lab system                   # hardware + software environment
inference-lab runtimes                 # probe every runtime
inference-lab matrix                   # runtime x device x precision, with reasons
inference-lab models                   # models with an adapter, and install status
inference-lab scenarios                # available scenarios

inference-lab benchmark run \
  --scenario single-image-detection \
  --model yolov8n-onnx \
  --device cpu --precision fp32 --iterations 20

inference-lab results list
inference-lab results show <run_id> --format markdown
inference-lab results export --output benchmarks/results/summary.csv
```

Or via make: `make bench`, `make bench-scenarios`, `make bench-results`, `make runtimes`,
`make system`.

Run it as a module during development: `python -m app.cli ...` from `backend/`.

## Verification

```bash
make test          # backend: pytest
make lint          # backend: ruff
cd frontend && npm run test && npm run typecheck && npm run lint && npm run build
```

Everything must pass before a commit. The suite covers metric arithmetic, capability
probes, adapter equivalence, failure handling, persistence round-trips, export formats
and WCAG contrast.

## Layout

```
backend/app/
  schemas/         versioned contracts — Measurement, BenchmarkRun, ScenarioSpec
  adapters/        model-specific code only
  runtimes/        execution backends, model-agnostic
  instrumentation/ timing, probes, sampling, energy, environment
  benchmark/       engine, scenarios, throughput, export
  storage/         SQLite + versioned migrations
  api/             FastAPI routers (lab.py is the InferenceLab surface)
  cli.py           shares the engine with the API
frontend/src/
  components/      LatencyDecomposition, MeasurementValue, UtilizationChart…
  pages/lab/       Overview, RunBenchmark, Results, RunDetail, Models, System
  theme/           contrast tests over the shipped stylesheet
benchmarks/scenarios/   versioned YAML run recipes
```

## Conventions

- **Never fabricate a metric.** Construct `Measurement.unavailable(reason=...)`. The
  schema rejects a valueless measurement with no reason, and an estimate with no
  documented methodology.
- **Measurement lives in `instrumentation/`.** Adapters execute; they do not time.
- **Runtime-specific code lives in `runtimes/`.** No adapter imports `onnxruntime`.
- **Schemas are versioned.** Bump `RESULT_SCHEMA_VERSION` on a breaking change and add
  a migration; migrations only ever add.
- **Env vars are allow-listed by name** before entering a result — exports get shared,
  and sweeping `os.environ` would leak credentials.
- **Tests assert the silent failure**, not just the crash. A bug that still returns
  plausible numbers is the one worth a test.

## Adding things

- A model → [MODEL_ADAPTERS.md](MODEL_ADAPTERS.md)
- A runtime → [RUNTIMES.md](RUNTIMES.md)
- A scenario → drop a YAML file in `benchmarks/scenarios/`; it is validated on load and
  duplicate ids are an error
- A metric → add it to the relevant schema as a `Measurement`, document it in
  [METRICS.md](METRICS.md), and state how it is measured in
  [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md)
