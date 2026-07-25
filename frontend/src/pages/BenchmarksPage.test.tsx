import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ComparisonTable } from './BenchmarksPage';
import type { BenchmarkComparisonRow } from '@/types';

const ROWS: BenchmarkComparisonRow[] = [
  {
    model_id: 'yolov8n-onnx', backend: 'onnxruntime', provider: 'CPUExecutionProvider',
    device: 'cpu', input_size: 640, precision: 'fp32', n: 3,
    median_fps: 10, median_p50_ms: 100, latest_ts: 1, latest_fps: 12,
    any_concurrent_traffic: false,
  },
  {
    model_id: 'yolov8s-onnx', backend: 'onnxruntime', provider: 'CPUExecutionProvider',
    device: 'cpu', input_size: 640, precision: 'fp32', n: 1,
    median_fps: 4, median_p50_ms: 250, latest_ts: 2, latest_fps: 4,
    any_concurrent_traffic: true,
  },
];

describe('ComparisonTable', () => {
  it('shows the run count so single-run rows are visibly weaker evidence', () => {
    render(<ComparisonTable rows={ROWS} />);
    expect(screen.getByText('yolov8n-onnx')).toBeDefined();
    expect(screen.getByText('n=3')).toBeDefined();
    expect(screen.getByText('n=1')).toBeDefined();
  });

  it('marks groups measured while live inference was running', () => {
    render(<ComparisonTable rows={ROWS} />);
    expect(screen.getAllByTitle(/measured while live inference/i)).toHaveLength(1);
  });

  it('renders an explicit empty state rather than a zero-filled table', () => {
    render(<ComparisonTable rows={[]} />);
    expect(screen.getByText(/no benchmarks recorded yet/i)).toBeDefined();
  });
});
