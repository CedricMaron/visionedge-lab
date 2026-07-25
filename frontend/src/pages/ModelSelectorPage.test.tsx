import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BenchmarkJobStatus } from './ModelSelectorPage';
import type { JobRecord } from '@/types';

function job(state: string, current = 0, total = 30): JobRecord {
  return {
    job_id: 'benchmark-abc', kind: 'benchmark', params: { runs: total },
    state, progress: total ? current / total : 0, total_steps: total,
    current_step: current, metrics: {}, error: null, checkpoint_path: null,
    created_at: 0, updated_at: 0,
  };
}

describe('BenchmarkJobStatus', () => {
  it('reports progress while a benchmark is running', () => {
    render(<BenchmarkJobStatus job={job('running', 12)} />);
    expect(screen.getByText(/benchmarking/i)).toBeDefined();
    expect(screen.getByText(/12\s*\/\s*30/)).toBeDefined();
  });

  it('renders nothing when there is no job', () => {
    const { container } = render(<BenchmarkJobStatus job={null} />);
    expect(container.textContent).toBe('');
  });

  it('surfaces a failed benchmark instead of hiding it', () => {
    const failed = { ...job('failed', 4), error: 'no detection backend loaded' };
    render(<BenchmarkJobStatus job={failed} />);
    expect(screen.getByText(/no detection backend loaded/i)).toBeDefined();
  });
});
