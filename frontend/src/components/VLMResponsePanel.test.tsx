import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { VLMResponsePanel } from './VLMResponsePanel';
import type { VLMResponse } from '@/types';

// Captured verbatim from a live GET of POST /api/vlm/ask (mock-vlm backend), so this
// pins the panel against what the backend actually sends — including prompt_tokens
// arriving as null, which the type previously declared as a plain number.
const REAL_RESPONSE: VLMResponse = {
  text: 'Based on detector grounding: The scene appears to contain 4 persons and 1 bus. A real VLM is needed to reason further about this question.',
  structured_output: null,
  model_id: 'mock-vlm',
  runtime: 'python',
  execution_location: 'pc_local',
  prompt_tokens: null,
  generated_tokens: 25,
  time_to_first_token_ms: 0.045,
  generation_latency_ms: 0.045,
  total_latency_ms: 0.045,
  memory_usage_mb: 380.08203125,
  warnings: ['mock VLM: deterministic output derived from detector context, not a real model'],
};

describe('VLMResponsePanel', () => {
  it('renders every performance metric for a real backend response', () => {
    render(<VLMResponsePanel response={REAL_RESPONSE} />);
    for (const label of [
      'Time to first token',
      'Generation',
      'Total latency',
      'Prompt tokens',
      'Generated tokens',
      'Memory',
    ]) {
      expect(screen.getByText(label)).toBeDefined();
    }
    expect(screen.getByText('25')).toBeDefined();
  });

  it('shows a placeholder for a metric the backend reports as null', () => {
    render(<VLMResponsePanel response={REAL_RESPONSE} />);
    expect(screen.getByText('—')).toBeDefined();
  });

  it('reads execution location off the response without crashing', () => {
    render(<VLMResponsePanel response={REAL_RESPONSE} />);
    expect(screen.getByText(/on-device/i)).toBeDefined();
  });
});
