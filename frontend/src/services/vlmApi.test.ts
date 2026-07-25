import { describe, it, expect, afterEach, vi } from 'vitest';
import { vlmApi } from './vlmApi';

// The backend wraps every VLM answer in an envelope:
//   {response: {...VLMResponse}, grounding, agreement, disclaimer}
// (see backend/app/api/vlm.py). These tests pin the wire contract so a field
// rename on either side fails here instead of at runtime in the UI.
const ENVELOPE = {
  response: {
    text: 'a bus and four people',
    structured_output: null,
    model_id: 'mock-vlm',
    runtime: 'python',
    execution_location: 'pc_local',
    prompt_tokens: 0,
    generated_tokens: 0,
    time_to_first_token_ms: 0,
    generation_latency_ms: 1,
    total_latency_ms: 1,
    memory_usage_mb: 0,
    warnings: [],
  },
  grounding: null,
  agreement: null,
  disclaimer: 'VLM output is a model-generated interpretation, not verified truth.',
};

function mockFetch() {
  const spy = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(ENVELOPE), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    }),
  );
  vi.stubGlobal('fetch', spy);
  return spy;
}

function sentForm(spy: ReturnType<typeof mockFetch>): FormData {
  const init = spy.mock.calls[0][1] as RequestInit;
  return init.body as FormData;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('vlmApi.analyzeImage', () => {
  it('sends the prompt under the field name the backend declares', async () => {
    const spy = mockFetch();
    await vlmApi.analyzeImage(new Blob(['x']), { prompt: 'How many buses?' });

    const form = sentForm(spy);
    // /api/vlm/analyze-image declares `prompt`; sending `question` silently
    // drops the user's instruction and the model gets the default prompt.
    expect(form.get('prompt')).toBe('How many buses?');
    expect(form.get('question')).toBeNull();
    expect(form.get('file')).toBeInstanceOf(Blob);
  });

  it('returns the whole envelope, not just its response member', async () => {
    mockFetch();
    const res = await vlmApi.analyzeImage(new Blob(['x']), { prompt: 'hi' });
    expect(res.response.execution_location).toBe('pc_local');
    expect(res.disclaimer).toContain('not verified truth');
  });
});

describe('vlmApi.ask', () => {
  it('sends file + question and returns the envelope', async () => {
    const spy = mockFetch();
    const res = await vlmApi.ask(new Blob(['x']), 'What is here?', true);

    const form = sentForm(spy);
    expect(form.get('question')).toBe('What is here?');
    expect(form.get('ground')).toBe('true');
    // The panel renders res.response — a bare VLMResponse at the top level
    // would leave execution_location undefined and crash on .toLowerCase().
    expect(res.response.execution_location).toBe('pc_local');
    expect(res.response.text).toBe('a bus and four people');
  });
});
