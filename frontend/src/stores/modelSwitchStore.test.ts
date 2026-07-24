import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { buildInferenceConfig, useModelSwitchStore, DEFAULT_DRAFT } from './modelSwitchStore';
import { api } from '@/services/api';
import type { SwitchResponse } from '@/types';

describe('buildInferenceConfig', () => {
  it('composes a valid InferenceConfig and sorts class ids', () => {
    const cfg = buildInferenceConfig(
      { ...DEFAULT_DRAFT, model_id: 'yolo', runtime: 'onnx', input_size: 640 },
      [5, 1, 3],
    );
    expect(cfg).toEqual({
      model_id: 'yolo',
      runtime: 'onnx',
      input_size: 640,
      confidence: 0.25,
      iou: 0.45,
      execution_location: 'server',
      allowed_class_ids: [1, 3, 5],
    });
  });
});

describe('modelSwitchStore.applySwitch', () => {
  beforeEach(() => {
    useModelSwitchStore.setState({
      draft: { ...DEFAULT_DRAFT, model_id: 'm1', runtime: 'onnx' },
      status: 'idle',
      message: '',
      activeConfig: null,
      rolledBack: false,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('sets success status on ok switch', async () => {
    const resp: SwitchResponse = {
      ok: true,
      rolled_back: false,
      message: 'switched',
      config: buildInferenceConfig(useModelSwitchStore.getState().draft, [0]),
    };
    vi.spyOn(api, 'switchDetection').mockResolvedValue(resp);

    await useModelSwitchStore.getState().applySwitch([0]);
    const state = useModelSwitchStore.getState();
    expect(state.status).toBe('success');
    expect(state.activeConfig?.model_id).toBe('m1');
    expect(state.rolledBack).toBe(false);
  });

  it('sets rolled_back status when backend rolls back', async () => {
    const resp: SwitchResponse = {
      ok: false,
      rolled_back: true,
      message: 'incompatible runtime',
      config: buildInferenceConfig(useModelSwitchStore.getState().draft, []),
    };
    vi.spyOn(api, 'switchDetection').mockResolvedValue(resp);

    await useModelSwitchStore.getState().applySwitch([]);
    const state = useModelSwitchStore.getState();
    expect(state.status).toBe('rolled_back');
    expect(state.rolledBack).toBe(true);
    expect(state.message).toContain('incompatible');
  });

  it('sets error status and rethrows on failure', async () => {
    vi.spyOn(api, 'switchDetection').mockRejectedValue(new Error('network down'));

    await expect(useModelSwitchStore.getState().applySwitch([1])).rejects.toThrow('network down');
    const state = useModelSwitchStore.getState();
    expect(state.status).toBe('error');
    expect(state.message).toBe('network down');
  });
});
