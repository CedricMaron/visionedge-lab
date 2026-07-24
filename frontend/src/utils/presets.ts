// Quick optimization presets for the Live page. Each preset sets the detector
// input size + confidence (applied via /api/detection/switch) and client-side
// streaming parameters (target FPS + JPEG quality).

export interface OptimizationPreset {
  key: string;
  label: string;
  description: string;
  inputSize: number;
  confidence: number;
  targetFps: number;
  jpegQuality: number;
}

export const PRESETS: OptimizationPreset[] = [
  {
    key: 'max_speed',
    label: 'Max speed',
    description: 'Smallest input, high confidence — highest FPS.',
    inputSize: 320,
    confidence: 0.35,
    targetFps: 30,
    jpegQuality: 0.7,
  },
  {
    key: 'balanced',
    label: 'Balanced',
    description: 'Good speed/quality trade-off for general use.',
    inputSize: 640,
    confidence: 0.25,
    targetFps: 20,
    jpegQuality: 0.8,
  },
  {
    key: 'max_quality',
    label: 'Max quality',
    description: 'Large input, low confidence — most detections.',
    inputSize: 1280,
    confidence: 0.2,
    targetFps: 10,
    jpegQuality: 0.9,
  },
  {
    key: 'battery',
    label: 'Battery',
    description: 'Low frame rate to reduce device load.',
    inputSize: 416,
    confidence: 0.3,
    targetFps: 6,
    jpegQuality: 0.7,
  },
  {
    key: 'low_bandwidth',
    label: 'Low bandwidth',
    description: 'Small frames + low JPEG quality for weak links.',
    inputSize: 320,
    confidence: 0.3,
    targetFps: 8,
    jpegQuality: 0.5,
  },
];

export const DEFAULT_PRESET = PRESETS[1];
