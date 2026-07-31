/**
 * Row construction for the latency decomposition.
 *
 * Separated from the component so it can be unit-tested directly and so the
 * component file exports only components (React Fast Refresh requirement).
 */
import type { DurationStats, PhaseBreakdown } from '@/types/lab';

/** Display order follows the real pipeline, not the size of the bars. */
const PHASE_ORDER = [
  'request_preparation',
  'input_loading',
  'input_decoding',
  'input_validation',
  'preprocessing',
  'tokenization',
  'host_memory_allocation',
  'host_to_device',
  'queue_wait',
  'model_execution',
  'device_synchronization',
  'device_to_host',
  'postprocessing',
  'output_serialization',
  'request_serialization',
  'dns_resolution',
  'connection_establishment',
  'tls_handshake',
  'upload',
  'server_queue',
  'server_preprocessing',
  'server_model_execution',
  'server_postprocessing',
  'response_serialization',
  'download',
  'client_parsing',
  'client_rendering',
];

const PHASE_LABELS: Record<string, string> = {
  request_preparation: 'Request preparation',
  input_loading: 'Input loading',
  input_decoding: 'Decode',
  input_validation: 'Input validation',
  preprocessing: 'Preprocessing',
  tokenization: 'Tokenization',
  host_memory_allocation: 'Host allocation',
  host_to_device: 'Host → device',
  queue_wait: 'Queue wait',
  model_execution: 'Model execution',
  device_synchronization: 'Device sync',
  device_to_host: 'Device → host',
  postprocessing: 'Postprocessing',
  output_serialization: 'Serialization',
  request_serialization: 'Request serialization',
  dns_resolution: 'DNS',
  connection_establishment: 'Connect',
  tls_handshake: 'TLS handshake',
  upload: 'Upload',
  server_queue: 'Server queue',
  server_preprocessing: 'Server preprocessing',
  server_model_execution: 'Server model execution',
  server_postprocessing: 'Server postprocessing',
  response_serialization: 'Response serialization',
  download: 'Download',
  client_parsing: 'Client parsing',
  client_rendering: 'Client rendering',
};

/** Colour by pipeline stage group, so the eye can find the model bar instantly. */
export function phaseTone(phase: string): string {
  if (phase === 'model_execution' || phase === 'server_model_execution') return 'bg-series-1';
  if (phase.includes('device') || phase === 'queue_wait') return 'bg-series-4';
  if (
    phase.includes('upload') ||
    phase.includes('download') ||
    phase.includes('tls') ||
    phase.includes('dns') ||
    phase.includes('connection')
  )
    return 'bg-series-5';
  if (phase.startsWith('server_')) return 'bg-series-6';
  if (phase.includes('post') || phase.includes('serial')) return 'bg-series-3';
  return 'bg-series-2';
}

export interface LatencyRow {
  phase: string;
  label: string;
  stats: DurationStats | null;
  valueMs: number;
  share: number;
  isResidual: boolean;
}

/**
 * Build the ordered rows. Exported for tests: the residual must survive to the UI,
 * and phases must not be reordered by magnitude.
 */
export function buildLatencyRows(breakdown: PhaseBreakdown): {
  rows: LatencyRow[];
  totalMs: number;
} {
  const totalMs = breakdown.total.mean_ms ?? 0;

  const known = Object.entries(breakdown.phases)
    .map(([phase, stats]) => ({ phase, stats }))
    .sort((a, b) => {
      const ai = PHASE_ORDER.indexOf(a.phase);
      const bi = PHASE_ORDER.indexOf(b.phase);
      // Unknown phases sort last rather than to the front.
      return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
    });

  const rows: LatencyRow[] = known.map(({ phase, stats }) => ({
    phase,
    label: PHASE_LABELS[phase] ?? phase.replace(/_/g, ' '),
    stats,
    valueMs: stats.mean_ms ?? 0,
    share: totalMs > 0 ? (stats.mean_ms ?? 0) / totalMs : 0,
    isResidual: false,
  }));

  // Residual is included whenever it is meaningfully non-zero. Hiding it would
  // imply the phases account for the whole measured total when they do not.
  const residual = breakdown.residual_ms;
  if (residual !== null && Math.abs(residual) > 0.005) {
    rows.push({
      phase: 'residual_overhead',
      label: 'Residual overhead',
      stats: null,
      valueMs: residual,
      share: totalMs > 0 ? residual / totalMs : 0,
      isResidual: true,
    });
  }

  return { rows, totalMs };
}

