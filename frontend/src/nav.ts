import type { IconName } from '@/components/Icon';

export interface NavItem {
  path: string;
  label: string;
  icon: IconName;
  status: 'live' | 'planned';
}

export interface NavSection {
  title: string;
  items: NavItem[];
}

export const NAV: NavSection[] = [
  {
    title: 'Benchmark',
    items: [
      { path: '/', label: 'Overview', icon: 'gauge', status: 'live' },
      { path: '/lab/run', label: 'Run Benchmark', icon: 'flask', status: 'live' },
      { path: '/lab/results', label: 'Results', icon: 'chart', status: 'live' },
    ],
  },
  {
    title: 'Environment',
    items: [
      { path: '/lab/models', label: 'Models', icon: 'layers', status: 'live' },
      { path: '/lab/system', label: 'System', icon: 'chip', status: 'live' },
      { path: '/capabilities', label: 'Device Capabilities', icon: 'server', status: 'live' },
    ],
  },
  {
    title: 'Vision',
    items: [
      { path: '/live', label: 'Live Inference', icon: 'live', status: 'live' },
      { path: '/models', label: 'Detection Models', icon: 'grid', status: 'live' },
      { path: '/classes', label: 'Class Selector', icon: 'tag', status: 'live' },
      { path: '/assistant', label: 'Multimodal Assistant', icon: 'sparkles', status: 'live' },
      { path: '/performance', label: 'Live Performance', icon: 'clock', status: 'live' },
      { path: '/benchmarks', label: 'Legacy Benchmarks', icon: 'scale', status: 'live' },
    ],
  },
  {
    title: 'Research (Planned)',
    items: [
      { path: '/temporal', label: 'Temporal Scene Analysis', icon: 'clock', status: 'planned' },
      { path: '/world-model', label: 'World Model Lab', icon: 'globe', status: 'planned' },
      { path: '/jepa', label: 'JEPA Training', icon: 'flask', status: 'planned' },
      { path: '/embeddings', label: 'Embedding Explorer', icon: 'grid', status: 'planned' },
      { path: '/anomaly', label: 'Anomaly Detection', icon: 'alert', status: 'planned' },
      { path: '/cross-modal', label: 'Cross-Modal Search', icon: 'search', status: 'planned' },
    ],
  },
  {
    title: 'Operations (Planned)',
    items: [
      { path: '/servers', label: 'Server Connections', icon: 'server', status: 'planned' },
      { path: '/logs', label: 'Logs', icon: 'terminal', status: 'planned' },
      { path: '/optimization', label: 'Optimization Advisor', icon: 'wand', status: 'planned' },
    ],
  },
  {
    title: 'Reference',
    items: [
      { path: '/architecture', label: 'Architecture', icon: 'blueprint', status: 'live' },
      { path: '/research', label: 'Research Notes', icon: 'book', status: 'live' },
      { path: '/settings', label: 'Settings', icon: 'settings', status: 'live' },
    ],
  },
];

export const ALL_NAV_ITEMS: NavItem[] = NAV.flatMap((s) => s.items);
