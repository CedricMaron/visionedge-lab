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
    title: 'Perception',
    items: [
      { path: '/', label: 'Live Inference', icon: 'live', status: 'live' },
      { path: '/models', label: 'Model Selector', icon: 'layers', status: 'live' },
      { path: '/classes', label: 'Class Selector', icon: 'tag', status: 'live' },
      { path: '/assistant', label: 'Multimodal Assistant', icon: 'sparkles', status: 'live' },
    ],
  },
  {
    title: 'Telemetry',
    items: [
      { path: '/capabilities', label: 'Device Capabilities', icon: 'chip', status: 'live' },
      { path: '/performance', label: 'Performance', icon: 'chart', status: 'live' },
      { path: '/benchmarks', label: 'Benchmarks', icon: 'gauge', status: 'live' },
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
      {
        path: '/multimodal-benchmarks',
        label: 'Multimodal Benchmarks',
        icon: 'gauge',
        status: 'planned',
      },
    ],
  },
  {
    title: 'Operations (Planned)',
    items: [
      { path: '/servers', label: 'Server Connections', icon: 'server', status: 'planned' },
      { path: '/logs', label: 'Logs', icon: 'terminal', status: 'planned' },
      { path: '/model-comparison', label: 'Model Comparison', icon: 'scale', status: 'planned' },
      {
        path: '/optimization',
        label: 'Optimization Advisor',
        icon: 'wand',
        status: 'planned',
      },
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
