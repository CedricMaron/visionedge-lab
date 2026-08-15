import type { IconName } from '@/components/Icon';

export interface NavItem {
  path: string;
  label: string;
  icon: IconName;
}

export interface NavSection {
  title: string;
  items: NavItem[];
}

/**
 * One entry per surface, grouped by what the visitor is trying to do: run
 * something, understand what happened, look something up, or check the machine.
 * Every entry is a page that works — there is no "soon" tier in the main nav.
 */
export const NAV: NavSection[] = [
  {
    title: 'Playground',
    items: [{ path: '/', label: 'Playground', icon: 'live' }],
  },
  {
    title: 'Analyze',
    items: [
      { path: '/pipeline', label: 'Pipeline', icon: 'blueprint' },
      { path: '/performance', label: 'Performance', icon: 'gauge' },
    ],
  },
  {
    title: 'Library',
    items: [{ path: '/models', label: 'Models', icon: 'layers' }],
  },
  {
    title: 'System',
    items: [{ path: '/environment', label: 'Environment', icon: 'chip' }],
  },
];

export interface FooterLink {
  label: string;
  icon: IconName;
  /** Internal route, or an absolute URL for an external destination. */
  path?: string;
  href?: string;
}

export const FOOTER_NAV: FooterLink[] = [
  { path: '/settings', label: 'Settings', icon: 'settings' },
  { path: '/about', label: 'About', icon: 'book' },
  { href: 'https://github.com/CedricMaron/visionedge-lab', label: 'GitHub', icon: 'terminal' },
];

export const ALL_NAV_ITEMS: NavItem[] = NAV.flatMap((s) => s.items);
