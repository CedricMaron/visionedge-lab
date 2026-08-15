import { NavLink } from 'react-router-dom';
import { FOOTER_NAV, NAV } from '@/nav';
import { Icon } from './Icon';

const linkClass = (isActive: boolean) =>
  `group flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition ${
    isActive ? 'bg-accent/15 text-accent' : 'text-secondary hover:bg-elevated hover:text-primary'
  }`;

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="flex h-full flex-col overflow-y-auto bg-elevated px-3 py-4">
      <div className="mb-5 flex items-center gap-2.5 px-2">
        <div className="rounded-lg bg-accent/15 p-1.5 text-accent">
          <Icon name="live" className="h-5 w-5" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold text-primary">InferenceLab</div>
          <div className="text-[11px] text-muted">inference · pipeline · benchmarks</div>
        </div>
      </div>

      <div className="flex-1 space-y-5">
        {NAV.map((section) => (
          <div key={section.title}>
            <div className="px-2 pb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted">
              {section.title}
            </div>
            <ul className="space-y-0.5">
              {section.items.map((item) => (
                <li key={item.path}>
                  <NavLink
                    to={item.path}
                    end={item.path === '/'}
                    onClick={onNavigate}
                    className={({ isActive }) => linkClass(isActive)}
                  >
                    <Icon name={item.icon} className="h-[18px] w-[18px] shrink-0" />
                    <span className="truncate">{item.label}</span>
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <ul className="mt-6 space-y-0.5 border-t border-subtle pt-3">
        {FOOTER_NAV.map((item) => (
          <li key={item.label}>
            {item.href ? (
              <a
                href={item.href}
                target="_blank"
                rel="noreferrer noopener"
                className={linkClass(false)}
              >
                <Icon name={item.icon} className="h-[18px] w-[18px] shrink-0" />
                <span className="truncate">{item.label}</span>
              </a>
            ) : (
              <NavLink
                to={item.path as string}
                onClick={onNavigate}
                className={({ isActive }) => linkClass(isActive)}
              >
                <Icon name={item.icon} className="h-[18px] w-[18px] shrink-0" />
                <span className="truncate">{item.label}</span>
              </NavLink>
            )}
          </li>
        ))}
      </ul>
    </nav>
  );
}
