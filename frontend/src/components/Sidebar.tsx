import { NavLink } from 'react-router-dom';
import { NAV } from '@/nav';
import { Icon } from './Icon';

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="flex h-full flex-col overflow-y-auto bg-surface-900 px-3 py-4">
      <div className="mb-5 flex items-center gap-2.5 px-2">
        <div className="rounded-lg bg-accent/15 p-1.5 text-accent">
          <Icon name="live" className="h-5 w-5" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold text-slate-100">VisionEdge Lab</div>
          <div className="text-[11px] text-slate-500">multimodal vision platform</div>
        </div>
      </div>

      <div className="flex-1 space-y-5">
        {NAV.map((section) => (
          <div key={section.title}>
            <div className="px-2 pb-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-600">
              {section.title}
            </div>
            <ul className="space-y-0.5">
              {section.items.map((item) => (
                <li key={item.path}>
                  <NavLink
                    to={item.path}
                    end={item.path === '/'}
                    onClick={onNavigate}
                    className={({ isActive }) =>
                      `group flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition ${
                        isActive
                          ? 'bg-accent/15 text-accent'
                          : 'text-slate-400 hover:bg-surface-800 hover:text-slate-200'
                      }`
                    }
                  >
                    <Icon name={item.icon} className="h-[18px] w-[18px] shrink-0" />
                    <span className="truncate">{item.label}</span>
                    {item.status === 'planned' && (
                      <span className="ml-auto rounded bg-surface-700 px-1.5 py-0.5 text-[9px] font-medium uppercase text-slate-500">
                        soon
                      </span>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="mt-4 px-2 text-[10px] text-slate-600">Phase 2 build · Detection slice live</div>
    </nav>
  );
}
