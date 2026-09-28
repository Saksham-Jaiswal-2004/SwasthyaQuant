import { X } from 'lucide-react'
import { useEffect } from 'react'
import { Link, NavLink } from 'react-router-dom'
import { cx } from '../../lib/format'
import { Brand } from './Brand'
import { NAV_GROUPS } from './nav'

export function SidebarContent({ collapsed, onNavigate }: { collapsed?: boolean; onNavigate?: () => void }) {
  return (
    <nav aria-label="Main" className="flex h-full flex-col">
      <div className={cx('flex h-16 items-center border-b border-white/10', collapsed ? 'justify-center px-2' : 'px-5')}>
        <Link to="/" onClick={onNavigate} aria-label="Swasthya Quant home"><Brand inverted compact={collapsed} /></Link>
      </div>
      <div className="flex-1 overflow-y-auto px-3 py-4">
        {NAV_GROUPS.map((g) => (
          <div key={g.title} className="mb-5">
            {!collapsed && <p className="mb-1.5 px-2 text-[10.5px] font-semibold uppercase tracking-[0.14em] text-slate-500">{g.title}</p>}
            <ul className="space-y-0.5">
              {g.items.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.end}
                    onClick={onNavigate}
                    title={collapsed ? item.label : undefined}
                    className={({ isActive }) => cx(
                      'group relative flex items-center gap-3 rounded-lg py-2 text-sm font-medium transition-colors',
                      collapsed ? 'justify-center px-2' : 'px-2.5',
                      isActive ? 'bg-white/10 text-white' : 'text-slate-300 hover:bg-white/5 hover:text-white',
                    )}
                  >
                    {({ isActive }) => (
                      <>
                        {isActive && <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-teal-400" aria-hidden />}
                        <item.icon className={cx('size-[18px] shrink-0', isActive ? 'text-teal-400' : 'text-slate-400 group-hover:text-slate-200')} aria-hidden />
                        <span className={collapsed ? 'sr-only' : ''}>{item.label}</span>
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      {!collapsed && (
        <div className="m-3 rounded-lg border border-white/10 bg-white/5 p-3 text-[11.5px] leading-relaxed text-slate-400">
          Research prototype. Outputs are model-estimated risk, not a medical diagnosis.
        </div>
      )}
    </nav>
  )
}

export function Sidebar({ collapsed }: { collapsed: boolean }) {
  return (
    <aside className={cx('fixed inset-y-0 left-0 z-30 hidden bg-navy-900 transition-[width] duration-200 md:block', collapsed ? 'w-[72px]' : 'w-60')}>
      <SidebarContent collapsed={collapsed} />
    </aside>
  )
}

export function MobileDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 md:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
      <button className="absolute inset-0 bg-navy-950/60 animate-fade" aria-label="Close navigation" onClick={onClose} />
      <div className="absolute inset-y-0 left-0 w-72 max-w-[85vw] bg-navy-900 shadow-xl animate-rise">
        <button onClick={onClose} className="absolute right-3 top-4 z-10 grid size-8 place-items-center rounded-md text-slate-300 hover:bg-white/10" aria-label="Close navigation">
          <X className="size-5" aria-hidden />
        </button>
        <SidebarContent onNavigate={onClose} />
      </div>
    </div>
  )
}
