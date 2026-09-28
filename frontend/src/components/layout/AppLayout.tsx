import { Menu, PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { Suspense, useEffect, useState } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import { DEMO_MODE } from '../../config'
import { cx } from '../../lib/format'
import { LoadingState } from '../ui/primitives'
import { Brand } from './Brand'
import { NAV_GROUPS } from './nav'
import { ServiceStatusPill } from './ServiceStatusPill'
import { MobileDrawer, Sidebar } from './Sidebar'

const COLLAPSE_KEY = 'swasthya-quant.sidebar-collapsed'

export function AppLayout() {
  const [drawer, setDrawer] = useState(false)
  const [collapsed, setCollapsed] = useState(() => {
    try {
      const saved = localStorage.getItem(COLLAPSE_KEY)
      if (saved !== null) return saved === '1'
    } catch { /* ignore */ }
    return typeof window !== 'undefined' && window.innerWidth < 1100
  })
  const { pathname } = useLocation()

  useEffect(() => {
    try { localStorage.setItem(COLLAPSE_KEY, collapsed ? '1' : '0') } catch { /* ignore */ }
  }, [collapsed])

  useEffect(() => { window.scrollTo(0, 0) }, [pathname])

  const current = NAV_GROUPS.flatMap((g) => g.items).find((i) => (i.end ? pathname === i.to : pathname.startsWith(i.to)))

  return (
    <div className="min-h-dvh">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[60] focus:rounded-md focus:bg-surface focus:px-3 focus:py-2 focus:shadow">Skip to content</a>
      <Sidebar collapsed={collapsed} />
      <MobileDrawer open={drawer} onClose={() => setDrawer(false)} />

      <div className={cx('transition-[padding] duration-200', collapsed ? 'md:pl-[72px]' : 'md:pl-60')}>
        <header className="sticky top-0 z-20 flex h-16 items-center gap-3 border-b border-line bg-surface/90 px-4 backdrop-blur sm:px-6">
          <button className="grid size-9 place-items-center rounded-md text-ink-2 hover:bg-slate-100 md:hidden" onClick={() => setDrawer(true)} aria-label="Open navigation">
            <Menu className="size-5" aria-hidden />
          </button>
          <button className="hidden size-9 place-items-center rounded-md text-ink-3 hover:bg-slate-100 md:grid" onClick={() => setCollapsed((c) => !c)} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
            {collapsed ? <PanelLeftOpen className="size-[18px]" aria-hidden /> : <PanelLeftClose className="size-[18px]" aria-hidden />}
          </button>
          <Link to="/" className="md:hidden"><Brand compact /></Link>
          <div className="hidden min-w-0 items-center gap-2 text-sm sm:flex">
            <span className="text-ink-3">Swasthya Quant</span>
            <span className="text-slate-300">/</span>
            <span className="truncate font-medium text-navy-900">{current?.label ?? 'Workspace'}</span>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {DEMO_MODE && <span className="hidden rounded-full bg-warn-bg px-2.5 py-1 text-xs font-semibold text-warn ring-1 ring-inset ring-warn/30 sm:inline">Demo mode on</span>}
            <ServiceStatusPill />
          </div>
        </header>

        <main id="main" className="mx-auto w-full max-w-[1280px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          <Suspense fallback={<LoadingState label="Loading…" className="min-h-[40vh]" />}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  )
}
