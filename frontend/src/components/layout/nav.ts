import { Atom, BarChart3, BookOpenText, ClipboardPlus, History, Info, LayoutDashboard, Lightbulb, type LucideIcon } from 'lucide-react'

export interface NavItem { to: string; label: string; icon: LucideIcon; end?: boolean }

export const NAV_GROUPS: { title: string; items: NavItem[] }[] = [
  {
    title: 'Clinical',
    items: [
      { to: '/app', label: 'Overview', icon: LayoutDashboard, end: true },
      { to: '/app/assess', label: 'Disease Assessment', icon: ClipboardPlus },
      { to: '/app/history', label: 'Prediction History', icon: History },
    ],
  },
  {
    title: 'Research',
    items: [
      { to: '/app/explain', label: 'Explainability', icon: Lightbulb },
      { to: '/app/benchmarks', label: 'Benchmarks', icon: BarChart3 },
      { to: '/app/quantum', label: 'Quantum Hardware', icon: Atom },
    ],
  },
  {
    title: 'Project',
    items: [
      { to: '/app/methodology', label: 'Methodology', icon: BookOpenText },
      { to: '/app/about', label: 'About', icon: Info },
    ],
  },
]
