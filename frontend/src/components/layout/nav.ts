import { Atom, BarChart3, HeartPulse, History, Lightbulb, type LucideIcon } from 'lucide-react'

export interface NavItem { to: string; label: string; icon: LucideIcon; end?: boolean; description: string }

export const NAV_GROUPS: { title: string; items: NavItem[] }[] = [
  {
    title: 'Assess',
    items: [
      { to: '/', label: 'Risk Assessment', icon: HeartPulse, end: true, description: 'Estimate cardiovascular risk for a patient' },
      { to: '/history', label: 'History', icon: History, description: 'Previous assessments' },
    ],
  },
  {
    title: 'Model',
    items: [
      { to: '/insights', label: 'Insights', icon: Lightbulb, description: 'What drives the prediction' },
      { to: '/performance', label: 'Model Performance', icon: BarChart3, description: 'Accuracy across models' },
      { to: '/quantum', label: 'Quantum Engine', icon: Atom, description: 'The quantum feature layer' },
    ],
  },
]
