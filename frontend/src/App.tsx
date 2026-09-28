import { lazy, Suspense, type ComponentType } from 'react'
import { Link, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/layout/AppLayout'
import { EmptyState, LoadingState } from './components/ui/primitives'
import { LandingPage } from './pages/LandingPage'

// Route-level code splitting keeps the charting library out of the landing bundle.
const page = <K extends string>(load: () => Promise<Record<K, ComponentType>>, name: K) =>
  lazy(() => load().then((m) => ({ default: m[name] })))

const DashboardPage = page(() => import('./pages/DashboardPage'), 'DashboardPage')
const AssessmentPage = page(() => import('./pages/AssessmentPage'), 'AssessmentPage')
const HistoryPage = page(() => import('./pages/HistoryPage'), 'HistoryPage')
const ExplainabilityPage = page(() => import('./pages/ExplainabilityPage'), 'ExplainabilityPage')
const BenchmarksPage = page(() => import('./pages/BenchmarksPage'), 'BenchmarksPage')
const QuantumPage = page(() => import('./pages/QuantumPage'), 'QuantumPage')
const MethodologyPage = page(() => import('./pages/MethodologyPage'), 'MethodologyPage')
const AboutPage = page(() => import('./pages/AboutPage'), 'AboutPage')

export default function App() {
  return (
    <Suspense fallback={<LoadingState label="Loading…" className="min-h-[50vh]" />}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app" element={<AppLayout />}>
          <Route index element={<DashboardPage />} />
          <Route path="assess" element={<AssessmentPage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="explain" element={<ExplainabilityPage />} />
          <Route path="benchmarks" element={<BenchmarksPage />} />
          <Route path="quantum" element={<QuantumPage />} />
          <Route path="methodology" element={<MethodologyPage />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="*" element={<NotFound />} />
        </Route>
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  )
}

function NotFound() {
  return (
    <div className="grid min-h-[60vh] place-items-center">
      <EmptyState title="Page not found" action={<Link to="/app" className="text-sm font-medium text-teal-700">Go to dashboard →</Link>}>
        The page you requested does not exist.
      </EmptyState>
    </div>
  )
}
