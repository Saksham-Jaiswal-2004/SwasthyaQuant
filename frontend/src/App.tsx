import { lazy, Suspense, type ComponentType } from 'react'
import { Link, Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/layout/AppLayout'
import { EmptyState, LoadingState } from './components/ui/primitives'
import { AssessmentPage } from './pages/AssessmentPage'

// Route-level code splitting keeps the charting library out of the first load.
const page = <K extends string>(load: () => Promise<Record<K, ComponentType>>, name: K) =>
  lazy(() => load().then((m) => ({ default: m[name] })))

const HistoryPage = page(() => import('./pages/HistoryPage'), 'HistoryPage')
const InsightsPage = page(() => import('./pages/ExplainabilityPage'), 'ExplainabilityPage')
const PerformancePage = page(() => import('./pages/BenchmarksPage'), 'BenchmarksPage')
const QuantumPage = page(() => import('./pages/QuantumPage'), 'QuantumPage')

export default function App() {
  return (
    <Suspense fallback={<LoadingState label="Loading…" className="min-h-[50vh]" />}>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<AssessmentPage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="insights" element={<InsightsPage />} />
          <Route path="performance" element={<PerformancePage />} />
          <Route path="quantum" element={<QuantumPage />} />
          <Route path="app/*" element={<Navigate to="/" replace />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  )
}

function NotFound() {
  return (
    <div className="grid min-h-[60vh] place-items-center">
      <EmptyState title="Page not found" action={<Link to="/" className="text-sm font-medium text-teal-700">Back to Risk Assessment →</Link>}>
        The page you requested does not exist.
      </EmptyState>
    </div>
  )
}
