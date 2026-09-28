import type { ValidationIssue } from '../types/api'

const BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''

export type ApiErrorKind = 'network' | 'unavailable' | 'validation' | 'server'

/** Normalised API failure. `detail` is the backend's message, never a stack trace. */
export class ApiError extends Error {
  kind: ApiErrorKind
  status: number
  detail?: string
  issues?: ValidationIssue[]

  constructor(kind: ApiErrorKind, status: number, message: string, extra?: Partial<ApiError>) {
    super(message)
    this.kind = kind
    this.status = status
    Object.assign(this, extra)
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { Accept: 'application/json', 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new ApiError('network', 0, 'The Swasthya Quant inference service could not be reached.')
  }

  const body: unknown = await res.json().catch(() => null)
  if (res.ok) return body as T

  const detail = (body as { detail?: unknown } | null)?.detail
  if (res.status === 422 && Array.isArray(detail)) {
    throw new ApiError('validation', 422, 'Some clinical values were rejected.', {
      issues: detail as ValidationIssue[],
    })
  }
  if (res.status === 503) {
    throw new ApiError('unavailable', 503, 'The prediction model is not loaded.', {
      detail: typeof detail === 'string' ? detail : undefined,
    })
  }
  // Vite's proxy answers 500/502/504 when the backend process is down.
  if (res.status >= 500 && body === null) {
    throw new ApiError('network', res.status, 'The Swasthya Quant inference service could not be reached.')
  }
  throw new ApiError('server', res.status, 'The service returned an unexpected error.', {
    detail: typeof detail === 'string' ? detail : undefined,
  })
}
