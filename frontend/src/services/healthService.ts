import { request } from '../lib/api'
import type { HealthResponse } from '../types/api'

export const getHealth = () => request<HealthResponse>('/api/health')
