/**
 * API communication hooks for the Behavioral Digital Twin backend.
 * All requests go to the FastAPI server — same origin when served via FastAPI static files.
 */
import { useState, useEffect, useCallback, useRef } from 'react'

export const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : ''

// ---------- Types ----------

export interface TimelinePoint {
  date: string
  actual: number
  predicted: number
  confidence: number
}

export interface DecisionRow {
  id: number
  timestamp: string
  domain: string
  predicted: string
  actual: string
  hit: boolean
  confidence: number
}

export interface DriftEvent {
  date: string
  domain: string
  note: string
}

export interface DataMaturity {
  count: number
  threshold: number
  status: 'learning' | 'ready'
  message: string
}

export interface HeatmapPoint {
  date: string
  count: number
}

export interface DashboardData {
  accuracy: number
  lastSynced: string
  timeline: TimelinePoint[]
  decisions: DecisionRow[]
  driftEvents: DriftEvent[]
  data_maturity: DataMaturity | null
  heatmap: HeatmapPoint[]
}

export interface PredictResult {
  predicted: string
  confidence: number
}

export interface UserProfile {
  user_id: string
  decision_counts: Record<string, number>
  embedding_summary: { dim: number; norm: number }
  last_updated: string | null
}

export interface RetrainResult {
  status: string
  metrics?: Record<string, { accuracy: number; macro_f1: number; brier: number; n?: number }>
  reason?: string
}

export interface LogDecisionPayload {
  domain: string
  decision_made: string
  location?: string
  weather?: string
  mood_energy?: number
  stress_level?: string
}

export interface LogDecisionResult {
  status: string
  user_id: string
  domain: string
  decision_made: string
  timestamp: string
}

export interface CSVImportResult {
  imported: number
  skipped: number
  errors: string[]
}

// ---------- Domain constants (must match data/schema.py) ----------

export const DOMAINS = [
  {
    id: 'focus',
    label: 'Focus Mode',
    options: ['pomodoro', 'flow_state', 'light_work', 'admin'],
  },
  {
    id: 'task',
    label: 'Task Choice',
    options: ['deep_work', 'email', 'meeting', 'break'],
  },
  {
    id: 'purchase',
    label: 'Purchase',
    options: ['coffee', 'snack', 'lunch', 'none'],
  },
] as const

// ---------- Hooks ----------

export function useDashboard(userId: string) {
  const [data, setData] = useState<DashboardData | null>(null)
  // `loading` is only true on the very first fetch for this userId — subsequent
  // refreshes (e.g. after logging a decision) use `refreshing` so the dashboard
  // stays visible and never flashes a blank loading screen.
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const isFirstFetch = useRef(true)

  const fetch_ = useCallback(async () => {
    // First fetch: show the full-screen spinner.
    // Subsequent fetches (refetch): keep existing data visible, show subtle indicator.
    if (isFirstFetch.current) {
      setLoading(true)
    } else {
      setRefreshing(true)
    }
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/twin/${encodeURIComponent(userId)}`, { cache: 'no-store' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setData(json)
      setError(null)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
      setRefreshing(false)
      isFirstFetch.current = false
    }
  }, [userId])

  // Reset on userId change so it shows the spinner again for the new user
  useEffect(() => {
    isFirstFetch.current = true
    setData(null)
    setLoading(true)
    fetch_()
  }, [fetch_])

  return { data, loading, refreshing, error, refetch: fetch_ }
}

export function useDataMaturity(userId: string) {
  const [maturity, setMaturity] = useState<DataMaturity | null>(null)

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/maturity/${encodeURIComponent(userId)}`, { cache: 'no-store' })
      if (res.ok) setMaturity(await res.json())
    } catch { /* ignore */ }
  }, [userId])

  useEffect(() => { refresh() }, [refresh])

  return { maturity, refresh }
}

export const connectIntegration = async (appId: string, token: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/integrations/${appId}/connect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token })
  })
  if (!res.ok) throw new Error('Failed to connect integration')
}

export const syncIntegration = async (appId: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/integrations/${appId}/sync`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Failed to sync integration')
}

// ---------- Domain Configurations ----------

export async function predictNext(userId: string, domain: string): Promise<PredictResult> {
  const res = await fetch(`${API_BASE}/predict_next_decision/${encodeURIComponent(userId)}?domain=${domain}`, {
    cache: 'no-store'
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `HTTP ${res.status}`)
  }
  const data = await res.json()
  return {
    predicted: data.predicted_decision || data.predicted || '',
    confidence: data.confidence || 0.0
  }
}

export async function retrain(userId: string) {
  const res = await fetch(`${API_BASE}/retrain/${encodeURIComponent(userId)}`, { method: 'POST' })
  if (!res.ok) throw new Error('Retrain failed')
  return res.json()
}

export async function fetchAIBriefing(userId: string) {
  const res = await fetch(`${API_BASE}/briefing/${encodeURIComponent(userId)}`, { cache: 'no-store' })
  if (!res.ok) throw new Error('Briefing failed')
  return res.json()
}

export async function logDecision(
  userId: string,
  payload: LogDecisionPayload,
): Promise<LogDecisionResult> {
  const res = await fetch(`${API_BASE}/decisions/${encodeURIComponent(userId)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function importCSV(userId: string, file: File): Promise<CSVImportResult> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/decisions/${encodeURIComponent(userId)}/import-csv`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export const deleteTwin = async (userId: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/decisions/${encodeURIComponent(userId)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to delete twin')
}

export const seedData = async (userId: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/seed/${encodeURIComponent(userId)}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Failed to seed data')
}
