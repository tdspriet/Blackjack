import type { CheckpointInfo, TrainingLogRow } from './protocol'
import { useLab } from './store'

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`)
  return (await r.json()) as T
}

export async function fetchCheckpoints(): Promise<CheckpointInfo[]> {
  if (useLab.getState().conn.mode === 'mock') {
    const m = await import('./mock/api')
    return m.mockCheckpoints()
  }
  return getJson<CheckpointInfo[]>('/api/checkpoints')
}

export async function fetchTraining(name: string): Promise<TrainingLogRow[]> {
  if (useLab.getState().conn.mode === 'mock') {
    const m = await import('./mock/api')
    return m.mockTrainingLog(name)
  }
  return getJson<TrainingLogRow[]>(`/api/training/${encodeURIComponent(name)}`)
}
