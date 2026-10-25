import type { CheckpointInfo, TrainingLogRow } from '../protocol'
import { Rng } from './rng'

/** Mock REST endpoints: GET /api/checkpoints and GET /api/training/{name}. */

export function mockCheckpoints(): CheckpointInfo[] {
  return [
    { name: 'v1-mock', created: '2026-10-25T18:12:00Z', metrics: { ev_per_round: 0.031, edge: 0.0021, agreement_basic: 0.962 } },
    { name: 'v0-mock', created: '2026-10-24T09:40:00Z', metrics: { ev_per_round: -0.084, edge: -0.0068, agreement_basic: 0.913 } },
  ]
}

export function mockTrainingLog(name: string): TrainingLogRow[] {
  const rng = new Rng(name.length * 7919)
  const rows: TrainingLogRow[] = []
  const good = name.startsWith('v1')
  for (let i = 0; i <= 200; i++) {
    const step = i * 1000
    const prog = 1 - Math.exp(-i / (good ? 45 : 70))
    const loss = 0.42 * (1 - 0.82 * prog) + rng.normal() * 0.012 * (1 - prog * 0.6)
    const evalEv = (good ? 0.002 : -0.0068) - 0.03 * (1 - prog) + rng.normal() * 0.0016
    rows.push({
      step,
      loss: Math.max(0.02, Math.round(loss * 10000) / 10000),
      eval_ev: i % 5 === 0 ? Math.round(evalEv * 100000) / 100000 : null,
      agreement_basic: i % 5 === 0 ? Math.round((0.78 + 0.18 * prog + rng.normal() * 0.005) * 1000) / 1000 : null,
      epsilon: Math.round(Math.max(0.02, 1 - i / 120) * 1000) / 1000,
    })
  }
  return rows
}
