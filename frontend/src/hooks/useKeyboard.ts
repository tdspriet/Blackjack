import { useEffect } from 'react'
import { useLab } from '../store'
import type { Action } from '../protocol'

const KEYMAP: Record<string, Action> = { h: 'hit', s: 'stand', d: 'double', p: 'split', c: 'cashout', i: 'insure', n: 'no_insurance' }

/** H/S/D/P/C actions, I/N insurance, Space = deal (rebet when no chips are placed). */
export function useKeyboard() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      if (el && (el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const s = useLab.getState()
      const legal = s.state?.game.legal_actions ?? []
      if (s.state?.ai.autoplay) return
      if (e.key === ' ') {
        if (legal.includes('deal')) {
          e.preventDefault()
          s.deal()
        }
        return
      }
      const a = KEYMAP[e.key.toLowerCase()]
      if (a && legal.includes(a)) {
        e.preventDefault()
        s.send({ cmd: 'action', action: a })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}
