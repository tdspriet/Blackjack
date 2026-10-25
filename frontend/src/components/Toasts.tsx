import { useLab } from '../store'

export function Toasts() {
  const toasts = useLab((s) => s.toasts)
  const dismiss = useLab((s) => s.dismiss)
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} onClick={() => dismiss(t.id)}>
          {t.message}
        </div>
      ))}
    </div>
  )
}
