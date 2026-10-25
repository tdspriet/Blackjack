import { CHIP_COLORS, chipColor } from '../../lib/colors'
import { compact } from '../../lib/format'

export function Chip({
  value,
  size = 40,
  chips,
  selected,
  onClick,
  disabled,
  label,
}: {
  value: number
  size?: number
  chips?: number[]
  selected?: boolean
  onClick?: () => void
  disabled?: boolean
  label?: string
}) {
  const c = chips ? chipColor(value, chips) : CHIP_COLORS[value] ?? CHIP_COLORS[1]
  const style = {
    width: size,
    height: size,
    '--chip-base': c.base,
    '--chip-edge': c.edge,
    '--chip-text': c.text,
    fontSize: Math.max(9, size * (String(compact(value)).length > 3 ? 0.24 : 0.3)),
  } as React.CSSProperties
  const Tag = onClick ? 'button' : 'div'
  return (
    <Tag
      className={`chip ${selected ? 'selected' : ''} ${onClick ? 'clickable' : ''}`}
      style={style}
      onClick={onClick}
      disabled={onClick ? disabled : undefined}
      aria-label={label ?? `€${value} chip`}
      aria-pressed={onClick ? selected : undefined}
    >
      <span>{label ?? compact(value)}</span>
    </Tag>
  )
}
