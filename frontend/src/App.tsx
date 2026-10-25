import { useEffect } from 'react'
import { useLab } from './store'
import { TopBar } from './components/TopBar'
import { NNPanel } from './components/nn/NNPanel'
import { GameTable } from './components/table/GameTable'
import { CountPanel } from './components/count/CountPanel'
import { Toasts } from './components/Toasts'
import { useKeyboard } from './hooks/useKeyboard'

export function App() {
  const connect = useLab((s) => s.connect)
  const ready = useLab((s) => s.state !== null && s.hello !== null)
  const conn = useLab((s) => s.conn)

  useEffect(() => {
    void connect()
  }, [connect])

  useKeyboard()

  return (
    <div className="app">
      <TopBar />
      {ready ? (
        <div className="columns">
          <NNPanel />
          <GameTable />
          <CountPanel />
        </div>
      ) : (
        <div className="boot">
          <div className="boot-spinner" />
          <div>{conn.mode === 'connecting' ? 'Connecting to the ONE Blackjack Lab backend…' : 'Waiting for the first state…'}</div>
        </div>
      )}
      <Toasts />
    </div>
  )
}
