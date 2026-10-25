import type { ClientCmd, ServerMsg } from './protocol'

export type TransportKind = 'live' | 'mock'

export interface Transport {
  kind: TransportKind
  send(cmd: ClientCmd): void
  close(): void
}

export interface TransportHandlers {
  onMessage(msg: ServerMsg): void
  onOpen?(): void
  onClose?(reason: string): void
}

export function wsUrl(): string {
  const env = import.meta.env.VITE_WS_URL as string | undefined
  if (env) return env
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}/ws`
}

/**
 * Open the real WebSocket. Resolves once the server's `hello` arrives (that is what makes
 * a connection "usable"); rejects on error/timeout so the caller can fall back to mock.
 */
export function connectLive(h: TransportHandlers, timeoutMs = 2500): Promise<Transport> {
  return new Promise((resolve, reject) => {
    let settled = false
    let ws: WebSocket
    try {
      ws = new WebSocket(wsUrl())
    } catch (e) {
      reject(e)
      return
    }
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true
        ws.close()
        reject(new Error('timeout'))
      }
    }, timeoutMs)
    const transport: Transport = {
      kind: 'live',
      send(cmd) {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(cmd))
      },
      close() {
        ws.onclose = null
        ws.close()
      },
    }
    ws.onmessage = (ev) => {
      let msg: ServerMsg
      try {
        msg = JSON.parse(String(ev.data)) as ServerMsg
      } catch {
        return
      }
      if (!settled && msg.type === 'hello') {
        settled = true
        clearTimeout(timer)
        resolve(transport)
        h.onOpen?.()
      }
      h.onMessage(msg)
    }
    ws.onerror = () => {
      if (!settled) {
        settled = true
        clearTimeout(timer)
        reject(new Error('websocket error'))
      }
    }
    ws.onclose = (ev) => {
      if (!settled) {
        settled = true
        clearTimeout(timer)
        reject(new Error(`closed (${ev.code})`))
        return
      }
      h.onClose?.(ev.reason || `code ${ev.code}`)
    }
  })
}

/** In-browser mock server (lazy-loaded so it is code-split and easy to drop later). */
export async function connectMock(h: TransportHandlers): Promise<Transport> {
  const { MockServer } = await import('./mock/server')
  const server = new MockServer()
  server.subscribe((m) => h.onMessage(m))
  ;(window as unknown as { __mock?: unknown }).__mock = server
  server.start()
  h.onOpen?.()
  return {
    kind: 'mock',
    send: (cmd) => server.send(cmd),
    close: () => server.close(),
  }
}
