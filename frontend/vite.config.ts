import { defineConfig, type Connect, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const CARDS_DIR = path.resolve(here, '../cards')
const BACKEND = process.env.ONEBJ_BACKEND ?? 'http://localhost:8000'

/**
 * Serve the repo's `cards/` folder at `/cards/*` directly from disk in dev/preview,
 * so card images work in mock mode without the backend (the backend serves the
 * very same files at the same URL in production).
 */
function serveCards(): Plugin {
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    const url = req.url ?? ''
    if (!url.startsWith('/cards/')) return next()
    const name = decodeURIComponent(url.slice('/cards/'.length).split('?')[0])
    if (!/^[A-Za-z0-9_]+\.png$/.test(name)) return next()
    const file = path.join(CARDS_DIR, name)
    if (!fs.existsSync(file)) {
      res.statusCode = 404
      return res.end()
    }
    res.setHeader('Content-Type', 'image/png')
    res.setHeader('Cache-Control', 'public, max-age=86400')
    fs.createReadStream(file).pipe(res)
  }
  return {
    name: 'onebj-serve-cards',
    configureServer(server) {
      server.middlewares.use(handler)
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler)
    },
  }
}

export default defineConfig({
  plugins: [serveCards(), react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/ws': { target: BACKEND.replace(/^http/, 'ws'), ws: true, changeOrigin: true },
    },
  },
  preview: {
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/ws': { target: BACKEND.replace(/^http/, 'ws'), ws: true, changeOrigin: true },
    },
  },
})
