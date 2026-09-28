import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { defineConfig, loadEnv } from 'vite'

// The FastAPI backend does not enable CORS, so the dev/preview servers proxy /api to it.
// Override the target with VITE_API_PROXY_TARGET in frontend/.env.local.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'
  const proxy = { '/api': { target, changeOrigin: true } }

  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      proxy,
      // Allow importing the research ledger from ../results as a build-time snapshot.
      fs: { allow: [fileURLToPath(new URL('..', import.meta.url))] },
    },
    preview: { port: 4173, proxy },
  }
})
