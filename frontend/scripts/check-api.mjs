#!/usr/bin/env node
// Frontend <-> backend integration check.
//
// By default it goes through the Vite dev server (http://127.0.0.1:5173), exercising the
// same /api proxy the browser uses. Point it straight at FastAPI with:
//   API_URL=http://127.0.0.1:8000 npm run test:api
//
// Checks each endpoint against the shapes in src/types/api.ts. Exit code 1 on any failure.

const BASE = (process.env.API_URL || 'http://127.0.0.1:5173').replace(/\/$/, '')

const VALID = { age: 58, height: 170, weight: 78, ap_hi: 145, ap_lo: 90, smoke: 0, alco: 0, active: 1, gender: 1, cholesterol: 1, gluc: 1 }

const results = []
const check = (name, ok, detail = '') => results.push({ name, ok: !!ok, detail })

async function call(method, path, body) {
  const started = performance.now()
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  const text = await res.text()
  let json = null
  try { json = JSON.parse(text) } catch { /* not JSON */ }
  return { status: res.status, json, text, ms: Math.round(performance.now() - started) }
}

const isNum = (v) => typeof v === 'number' && Number.isFinite(v)
const locs = (json) => (Array.isArray(json?.detail) ? json.detail.map((d) => (d.loc ?? []).at(-1)) : [])

async function main() {
  console.log(`\nSwasthya Quant API check -> ${BASE}\n`)

  // 0. Reachability (and, via the dev server, that the frontend itself is served).
  try {
    if (!process.env.API_URL) {
      const page = await fetch(`${BASE}/`).then((r) => r.text())
      check('Frontend served at /', page.includes('<div id="root">'), 'index.html contains #root')
    }
  } catch {
    console.error(`Cannot reach ${BASE}. Start the dev server (npm run dev) and the backend first.`)
    process.exitCode = 1
    return
  }

  // 1. Health
  let modelLoaded = false
  try {
    const h = await call('GET', '/api/health')
    modelLoaded = h.json?.model_loaded === true
    check('GET /api/health -> 200', h.status === 200, `status ${h.status}${h.status >= 500 && !h.json ? ' (proxy could not reach backend on :8000?)' : ''}`)
    check('health shape {status, model_loaded}', typeof h.json?.status === 'string' && typeof h.json?.model_loaded === 'boolean', JSON.stringify(h.json))
  } catch (e) { check('GET /api/health', false, String(e)) }

  // 2. Valid prediction
  try {
    const p = await call('POST', '/api/predict', VALID)
    if (modelLoaded) {
      const j = p.json ?? {}
      check('POST /api/predict (valid) -> 200', p.status === 200, `status ${p.status}, ${p.ms} ms`)
      check('prediction is 0 or 1', j.prediction === 0 || j.prediction === 1, `prediction=${j.prediction}`)
      check('risk_probability in [0, 1]', isNum(j.risk_probability) && j.risk_probability >= 0 && j.risk_probability <= 1, `p=${j.risk_probability}`)
      check('risk_percentage = round(p × 100)', j.risk_percentage === Math.round(j.risk_probability * 100), `${j.risk_percentage}%`)
      check('prediction matches 0.5 threshold', j.prediction === (j.risk_probability >= 0.5 ? 1 : 0))
      check('risk_label and disclaimer are text', typeof j.risk_label === 'string' && typeof j.disclaimer === 'string', j.risk_label)
    } else {
      check('POST /api/predict -> 503 while model not loaded', p.status === 503, `status ${p.status}: ${p.json?.detail ?? ''}`.slice(0, 120))
    }
  } catch (e) { check('POST /api/predict (valid)', false, String(e)) }

  // 3. Validation errors map to fields (the form highlights these)
  try {
    const bad = await call('POST', '/api/predict', { ...VALID, age: 10, ap_hi: 80, ap_lo: 90 })
    const fields = locs(bad.json)
    check('invalid input -> 422', bad.status === 422, `status ${bad.status}`)
    check('422 names the offending fields', fields.includes('age') && (fields.includes('ap_hi') || fields.includes('ap_lo')), `fields: ${fields.join(', ')}`)
    const extra = await call('POST', '/api/predict', { ...VALID, cholesterol_mg: 240 })
    check('unknown field rejected (extra="forbid")', extra.status === 422, `status ${extra.status}`)
    const missing = await call('POST', '/api/predict', { ...VALID, gluc: undefined })
    check('missing field rejected', missing.status === 422 && locs(missing.json).includes('gluc'), `status ${missing.status}`)
  } catch (e) { check('validation errors', false, String(e)) }

  // 4. Benchmark
  try {
    const b = await call('GET', '/api/benchmark')
    const count = b.json?.models ? Object.keys(b.json.models).length : -1
    check('GET /api/benchmark -> 200', b.status === 200, `status ${b.status}`)
    check('benchmark shape {status, models}', typeof b.json?.status === 'string' && typeof b.json?.models === 'object', `status=${b.json?.status}, models=${count}`)
    if (count === 0) console.log('  note: /api/benchmark returned 0 models; the UI falls back to the bundled ledger (known backend CSV-parsing issue).')
  } catch (e) { check('GET /api/benchmark', false, String(e)) }

  const w = Math.max(...results.map((r) => r.name.length))
  for (const r of results) console.log(`  ${r.ok ? 'PASS' : 'FAIL'}  ${r.name.padEnd(w)}  ${r.detail}`)
  const failed = results.filter((r) => !r.ok).length
  console.log(`\n${results.length - failed}/${results.length} checks passed${modelLoaded ? '' : ' (model not loaded: prediction success path not exercised)'}\n`)
  process.exitCode = failed ? 1 : 0
}

main()
