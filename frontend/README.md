# Swasthya Quant: frontend

React 19 + TypeScript + Vite + Tailwind CSS v4 + Recharts.

```bash
npm install
npm run dev        # http://localhost:5173, proxies /api → http://127.0.0.1:8000
npm run build      # type-check + production build (dist/)
npm run preview    # serve dist/ on :4173 with the same proxy
npm run lint
```

Start the FastAPI backend first (see the root `README.md`). Environment options are documented in `.env.example`.

## Data rules

- API contracts are mirrored in `src/types/api.ts`. Do not add fields the backend does not send.
- Classical benchmark metrics are aggregated from `../results/runs/ledger.csv`, which is imported at build time (`src/data/research.ts`). Rebuild after the ledger changes.
- Missing results render as "Awaiting benchmark" or "Not available". Never replace them with sample numbers.
- `src/mocks/mockData.ts` is demo-only. It is used only when `VITE_DEMO_MODE=true` and the model is unavailable, and its output is always watermarked.

See the root README for routes, structure, and known backend issues.
