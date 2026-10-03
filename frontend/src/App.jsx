import React, { useEffect, useState } from 'react'
import Restoration from './pages/Restoration.jsx'
import Sketch from './pages/Sketch.jsx'
import { getHealth } from './api.js'

const NAV = [
  { id: 'universal', label: 'Universal Restoration', hint: 'Task 1', icon: 'M4 12a8 8 0 1 0 16 0 8 8 0 0 0-16 0Zm8-4v4l3 2' },
  { id: 'hard', label: 'Hard-Routed Restoration', hint: 'Task 2', icon: 'M6 4v6a4 4 0 0 0 4 4h8m0 0-3-3m3 3-3 3M6 20v-3' },
  { id: 'soft', label: 'Soft Mixture-of-Experts', hint: 'Task 3', icon: 'M4 7h6m4 0h6M4 12h10m4 0h2M4 17h3m4 0h9' },
  { id: 'sketch', label: 'Face-to-Sketch Generator', hint: 'Task 4', icon: 'm4 20 4-1L19 8a2.1 2.1 0 0 0-3-3L5 16l-1 4Z' },
]

function useHash() {
  const read = () => (window.location.hash || '#universal').slice(1)
  const [page, setPage] = useState(read)
  useEffect(() => {
    const h = () => setPage(read())
    window.addEventListener('hashchange', h)
    return () => window.removeEventListener('hashchange', h)
  }, [])
  return page
}

function HealthBadge() {
  const [h, setH] = useState(null)
  useEffect(() => {
    const poll = () => getHealth().then(setH).catch(() => setH({ status: 'down' }))
    poll()
    const t = setInterval(poll, 10000)
    return () => clearInterval(t)
  }, [])
  const ok = h?.status === 'ok'
  const missing = h?.models ? Object.entries(h.models).filter(([, v]) => !v).map(([k]) => k) : []
  return (
    <div className="flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1.5 text-sm" title={missing.length ? `Missing models: ${missing.join(', ')}` : ''}>
      <span className={`h-2 w-2 rounded-full ${!h ? 'bg-slate-500' : ok && !missing.length ? 'bg-tertiary' : ok ? 'bg-amber-400' : 'bg-red-500'}`} />
      <span className="text-muted">
        {!h ? 'Connecting…' : !ok ? 'Backend offline' : missing.length ? `Backend online · ${missing.length} model(s) missing` : 'Backend online · all models loaded'}
      </span>
    </div>
  )
}

export default function App() {
  const page = useHash()
  const current = NAV.find((n) => n.id === page) || NAV[0]
  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-72 shrink-0 flex-col border-r border-line bg-surface p-5 lg:flex">
        <div className="mb-8 flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-primary to-secondary font-display text-lg font-bold">R</div>
          <div>
            <div className="font-display text-lg font-bold leading-tight">Restoration Studio</div>
            <div className="text-xs text-muted">Generative AI · Assignment 1</div>
          </div>
        </div>
        <nav className="space-y-1.5">
          {NAV.map((n) => (
            <a key={n.id} href={`#${n.id}`}
               className={`flex items-center gap-3 rounded-xl px-3 py-3 transition ${current.id === n.id ? 'bg-primary/15 text-white shadow-glow' : 'text-slate-300 hover:bg-white/5'}`}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" className={current.id === n.id ? 'text-secondary' : 'text-muted'}><path d={n.icon} /></svg>
              <div>
                <div className="text-sm font-medium">{n.label}</div>
                <div className="text-xs text-muted">{n.hint}</div>
              </div>
            </a>
          ))}
        </nav>
        <div className="mt-auto text-xs text-muted">Models run as ONNX on the FastAPI backend.</div>
      </aside>

      <main className="min-w-0 flex-1">
        <header className="flex items-center justify-between border-b border-line px-6 py-4">
          <div className="lg:hidden">
            <select value={current.id} onChange={(e) => { window.location.hash = e.target.value }}
                    className="rounded-lg border border-line bg-surface px-3 py-2 text-sm">
              {NAV.map((n) => <option key={n.id} value={n.id}>{n.label}</option>)}
            </select>
          </div>
          <div className="hidden text-sm text-muted lg:block">{current.hint} · {current.label}</div>
          <HealthBadge />
        </header>
        <div className="p-6 lg:p-8">
          {current.id === 'sketch' ? <Sketch /> : <Restoration key={current.id} variant={current.id} />}
        </div>
      </main>
    </div>
  )
}
