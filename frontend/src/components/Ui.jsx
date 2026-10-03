import React from 'react'

export function Segmented({ options, value, onChange, disabled }) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          disabled={disabled}
          onClick={() => onChange(o.value)}
          className={`chip ${value === o.value ? 'chip-on' : 'text-muted hover:text-white'} disabled:opacity-40`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Panel({ title, src, hint, loading, download }) {
  return (
    <div className="card overflow-hidden">
      <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
        <span className="label">{title}</span>
        {download && src && (
          <a href={src} download={download} className="text-xs text-secondary hover:underline">Download</a>
        )}
      </div>
      <div className="flex aspect-square items-center justify-center bg-surface">
        {src ? (
          <img src={src} alt={title} className="h-full w-full object-contain" style={{ imageRendering: 'pixelated' }} />
        ) : (
          <span className="px-6 text-center text-sm text-muted">{loading ? 'Running model…' : hint || 'Not available'}</span>
        )}
      </div>
    </div>
  )
}

export function Bars({ title, values, highlight }) {
  return (
    <div className="card p-5">
      <div className="label mb-4">{title}</div>
      <div className="space-y-3">
        {Object.entries(values).map(([name, v]) => (
          <div key={name}>
            <div className="mb-1 flex justify-between text-sm">
              <span className={name === highlight ? 'font-semibold text-white' : 'text-slate-300'}>{name}</span>
              <span className="tabular-nums text-muted">{(v * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-surface">
              <div
                className={`h-full rounded-full transition-all duration-500 ${name === highlight ? 'bg-gradient-to-r from-primary to-secondary' : 'bg-primary/40'}`}
                style={{ width: `${Math.max(1, v * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function Stat({ label, value }) {
  return (
    <div className="rounded-xl border border-line bg-surface px-4 py-3">
      <div className="label">{label}</div>
      <div className="mt-1 font-display text-xl font-semibold tabular-nums">{value ?? '—'}</div>
    </div>
  )
}

export function Spinner() {
  return <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
}
