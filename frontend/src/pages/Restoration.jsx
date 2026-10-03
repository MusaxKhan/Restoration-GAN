import React, { useState } from 'react'
import ImagePicker from '../components/ImagePicker.jsx'
import { Bars, Panel, Segmented, Spinner, Stat } from '../components/Ui.jsx'
import { postForm } from '../api.js'

const CORRUPTIONS = [
  { value: 'clean', label: 'Clean' },
  { value: 'salt_pepper', label: 'Salt-and-pepper' },
  { value: 'blur', label: 'Gaussian blur' },
  { value: 'occlusion', label: 'Rectangular occlusion' },
]
const LEVELS = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
]

const VARIANTS = {
  universal: {
    endpoint: '/api/universal',
    title: 'Universal Restoration',
    blurb: 'One denoising autoencoder restores clean, noisy, blurred and occluded images without being told the corruption type.',
  },
  hard: {
    endpoint: '/api/hard-route',
    title: 'Hard-Routed Restoration',
    blurb: 'A classifier detects the corruption, then exactly one specialist autoencoder restores the image (clean images bypass restoration).',
  },
  soft: {
    endpoint: '/api/soft-moe',
    title: 'Soft Mixture-of-Experts Restoration',
    blurb: 'A gating network assigns a weight to the identity branch and each specialist; the output is their weighted blend.',
  },
}

export default function Restoration({ variant }) {
  const cfg = VARIANTS[variant]
  const [image, setImage] = useState(null)
  const [corruption, setCorruption] = useState('salt_pepper')
  const [level, setLevel] = useState('medium')
  const [asUploaded, setAsUploaded] = useState(false)
  const [seed, setSeed] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [res, setRes] = useState(null)

  async function run() {
    if (!image) return setError('Choose or upload an image first.')
    setBusy(true); setError('')
    const f = new FormData()
    f.append('file', image.file)
    f.append('mode', asUploaded ? 'uploaded' : 'corrupt')
    f.append('corruption', corruption)
    f.append('level', level)
    f.append('seed', String(seed))
    try { setRes(await postForm(cfg.endpoint, f)) } catch (e) { setError(e.message); setRes(null) } finally { setBusy(false) }
  }

  const s = res?.settings
  return (
    <div>
      <h1 className="font-display text-3xl font-bold">{cfg.title}</h1>
      <p className="mt-2 max-w-3xl text-muted">{cfg.blurb}</p>

      <div className="mt-8 grid gap-6 xl:grid-cols-[360px_1fr]">
        <aside className="card h-fit space-y-6 p-5">
          <div>
            <div className="label mb-3">1 · Image</div>
            <ImagePicker value={image} onChange={setImage} />
          </div>

          <div>
            <label className="flex cursor-pointer items-center gap-3 text-sm">
              <input type="checkbox" checked={asUploaded} onChange={(e) => setAsUploaded(e.target.checked)} className="h-4 w-4 accent-primary" />
              Use image as uploaded (already corrupted)
            </label>
          </div>

          <div className={asUploaded ? 'pointer-events-none opacity-40' : ''}>
            <div className="label mb-3">2 · Corruption</div>
            <Segmented options={CORRUPTIONS} value={corruption} onChange={setCorruption} />
            <div className="label mb-3 mt-5">3 · Severity</div>
            <Segmented options={LEVELS} value={level} onChange={setLevel} disabled={corruption === 'clean'} />
            <button type="button" className="btn-ghost mt-4 w-full" onClick={() => setSeed(Math.floor(Math.random() * 1e6))}>
              New random pattern (seed {seed})
            </button>
          </div>

          <button type="button" className="btn-primary w-full" onClick={run} disabled={busy}>
            {busy ? <><Spinner /> Restoring…</> : 'Restore image'}
          </button>
          {error && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">{error}</div>}
        </aside>

        <section className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-4">
            <Panel title="Clean reference" src={res?.clean} hint={asUploaded ? 'No reference for an uploaded image' : 'Result appears here'} loading={busy} />
            <Panel title="Corrupted input" src={res?.input} loading={busy} hint="Result appears here" />
            <Panel title="Restored output" src={res?.restored} loading={busy} hint="Result appears here" download="restored.png" />
            <Panel title="Absolute error ×4" src={res?.error_map} loading={busy} hint={asUploaded ? 'Needs a clean reference' : 'Result appears here'} />
          </div>

          {res && (
            <>
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <Stat label="Inference time" value={`${res.inference_ms} ms`} />
                <Stat label="PSNR input → restored" value={res.metrics ? `${res.metrics.psnr_input} → ${res.metrics.psnr_restored} dB` : 'n/a'} />
                {variant === 'hard' && <Stat label="Classifier / expert" value={`${res.classifier_ms} / ${res.expert_ms} ms`} />}
                <Stat label="Corruption" value={s.type} />
              </div>

              <div className="card p-5">
                <div className="label mb-3">Selected corruption settings</div>
                <div className="flex flex-wrap gap-2 text-sm">
                  {Object.entries(s).filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => (
                    <span key={k} className="rounded-lg border border-line bg-surface px-3 py-1.5"><span className="text-muted">{k}: </span>{String(v)}</span>
                  ))}
                </div>
              </div>

              {variant === 'hard' && (
                <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
                  <Bars title="Classifier probabilities" values={res.probabilities} highlight={res.predicted} />
                  <div className="card space-y-4 p-5">
                    <div><div className="label mb-2">Predicted corruption</div>
                      <span className="rounded-full bg-primary/20 px-3 py-1 font-display font-semibold text-primary">{res.predicted}</span></div>
                    <div><div className="label mb-2">Selected expert</div><div className="font-medium">{res.selected_expert}</div></div>
                    {res.true_condition && (
                      <div><div className="label mb-2">True condition</div>
                        <span className={res.true_condition === res.predicted ? 'text-tertiary' : 'text-red-400'}>
                          {res.true_condition} {res.true_condition === res.predicted ? '✓ routed correctly' : '✗ misrouted'}
                        </span></div>
                    )}
                  </div>
                </div>
              )}

              {variant === 'soft' && (
                <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
                  <Bars title="Routing weights" values={res.weights} highlight={res.dominant_branch} />
                  <div className="card p-5">
                    <div className="label mb-3">Strongest contributors</div>
                    <ol className="space-y-2">
                      {res.contributions_ranked.map((n, i) => (
                        <li key={n} className="flex items-center gap-3">
                          <span className={`grid h-6 w-6 place-items-center rounded-full text-xs font-semibold ${i === 0 ? 'bg-primary text-white' : 'bg-surface text-muted'}`}>{i + 1}</span>
                          <span className={i === 0 ? 'font-semibold' : 'text-slate-300'}>{n}</span>
                          <span className="ml-auto tabular-nums text-muted">{(res.weights[n] * 100).toFixed(0)}%</span>
                        </li>
                      ))}
                    </ol>
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  )
}
