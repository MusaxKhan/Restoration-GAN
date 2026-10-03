import React, { useState } from 'react'
import ImagePicker from '../components/ImagePicker.jsx'
import { Panel, Segmented, Spinner, Stat } from '../components/Ui.jsx'
import { postForm } from '../api.js'

const STYLES = [
  { value: 1, label: 'Style 1' },
  { value: 2, label: 'Style 2' },
  { value: 3, label: 'Style 3' },
]

export default function Sketch() {
  const [image, setImage] = useState(null)
  const [style, setStyle] = useState(1)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [res, setRes] = useState(null)

  async function run() {
    if (!image) return setError('Upload a photograph, pick a sample, or capture one with the webcam.')
    setBusy(true); setError('')
    const f = new FormData()
    f.append('file', image.file)
    f.append('style', String(style))
    try { setRes(await postForm('/api/sketch', f)) } catch (e) { setError(e.message); setRes(null) } finally { setBusy(false) }
  }

  return (
    <div>
      <h1 className="font-display text-3xl font-bold">Face-to-Sketch Generator</h1>
      <p className="mt-2 max-w-3xl text-muted">
        A style-conditioned pix2pix generator (U-Net + PatchGAN) turns a face photograph into a pencil sketch in one of three FS2K styles.
      </p>

      <div className="mt-8 grid gap-6 xl:grid-cols-[360px_1fr]">
        <aside className="card h-fit space-y-6 p-5">
          <div>
            <div className="label mb-3">1 · Photograph</div>
            <ImagePicker value={image} onChange={setImage} webcam />
          </div>
          <div>
            <div className="label mb-3">2 · Sketch style</div>
            <Segmented options={STYLES} value={style} onChange={setStyle} />
          </div>
          <button type="button" className="btn-primary w-full" onClick={run} disabled={busy}>
            {busy ? <><Spinner /> Generating…</> : 'Generate sketch'}
          </button>
          {error && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">{error}</div>}
        </aside>

        <section className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <Panel title="Original photograph" src={res?.input || image?.url} hint="Choose a photograph" />
            <Panel title={`Generated sketch${res ? ' · ' + res.style : ''}`} src={res?.sketch} loading={busy} hint="The result appears here" />
          </div>
          {res && (
            <div className="flex flex-wrap items-center gap-4">
              <Stat label="Inference time" value={`${res.inference_ms} ms`} />
              <Stat label="Style" value={res.style} />
              <a className="btn-primary ml-auto" href={res.sketch} download={`sketch_${res.style.replace(' ', '').toLowerCase()}.png`}>
                Download result
              </a>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
