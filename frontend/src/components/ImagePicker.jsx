import React, { useEffect, useRef, useState } from 'react'
import { dataUrlToFile, getSamples } from '../api.js'

const MAX_MB = 10

/** Upload (click / drag & drop), choose a clean sample, or (optionally) capture from the webcam. */
export default function ImagePicker({ value, onChange, webcam = false }) {
  const [samples, setSamples] = useState([])
  const [error, setError] = useState('')
  const [cam, setCam] = useState(false)
  const inputRef = useRef(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  useEffect(() => { getSamples().then(setSamples).catch(() => {}) }, [])
  useEffect(() => () => stopCam(), [])

  const take = (file, label) => {
    setError('')
    if (!file.type.startsWith('image/')) return setError('Please choose an image file.')
    if (file.size > MAX_MB * 1024 * 1024) return setError(`File is larger than ${MAX_MB} MB.`)
    onChange({ file, url: URL.createObjectURL(file), name: label || file.name })
  }

  const pickSample = async (s) => take(await dataUrlToFile(s.image, `${s.name}.jpg`), s.name)

  async function startCam() {
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      setCam(true)
      setTimeout(() => { if (videoRef.current) videoRef.current.srcObject = streamRef.current }, 50)
    } catch { setError('Webcam unavailable or permission denied.') }
  }
  function stopCam() {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setCam(false)
  }
  function capture() {
    const v = videoRef.current
    const c = document.createElement('canvas')
    c.width = v.videoWidth; c.height = v.videoHeight
    c.getContext('2d').drawImage(v, 0, 0)
    c.toBlob((b) => { take(new File([b], 'webcam.png', { type: 'image/png' }), 'webcam capture'); stopCam() }, 'image/png')
  }

  return (
    <div className="space-y-3">
      {cam ? (
        <div className="space-y-2">
          <video ref={videoRef} autoPlay playsInline muted className="w-full rounded-xl bg-black" />
          <div className="flex gap-2">
            <button type="button" className="btn-primary flex-1 !py-2" onClick={capture}>Capture photo</button>
            <button type="button" className="btn-ghost" onClick={stopCam}>Cancel</button>
          </div>
        </div>
      ) : (
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); if (e.dataTransfer.files[0]) take(e.dataTransfer.files[0]) }}
          className="flex h-40 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-line bg-surface text-center transition hover:border-primary/70"
        >
          {value ? (
            <img src={value.url} alt="selected" className="h-full w-full rounded-xl object-contain p-1" />
          ) : (
            <>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="text-primary"><path d="M12 16V4m0 0 4 4m-4-4L8 8M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></svg>
              <div className="text-sm">Drop an image or <span className="text-primary">browse</span></div>
              <div className="text-xs text-muted">JPEG · PNG · WEBP · up to {MAX_MB} MB</div>
            </>
          )}
          <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp,image/bmp" hidden
                 onChange={(e) => e.target.files[0] && take(e.target.files[0])} />
        </div>
      )}
      {webcam && !cam && <button type="button" className="btn-ghost w-full" onClick={startCam}>Use webcam</button>}
      {error && <div className="text-sm text-red-400">{error}</div>}
      {samples.length > 0 && (
        <div>
          <div className="label mb-2">Sample images</div>
          <div className="grid grid-cols-4 gap-2">
            {samples.map((s) => (
              <button type="button" key={s.name} onClick={() => pickSample(s)}
                      className={`overflow-hidden rounded-lg border ${value?.name === s.name ? 'border-primary shadow-glow' : 'border-line'} hover:border-primary/70`}>
                <img src={s.image} alt={s.name} className="aspect-square w-full object-cover" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
