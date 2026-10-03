// Thin client for the FastAPI backend (same-origin /api; proxied by nginx or the Vite dev server).
async function handle(r) {
  let body = null
  try { body = await r.json() } catch { /* non-JSON error body */ }
  if (!r.ok) throw new Error((body && body.detail) || `${r.status} ${r.statusText}`)
  return body
}

export const getHealth = () => fetch('/api/health').then(handle)
export const getSamples = () => fetch('/api/samples').then(handle)
export const postForm = (path, form) => fetch(path, { method: 'POST', body: form }).then(handle)

export async function dataUrlToFile(dataUrl, name) {
  const blob = await (await fetch(dataUrl)).blob()
  return new File([blob], name, { type: blob.type })
}
