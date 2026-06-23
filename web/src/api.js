/**
 * FedShield API client — handles JWT auth automatically
 * Stores token in memory (not localStorage for security)
 */

let token = null

export async function login(username = 'fedshield', password = 'shield2025') {
  const form = new URLSearchParams()
  form.append('username', username)
  form.append('password', password)

  const res = await fetch('/auth/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: form,
  })

  if (!res.ok) throw new Error('Login failed')
  const data = await res.json()
  token = data.access_token
  return data
}

async function get(path) {
  if (!token) await login()
  const res = await fetch(path, {
    headers: { Authorization: `Bearer ${token}` }
  })
  if (res.status === 401) {
    token = null
    await login()
    return get(path)
  }
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export const api = {
  stats:     () => get('/api/stats'),
  feed:      (limit = 50) => get(`/api/feed?limit=${limit}`),
  breakdown: () => get('/api/breakdown'),
  timeline:  () => get('/api/timeline'),
  blocked:   () => get('/api/blocked'),
  training:  () => get('/api/training'),
  shap:      () => get('/api/shap'),
  drift:     () => get('/api/drift'),
  health:    () => fetch('/api/health').then(r => r.json()),
}