/**
 * FedShield API client
 *
 * Handles JWT authentication automatically.
 * JWT token is kept in memory only and is never stored in localStorage.
 */

let token = null


/**
 * Login to the FedShield API.
 *
 * The username and password are supplied by the login UI.
 * We deliberately do NOT store credentials in frontend source code.
 */
export async function login(username, password) {
  if (!username || !password) {
    throw new Error('Username and password are required')
  }

  const form = new URLSearchParams()

  form.append('username', username)
  form.append('password', password)

  const res = await fetch('/auth/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: form,
  })

  if (!res.ok) {
    let message = 'Login failed'

    try {
      const data = await res.json()

      if (data?.detail) {
        message = data.detail
      }
    } catch {
      // Keep the default error message.
    }

    throw new Error(message)
  }

  const data = await res.json()

  token = data.access_token

  return data
}


/**
 * Clear the in-memory JWT token.
 */
export function logout() {
  token = null
}


/**
 * Check whether the client currently has a JWT token.
 */
export function isAuthenticated() {
  return token !== null
}


/**
 * Return the current JWT token.
 *
 * Useful if the application needs to inspect authentication state.
 */
export function getToken() {
  return token
}


/**
 * Perform an authenticated GET request.
 */
async function get(path) {
  if (!token) {
    throw new Error('Authentication required')
  }

  let res = await fetch(path, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })

  /*
   * If the JWT has expired or is invalid, clear it.
   *
   * We intentionally do NOT automatically log in again because
   * the frontend no longer contains a hardcoded password.
   *
   * The UI should ask the user to log in again.
   */
  if (res.status === 401) {
    token = null
    throw new Error('Session expired. Please log in again.')
  }

  if (!res.ok) {
    let message = `API error: ${res.status}`

    try {
      const data = await res.json()

      if (data?.detail) {
        message = data.detail
      }
    } catch {
      // Keep the default API error.
    }

    throw new Error(message)
  }

  return res.json()
}


/**
 * Public API methods.
 */
export const api = {

  // System
  health: () =>
    fetch('/api/health').then(async (res) => {
      if (!res.ok) {
        throw new Error(`Health check failed: ${res.status}`)
      }

      return res.json()
    }),

  // Public experiment summary used by the unauthenticated landing page.
  publicSummary: () =>
    fetch('/api/public-summary').then(async (res) => {
      if (!res.ok) {
        throw new Error(`Summary request failed: ${res.status}`)
      }

      return res.json()
    }),

  // SOC
  stats: () =>
    get('/api/stats'),

  feed: (limit = 50) =>
    get(`/api/feed?limit=${limit}`),

  breakdown: () =>
    get('/api/breakdown'),

  timeline: () =>
    get('/api/timeline'),

  blocked: () =>
    get('/api/blocked'),

  // ML
  training: () =>
    get('/api/training'),

  shap: () =>
    get('/api/shap'),

  drift: () =>
    get('/api/drift'),
}