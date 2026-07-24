import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
})

// Attach JWT token from localStorage on every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('ckd_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Handle 401 globally — redirect to login
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('ckd_token')
      localStorage.removeItem('ckd_user')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

// ── Auth ──────────────────────────────────────────────
export const authAPI = {
  login: (username, password) => {
    const form = new URLSearchParams()
    form.append('username', username)
    form.append('password', password)
    return api.post('/auth/login', form, { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } })
  },
  register: (data) => api.post('/auth/register', data),
  me: () => api.get('/auth/me'),
}

// ── Predictions ────────────────────────────────────────
export const predictAPI = {
  tabular: (payload) => api.post('/predict/tabular', payload),
  image: (file, imagePath) => {
    const form = new FormData()
    if (file) form.append('file', file)
    if (imagePath) form.append('image_path', imagePath)
    return api.post('/predict/image', form)
  },
  fusion: (patientData, file, imagePath, options = {}) => {
    const form = new FormData()
    form.append('patient_data', JSON.stringify(patientData))
    if (file) form.append('file', file)
    if (imagePath) form.append('image_path', imagePath)
    form.append('fusion_type', options.fusionType || 'late')
    if (options.tabularWeight != null) form.append('tabular_weight', options.tabularWeight)
    if (options.imageWeight != null) form.append('image_weight', options.imageWeight)
    return api.post('/predict/fusion', form)
  },
}

// ── Training ───────────────────────────────────────────
export const trainAPI = {
  startImageTraining: () => api.post('/train/image'),
  getStatus: () => api.get('/train/status'),
  getModelStatus: () => api.get('/model/status'),
  getMetrics: () => api.get('/metrics'),
  runGenetic: (popSize = 15, generations = 5) =>
    api.post(`/train/optimize-genetic?pop_size=${popSize}&generations=${generations}`),
}

// ── History ────────────────────────────────────────────
export const historyAPI = {
  getHistory: (params) => api.get('/history', { params }),
  getMeta: () => api.get('/predictions/meta'),
}

// ── Upload ─────────────────────────────────────────────
export const uploadAPI = {
  upload: (file, patientId) => {
    const form = new FormData()
    form.append('file', file)
    if (patientId) form.append('patient_id', patientId)
    return api.post('/upload', form)
  },
}

// ── Health ─────────────────────────────────────────────
export const healthAPI = {
  check: () => api.get('/health'),
}

export default api
