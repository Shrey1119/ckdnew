import { useState, useEffect } from 'react'
import DashboardLayout from '../layouts/DashboardLayout'
import { historyAPI } from '../services/api'
import { Clock, AlertCircle, CheckCircle, ChevronLeft, ChevronRight } from 'lucide-react'

export default function History() {
  const [items, setItems]   = useState([])
  const [total, setTotal]   = useState(0)
  const [page, setPage]     = useState(1)
  const [loading, setLoading] = useState(true)
  const [label, setLabel]   = useState('')
  const [risk, setRisk]     = useState('')
  const SIZE = 10

  useEffect(() => {
    setLoading(true)
    historyAPI.getHistory({ page, size: SIZE, label: label || undefined, risk: risk || undefined })
      .then(res => {
        setItems(res.data.items || [])
        setTotal(res.data.total || 0)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [page, label, risk])

  const totalPages = Math.ceil(total / SIZE)

  return (
    <DashboardLayout title="Prediction History">
      <div className="max-w-6xl mx-auto space-y-5">

        {/* Filters */}
        <div className="glass-card p-4 flex flex-wrap gap-4 items-center">
          <Clock size={18} className="text-cyan-400" />
          <p className="text-white font-semibold flex-1">All Predictions</p>
          <select
            className="form-input w-auto"
            value={label}
            onChange={e => { setLabel(e.target.value); setPage(1) }}
          >
            <option value="">All Labels</option>
            <option value="ckd">CKD</option>
            <option value="not_ckd">Not CKD</option>
          </select>
          <select
            className="form-input w-auto"
            value={risk}
            onChange={e => { setRisk(e.target.value); setPage(1) }}
          >
            <option value="">All Risk</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>

        {/* Table */}
        <div className="glass-card overflow-hidden">
          {loading ? (
            <div className="p-8 flex items-center justify-center">
              <div className="w-8 h-8 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
            </div>
          ) : items.length === 0 ? (
            <div className="p-12 text-center">
              <Clock size={40} className="mx-auto mb-3 text-gray-600" />
              <p className="text-gray-500">No predictions found. Run a tabular, image, or fusion prediction first.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Patient</th>
                    <th>Label</th>
                    <th>Confidence</th>
                    <th>Risk</th>
                    <th>Tabular Prob</th>
                    <th>Image Prob</th>
                    <th>Fusion Prob</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map(p => (
                    <tr key={p.id}>
                      <td className="font-mono text-gray-500">#{p.id}</td>
                      <td>#{p.patient_id}</td>
                      <td>
                        <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-lg text-xs font-semibold
                          ${p.prediction_label === 'ckd' ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                          {p.prediction_label === 'ckd' ? <AlertCircle size={11} /> : <CheckCircle size={11} />}
                          {p.prediction_label.toUpperCase()}
                        </span>
                      </td>
                      <td className="capitalize text-gray-300">{p.confidence}</td>
                      <td>
                        <span className={`font-semibold capitalize text-sm risk-${p.risk_level}`}>{p.risk_level}</span>
                      </td>
                      <td className="font-mono text-gray-300">
                        {p.tabular_prob != null ? `${(p.tabular_prob * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="font-mono text-gray-300">
                        {p.image_prob != null ? `${(p.image_prob * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="font-mono text-cyan-300">
                        {p.fusion_prob != null ? `${(p.fusion_prob * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="text-gray-500 text-xs">{new Date(p.created_at).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between">
            <p className="text-gray-500 text-sm">Showing {(page - 1) * SIZE + 1}–{Math.min(page * SIZE, total)} of {total}</p>
            <div className="flex gap-2">
              <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
                className="p-2 rounded-xl bg-gray-800 text-gray-400 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
                <ChevronLeft size={16} />
              </button>
              <span className="px-4 py-2 rounded-xl bg-gray-800 text-white text-sm">{page} / {totalPages}</span>
              <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}
                className="p-2 rounded-xl bg-gray-800 text-gray-400 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  )
}
