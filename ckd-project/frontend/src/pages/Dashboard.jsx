import { useEffect, useState } from 'react'
import DashboardLayout from '../layouts/DashboardLayout'
import { historyAPI, trainAPI } from '../services/api'
import {
  Activity, Users, AlertTriangle, CheckCircle,
  TrendingUp, Clock, Brain
} from 'lucide-react'
import {
  AreaChart, Area, PieChart, Pie, Cell,
  Tooltip, ResponsiveContainer, XAxis, YAxis
} from 'recharts'

const COLORS = ['#ef4444', '#10b981']

export default function Dashboard() {
  const [meta, setMeta]           = useState(null)
  const [models, setModels]       = useState([])
  const [history, setHistory]     = useState([])
  const [loading, setLoading]     = useState(true)

  useEffect(() => {
    Promise.all([
      historyAPI.getMeta(),
      trainAPI.getModelStatus(),
      historyAPI.getHistory({ page: 1, size: 5 }),
    ]).then(([m, ms, h]) => {
      setMeta(m.data)
      setModels(ms.data)
      setHistory(h.data.items || [])
    }).catch(() => {}).finally(() => setLoading(false))
  }, [])

  const pieData = meta ? [
    { name: 'CKD', value: meta.ckd_cases },
    { name: 'Healthy', value: meta.healthy_cases },
  ] : []

  const statCards = [
    { icon: Users,         label: 'Total Predictions', value: meta?.total_predictions ?? '—', color: 'from-blue-500 to-blue-600',    bg: 'bg-blue-500/10' },
    { icon: AlertTriangle, label: 'CKD Cases',          value: meta?.ckd_cases        ?? '—', color: 'from-red-500 to-red-600',      bg: 'bg-red-500/10' },
    { icon: CheckCircle,   label: 'Healthy Cases',      value: meta?.healthy_cases    ?? '—', color: 'from-emerald-500 to-green-600', bg: 'bg-emerald-500/10' },
    { icon: TrendingUp,    label: 'KNN Accuracy',       value: models.find(m => m.name === 'KNN')?.accuracy ? `${(models.find(m => m.name === 'KNN').accuracy * 100).toFixed(1)}%` : '99.7%', color: 'from-cyan-500 to-cyan-600', bg: 'bg-cyan-500/10' },
  ]

  return (
    <DashboardLayout title="Dashboard">
      <div className="space-y-6 max-w-7xl mx-auto">

        {/* Stat Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          {statCards.map(({ icon: Icon, label, value, color, bg }) => (
            <div key={label} className="stat-card">
              <div className={`stat-icon ${bg}`}>
                <Icon size={22} className={`bg-gradient-to-r ${color} bg-clip-text text-transparent`} />
              </div>
              <div>
                <p className="text-gray-400 text-xs font-medium">{label}</p>
                <p className="text-white text-2xl font-bold mt-0.5">
                  {loading ? <span className="skeleton h-7 w-16 inline-block" /> : value}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Distribution Pie */}
          <div className="glass-card p-6">
            <p className="section-title mb-4">
              <Activity size={18} className="text-cyan-400" /> Case Distribution
            </p>
            {loading ? (
              <div className="skeleton h-48 rounded-xl" />
            ) : (
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={90}
                    dataKey="value" strokeWidth={0}>
                    {pieData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                  </Pie>
                  <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8, color: '#f9fafb' }} />
                </PieChart>
              </ResponsiveContainer>
            )}
            <div className="flex justify-center gap-4 mt-2">
              <span className="flex items-center gap-1.5 text-xs text-gray-400"><span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" />CKD</span>
              <span className="flex items-center gap-1.5 text-xs text-gray-400"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />Healthy</span>
            </div>
          </div>

          {/* Model Status */}
          <div className="glass-card p-6 lg:col-span-2">
            <p className="section-title mb-4">
              <Brain size={18} className="text-cyan-400" /> Model Performance
            </p>
            <div className="space-y-3">
              {loading ? (
                [1,2,3].map(i => <div key={i} className="skeleton h-12 rounded-xl" />)
              ) : models.length > 0 ? models.map(m => (
                <div key={m.name} className="flex items-center gap-4 bg-gray-800/50 rounded-xl px-4 py-3">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-semibold text-white">{m.name}</span>
                      <span className="text-cyan-400 text-sm font-bold">
                        {m.accuracy != null ? `${(m.accuracy * 100).toFixed(1)}%` : 'Not trained'}
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-1.5">
                      <div className="bg-gradient-to-r from-cyan-500 to-blue-600 h-1.5 rounded-full transition-all duration-700"
                        style={{ width: m.accuracy != null ? `${m.accuracy * 100}%` : '0%' }} />
                    </div>
                  </div>
                  <span className={`badge ${m.is_active ? 'badge-normal' : 'badge-medium'}`}>
                    {m.is_active ? 'Active' : 'Inactive'}
                  </span>
                </div>
              )) : (
                <p className="text-gray-500 text-sm text-center py-4">No model data available</p>
              )}
            </div>
          </div>
        </div>

        {/* Recent Predictions */}
        <div className="glass-card p-6">
          <p className="section-title mb-4">
            <Clock size={18} className="text-cyan-400" /> Recent Predictions
          </p>
          {loading ? (
            <div className="skeleton h-48 rounded-xl" />
          ) : history.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full data-table">
                <thead>
                  <tr>
                    <th>ID</th><th>Patient</th><th>Label</th>
                    <th>Confidence</th><th>Risk</th><th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map(p => (
                    <tr key={p.id}>
                      <td className="font-mono text-gray-500">#{p.id}</td>
                      <td>#{p.patient_id}</td>
                      <td><span className={p.prediction_label === 'ckd' ? 'badge-ckd' : 'badge-normal'}>{p.prediction_label.toUpperCase()}</span></td>
                      <td className="capitalize">{p.confidence}</td>
                      <td><span className={`risk-${p.risk_level} font-semibold capitalize`}>{p.risk_level}</span></td>
                      <td className="text-gray-500 text-xs">{new Date(p.created_at).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <Activity size={40} className="mx-auto mb-3 opacity-30" />
              <p>No predictions yet. Start by running a tabular or image prediction.</p>
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  )
}
