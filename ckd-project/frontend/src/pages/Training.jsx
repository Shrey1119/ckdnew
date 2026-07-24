import { useState, useEffect } from 'react'
import DashboardLayout from '../layouts/DashboardLayout'
import { trainAPI } from '../services/api'
import { useAuth } from '../context/AuthContext'
import { Cpu, Play, RefreshCw, TrendingUp, Dna, AlertCircle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'
import toast from 'react-hot-toast'

export default function Training() {
  const { isAdmin } = useAuth()
  const [status, setStatus]   = useState(null)
  const [models, setModels]   = useState([])
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [trainLoading, setTrainLoading] = useState(false)
  const [genLoading, setGenLoading]     = useState(false)
  const [genResults, setGenResults]     = useState(null)

  const fetchStatus = async () => {
    try {
      const [s, m, met] = await Promise.all([
        trainAPI.getStatus(),
        trainAPI.getModelStatus(),
        trainAPI.getMetrics(),
      ])
      setStatus(s.data)
      setModels(m.data)
      setMetrics(met.data)
    } catch { /* backend may not be running */ }
    finally { setLoading(false) }
  }

  useEffect(() => { fetchStatus() }, [])

  // Poll while training
  useEffect(() => {
    if (!status?.is_training) return
    const id = setInterval(fetchStatus, 3000)
    return () => clearInterval(id)
  }, [status?.is_training])

  const handleStartTraining = async () => {
    setTrainLoading(true)
    try {
      await trainAPI.startImageTraining()
      toast.success('ResNet-18 training started in background!')
      fetchStatus()
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Training failed to start.')
    } finally {
      setTrainLoading(false)
    }
  }

  const handleRunGenetic = async () => {
    setGenLoading(true)
    setGenResults(null)
    try {
      const res = await trainAPI.runGenetic(15, 5)
      setGenResults(res.data.results)
      toast.success('Genetic optimization complete!')
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Genetic optimization failed.')
    } finally {
      setGenLoading(false)
    }
  }

  const chartData = status?.logs?.map(l => ({
    epoch: l.epoch,
    trainAcc: parseFloat((l.accuracy * 100).toFixed(2)),
    valAcc:   parseFloat((l.val_accuracy * 100).toFixed(2)),
    trainLoss: parseFloat(l.loss.toFixed(4)),
    valLoss:   parseFloat(l.val_loss.toFixed(4)),
  })) || []

  return (
    <DashboardLayout title="Model Training">
      <div className="max-w-5xl mx-auto space-y-6">

        {/* Model Status Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {loading ? [1,2,3].map(i => <div key={i} className="skeleton h-24 rounded-2xl" />)
          : models.map(m => (
            <div key={m.name} className="glass-card p-5">
              <div className="flex items-center justify-between mb-2">
                <p className="text-gray-400 text-sm">{m.name}</p>
                <span className={`text-xs px-2 py-0.5 rounded-full ${m.is_active ? 'bg-emerald-500/20 text-emerald-400' : 'bg-gray-700 text-gray-500'}`}>
                  {m.is_active ? 'Active' : 'Inactive'}
                </span>
              </div>
              <p className="text-white text-2xl font-bold">
                {m.accuracy != null ? `${(m.accuracy * 100).toFixed(1)}%` : '—'}
              </p>
              <p className="text-gray-500 text-xs mt-1">F1: {m.f1 != null ? m.f1.toFixed(3) : '—'}</p>
            </div>
          ))}
        </div>

        {/* Training Controls */}
        {isAdmin && (
          <div className="glass-card p-6 space-y-4">
            <p className="section-title"><Cpu size={18} className="text-cyan-400" /> Image Model Training</p>
            <p className="text-gray-400 text-sm">
              Triggers ResNet-18 fine-tuning on the kidney CT dataset in a background thread.
              Training will run for up to {status?.total_epochs || 10} epochs with early stopping.
            </p>

            {status?.is_training && (
              <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-4">
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-4 h-4 border-2 border-cyan-500/40 border-t-cyan-500 rounded-full animate-spin" />
                  <p className="text-cyan-300 font-medium text-sm">Training in progress...</p>
                </div>
                <p className="text-gray-400 text-xs">
                  Epoch {status.current_epoch} / {status.total_epochs}
                </p>
                <div className="w-full bg-gray-700 rounded-full h-1.5 mt-2">
                  <div className="bg-cyan-500 h-1.5 rounded-full transition-all duration-700"
                    style={{ width: `${(status.current_epoch / status.total_epochs) * 100}%` }} />
                </div>
              </div>
            )}

            <button
              onClick={handleStartTraining}
              disabled={trainLoading || status?.is_training}
              className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {trainLoading ? <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" /> : <Play size={16} />}
              {status?.is_training ? 'Training Running...' : 'Start ResNet-18 Training'}
            </button>
          </div>
        )}

        {/* Training Curve Chart */}
        {chartData.length > 0 && (
          <div className="glass-card p-6">
            <p className="section-title mb-4"><TrendingUp size={18} className="text-cyan-400" /> Training Curves</p>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 12 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#6b7280' }} />
                <YAxis stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 12 }} unit="%" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8, color: '#f9fafb' }}
                  formatter={(v) => [`${v}%`]}
                />
                <Legend wrapperStyle={{ color: '#9ca3af', fontSize: 12 }} />
                <Line type="monotone" dataKey="trainAcc" name="Train Acc" stroke="#06b6d4" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="valAcc"   name="Val Acc"   stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Genetic Optimization */}
        <div className="glass-card p-6 space-y-4">
          <p className="section-title"><Dna size={18} className="text-cyan-400" /> Genetic Weight Optimization</p>
          <p className="text-gray-400 text-sm">
            Runs a DEAP genetic algorithm (15 individuals, 5 generations) to find the optimal feature subset
            and tabular/image fusion weights that maximize F1 score.
          </p>
          <button
            onClick={handleRunGenetic}
            disabled={genLoading}
            className="btn-primary flex items-center gap-2 disabled:opacity-50"
          >
            {genLoading ? <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" /> : <RefreshCw size={16} />}
            {genLoading ? 'Optimizing...' : 'Run Genetic Optimization'}
          </button>

          {genResults && (
            <div className="bg-gray-800/60 rounded-xl p-4 space-y-3 animate-slide-up">
              <div className="grid grid-cols-3 gap-3 text-center text-xs">
                <div>
                  <p className="text-gray-400">Best F1</p>
                  <p className="text-white font-bold text-lg">{genResults.best_f1?.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-gray-400">Tabular Weight</p>
                  <p className="text-cyan-300 font-bold text-lg">{genResults.tabular_weight?.toFixed(3)}</p>
                </div>
                <div>
                  <p className="text-gray-400">Image Weight</p>
                  <p className="text-blue-300 font-bold text-lg">{genResults.image_weight?.toFixed(3)}</p>
                </div>
              </div>
              <div>
                <p className="text-gray-400 text-xs mb-1">Selected Features ({genResults.selected_features?.length})</p>
                <div className="flex flex-wrap gap-1">
                  {genResults.selected_features?.map(f => (
                    <span key={f} className="px-2 py-0.5 bg-cyan-500/20 text-cyan-300 rounded text-xs font-mono">{f}</span>
                  ))}
                </div>
              </div>
              {genResults.dropped_features?.length > 0 && (
                <div>
                  <p className="text-gray-400 text-xs mb-1">Dropped Features</p>
                  <div className="flex flex-wrap gap-1">
                    {genResults.dropped_features.map(f => (
                      <span key={f} className="px-2 py-0.5 bg-gray-700 text-gray-500 rounded text-xs font-mono">{f}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Confusion Matrix */}
        {metrics?.confusion_matrix && (
          <div className="glass-card p-6">
            <p className="section-title mb-4">📊 Image Model Confusion Matrix</p>
            <div className="overflow-x-auto">
              <table className="mx-auto text-center">
                <thead>
                  <tr>
                    <th className="px-3 py-2 text-gray-500 text-xs">Pred →</th>
                    {['Normal','Cyst','Tumor','Stone'].map(c => (
                      <th key={c} className="px-3 py-2 text-cyan-400 text-xs">{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics.confusion_matrix.map((row, i) => (
                    <tr key={i}>
                      <td className="px-3 py-2 text-emerald-400 text-xs font-semibold">
                        {['Normal','Cyst','Tumor','Stone'][i]}
                      </td>
                      {row.map((val, j) => (
                        <td key={j} className={`px-3 py-2 text-sm font-bold rounded
                          ${i === j ? 'bg-cyan-500/20 text-cyan-300' : 'text-gray-400'}`}>
                          {val}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  )
}
