import { useState } from 'react'
import DashboardLayout from '../layouts/DashboardLayout'
import { predictAPI } from '../services/api'
import { Activity, AlertCircle, CheckCircle, ChevronRight } from 'lucide-react'
import toast from 'react-hot-toast'

const FIELDS = [
  { key: 'age',   label: 'Age',              unit: 'years',       type: 'number', placeholder: '45' },
  { key: 'bp',    label: 'Blood Pressure',   unit: 'mm/Hg',       type: 'number', placeholder: '80' },
  { key: 'sg',    label: 'Specific Gravity', unit: '',            type: 'number', placeholder: '1.020', step: '0.001' },
  { key: 'al',    label: 'Albumin',          unit: '0–5',         type: 'number', placeholder: '0' },
  { key: 'su',    label: 'Sugar',            unit: '0–5',         type: 'number', placeholder: '0' },
  { key: 'bgr',   label: 'Blood Glucose',    unit: 'mgs/dl',      type: 'number', placeholder: '121' },
  { key: 'bu',    label: 'Blood Urea',       unit: 'mgs/dl',      type: 'number', placeholder: '36' },
  { key: 'sc',    label: 'Serum Creatinine', unit: 'mgs/dl',      type: 'number', placeholder: '1.2' },
  { key: 'sod',   label: 'Sodium',           unit: 'mEq/L',       type: 'number', placeholder: '137' },
  { key: 'pot',   label: 'Potassium',        unit: 'mEq/L',       type: 'number', placeholder: '4.5' },
  { key: 'hemo',  label: 'Hemoglobin',       unit: 'gms',         type: 'number', placeholder: '15.4' },
  { key: 'pcv',   label: 'Packed Cell Vol.', unit: '',            type: 'number', placeholder: '44' },
  { key: 'wc',    label: 'WBC Count',        unit: 'cells/cumm',  type: 'number', placeholder: '7800' },
  { key: 'rc',    label: 'RBC Count',        unit: 'millions/cmm',type: 'number', placeholder: '5.2' },
]

const SELECT_FIELDS = [
  { key: 'rbc',   label: 'Red Blood Cells',   options: ['normal', 'abnormal'] },
  { key: 'pc',    label: 'Pus Cell',           options: ['normal', 'abnormal'] },
  { key: 'pcc',   label: 'Pus Cell Clumps',   options: ['notpresent', 'present'] },
  { key: 'ba',    label: 'Bacteria',           options: ['notpresent', 'present'] },
  { key: 'htn',   label: 'Hypertension',       options: ['no', 'yes'] },
  { key: 'dm',    label: 'Diabetes Mellitus',  options: ['no', 'yes'] },
  { key: 'cad',   label: 'Coronary Artery Dis.',options: ['no', 'yes'] },
  { key: 'appet', label: 'Appetite',           options: ['good', 'poor'] },
  { key: 'pe',    label: 'Pedal Edema',        options: ['no', 'yes'] },
  { key: 'ane',   label: 'Anemia',             options: ['no', 'yes'] },
]

export default function TabularPredict() {
  const [form, setForm] = useState({})
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleChange = (key, value) => setForm(prev => ({ ...prev, [key]: value }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await predictAPI.tabular(form)
      setResult(res.data)
      toast.success('Prediction complete!')
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <DashboardLayout title="Clinical Prediction">
      <div className="max-w-5xl mx-auto space-y-6">

        {/* Header */}
        <div className="glass-card p-5 flex items-center gap-3">
          <Activity size={22} className="text-cyan-400 shrink-0" />
          <div>
            <p className="text-white font-semibold">KNN Tabular Prediction</p>
            <p className="text-gray-400 text-sm">Enter clinical lab values to predict CKD risk using the trained K-Nearest Neighbours model.</p>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* Numeric fields */}
            <div className="glass-card p-6 space-y-4">
              <p className="section-title">🔬 Lab Values</p>
              {FIELDS.map(f => (
                <div key={f.key}>
                  <label className="form-label">{f.label} {f.unit && <span className="text-gray-500 text-xs">({f.unit})</span>}</label>
                  <input
                    type={f.type}
                    step={f.step || 'any'}
                    placeholder={f.placeholder}
                    className="form-input"
                    value={form[f.key] ?? ''}
                    onChange={e => handleChange(f.key, e.target.value === '' ? undefined : parseFloat(e.target.value))}
                  />
                </div>
              ))}
            </div>

            {/* Select fields + Results */}
            <div className="space-y-6">
              <div className="glass-card p-6 space-y-4">
                <p className="section-title">🏥 Clinical History</p>
                {SELECT_FIELDS.map(f => (
                  <div key={f.key}>
                    <label className="form-label">{f.label}</label>
                    <select
                      className="form-input"
                      value={form[f.key] ?? ''}
                      onChange={e => handleChange(f.key, e.target.value || undefined)}
                    >
                      <option value="">— Select —</option>
                      {f.options.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </div>
                ))}
              </div>

              {/* Submit */}
              <button type="submit" disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2">
                {loading
                  ? <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                  : <ChevronRight size={18} />
                }
                {loading ? 'Analysing...' : 'Run Prediction'}
              </button>

              {/* Error */}
              {error && (
                <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 text-red-400 rounded-xl px-4 py-3 text-sm">
                  <AlertCircle size={16} /> {error}
                </div>
              )}

              {/* Result Card */}
              {result && (
                <div className={`glass-card p-6 border-2 ${result.prediction === 1 ? 'border-red-500/40' : 'border-emerald-500/40'} animate-slide-up`}>
                  <div className="flex items-center gap-3 mb-4">
                    {result.prediction === 1
                      ? <AlertCircle size={24} className="text-red-400" />
                      : <CheckCircle size={24} className="text-emerald-400" />
                    }
                    <div>
                      <p className="text-white font-bold text-lg">{result.label === 'ckd' ? 'CKD Detected' : 'No CKD Detected'}</p>
                      <p className="text-gray-400 text-sm">Confidence: <span className="capitalize text-white">{result.confidence}</span> · Risk: <span className={`capitalize font-semibold risk-${result.risk_level}`}>{result.risk_level}</span></p>
                    </div>
                  </div>

                  {/* Probability Bar */}
                  <div className="mb-4">
                    <div className="flex justify-between text-xs text-gray-400 mb-1">
                      <span>CKD Probability</span>
                      <span className="font-mono text-white">{(result.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-700 ${result.prediction === 1 ? 'bg-gradient-to-r from-red-500 to-red-400' : 'bg-gradient-to-r from-emerald-500 to-green-400'}`}
                        style={{ width: `${result.probability * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* SHAP explainability top features */}
                  {result.explainability?.prediction_impact && (
                    <div>
                      <p className="text-gray-400 text-xs font-medium mb-2">Top Contributing Features (SHAP)</p>
                      <div className="space-y-1">
                        {result.explainability.prediction_impact.slice(0, 5).map(item => (
                          <div key={item.feature} className="flex items-center justify-between text-xs">
                            <span className="text-gray-300 font-mono">{item.feature}</span>
                            <span className={`font-mono ${item.shap > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                              {item.shap > 0 ? '+' : ''}{item.shap.toFixed(3)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </form>
      </div>
    </DashboardLayout>
  )
}
