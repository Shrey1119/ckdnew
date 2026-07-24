import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import DashboardLayout from '../layouts/DashboardLayout'
import { predictAPI } from '../services/api'
import { Layers, AlertCircle, CheckCircle, Upload, X, ChevronRight } from 'lucide-react'
import toast from 'react-hot-toast'

const NUMERIC_FIELDS = [
  { key: 'age', label: 'Age', placeholder: '45' },
  { key: 'bp',  label: 'Blood Pressure', placeholder: '80' },
  { key: 'sc',  label: 'Serum Creatinine', placeholder: '1.2' },
  { key: 'hemo',label: 'Hemoglobin', placeholder: '15.4' },
  { key: 'bgr', label: 'Blood Glucose', placeholder: '121' },
  { key: 'bu',  label: 'Blood Urea', placeholder: '36' },
]

export default function FusionPredict() {
  const [step, setStep] = useState(1) // 1: Clinical, 2: Image, 3: Review & Submit
  const [form, setForm] = useState({})
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [fusionType, setFusionType] = useState('late')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const onDrop = useCallback((accepted) => {
    if (accepted.length > 0) {
      const f = accepted[0]
      setFile(f)
      setPreview(URL.createObjectURL(f))
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [] },
    maxFiles: 1,
  })

  const handleChange = (key, value) => setForm(prev => ({ ...prev, [key]: value }))

  const handleSubmit = async () => {
    if (!file) { setError('Please upload a CT scan image.'); return }
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await predictAPI.fusion(form, file, null, { fusionType })
      setResult(res.data)
      toast.success('Multimodal fusion complete!')
    } catch (err) {
      setError(err.response?.data?.detail || 'Fusion prediction failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const steps = ['Clinical Data', 'CT Scan Upload', 'Review & Submit']

  return (
    <DashboardLayout title="Multimodal Fusion">
      <div className="max-w-3xl mx-auto space-y-6">

        {/* Header */}
        <div className="glass-card p-5 flex items-center gap-3">
          <Layers size={22} className="text-cyan-400 shrink-0" />
          <div>
            <p className="text-white font-semibold">Multimodal Fusion Prediction</p>
            <p className="text-gray-400 text-sm">Combine clinical lab values and CT scan for a joint AI prediction.</p>
          </div>
        </div>

        {/* Step Wizard */}
        <div className="glass-card p-4">
          <div className="flex items-center gap-2">
            {steps.map((label, i) => (
              <div key={i} className="flex items-center gap-2 flex-1">
                <div className={`flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold shrink-0 transition-all
                  ${i + 1 < step ? 'bg-emerald-500 text-white' : i + 1 === step ? 'bg-cyan-500 text-white' : 'bg-gray-700 text-gray-400'}`}>
                  {i + 1 < step ? '✓' : i + 1}
                </div>
                <span className={`text-sm ${i + 1 === step ? 'text-white font-medium' : 'text-gray-500'}`}>{label}</span>
                {i < steps.length - 1 && <div className="flex-1 h-px bg-gray-700 mx-1" />}
              </div>
            ))}
          </div>
        </div>

        {/* Step 1: Clinical Data */}
        {step === 1 && (
          <div className="glass-card p-6 space-y-4 animate-fade-in">
            <p className="section-title">🔬 Enter Clinical Values (key fields)</p>
            <div className="grid grid-cols-2 gap-4">
              {NUMERIC_FIELDS.map(f => (
                <div key={f.key}>
                  <label className="form-label">{f.label}</label>
                  <input
                    type="number" step="any" placeholder={f.placeholder}
                    className="form-input"
                    value={form[f.key] ?? ''}
                    onChange={e => handleChange(f.key, e.target.value === '' ? undefined : parseFloat(e.target.value))}
                  />
                </div>
              ))}
            </div>
            <button onClick={() => setStep(2)} className="btn-primary w-full flex items-center justify-center gap-2 mt-2">
              Next: Upload CT Scan <ChevronRight size={16} />
            </button>
          </div>
        )}

        {/* Step 2: Image Upload */}
        {step === 2 && (
          <div className="glass-card p-6 space-y-4 animate-fade-in">
            <p className="section-title">🖼️ Upload CT Scan Image</p>
            {!preview ? (
              <div {...getRootProps()} className={`p-10 border-2 border-dashed rounded-xl flex flex-col items-center gap-3 cursor-pointer transition-all
                ${isDragActive ? 'border-cyan-400 bg-cyan-500/5' : 'border-gray-700 hover:border-gray-500'}`}>
                <input {...getInputProps()} />
                <Upload size={36} className="text-gray-500" />
                <p className="text-gray-400 text-sm">{isDragActive ? 'Drop here!' : 'Drag & drop or click — JPEG/PNG'}</p>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <p className="text-white text-sm">{file.name}</p>
                  <button onClick={() => { setFile(null); setPreview(null) }} className="text-gray-500 hover:text-red-400">
                    <X size={16} />
                  </button>
                </div>
                <img src={preview} alt="preview" className="w-full rounded-xl max-h-56 object-cover" />
              </div>
            )}
            <div className="flex gap-3 mt-2">
              <button onClick={() => setStep(1)} className="btn-secondary flex-1">← Back</button>
              <button onClick={() => setStep(3)} disabled={!file} className="btn-primary flex-1 disabled:opacity-50">Next: Review →</button>
            </div>
          </div>
        )}

        {/* Step 3: Review & Submit */}
        {step === 3 && (
          <div className="glass-card p-6 space-y-5 animate-fade-in">
            <p className="section-title">🧠 Review & Run Fusion</p>
            <div className="flex gap-4">
              <div className="flex-1 bg-gray-800/60 rounded-xl p-3">
                <p className="text-gray-400 text-xs mb-2">Clinical Fields Entered</p>
                {Object.keys(form).length === 0
                  ? <p className="text-gray-500 text-xs">No fields filled</p>
                  : Object.entries(form).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-xs">
                      <span className="text-gray-400 font-mono">{k}</span>
                      <span className="text-white font-mono">{v}</span>
                    </div>
                  ))}
              </div>
              <div className="flex-1 bg-gray-800/60 rounded-xl p-3">
                <p className="text-gray-400 text-xs mb-2">CT Scan</p>
                {preview && <img src={preview} alt="CT scan" className="w-full rounded-lg max-h-36 object-cover" />}
              </div>
            </div>

            {/* Fusion type toggle */}
            <div>
              <p className="form-label">Fusion Strategy</p>
              <div className="flex gap-3">
                {['late', 'early'].map(t => (
                  <button key={t} type="button"
                    onClick={() => setFusionType(t)}
                    className={`px-4 py-2 rounded-xl text-sm font-medium border transition-all capitalize
                      ${fusionType === t ? 'bg-cyan-500/20 border-cyan-500/60 text-cyan-300' : 'border-gray-700 text-gray-400 hover:border-gray-500'}`}>
                    {t} Fusion
                  </button>
                ))}
              </div>
              <p className="text-gray-500 text-xs mt-1">
                {fusionType === 'late' ? 'Weighted average of tabular KNN and image ResNet probabilities.' : 'MLP trained on concatenated tabular + image embeddings (requires trained model).'}
              </p>
            </div>

            <div className="flex gap-3">
              <button onClick={() => setStep(2)} className="btn-secondary flex-1">← Back</button>
              <button onClick={handleSubmit} disabled={loading} className="btn-primary flex-1 flex items-center justify-center gap-2">
                {loading ? <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" /> : <Layers size={16} />}
                {loading ? 'Running fusion...' : 'Run Fusion'}
              </button>
            </div>

            {error && (
              <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 text-red-400 rounded-xl px-4 py-3 text-sm">
                <AlertCircle size={16} /> {error}
              </div>
            )}

            {result && (
              <div className={`p-5 rounded-xl border-2 animate-slide-up ${result.prediction === 1 ? 'border-red-500/40 bg-red-500/5' : 'border-emerald-500/40 bg-emerald-500/5'}`}>
                <div className="flex items-center gap-3 mb-4">
                  {result.prediction === 1 ? <AlertCircle className="text-red-400" size={22} /> : <CheckCircle className="text-emerald-400" size={22} />}
                  <div>
                    <p className="text-white font-bold">{result.label === 'ckd' ? 'CKD Detected' : 'No CKD Detected'}</p>
                    <p className="text-gray-400 text-xs capitalize">Confidence: {result.confidence} · Risk: {result.risk_level} · Strategy: {result.fusion_type}</p>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-3 text-center text-xs">
                  <div className="bg-gray-800/60 rounded-xl p-3">
                    <p className="text-gray-400">Tabular Prob</p>
                    <p className="text-white font-bold text-lg">{(result.tabular_prob * 100).toFixed(1)}%</p>
                  </div>
                  <div className="bg-gray-800/60 rounded-xl p-3">
                    <p className="text-gray-400">Image Prob</p>
                    <p className="text-white font-bold text-lg">{(result.image_prob * 100).toFixed(1)}%</p>
                  </div>
                  <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-3">
                    <p className="text-cyan-400">Fused Prob</p>
                    <p className="text-white font-bold text-lg">{(result.probability * 100).toFixed(1)}%</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </DashboardLayout>
  )
}
