import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import DashboardLayout from '../layouts/DashboardLayout'
import { predictAPI } from '../services/api'
import { Image, Upload, AlertCircle, CheckCircle, X } from 'lucide-react'
import toast from 'react-hot-toast'

const CLASS_COLORS = {
  Normal: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
  Cyst:   'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
  Stone:  'text-orange-400 bg-orange-500/10 border-orange-500/30',
  Tumor:  'text-red-400 bg-red-500/10 border-red-500/30',
}

const CLASS_BAR_COLOR = {
  Normal: 'bg-emerald-500',
  Cyst:   'bg-yellow-500',
  Stone:  'bg-orange-500',
  Tumor:  'bg-red-500',
}

export default function ImagePredict() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const onDrop = useCallback((accepted) => {
    if (accepted.length > 0) {
      const f = accepted[0]
      setFile(f)
      setPreview(URL.createObjectURL(f))
      setResult(null)
      setError('')
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [] },
    maxFiles: 1,
  })

  const handleClear = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError('')
  }

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await predictAPI.image(file, null)
      setResult(res.data)
      toast.success('Image analysis complete!')
    } catch (err) {
      setError(err.response?.data?.detail || 'Image prediction failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const topClass = result ? Object.entries(result.confidence_probs).sort((a, b) => b[1] - a[1])[0][0] : null

  return (
    <DashboardLayout title="Image Prediction">
      <div className="max-w-4xl mx-auto space-y-6">

        {/* Header */}
        <div className="glass-card p-5 flex items-center gap-3">
          <Image size={22} className="text-cyan-400 shrink-0" />
          <div>
            <p className="text-white font-semibold">CT Scan Classification (ResNet-18)</p>
            <p className="text-gray-400 text-sm">Upload a kidney CT scan image to classify it as Normal, Cyst, Tumor, or Stone.</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload zone */}
          <div className="space-y-4">
            {!preview ? (
              <div
                {...getRootProps()}
                className={`glass-card p-10 border-2 border-dashed flex flex-col items-center justify-center gap-4 cursor-pointer transition-all duration-200
                  ${isDragActive ? 'border-cyan-400 bg-cyan-500/5' : 'border-gray-700 hover:border-gray-500'}`}
              >
                <input {...getInputProps()} />
                <Upload size={40} className="text-gray-500" />
                <div className="text-center">
                  <p className="text-white font-medium">{isDragActive ? 'Drop it here!' : 'Drag & drop a CT scan'}</p>
                  <p className="text-gray-500 text-sm mt-1">or click to browse — JPEG / PNG</p>
                </div>
              </div>
            ) : (
              <div className="glass-card p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-white text-sm font-medium">{file.name}</p>
                  <button onClick={handleClear} className="text-gray-500 hover:text-red-400 transition-colors">
                    <X size={16} />
                  </button>
                </div>
                <img src={preview} alt="CT scan preview" className="w-full rounded-xl object-cover max-h-64" />

                {/* Grad-CAM overlay if available */}
                {result?.gradcam_url && (
                  <div>
                    <p className="text-gray-400 text-xs mb-2">Grad-CAM Activation Map</p>
                    <img
                      src={`http://localhost:8000${result.gradcam_url}`}
                      alt="Grad-CAM"
                      className="w-full rounded-xl object-cover max-h-64"
                    />
                  </div>
                )}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={!file || loading}
              className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading
                ? <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                : <Image size={18} />
              }
              {loading ? 'Analysing CT scan...' : 'Classify Image'}
            </button>

            {error && (
              <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 text-red-400 rounded-xl px-4 py-3 text-sm">
                <AlertCircle size={16} /> {error}
              </div>
            )}
          </div>

          {/* Results */}
          <div className="space-y-4">
            {!result && !loading && (
              <div className="glass-card p-10 flex flex-col items-center justify-center gap-3 text-center h-full">
                <Image size={40} className="text-gray-600" />
                <p className="text-gray-500 text-sm">Upload and classify a CT scan<br />to see results here.</p>
              </div>
            )}

            {loading && (
              <div className="glass-card p-10 flex flex-col items-center justify-center gap-4 h-full">
                <div className="w-12 h-12 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
                <p className="text-gray-400 text-sm">Running ResNet-18 inference...</p>
              </div>
            )}

            {result && (
              <div className="glass-card p-6 space-y-5 animate-slide-up">
                {/* Predicted class badge */}
                <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl border text-sm font-semibold ${CLASS_COLORS[topClass] || ''}`}>
                  {topClass === 'Normal' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
                  {topClass} Detected
                </div>

                <div>
                  <p className="text-gray-400 text-xs mb-1">Confidence</p>
                  <p className="text-white font-semibold capitalize">{result.confidence}</p>
                </div>

                {/* Class probability bars */}
                <div>
                  <p className="text-gray-400 text-xs font-medium mb-3">Class Probabilities</p>
                  <div className="space-y-3">
                    {Object.entries(result.confidence_probs)
                      .sort((a, b) => b[1] - a[1])
                      .map(([cls, prob]) => (
                        <div key={cls}>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-gray-300">{cls}</span>
                            <span className="font-mono text-white">{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-700 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full transition-all duration-700 ${CLASS_BAR_COLOR[cls] || 'bg-cyan-500'}`}
                              style={{ width: `${prob * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  )
}
