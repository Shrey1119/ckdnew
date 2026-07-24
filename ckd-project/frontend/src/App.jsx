import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './context/AuthContext'

// Pages
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import TabularPredict from './pages/TabularPredict'
import ImagePredict from './pages/ImagePredict'
import FusionPredict from './pages/FusionPredict'
import History from './pages/History'
import Training from './pages/Training'

/**
 * ProtectedRoute — redirects to /login if the user is not authenticated.
 * Shows a loading screen while auth state is being resolved.
 */
function ProtectedRoute({ children }) {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
          <p className="text-gray-400 text-sm">Loading...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/login" replace />
  }

  return children
}

export default function App() {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/login" element={<Login />} />

      {/* Protected routes */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/predict/tabular"
        element={
          <ProtectedRoute>
            <TabularPredict />
          </ProtectedRoute>
        }
      />
      <Route
        path="/predict/image"
        element={
          <ProtectedRoute>
            <ImagePredict />
          </ProtectedRoute>
        }
      />
      <Route
        path="/predict/fusion"
        element={
          <ProtectedRoute>
            <FusionPredict />
          </ProtectedRoute>
        }
      />
      <Route
        path="/history"
        element={
          <ProtectedRoute>
            <History />
          </ProtectedRoute>
        }
      />
      <Route
        path="/training"
        element={
          <ProtectedRoute>
            <Training />
          </ProtectedRoute>
        }
      />

      {/* Catch-all fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
