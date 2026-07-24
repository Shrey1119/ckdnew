import { createContext, useContext, useState, useEffect } from 'react'
import { authAPI } from '../services/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser]       = useState(null)
  const [token, setToken]     = useState(localStorage.getItem('ckd_token'))
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const stored = localStorage.getItem('ckd_user')
    if (stored && token) {
      try { setUser(JSON.parse(stored)) } catch { /* ignore */ }
    }
    setLoading(false)
  }, [token])

  const login = async (username, password) => {
    const res = await authAPI.login(username, password)
    const { access_token, role, username: uname } = res.data
    localStorage.setItem('ckd_token', access_token)
    const userObj = { username: uname, role }
    localStorage.setItem('ckd_user', JSON.stringify(userObj))
    setToken(access_token)
    setUser(userObj)
    return userObj
  }

  const logout = () => {
    localStorage.removeItem('ckd_token')
    localStorage.removeItem('ckd_user')
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, token, loading, login, logout, isAdmin: user?.role === 'admin' }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
