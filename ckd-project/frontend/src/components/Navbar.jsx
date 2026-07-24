import { Wifi, WifiOff } from 'lucide-react'
import { useEffect, useState } from 'react'
import { healthAPI } from '../services/api'

export default function Navbar({ title }) {
  const [online, setOnline] = useState(null)

  useEffect(() => {
    healthAPI.check()
      .then(() => setOnline(true))
      .catch(() => setOnline(false))
  }, [])

  return (
    <header className="h-16 bg-gray-900/60 backdrop-blur-md border-b border-gray-800 flex items-center justify-between px-6">
      <h1 className="text-white font-bold text-xl">{title}</h1>
      <div className="flex items-center gap-4">
        {/* API Status indicator */}
        <div className="flex items-center gap-2 text-xs">
          {online === null ? (
            <span className="text-gray-500">Checking API...</span>
          ) : online ? (
            <>
              <Wifi size={14} className="text-emerald-400" />
              <span className="text-emerald-400 font-medium">API Online</span>
            </>
          ) : (
            <>
              <WifiOff size={14} className="text-red-400" />
              <span className="text-red-400 font-medium">API Offline</span>
            </>
          )}
        </div>

        <div className="w-px h-5 bg-gray-700" />

        {/* Current time */}
        <Clock />
      </div>
    </header>
  )
}

function Clock() {
  const [time, setTime] = useState(new Date())
  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  return (
    <span className="text-gray-400 text-sm font-mono">
      {time.toLocaleTimeString()}
    </span>
  )
}
