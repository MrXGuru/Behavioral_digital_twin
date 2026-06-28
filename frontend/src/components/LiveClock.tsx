import { useState, useEffect } from 'react'

export default function LiveClock({ className = '' }: { className?: string }) {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  const timeStr = time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  const dateStr = time.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <span className="font-mono font-medium tracking-wider">{timeStr}</span>
      <span className="opacity-50">•</span>
      <span className="font-medium text-xs opacity-70 uppercase tracking-widest">{dateStr}</span>
    </div>
  )
}
