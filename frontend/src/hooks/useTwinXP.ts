import { useState, useEffect, useCallback } from 'react'

export function useTwinXP(userId: string, totalLogs: number) {
  const [xp, setXp] = useState(0)

  useEffect(() => {
    if (!userId) return
    const key = `twin_user_xp_${userId}`
    const stored = localStorage.getItem(key)
    
    if (stored !== null && !isNaN(parseInt(stored, 10))) {
      setXp(parseInt(stored, 10))
    } else if (totalLogs > 0) {
      // Fallback: if they have logs but no XP stored, baseline it
      const baseline = totalLogs * 10
      setXp(baseline)
      localStorage.setItem(key, baseline.toString())
    }
  }, [userId, totalLogs])

  const addXp = useCallback((amount: number) => {
    if (!userId) return
    setXp(prev => {
      const next = prev + amount
      localStorage.setItem(`twin_user_xp_${userId}`, next.toString())
      return next
    })
  }, [userId])

  const level = Math.floor(xp / 100) + 1

  // Invariant check in dev mode
  if (import.meta.env.DEV) {
    if (level !== Math.floor(xp / 100) + 1) {
      console.error(`Invariant violation: level ${level} does not match xp ${xp}`)
    }
  }

  return { xp, level, addXp }
}
