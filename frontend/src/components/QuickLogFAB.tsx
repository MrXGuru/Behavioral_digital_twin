import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Zap, X, Timer, Play, Square, CheckCircle2 } from 'lucide-react'
import { logDecision } from '../hooks/useApi'

const FOCUS_DURATION = 25 * 60

export default function QuickLogFAB({ userId, onLog, addXp }: { userId: string, onLog: () => void, addXp: (amount: number) => void }) {
  const [isOpen, setIsOpen] = useState(false)
  const [loadingAction, setLoadingAction] = useState<string | null>(null)
  const [xpGained, setXpGained] = useState<number | null>(null)
  
  // Timer state
  const [timeLeft, setTimeLeft] = useState(FOCUS_DURATION)
  const [timerActive, setTimerActive] = useState(false)

  // External control listener
  useEffect(() => {
    const handleStartTimer = () => {
      setIsOpen(true)
      setTimerActive(true)
      setTimeLeft(FOCUS_DURATION)
    }
    window.addEventListener('start-focus-timer', handleStartTimer)
    return () => window.removeEventListener('start-focus-timer', handleStartTimer)
  }, [])

  // Timer logic
  useEffect(() => {
    if (!timerActive) return
    if (timeLeft <= 0) {
      setTimerActive(false)
      handleLog('focus', 'deep_work', true)
      setTimeLeft(FOCUS_DURATION)
      return
    }
    const interval = setInterval(() => {
      setTimeLeft(prev => prev - 1)
    }, 1000)
    return () => clearInterval(interval)
  }, [timerActive, timeLeft])

  const handleLog = async (domain: string, decision: string, isTimer: boolean = false) => {
    setLoadingAction(decision)
    try {
      await logDecision(userId, {
        domain,
        decision_made: decision,
      })
      onLog()
      const amount = isTimer ? 50 : 20
      setXpGained(amount)
      addXp(amount)
      setTimeout(() => {
        setXpGained(null)
        setIsOpen(false)
        setLoadingAction(null)
      }, 1500)
    } catch (e) {
      console.error(e)
      setLoadingAction(null)
    }
  }

  const toggleTimer = () => {
    setTimerActive(!timerActive)
    if (timerActive) {
      setTimeLeft(FOCUS_DURATION) // reset if stopped
    }
  }

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0')
    const s = (secs % 60).toString().padStart(2, '0')
    return `${m}:${s}`
  }

  const timerProgress = timerActive ? ((FOCUS_DURATION - timeLeft) / FOCUS_DURATION) * 100 : 0
  const dashArray = 2 * Math.PI * 22

  return (
    <>
      <div className="fixed bottom-4 left-4 z-50 flex items-center justify-center">
        <motion.button
          layoutId={!isOpen ? "quick-log-container" : undefined}
          initial={{ scale: 0 }}
          animate={{ scale: isOpen ? 0 : 1 }}
          onClick={() => setIsOpen(true)}
          className="w-14 h-14 bg-[#1a1a1a] border border-[#ffffff14] hover:bg-[#2a2a2a] hover:border-purple-500/50 rounded-full flex items-center justify-center shadow-2xl transition-colors group preserve-3d relative"
        >
          {/* Timer Progress Ring */}
          <svg className="absolute inset-0 w-full h-full -rotate-90 pointer-events-none" viewBox="0 0 48 48">
            {timerActive && (
              <circle
                cx="24"
                cy="24"
                r="22"
                fill="none"
                stroke="rgba(168, 85, 247, 0.4)"
                strokeWidth="2"
                strokeLinecap="round"
                strokeDasharray={dashArray}
                strokeDashoffset={dashArray - (dashArray * timerProgress) / 100}
                className="transition-all duration-1000 ease-linear"
              />
            )}
          </svg>
          {timerActive ? (
            <div className="flex flex-col items-center justify-center translate-z-10 text-purple-400">
              <span className="text-[10px] font-mono font-bold leading-none">{formatTime(timeLeft)}</span>
            </div>
          ) : (
            <Zap className="w-5 h-5 text-purple-400 group-hover:scale-110 transition-transform translate-z-10" />
          )}
        </motion.button>
      </div>

      <AnimatePresence>
        {isOpen && (
          <div className="fixed inset-0 z-50 flex items-end justify-start p-4 perspective-1000 pointer-events-none">
            <motion.div
              layoutId="quick-log-container"
              initial={{ opacity: 0, scale: 0.85, rotateX: -15, y: 40 }}
              animate={{ opacity: 1, scale: 1, rotateX: 0, y: 0 }}
              exit={{ opacity: 0, scale: 0.85, rotateX: 10, y: 20, transition: { duration: 0.25, ease: "easeIn" } }}
              transition={{ type: "spring", stiffness: 250, damping: 20 }}
              className="glass-card bg-[#0a0a0a]/90 border border-[#ffffff14] rounded-2xl overflow-hidden shadow-2xl backdrop-blur-xl pointer-events-auto preserve-3d w-[280px]"
            >
              {/* Header */}
              <div className="flex items-center justify-between p-3 border-b border-[#ffffff0a] bg-[#141414]/50">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-purple-400" />
                  <span className="text-xs font-bold text-white tracking-wider uppercase">Quick Log</span>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1 hover:bg-[#ffffff14] rounded-md transition-colors text-[#a1a1aa] hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Content */}
              <div className="p-4 relative flex flex-col gap-3">
                {xpGained ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8, y: 10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    className="flex flex-col items-center justify-center py-6 preserve-3d"
                  >
                    <CheckCircle2 className="w-12 h-12 text-emerald-400 mb-2 translate-z-10" />
                    <span className="text-emerald-400 font-bold text-lg translate-z-20">+{xpGained} XP</span>
                    <span className="text-[#a1a1aa] text-xs mt-1 translate-z-10">Decision Logged</span>
                  </motion.div>
                ) : (
                  <>
                    {/* Action Buttons */}
                    <div className="grid grid-cols-1 gap-2">
                      <button
                        disabled={loadingAction !== null}
                        onClick={() => handleLog('focus', 'deep_work')}
                        className="flex items-center justify-between p-3 rounded-xl bg-[#ffffff05] border border-[#ffffff0a] hover:bg-purple-500/10 hover:border-purple-500/30 transition-all group preserve-3d disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <span className="text-sm font-semibold text-white group-hover:text-purple-300 translate-z-10">Deep Work</span>
                        <span className="text-xs text-[#a1a1aa] font-mono translate-z-10">Focus</span>
                      </button>
                      <button
                        disabled={loadingAction !== null}
                        onClick={() => handleLog('task', 'email')}
                        className="flex items-center justify-between p-3 rounded-xl bg-[#ffffff05] border border-[#ffffff0a] hover:bg-blue-500/10 hover:border-blue-500/30 transition-all group preserve-3d disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <span className="text-sm font-semibold text-white group-hover:text-blue-300 translate-z-10">Email / Comms</span>
                        <span className="text-xs text-[#a1a1aa] font-mono translate-z-10">Task</span>
                      </button>
                      <button
                        disabled={loadingAction !== null}
                        onClick={() => handleLog('purchase', 'coffee')}
                        className="flex items-center justify-between p-3 rounded-xl bg-[#ffffff05] border border-[#ffffff0a] hover:bg-amber-500/10 hover:border-amber-500/30 transition-all group preserve-3d disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <span className="text-sm font-semibold text-white group-hover:text-amber-300 translate-z-10">Coffee</span>
                        <span className="text-xs text-[#a1a1aa] font-mono translate-z-10">Purchase</span>
                      </button>
                    </div>

                    <div className="h-px w-full bg-gradient-to-r from-transparent via-[#ffffff14] to-transparent my-1" />

                    {/* Timer Controls */}
                    <div className="flex items-center justify-between p-3 rounded-xl bg-[#141414] border border-[#ffffff0a] preserve-3d">
                      <div className="flex items-center gap-2 translate-z-10">
                        <Timer className="w-4 h-4 text-purple-400" />
                        <div className="flex flex-col">
                          <span className="text-xs font-semibold text-white">Focus Timer</span>
                          <span className="text-[10px] text-[#a1a1aa] font-mono">{formatTime(timeLeft)}</span>
                        </div>
                      </div>
                      <button
                        onClick={toggleTimer}
                        className={`p-2 rounded-lg transition-colors translate-z-20 ${
                          timerActive 
                            ? 'bg-rose-500/20 text-rose-400 hover:bg-rose-500/30' 
                            : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30'
                        }`}
                      >
                        {timerActive ? <Square className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                      </button>
                    </div>
                  </>
                )}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  )
}
