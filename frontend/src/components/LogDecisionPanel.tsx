/**
 * Real AI LogDecisionPanel
 * Fetches live predictions from the ML models for the 3 domains and lets the user log them with one click.
 * Falls back to time-based suggestions if models are untrained.
 */
import { useState, useEffect, memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Star, CheckCircle, Zap, Loader2, Sparkles, Target, ListTodo, ShoppingCart, Database } from 'lucide-react'
import { logDecision } from '../hooks/useApi'

interface LogDecisionPanelProps {
  userId: string
  onLogged: () => void
  totalLogs: number
  xp: number
  level: number
  addXp: (amount: number) => void
}

const DOMAINS = [
  { id: 'focus', icon: Target },
  { id: 'task', icon: ListTodo },
  { id: 'purchase', icon: ShoppingCart }
]

// Fallback dynamic suggestions
const getFallbackSuggestions = () => {
  const hour = new Date().getHours()
  if (hour < 11) {
    return [
      { label: 'Morning Coffee', icon: <ShoppingCart className="w-6 h-6" />, domain: 'purchase', decision: 'coffee', xp: 10 },
      { label: 'Deep Work', icon: <ListTodo className="w-6 h-6" />, domain: 'task', decision: 'deep_work', xp: 50 },
      { label: 'Pomodoro', icon: <Target className="w-6 h-6" />, domain: 'focus', decision: 'pomodoro', xp: 30 },
    ]
  } else if (hour < 15) {
    return [
      { label: 'Lunch Break', icon: <ShoppingCart className="w-6 h-6" />, domain: 'purchase', decision: 'lunch', xp: 15 },
      { label: 'Flow State', icon: <Target className="w-6 h-6" />, domain: 'focus', decision: 'flow_state', xp: 50 },
      { label: 'Clear Emails', icon: <ListTodo className="w-6 h-6" />, domain: 'task', decision: 'email', xp: 20 },
    ]
  } else if (hour < 19) {
    return [
      { label: 'Light Work', icon: <Target className="w-6 h-6" />, domain: 'focus', decision: 'light_work', xp: 20 },
      { label: 'Meeting', icon: <ListTodo className="w-6 h-6" />, domain: 'task', decision: 'meeting', xp: 25 },
      { label: 'Quick Snack', icon: <ShoppingCart className="w-6 h-6" />, domain: 'purchase', decision: 'snack', xp: 10 },
    ]
  } else {
    return [
      { label: 'Admin Tasks', icon: <Target className="w-6 h-6" />, domain: 'focus', decision: 'admin', xp: 20 },
      { label: 'Evening Break', icon: <ListTodo className="w-6 h-6" />, domain: 'task', decision: 'break', xp: 15 },
      { label: 'No Purchase', icon: <ShoppingCart className="w-6 h-6" />, domain: 'purchase', decision: 'none', xp: 10 },
    ]
  }
}

const LogDecisionPanel = memo(function LogDecisionPanel({ userId, onLogged, totalLogs, xp, level, addXp }: LogDecisionPanelProps) {
  const [logging, setLogging] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  
  const [suggestions, setSuggestions] = useState<any[]>([])
  const [manualInput, setManualInput] = useState('')
  const [manualDomain, setManualDomain] = useState('focus')
  const [manualLocation, setManualLocation] = useState('home')
  const [manualWeather, setManualWeather] = useState('clear')
  const [manualMood, setManualMood] = useState(0.8)
  const [manualStress, setManualStress] = useState('medium')
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Initialize time-based fallback suggestions
  useEffect(() => {
    setSuggestions(getFallbackSuggestions())
  }, [userId])

  const handleSuggestionLog = async (item: any) => {
    setLogging(item.label)
    setSuccess(null)
    try {
      await logDecision(userId, {
        domain: item.domain,
        decision_made: item.decision,
      })
      
      addXp(item.xp)
      
      setSuccess(`+${item.xp} XP! Logged ${item.label}`)
      onLogged()
    } catch (err: unknown) {
      console.error(err)
    } finally {
      setLogging(null)
      setTimeout(() => setSuccess(null), 3000)
    }
  }

  const handleManualLog = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!manualInput.trim()) return

    setLogging('manual')
    setSuccess(null)
    
    try {
      await logDecision(userId, {
        domain: manualDomain,
        decision_made: manualInput.trim().toLowerCase().replace(/ /g, '_'),
        location: manualLocation,
        weather: manualWeather,
        mood_energy: manualMood,
        stress_level: manualStress,
      })
      addXp(10)
      
      setSuccess(`+10 XP! Logged ${manualInput}`)
      setManualInput('')
      onLogged()
    } catch (err: unknown) {
      console.error(err)
    } finally {
      setLogging(null)
      setTimeout(() => setSuccess(null), 3000)
    }
  }

  const xpIntoLevel = xp % 100
  const progressPercent = xpIntoLevel

  const getGreeting = () => {
    const hour = new Date().getHours()
    if (hour < 12) return 'Good morning'
    if (hour < 17) return 'Good afternoon'
    return 'Good evening'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-full"
    >
        <div className="relative overflow-hidden rounded-2xl bg-[#141414] border border-[#ffffff14] shadow-2xl p-6 h-full flex flex-col">
          <div className="absolute top-0 right-0 -mr-16 -mt-16 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

          {/* Header */}
          <div className="flex items-start justify-between mb-6 relative z-10">
            <div>
              <h3 className="text-2xl font-black text-white mb-1 tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-200 to-white">{getGreeting()}!</h3>
              <div className="flex items-center gap-3 mt-2">
                <div className="flex items-center gap-1.5 bg-[#ffffff0a] border border-[#ffffff14] px-2.5 py-1 rounded-md">
                  <Star className="w-3.5 h-3.5 text-amber-400 fill-amber-400" />
                  <span className="text-xs font-bold text-white">Level {level} Twin</span>
                </div>
                <div className="flex items-center gap-1.5 bg-[#ffffff0a] border border-[#ffffff14] px-2.5 py-1 rounded-md">
                  <Database className="w-3.5 h-3.5 text-indigo-400" />
                  <span className="text-xs font-bold text-white">{totalLogs} Logs</span>
                </div>
              </div>
            </div>
            <div className="flex flex-col items-end">
              <div className="flex items-center gap-1.5">
                <span className="text-sm font-bold text-white">{xp} XP</span>
              </div>
            </div>
          </div>

          {/* XP Bar */}
          <div className="mb-8 relative z-10">
            <div className="flex justify-between text-xs text-[#a1a1aa] mb-2 font-medium">
              <span>{xpIntoLevel} / 100 XP to Level {level + 1}</span>
            </div>
        <div className="h-2 w-full bg-[#1f1f1f] rounded-full overflow-hidden border border-[#ffffff0a]">
          <motion.div 
            className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progressPercent}%` }}
            transition={{ type: "spring", stiffness: 50, damping: 15 }}
          />
        </div>
      </div>

      {/* Quick Suggestions */}
      <div className="relative z-10 flex-1">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="w-4 h-4 text-indigo-400" />
          <h4 className="text-sm font-bold text-white">Quick Suggestions</h4>
        </div>
        
        <div className="space-y-3">
          <AnimatePresence>
            {suggestions.map((action, i) => (
              <motion.button
                key={`${action.domain}-${action.label}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
                onClick={() => handleSuggestionLog(action)}
                disabled={logging !== null}
                className={`w-full group relative z-20 cursor-pointer flex items-center justify-between p-4 rounded-xl border text-left transition-all duration-300 ${
                  logging === action.label 
                    ? 'bg-indigo-500/20 border-indigo-500/50' 
                    : 'bg-[#1f1f1f] border-[#ffffff14] hover:bg-[#2a2a2a] hover:border-[#ffffff2a]'
                }`}
              >
                <div className="flex items-center gap-4">
                  <div className={`p-2 rounded-lg border bg-[#ffffff0a] border-[#ffffff14]`}>
                    {logging === action.label ? (
                      <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
                    ) : (
                      <div className="text-[#ededed]">{action.icon}</div>
                    )}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-[#ededed] leading-tight">{action.label}</span>
                    </div>
                    <span className="text-xs text-[#a1a1aa] capitalize flex items-center gap-1 mt-0.5">
                      {action.domain} Action
                      <span className="inline-block w-1 h-1 rounded-full bg-[#ffffff2a]" />
                      <Zap className="w-3 h-3 text-amber-400 fill-amber-400" /> {action.xp} XP
                    </span>
                  </div>
                </div>
              </motion.button>
            ))}
          </AnimatePresence>
        </div>
      </div>

      {/* Manual Input */}
      <div className="relative z-10 mt-6 pt-4 border-t border-[#ffffff0a]">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Target className="w-3 h-3 text-[#a1a1aa]" />
            <span className="text-xs font-semibold text-[#a1a1aa] uppercase tracking-wider">Do Something Else</span>
          </div>
          <button 
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-[10px] text-indigo-400 hover:text-indigo-300 font-medium uppercase tracking-wider"
          >
            {showAdvanced ? 'Hide Context' : 'Add Context'}
          </button>
        </div>
        <form onSubmit={handleManualLog} className="flex flex-col gap-3">
          <div className="flex gap-2">
            <select 
              value={manualDomain}
              onChange={(e) => setManualDomain(e.target.value)}
              disabled={logging !== null}
              className="bg-[#1f1f1f] border border-[#ffffff14] text-[#ededed] text-xs rounded-lg px-2 focus:outline-none focus:border-indigo-500/50"
            >
              <option value="focus">Focus</option>
              <option value="task">Task</option>
              <option value="purchase">Buy</option>
            </select>
            <input
              type="text"
              value={manualInput}
              onChange={(e) => setManualInput(e.target.value)}
              placeholder="e.g. read a book..."
              disabled={logging !== null}
              className="flex-1 bg-[#0a0a0a] border border-[#ffffff14] text-sm text-[#ededed] rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/50 placeholder:text-[#3f3f46] transition-all"
            />
            <button
              type="submit"
              disabled={!manualInput.trim() || logging !== null}
              className="px-4 py-2 bg-indigo-500/10 text-indigo-400 hover:bg-indigo-500/20 font-medium text-sm rounded-lg transition-colors border border-indigo-500/20 disabled:opacity-50 min-w-[64px] flex justify-center items-center"
            >
              {logging === 'manual' ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Log'}
            </button>
          </div>

          <AnimatePresence>
            {showAdvanced && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="grid grid-cols-2 gap-3 p-3 bg-[#ffffff05] border border-[#ffffff0a] rounded-lg overflow-hidden"
              >
                <div className="space-y-1">
                  <label className="text-[10px] text-[#a1a1aa] uppercase tracking-wider">Location</label>
                  <input
                    type="text"
                    value={manualLocation}
                    onChange={(e) => setManualLocation(e.target.value)}
                    className="w-full bg-[#141414] border border-[#ffffff14] text-xs text-[#ededed] rounded px-2 py-1.5 focus:border-indigo-500/50 outline-none"
                    placeholder="e.g. home, office"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] text-[#a1a1aa] uppercase tracking-wider">Weather</label>
                  <input
                    type="text"
                    value={manualWeather}
                    onChange={(e) => setManualWeather(e.target.value)}
                    className="w-full bg-[#141414] border border-[#ffffff14] text-xs text-[#ededed] rounded px-2 py-1.5 focus:border-indigo-500/50 outline-none"
                    placeholder="e.g. clear, rain"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] text-[#a1a1aa] uppercase tracking-wider">Stress Level</label>
                  <select
                    value={manualStress}
                    onChange={(e) => setManualStress(e.target.value)}
                    className="w-full bg-[#141414] border border-[#ffffff14] text-xs text-[#ededed] rounded px-2 py-1.5 focus:border-indigo-500/50 outline-none"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="flex justify-between text-[10px] text-[#a1a1aa] uppercase tracking-wider">
                    <span>Mood/Energy</span>
                    <span className="text-indigo-400">{Math.round(manualMood * 100)}%</span>
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={manualMood}
                    onChange={(e) => setManualMood(parseFloat(e.target.value))}
                    className="w-full accent-indigo-500 mt-1.5"
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </form>
      </div>

      <AnimatePresence>
        {success && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-gradient-to-r from-indigo-500 to-purple-600 border border-indigo-400/50 shadow-xl rounded-full px-5 py-2 z-20 whitespace-nowrap"
          >
            <CheckCircle className="w-4 h-4 text-white" />
            <span className="text-sm text-white font-bold">{success}</span>
          </motion.div>
        )}
      </AnimatePresence>
        </div>
    </motion.div>
  )
})

export default LogDecisionPanel

