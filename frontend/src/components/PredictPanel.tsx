import { useState } from 'react'
import { createPortal } from 'react-dom'
import { motion, AnimatePresence } from 'framer-motion'
import TiltCard from './TiltCard'
import { Sparkles, Loader2, Target, ListTodo, ShoppingCart, Brain, Zap, Timer } from 'lucide-react'
import { predictNext, type PredictResult } from '../hooks/useApi'

interface PredictPanelProps {
  userId: string
}

const DOMAINS = [
  { id: 'focus', label: 'Focus', icon: Target, color: 'text-indigo-400', bg: 'bg-indigo-500/10', border: 'border-indigo-500/30' },
  { id: 'task', label: 'Task', icon: ListTodo, color: 'text-purple-400', bg: 'bg-purple-500/10', border: 'border-purple-500/30' },
  { id: 'purchase', label: 'Purchase', icon: ShoppingCart, color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30' },
]

function confidenceColor(c: number) {
  if (c >= 0.8) return 'text-emerald-400'
  if (c >= 0.5) return 'text-amber-400'
  return 'text-rose-400'
}
function confidenceBg(c: number) {
  if (c >= 0.8) return 'bg-emerald-400'
  if (c >= 0.5) return 'bg-amber-400'
  return 'bg-rose-400'
}
function confidenceLabel(c: number) {
  if (c >= 0.9) return 'Very confident'
  if (c >= 0.7) return 'Confident'
  if (c >= 0.5) return 'Moderate'
  return 'Low confidence'
}

export default function PredictPanel({ userId }: PredictPanelProps) {
  const [selectedDomain, setSelectedDomain] = useState('focus')
  const [result, setResult] = useState<PredictResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const isModalOpen = !!(result || error)

  const handlePredict = async () => {
    setLoading(true)
    window.dispatchEvent(new Event('ai-processing-start'))
    setError(null)
    setResult(null)
    try {
      const res = await predictNext(userId, selectedDomain)
      setResult(res)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
      window.dispatchEvent(new Event('ai-processing-end'))
    }
  }

  const activeDomain = DOMAINS.find(d => d.id === selectedDomain)!

  // Intervention logic
  const isLowFocus = result && activeDomain.id === 'focus' && (result.confidence < 0.6 || ['break', 'light_work', 'email', 'none'].includes(result.predicted))


  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.45 }}
      className="h-full preserve-3d"
    >
      <div className="glass-card-hover p-5 h-full flex flex-col relative overflow-hidden preserve-3d">
        {/* Glow */}
        <div className="absolute top-0 left-0 w-48 h-48 bg-indigo-500/5 rounded-full blur-3xl -z-10 pointer-events-none" />

        {/* Header */}
        <div className="flex items-center gap-2.5 mb-4 translate-z-10">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
            <Brain className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-[#ededed]">Live Prediction</h3>
            <p className="text-[10px] text-[#71717a]">What will you do next?</p>
          </div>
        </div>

        {/* Domain tabs */}
        <div className="flex gap-1.5 mb-6 p-1 bg-[#0a0a0a] rounded-xl border border-[#ffffff08] translate-z-10">
          {DOMAINS.map(domain => (
            <button
              key={domain.id}
              onClick={() => { setSelectedDomain(domain.id); setResult(null); setError(null) }}
              className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg text-[11px] font-semibold transition-all ${
                selectedDomain === domain.id
                  ? `${domain.bg} ${domain.color} border ${domain.border} shadow-sm`
                  : 'text-[#71717a] hover:text-[#a1a1aa] hover:bg-white/5'
              }`}
            >
              <domain.icon className="w-3 h-3" />
              {domain.label}
            </button>
          ))}
        </div>

        {/* Predict button */}
        <motion.button
          layoutId={!isModalOpen ? "predict-container" : undefined}
          onClick={handlePredict}
          disabled={loading}
          whileTap={{ scale: 0.95, y: 2 }}
          className="mt-auto w-full py-3 px-4 bg-gradient-to-br from-indigo-500 to-purple-600 text-white text-sm font-bold rounded-xl flex justify-center items-center gap-2 transition-all shadow-[0_4px_15px_rgba(79,70,229,0.3)] hover:shadow-[0_8px_25px_rgba(79,70,229,0.5)] disabled:opacity-50 disabled:cursor-not-allowed preserve-3d translate-z-10"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Querying Brain...
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              Predict {activeDomain.label}
            </>
          )}
        </motion.button>
      </div>

      {/* 3D Unfolding Modal for Result */}
      {createPortal(
      <AnimatePresence>
        {(result || error) && !loading && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 perspective-1000">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, transition: { duration: 0.2 } }}
              onClick={() => { setResult(null); setError(null); }}
              className="absolute inset-0 bg-black/40 backdrop-blur-md"
            />
            
            <TiltCard intensity={10} className="w-full max-w-sm relative z-10">
            <motion.div
              layoutId="predict-container"
              initial={{ opacity: 0, scale: 0.85, rotateX: -15, y: 40 }}
              animate={{ opacity: 1, scale: 1, rotateX: 0, y: 0 }}
              exit={{ opacity: 0, scale: 0.85, rotateX: 10, y: 20, transition: { duration: 0.25, ease: "easeIn" } }}
              transition={{ type: "spring", stiffness: 250, damping: 20 }}
              className="glass-card w-full overflow-hidden bg-[#0d0d0d]/90 shadow-2xl"
            >
              {error ? (
                <div className="p-6 text-center">
                  <div className="w-12 h-12 rounded-full bg-rose-500/10 flex items-center justify-center mx-auto mb-4">
                    <Zap className="w-6 h-6 text-rose-400" />
                  </div>
                  <h3 className="text-lg font-bold text-white mb-2">Prediction Failed</h3>
                  <p className="text-sm text-rose-400">{error}</p>
                  <button onClick={() => setError(null)} className="mt-6 w-full py-2 btn-secondary">Close</button>
                </div>
              ) : result ? (
                <div>
                  <div className="p-6 border-b border-[#ffffff0a] bg-gradient-to-b from-indigo-500/10 to-transparent text-center relative">
                    <div className="absolute top-4 right-4">
                      <button onClick={() => setResult(null)} className="text-[#71717a] hover:text-white transition-colors">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
                      </button>
                    </div>
                    <div className="w-12 h-12 rounded-full bg-indigo-500/20 flex items-center justify-center mx-auto mb-4 shadow-[0_0_20px_rgba(99,102,241,0.4)]">
                      <Sparkles className="w-6 h-6 text-indigo-400" />
                    </div>
                    <div className="text-xs font-bold text-indigo-400 uppercase tracking-widest mb-2">AI Predicts</div>
                    <div className="text-3xl font-black text-white capitalize tracking-tight mb-1">
                      {result.predicted.replace(/_/g, ' ')}
                    </div>
                    <div className="text-xs text-[#a1a1aa] flex items-center justify-center gap-2">
                      <activeDomain.icon className="w-3 h-3" /> {activeDomain.label} Domain
                    </div>
                  </div>
                  <div className="p-6 bg-[#050505]">
                    <div className="flex justify-between items-end mb-2">
                      <div>
                        <div className="text-[10px] font-bold text-[#71717a] uppercase tracking-wider mb-1">Confidence Score</div>
                        <div className={`text-sm font-semibold ${confidenceColor(result.confidence)}`}>
                          {confidenceLabel(result.confidence)}
                        </div>
                      </div>
                      <div className={`text-2xl font-black font-mono tracking-tighter ${confidenceColor(result.confidence)}`}>
                        {(result.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="h-3 bg-[#141414] rounded-full overflow-hidden border border-[#ffffff0a] shadow-inner">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence * 100}%` }}
                        transition={{ delay: 0.2, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                        className={`h-full rounded-full ${confidenceBg(result.confidence)} shadow-[0_0_10px_currentColor]`}
                      />
                    </div>

                    {/* Intervention Card */}
                    {isLowFocus && (
                      <motion.div 
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.6 }}
                        className="mt-6 p-4 bg-purple-500/10 border border-purple-500/20 rounded-xl text-left shadow-[0_0_15px_rgba(168,85,247,0.1)]"
                      >
                        <div className="flex items-start gap-3">
                          <div className="bg-purple-500/20 p-2 rounded-lg text-purple-400 shrink-0">
                            <Timer className="w-4 h-4" />
                          </div>
                          <div>
                            <p className="text-sm font-semibold text-purple-200 mb-1">Model thinks focus will be low next hour.</p>
                            <p className="text-xs text-purple-300/70 mb-3">Want a 25-min timer instead to lock in?</p>
                            <button 
                              onClick={() => {
                                window.dispatchEvent(new Event('start-focus-timer'));
                                setResult(null);
                              }}
                              className="px-4 py-1.5 bg-purple-600 hover:bg-purple-500 text-white text-xs font-bold rounded-lg transition-colors shadow-lg"
                            >
                              Start Focus Timer
                            </button>
                          </div>
                        </div>
                      </motion.div>
                    )}

                    <button onClick={() => setResult(null)} className="w-full mt-6 py-2.5 btn-secondary">Done</button>
                  </div>
                </div>
              ) : null}
            </motion.div>
            </TiltCard>
          </div>
        )}
      </AnimatePresence>,
      document.body
    )}
    </motion.div>
  )
}
