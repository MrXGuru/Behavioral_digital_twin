import { useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { motion, AnimatePresence } from 'framer-motion'
import TiltCard from './TiltCard'
import {
  RefreshCw, Loader2, CheckCircle, XCircle, Brain, Database,
  Trophy, Sparkles, AlertTriangle
} from 'lucide-react'
import { retrain, seedData, type RetrainResult } from '../hooks/useApi'

interface RetrainPanelProps {
  userId: string
  onRetrained: () => void
}

const loadingPhases = [
  { text: 'Loading behavioral records...', duration: 200 },
  { text: 'Extracting features & n-grams...', duration: 200 },
  { text: 'Training Baseline Model...', duration: 300 },
  { text: 'Training Sequence Model...', duration: 300 },
  { text: 'Evaluating on held-out data...', duration: 200 },
  { text: 'Compiling MLOps report...', duration: 100 },
]

// Colour helpers
function accColor(acc: number) {
  if (acc >= 0.85) return 'text-emerald-400'
  if (acc >= 0.65) return 'text-amber-400'
  return 'text-rose-400'
}
function accBg(acc: number) {
  if (acc >= 0.85) return 'bg-emerald-400'
  if (acc >= 0.65) return 'bg-amber-400'
  return 'bg-rose-400'
}
function accLabel(acc: number) {
  if (acc >= 0.85) return 'Excellent'
  if (acc >= 0.65) return 'Good'
  if (acc >= 0.45) return 'Fair'
  return 'Needs more data'
}

function MetricBar({ value, label }: { value: number; label: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px]">
        <span className="text-slate-500">{label}</span>
        <span className={`font-mono font-bold ${accColor(value)}`}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-[#1f1f1f] rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className={`h-full rounded-full ${accBg(value)}`}
        />
      </div>
    </div>
  )
}

export default function RetrainPanel({ userId, onRetrained }: RetrainPanelProps) {
  const [loading, setLoading] = useState(false)
  const [seeding, setSeeding] = useState(false)
  const [phaseIdx, setPhaseIdx] = useState(0)
  const [result, setResult] = useState<RetrainResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeDomain, setActiveDomain] = useState<string>('focus')
  const [timeLeft, setTimeLeft] = useState(0)
  const isModalOpen = !!(result || error)

  useEffect(() => {
    if (result?.metrics) {
      setActiveDomain(Object.keys(result.metrics)[0])
    }
  }, [result])

  const handleSeed = async () => {
    setSeeding(true)
    window.dispatchEvent(new Event('ai-processing-start'))
    setError(null)
    try {
      await seedData(userId)
      onRetrained()
      await handleRetrain()
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Seeding failed')
    } finally {
      setSeeding(false)
      window.dispatchEvent(new Event('ai-processing-end'))
    }
  }

  useEffect(() => {
    if (!loading) return
    let timeoutId: ReturnType<typeof setTimeout>
    const advancePhase = (idx: number) => {
      if (idx >= loadingPhases.length - 1) return
      timeoutId = setTimeout(() => {
        setPhaseIdx(idx + 1)
        advancePhase(idx + 1)
      }, loadingPhases[idx].duration)
    }
    advancePhase(0)
    return () => clearTimeout(timeoutId)
  }, [loading])

  useEffect(() => {
    if (!loading || timeLeft <= 0) return
    const interval = setInterval(() => {
      setTimeLeft(prev => Math.max(0, prev - 100))
    }, 100)
    return () => clearInterval(interval)
  }, [loading, timeLeft])

  const handleRetrain = async () => {
    setLoading(true)
    window.dispatchEvent(new Event('ai-processing-start'))
    setPhaseIdx(0)
    setError(null)
    setResult(null)
    const totalTime = loadingPhases.reduce((acc, p) => acc + p.duration, 0)
    setTimeLeft(totalTime)
    try {
      const res = await retrain(userId)
      setResult(res)
      setLoading(false)
      onRetrained()
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Retrain failed')
      setLoading(false)
      window.dispatchEvent(new Event('ai-processing-end'))
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.55 }}
      className="h-full preserve-3d"
    >
      <div className="glass-card-hover p-6 border border-indigo-500/20 shadow-[0_0_30px_rgba(99,102,241,0.1)] relative overflow-hidden h-full flex flex-col preserve-3d">
        <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/5 rounded-full blur-3xl -z-10 pointer-events-none" />

        {/* Header */}
        <div className="flex items-center gap-3 mb-5 translate-z-10">
          <div className="bg-indigo-500/10 border border-indigo-500/20 p-2.5 rounded-xl shadow-inner">
            <Brain className={`w-5 h-5 text-indigo-400 ${loading ? 'animate-pulse' : ''}`} />
          </div>
          <div>
            <h3 className="text-title text-indigo-100">AI Engine Training</h3>
            <p className="text-body text-indigo-300/70 text-xs mt-0.5">
              {loading ? loadingPhases[phaseIdx].text : 'Retrain models on your latest behavioral patterns'}
            </p>
          </div>
        </div>

        {/* Progress bar when loading */}
        {loading && (
          <div className="mb-5 translate-z-10">
            <div className="h-1 bg-[#1f1f1f] rounded-full overflow-hidden">
              <motion.div
                animate={{ width: `${((phaseIdx + 1) / loadingPhases.length) * 100}%` }}
                transition={{ duration: 0.5 }}
                className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
              />
            </div>
            <div className="flex justify-between items-center mt-1.5">
              <p className="text-[10px] text-slate-500">
                Step {phaseIdx + 1} / {loadingPhases.length}
              </p>
              <p className="text-[10px] font-mono text-indigo-400 font-bold">
                {(timeLeft / 1000).toFixed(1)}s remaining
              </p>
            </div>
          </div>
        )}

        {/* Action buttons */}
        <motion.div layoutId={!isModalOpen ? "retrain-container" : undefined} className="flex flex-col gap-2.5 translate-z-10 mt-auto">
          <motion.button
            whileTap={{ scale: 0.95, y: 2 }}
            onClick={handleRetrain}
            disabled={loading || seeding}
            className="w-full py-3 px-4 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-xl flex justify-center items-center gap-2 transition-all shadow-[0_4px_15px_rgba(79,70,229,0.3)] hover:shadow-[0_8px_25px_rgba(79,70,229,0.5)] disabled:opacity-50 disabled:cursor-not-allowed preserve-3d"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                {loadingPhases[phaseIdx].text}
              </>
            ) : (
              <>
                <RefreshCw className="w-4 h-4" />
                Retrain AI Models
              </>
            )}
          </motion.button>

          <motion.button
            whileTap={{ scale: 0.95, y: 2 }}
            onClick={handleSeed}
            disabled={loading || seeding}
            className="w-full py-2.5 px-4 bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 font-semibold rounded-xl flex justify-center items-center gap-2 transition-all border border-indigo-500/20 disabled:opacity-50 disabled:cursor-not-allowed text-sm preserve-3d"
          >
            {seeding ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {seeding ? 'Generating sample data...' : 'Auto-Generate Sample Data'}
          </motion.button>
        </motion.div>
      </div>

      {createPortal(
      <AnimatePresence>
        {error && !loading && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 perspective-1000">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/40 backdrop-blur-md"
              onClick={() => setError(null)}
            />
            <TiltCard intensity={10} className="w-full max-w-sm relative z-10">
            <motion.div
              layoutId="retrain-container"
              initial={{ opacity: 0, scale: 0.85, rotateX: -15, y: 40 }}
              animate={{ opacity: 1, scale: 1, rotateX: 0, y: 0 }}
              exit={{ opacity: 0, scale: 0.85, rotateX: 10, y: 20, transition: { duration: 0.25, ease: "easeIn" } }}
              transition={{ type: "spring", stiffness: 250, damping: 20 }}
              className="glass-card w-full p-6 text-center bg-[#0d0d0d]/90 shadow-2xl"
            >
              <div className="w-12 h-12 rounded-full bg-rose-500/10 flex items-center justify-center mx-auto mb-4">
                <XCircle className="w-6 h-6 text-[#f43f5e]" />
              </div>
              <h3 className="text-lg font-bold text-white mb-2">Training Failed</h3>
              <p className="text-sm text-rose-400">{error}</p>
              <button onClick={() => setError(null)} className="mt-6 w-full py-2 btn-secondary">Close</button>
            </motion.div>
            </TiltCard>
          </div>
        )}

        {result && !loading && !error && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 perspective-1000 overflow-y-auto pt-24 pb-10">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/60 backdrop-blur-md"
              onClick={() => setResult(null)}
            />
            <TiltCard intensity={10} className="w-full max-w-lg relative z-10">
            <motion.div
              layoutId="retrain-container"
              initial={{ opacity: 0, scale: 0.85, rotateX: -15, y: 40 }}
              animate={{ opacity: 1, scale: 1, rotateX: 0, y: 0 }}
              exit={{ opacity: 0, scale: 0.85, rotateX: 10, y: 20, transition: { duration: 0.25, ease: "easeIn" } }}
              transition={{ type: "spring", stiffness: 250, damping: 20 }}
              className="glass-card w-full max-w-lg relative z-10 bg-[#0d0d0d]/95 shadow-[0_0_50px_rgba(99,102,241,0.2)]"
            >
              {/* Header */}
              <div className="p-6 border-b border-[#ffffff0a] flex items-center justify-between sticky top-0 bg-[#0d0d0d]/90 backdrop-blur z-20">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-indigo-500/20 flex items-center justify-center shadow-[0_0_15px_rgba(99,102,241,0.4)]">
                    <Sparkles className="w-5 h-5 text-indigo-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-white leading-tight">Training Complete</h2>
                    <p className="text-[11px] text-slate-400">MLOps Retrain Report</p>
                  </div>
                </div>
                <button onClick={() => setResult(null)} className="p-2 rounded-full hover:bg-white/5 transition-colors">
                  <XCircle className="w-5 h-5 text-slate-400" />
                </button>
              </div>

              <div className="p-6 space-y-6">
                {/* Status badge */}
                <div className={`flex items-center gap-2 px-4 py-3 rounded-xl border ${
                  result.status === 'completed' || result.status === 'retrained'
                    ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.1)]'
                    : result.status === 'skipped'
                    ? 'bg-amber-500/10 border-amber-500/20 text-amber-400 shadow-[0_0_15px_rgba(245,158,11,0.1)]'
                    : 'bg-slate-500/10 border-slate-500/20 text-slate-300'
                }`}>
                  {result.status === 'skipped'
                    ? <AlertTriangle className="w-5 h-5" />
                    : <CheckCircle className="w-5 h-5" />}
                  <div className="w-full">
                    <span className="text-sm font-semibold block leading-tight">
                      {result.status === 'completed' || result.status === 'retrained'
                        ? 'Models Successfully Updated'
                        : result.status === 'skipped'
                        ? 'Insufficient Data — Log more decisions'
                        : result.status}
                    </span>
                    {result.reason && <span className="text-xs opacity-80 mt-0.5 block">{result.reason}</span>}
                    

                  </div>
                </div>

                {/* Per-domain MLOps cards */}
                {result.metrics && Object.keys(result.metrics).length > 0 && (
                  <div className="space-y-4">
                    {/* Domain Tabs */}
                    <div className="flex gap-1.5 p-1 bg-[#0a0a0a] rounded-xl border border-[#ffffff08]">
                      {Object.keys(result.metrics).map(d => (
                        <button
                          key={d}
                          onClick={() => setActiveDomain(d)}
                          className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-semibold capitalize transition-all ${
                            activeDomain === d
                              ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 shadow-sm'
                              : 'text-[#71717a] hover:text-[#a1a1aa] hover:bg-white/5 border border-transparent'
                          }`}
                        >
                          {d}
                        </button>
                      ))}
                    </div>

                    <AnimatePresence mode="wait">
                    {(() => {
                      const domain = activeDomain
                      const m: any = result.metrics[domain]
                      if (!m) return null

                      const winnerAcc: number = m[m.winner]?.accuracy ?? 0
                      const valSamples = m.baseline?.n ?? 0
                      const isSmallSample = valSamples < 5

                      return (
                        <motion.div 
                          key={domain}
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                          transition={{ duration: 0.2 }}
                          className="bg-[#141414] border border-[#ffffff10] rounded-xl overflow-hidden preserve-3d"
                        >
                          {/* Domain header */}
                          <div className="px-4 py-3 bg-[#1a1a1a] border-b border-[#ffffff0a] flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Database className="w-4 h-4 text-indigo-400" />
                              <span className="text-sm font-bold text-white capitalize tracking-wide">{domain} Domain</span>
                            </div>
                            <div className="flex items-center gap-2">
                              {isSmallSample && (
                                <span className="flex items-center gap-1 text-[10px] font-bold text-amber-400 bg-amber-400/10 border border-amber-400/20 px-2 py-0.5 rounded uppercase">
                                  <AlertTriangle className="w-3 h-3" /> Small sample
                                </span>
                              )}
                              <span className="text-xs text-slate-500">{valSamples} val samples</span>
                            </div>
                          </div>

                          <div className="p-4 space-y-4">
                            {/* Winner badge */}
                            <div className="flex items-center gap-3 p-3 bg-indigo-500/10 border border-indigo-500/20 rounded-lg shadow-inner">
                              <Trophy className="w-5 h-5 text-yellow-400 shrink-0" />
                              <div className="flex-1 min-w-0">
                                <span className="text-xs text-indigo-200/70 block mb-0.5">Winning Architecture</span>
                                <div className="flex items-end gap-3">
                                  <span className="text-sm font-bold text-white capitalize leading-none">{m.winner} Model</span>
                                  <span className={`text-sm font-mono font-bold leading-none ${accColor(winnerAcc)}`}>
                                    {(winnerAcc * 100).toFixed(1)}% ({accLabel(winnerAcc)})
                                  </span>
                                </div>
                              </div>
                            </div>

                            {/* Model comparison */}
                            <div className="grid grid-cols-2 gap-3">
                              {/* Baseline */}
                              <div className={`p-4 rounded-xl border space-y-3 ${
                                m.winner === 'baseline'
                                  ? 'bg-indigo-500/10 border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.15)]'
                                  : 'bg-[#0a0a0a] border-[#ffffff08]'
                              }`}>
                                <div className="flex items-center justify-between mb-1">
                                  <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Baseline</span>
                                  {m.winner === 'baseline' && <Trophy className="w-4 h-4 text-yellow-400" />}
                                </div>
                                <MetricBar value={m.baseline?.accuracy ?? 0} label="Accuracy" />
                                <MetricBar value={m.baseline?.macro_f1 ?? 0} label="Macro F1" />
                              </div>

                              {/* Sequence */}
                              <div className={`p-4 rounded-xl border space-y-3 ${
                                m.winner === 'sequence'
                                  ? 'bg-indigo-500/10 border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.15)]'
                                  : 'bg-[#0a0a0a] border-[#ffffff08]'
                              }`}>
                                <div className="flex items-center justify-between mb-1">
                                  <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Sequence</span>
                                  {m.winner === 'sequence' && <Trophy className="w-4 h-4 text-yellow-400" />}
                                </div>
                                <MetricBar value={m.sequence?.accuracy ?? 0} label="Accuracy" />
                                <MetricBar value={m.sequence?.macro_f1 ?? 0} label="Macro F1" />
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      )
                    })()}
                    </AnimatePresence>
                  </div>
                )}
                
                <button onClick={() => setResult(null)} className="w-full mt-2 py-3 btn-secondary">Done</button>
              </div>
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
