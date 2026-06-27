import { useState, useEffect, memo, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BrainCircuit, RefreshCw, Zap, ShieldAlert, CheckCircle2, TrendingUp, Sparkles, Activity } from 'lucide-react'

import { fetchAIBriefing } from '../hooks/useApi'

// Map icon names from backend to Lucide components
const IconMap: Record<string, any> = {
  TrendingUp, ShieldAlert, Zap, CheckCircle2, Sparkles, Activity
}

export default memo(function AIBriefingPanel({ userId }: { userId: string }) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analyzed, setAnalyzed] = useState(false)
  const [visibleInsights, setVisibleInsights] = useState<any[]>([])

  const hasRun = useRef(false)

  // Auto-run once on mount
  useEffect(() => {
    if (hasRun.current) return
    hasRun.current = true
    handleSync()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleSync = async () => {
    if (isAnalyzing) return
    setIsAnalyzing(true)
    setAnalyzed(false)
    setVisibleInsights([])

    // Simulate multi-step analysis
    await new Promise(res => setTimeout(res, 800))
    
    try {
      const realInsights = await fetchAIBriefing(userId)
      // Reveal insights one by one
      for (let i = 0; i < realInsights.length; i++) {
        await new Promise(res => setTimeout(res, 400))
        setVisibleInsights(prev => [...prev, realInsights[i]])
      }
    } catch (e) {
      console.error(e)
    }
    
    await new Promise(res => setTimeout(res, 300))
    setIsAnalyzing(false)
    setAnalyzed(true)
  }

  return (
    <div className="glass-card p-0 overflow-hidden relative border border-[#ffffff14]">
      {/* Background Effect */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-emerald-500 opacity-50" />
      
      {isAnalyzing && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute top-0 left-0 w-full h-[2px] bg-indigo-500 shimmer z-10"
        />
      )}

      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/30 flex items-center justify-center shadow-[0_0_15px_rgba(99,102,241,0.15)] relative">
              <BrainCircuit className={`w-5 h-5 text-indigo-400 ${isAnalyzing ? 'animate-pulse' : ''}`} />
              {analyzed && !isAnalyzing && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-500 border-2 border-[#141414] rounded-full" />
              )}
            </div>
            <div>
              <h3 className="text-lg font-bold text-[#ededed] tracking-tight">Autonomous Briefing</h3>
              <p className="text-xs text-[#a1a1aa]">
                {isAnalyzing ? 'Syncing integrations and running neural weights...' : 'Live multi-source context analysis'}
              </p>
            </div>
          </div>
          <button 
            onClick={handleSync}
            disabled={isAnalyzing}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#1f1f1f] border border-[#ffffff14] hover:bg-[#ffffff0a] hover:border-[#ffffff2a] transition-all text-xs font-medium text-[#ededed] disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isAnalyzing ? 'animate-spin text-indigo-400' : ''}`} />
            {isAnalyzing ? 'Analyzing...' : 'Sync & Analyze'}
          </button>
        </div>

        <div className="space-y-3">
          <AnimatePresence>
            {visibleInsights.map((insight) => {
              const Icon = IconMap[insight.iconName] || Activity
              return (
                <motion.div
                  key={insight.id}
                  initial={{ opacity: 0, x: -10, y: 10 }}
                  animate={{ opacity: 1, x: 0, y: 0 }}
                  transition={{ type: "spring", stiffness: 100, damping: 15 }}
                  className="flex gap-4 p-4 rounded-xl bg-[#141414] border border-[#ffffff0a] relative overflow-hidden group hover:bg-[#1a1a1a] hover:border-[#ffffff14] transition-all"
                >
                  {/* Subtle left border accent */}
                  <div className={`absolute left-0 top-0 bottom-0 w-1 ${insight.bg} opacity-50`} />
                  
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${insight.bg}`}>
                    <Icon className={`w-4 h-4 ${insight.color}`} />
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-bold text-[#ededed] mb-1">{insight.title}</h4>
                    <p className="text-sm text-[#a1a1aa] leading-relaxed">{insight.desc}</p>
                  </div>
                </motion.div>
              )
            })}
          </AnimatePresence>

          {!isAnalyzing && visibleInsights.length === 0 && (
            <div className="py-8 text-center border border-dashed border-[#ffffff14] rounded-xl">
              <p className="text-sm text-[#71717a]">Awaiting sync...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
})
