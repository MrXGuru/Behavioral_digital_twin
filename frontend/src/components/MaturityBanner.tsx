/**
 * MaturityBanner — shown when user has fewer than DATA_MATURITY_THRESHOLD decisions.
 *
 * The spec explicitly requires the system to show "still learning" / low-confidence
 * rather than polished-looking predictions when data is insufficient.  This component
 * makes that honest state visible and actionable.
 */
import { memo } from 'react'
import { motion } from 'framer-motion'
import { BookOpen, TrendingUp } from 'lucide-react'
import type { DataMaturity } from '../hooks/useApi'

interface MaturityBannerProps {
  maturity: DataMaturity
}

const MaturityBanner = memo(function MaturityBanner({ maturity }: MaturityBannerProps) {
  if (maturity.status === 'ready') return null

  const pct = Math.min(100, Math.round((maturity.count / maturity.threshold) * 100))
  const remaining = maturity.threshold - maturity.count

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-5 border-amber-500/20 bg-amber-500/5"
      style={{ borderColor: 'rgba(245,158,11,0.2)' }}
    >
      <div className="flex items-start gap-4">
        <div className="bg-amber-500/15 p-2.5 rounded-xl shrink-0">
          <BookOpen className="w-5 h-5 text-amber-400" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-semibold text-amber-300">Twin is Still Learning</span>
            <span className="text-xs text-amber-400/70 bg-amber-500/10 px-2 py-0.5 rounded-full">
              {pct}%
            </span>
          </div>
          <p className="text-sm text-slate-400 leading-relaxed">
            {maturity.message}
          </p>
          {/* Progress bar */}
          <div className="mt-3">
            <div className="flex items-center justify-between text-xs text-slate-500 mb-1.5">
              <span>{maturity.count} decisions logged</span>
              <span>{remaining} more needed</span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
                className="h-full bg-gradient-to-r from-amber-500 to-orange-400 rounded-full"
              />
            </div>
          </div>
        </div>
        <div className="shrink-0 text-right">
          <div className="text-2xl font-bold text-amber-400">{maturity.count}</div>
          <div className="text-xs text-slate-500">of {maturity.threshold}</div>
        </div>
      </div>
      <div className="mt-4 flex items-center gap-2 text-xs text-slate-500">
        <TrendingUp className="w-3.5 h-3.5 text-amber-500/60" />
        <span>
          Log real decisions using the form below — each one makes predictions more accurate.
        </span>
      </div>
    </motion.div>
  )
})

export default MaturityBanner
