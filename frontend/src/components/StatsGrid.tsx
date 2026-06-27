import { motion } from 'framer-motion'
import { Activity, TrendingUp, AlertTriangle, Clock, BarChart3, Zap } from 'lucide-react'
import type { DashboardData } from '../hooks/useApi'
import TiltCard from './TiltCard'

interface StatsGridProps {
  data: DashboardData
}

export default function StatsGrid({ data }: StatsGridProps) {
  const totalDecisions = data.decisions.length
  const correctPredictions = data.decisions.filter(d => d.hit).length
  const driftCount = data.driftEvents.length
  const avgConfidence =
    data.decisions.length > 0
      ? data.decisions.reduce((s, d) => s + d.confidence, 0) / data.decisions.length
      : 0

  const domainCounts = data.decisions.reduce<Record<string, number>>((acc, d) => {
    acc[d.domain] = (acc[d.domain] || 0) + 1
    return acc
  }, {})
  const topDomain = Object.entries(domainCounts).sort((a, b) => b[1] - a[1])[0]

  const stats = [
    {
      label: 'Model Accuracy',
      value: `${(data.accuracy * 100).toFixed(1)}%`,
      subtext: `${correctPredictions} / ${totalDecisions} correct`,
      icon: TrendingUp,
      trend: data.accuracy >= 0.5 ? 'up' : 'down',
    },
    {
      label: 'Total Predictions',
      value: totalDecisions.toLocaleString(),
      subtext: `Across ${Object.keys(domainCounts).length} domains`,
      icon: BarChart3,
      trend: 'up',
    },
    {
      label: 'Avg Confidence',
      value: `${(avgConfidence * 100).toFixed(1)}%`,
      subtext: topDomain ? `Top: ${topDomain[0]} (${topDomain[1]})` : '—',
      icon: Zap,
      trend: avgConfidence >= 0.6 ? 'up' : 'neutral',
    },
    {
      label: 'Drift Alerts',
      value: driftCount.toString(),
      subtext: driftCount > 0 ? 'Behavior shifts detected' : 'No drift detected',
      icon: driftCount > 0 ? AlertTriangle : Activity,
      trend: driftCount > 0 ? 'down' : 'neutral',
    },
  ]

  const lastSyncedFormatted = (() => {
    try {
      const date = new Date(data.lastSynced)
      return date.toLocaleString()
    } catch {
      return data.lastSynced
    }
  })()

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05, duration: 0.2 }}
            className="h-full"
          >
            <TiltCard intensity={10} className="h-full">
              <div className="glass-card-hover p-4 flex flex-col justify-between h-full bg-[#141414]/50 border border-[#ffffff14] rounded-2xl">
                <div className="flex items-start justify-between mb-4">
                  <stat.icon className="w-4 h-4 text-[#71717a]" />
                  {stat.trend === 'up' && (
                    <span className="text-[10px] font-medium tracking-wide uppercase text-emerald-500/80">
                      Good
                    </span>
                  )}
                  {stat.trend === 'down' && (
                    <span className="text-[10px] font-medium tracking-wide uppercase text-rose-500/80">
                      Alert
                    </span>
                  )}
                </div>
                <div className="preserve-3d">
                  <motion.div 
                    key={stat.value}
                    initial={{ scale: 1.5, z: 80, textShadow: "0 0 30px rgba(168, 85, 247, 1)" }}
                    animate={{ scale: 1, z: 40, textShadow: "0 0 0px rgba(168, 85, 247, 0)" }}
                    transition={{ type: "spring", stiffness: 300, damping: 15 }}
                    className="text-display mb-1 inline-block"
                  >
                    {stat.value}
                  </motion.div>
                  <div className="text-body text-[#ededed] font-medium translate-z-10">{stat.label}</div>
                  <div className="text-caption mt-1">{stat.subtext}</div>
                </div>
              </div>
            </TiltCard>
          </motion.div>
        ))}
      </div>
      <div className="flex items-center gap-1.5 text-caption px-1">
        <Clock className="w-3.5 h-3.5" />
        <span>Last synced: {lastSyncedFormatted}</span>
      </div>
    </div>
  )
}
