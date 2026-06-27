import { motion } from 'framer-motion'
import { AlertTriangle, Radio, Info } from 'lucide-react'
import type { DriftEvent } from '../hooks/useApi'

interface DriftPanelProps {
  events: DriftEvent[]
}

const DOMAIN_STYLES: Record<string, string> = {
  focus:    'text-blue-500',
  task:     'text-purple-500',
  purchase: 'text-emerald-500',
}

export default function DriftPanel({ events }: DriftPanelProps) {
  if (events.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="glass-card p-6"
      >
        <div className="flex items-center gap-3 mb-6">
          <div className="bg-[#1f1f1f] border border-[#ffffff0a] p-2 rounded-lg">
            <Radio className="w-4 h-4 text-[#ededed]" />
          </div>
          <div>
            <h3 className="text-title">Drift Detection</h3>
            <p className="text-body">Monitoring behavioral consistency</p>
          </div>
        </div>
        <div className="flex items-center gap-3 bg-[#141414] border border-[#ffffff14] rounded-lg px-3 py-2">
          <div className="w-2 h-2 rounded-full bg-[#10b981]" />
          <span className="text-sm font-medium text-[#ededed]">All systems stable — no drift detected</span>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5 }}
      className="glass-card p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="bg-[#1f1f1f] border border-[#ffffff0a] p-2 rounded-lg">
            <AlertTriangle className="w-4 h-4 text-[#ededed]" />
          </div>
          <div>
            <h3 className="text-title">Drift Alerts</h3>
            <p className="text-body">{events.length} behavior shift{events.length !== 1 ? 's' : ''} detected</p>
          </div>
        </div>
        <span className="flex items-center gap-1.5 text-caption bg-[#f43f5e]/10 px-2 py-1 rounded-md text-[#f43f5e] border border-[#f43f5e]/20">
          <span className="w-1.5 h-1.5 rounded-full bg-[#f43f5e]" />
          Active
        </span>
      </div>

      <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
        {events.map((evt, i) => {
          const styleColor = DOMAIN_STYLES[evt.domain] || 'text-[#ededed]'
          return (
            <motion.div
              key={`${evt.date}-${evt.domain}-${i}`}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              className="bg-[#141414] border border-[#ffffff14] rounded-lg p-3"
            >
              <div className="flex items-start gap-3">
                <Info className="w-4 h-4 text-[#71717a] mt-0.5 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-caption font-mono">{evt.date}</span>
                    <span className={`text-[10px] font-medium uppercase tracking-wide bg-[#1f1f1f] border border-[#ffffff0a] px-1.5 py-0.5 rounded ${styleColor}`}>
                      {evt.domain}
                    </span>
                  </div>
                  <p className="text-sm text-[#ededed] leading-relaxed">{evt.note}</p>
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>
    </motion.div>
  )
}
