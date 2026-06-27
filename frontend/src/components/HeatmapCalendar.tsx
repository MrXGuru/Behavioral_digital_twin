import { memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CalendarDays, AlertTriangle } from 'lucide-react'
import { HeatmapPoint } from '../hooks/useApi'

interface HeatmapCalendarProps {
  heatmap: HeatmapPoint[]
}

const HeatmapCalendar = memo(function HeatmapCalendar({ heatmap }: HeatmapCalendarProps) {
  // Generate last 90 days array
  const today = new Date()
  const days = []
  for (let i = 89; i >= 0; i--) {
    const d = new Date(today)
    d.setDate(today.getDate() - i)
    days.push(d.toISOString().split('T')[0])
  }

  // Create lookup map
  const countMap = new Map(heatmap.map(p => [p.date, p.count]))

  // Function to determine color based on count
  const getColor = (count: number) => {
    if (count === 0) return 'bg-[#141414] border border-[#ffffff0a]'
    if (count < 3) return 'bg-[#10b981]/20 border border-[#10b981]/10'
    if (count < 6) return 'bg-[#10b981]/40 border border-[#10b981]/20'
    if (count < 10) return 'bg-[#10b981]/60 border border-[#10b981]/30'
    return 'bg-[#10b981] border border-[#10b981]/40 shadow-[0_0_8px_rgba(16,185,129,0.4)]'
  }

  // Early warning signal: check if focus dropped over the last 3 days
  const last3Days = days.slice(-3)
  const c1 = countMap.get(last3Days[0]) || 0
  const c2 = countMap.get(last3Days[1]) || 0
  const c3 = countMap.get(last3Days[2]) || 0
  const isDropping = (c1 > c2 && c2 > c3) || (c1 === 0 && c2 === 0 && c3 === 0 && countMap.size > 0) // Also show if 3 days of zero, but they have some history


  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-full"
    >
      <div className="glass-card-hover p-6 h-full preserve-3d">
        <div className="flex items-center gap-3 mb-8 preserve-3d translate-z-10">
          <div className="bg-gradient-to-br from-emerald-500 to-emerald-700 shadow-[0_0_15px_rgba(16,185,129,0.3)] p-2 rounded-lg">
            <CalendarDays className="w-4 h-4 text-white" />
          </div>
          <div>
            <h3 className="text-title">Focus Streaks</h3>
            <p className="text-body">Your positive habits over the last 90 days</p>
          </div>
        </div>

        {/* 3D Isometric Grid Container */}
        <div className="relative w-full h-[140px] flex items-center justify-center preserve-3d">
          <motion.div 
            initial={{ rotateX: 60, rotateZ: -45, scale: 0.8 }}
            animate={{ rotateX: 60, rotateZ: -45, scale: 0.9 }}
            transition={{ type: "spring", stiffness: 50, damping: 20 }}
            className="flex gap-[4px] flex-wrap w-[220px] preserve-3d"
          >
            {days.map((date, i) => {
              const count = countMap.get(date) || 0
              return (
                <motion.div
                  key={date}
                  initial={{ z: 0 }}
                  animate={{ z: count * 8 }} // Extrude on Z axis based on activity count
                  transition={{ delay: i * 0.005, type: "spring", stiffness: 200, damping: 15 }}
                  title={`${date}: ${count} positive habits`}
                  className={`w-3 h-3 rounded-sm preserve-3d relative ${getColor(count)}`}
                  style={{
                    boxShadow: count > 0 ? `-3px 3px 6px rgba(0,0,0,0.6)` : 'none'
                  }}
                />
              )
            })}
          </motion.div>
        </div>
        
        <div className="flex items-center justify-end gap-2 mt-6 text-[10px] font-semibold tracking-wider text-[#a1a1aa] uppercase translate-z-10 preserve-3d">
          <span>Less</span>
          <div className="w-3 h-3 rounded-[2px] bg-[#141414] border border-[#ffffff0a]" />
          <div className="w-3 h-3 rounded-[2px] bg-[#10b981]/20 border border-[#10b981]/10" />
          <div className="w-3 h-3 rounded-[2px] bg-[#10b981]/40 border border-[#10b981]/20" />
          <div className="w-3 h-3 rounded-[2px] bg-[#10b981]/60 border border-[#10b981]/30" />
          <div className="w-3 h-3 rounded-[2px] bg-[#10b981] border border-[#10b981]/40" />
          <span>More</span>
        </div>

        {/* Early Warning Banner */}
        <AnimatePresence>
          {isDropping && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl flex items-center gap-3 translate-z-10 preserve-3d"
            >
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              <span className="text-xs font-medium text-amber-200/90">
                Focus dropped 3 days in a row — that's unusual for you.
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
})

export default HeatmapCalendar
