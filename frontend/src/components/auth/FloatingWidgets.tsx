import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Brain, Activity, Target, Sun, Flame, Zap, ChevronUp } from 'lucide-react'

const floatVariants = (delay: number, duration: number, yRange: number) => ({
  initial: { y: 0, opacity: 0, scale: 0.8 },
  animate: {
    y: [0, -yRange, 0],
    opacity: 1,
    scale: 1,
    transition: {
      y: { repeat: Infinity, duration, ease: "easeInOut", delay },
      opacity: { duration: 1, delay },
      scale: { type: "spring", stiffness: 100, damping: 20, delay }
    }
  }
})

function WidgetCard({ children, className, delay, yRange, duration }: any) {
  return (
    <motion.div
      variants={floatVariants(delay, duration, yRange)}
      initial="initial"
      animate="animate"
      whileHover={{ scale: 1.05, y: -5, transition: { duration: 0.2 } }}
      className={`absolute bg-white/5 backdrop-blur-[20px] border border-[#ffffff1a] rounded-2xl shadow-2xl p-4 flex flex-col gap-2 pointer-events-auto cursor-default group overflow-hidden ${className}`}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      <div className="relative z-10">
        {children}
      </div>
    </motion.div>
  )
}

function LiveChart({ color }: { color: string }) {
  const [data, setData] = useState([40, 60, 45, 80, 50, 90, 70])
  useEffect(() => {
    const int = setInterval(() => {
      setData(prev => [...prev.slice(1), Math.floor(Math.random() * 60) + 40])
    }, 1500)
    return () => clearInterval(int)
  }, [])
  
  return (
    <div className="flex items-end gap-1 h-8 mt-1">
      {data.map((h, i) => (
        <motion.div 
          key={i} 
          animate={{ height: `${h}%` }} 
          className={`w-[6px] rounded-t-sm ${color} opacity-80`}
          transition={{ type: "spring", stiffness: 100 }}
        />
      ))}
    </div>
  )
}

export default function FloatingWidgets() {
  const [liveScore, setLiveScore] = useState(87)

  useEffect(() => {
    const int = setInterval(() => {
      setLiveScore(s => Math.min(99, Math.max(70, s + (Math.random() > 0.5 ? 1 : -1))))
    }, 3000)
    return () => clearInterval(int)
  }, [])

  return (
    <div className="absolute inset-0 pointer-events-none z-10 hidden lg:block overflow-hidden perspective-[1000px]">
      
      {/* 1. Behavior Score */}
      <WidgetCard delay={0.2} duration={6} yRange={15} className="top-[15%] left-[8%] w-52">
        <div className="flex items-center gap-2 text-indigo-400 mb-1">
          <Brain className="w-4 h-4" />
          <span className="text-xs font-bold uppercase tracking-wider">Behavior Score</span>
        </div>
        <div className="flex items-baseline gap-2">
          <motion.span className="text-3xl font-black text-white">{liveScore}</motion.span>
          <span className="text-emerald-400 flex items-center text-xs font-medium"><ChevronUp className="w-3 h-3"/> 2.4%</span>
        </div>
        <div className="w-full h-1 bg-white/10 rounded-full mt-2 overflow-hidden">
          <motion.div className="h-full bg-indigo-500 rounded-full" animate={{ width: `${liveScore}%` }} />
        </div>
      </WidgetCard>

      {/* 2. Today's Focus */}
      <WidgetCard delay={1.5} duration={7} yRange={20} className="top-[45%] left-[3%] w-56">
        <div className="flex items-center gap-2 text-rose-400 mb-1">
          <Target className="w-4 h-4" />
          <span className="text-xs font-bold uppercase tracking-wider">Focus Priority</span>
        </div>
        <p className="text-sm font-medium text-white mb-2">Deep Work Session</p>
        <div className="flex -space-x-2">
          {[1,2,3].map(i => <div key={i} className="w-6 h-6 rounded-full bg-rose-500/20 border border-rose-500/50 backdrop-blur-md" />)}
        </div>
      </WidgetCard>

      {/* 3. Live AI Prediction */}
      <WidgetCard delay={0.8} duration={5} yRange={12} className="bottom-[15%] left-[10%] w-48">
        <div className="flex items-center gap-2 text-amber-400 mb-1">
          <Zap className="w-4 h-4" />
          <span className="text-xs font-bold uppercase tracking-wider">Prediction</span>
        </div>
        <p className="text-xs text-[#a1a1aa]">Next optimal action:</p>
        <p className="text-base font-bold text-white mt-0.5">Pomodoro Block</p>
        <p className="text-[10px] text-amber-400 mt-1 font-mono">94% Confidence</p>
      </WidgetCard>

      {/* 4. Activity Graph */}
      <WidgetCard delay={1.2} duration={8} yRange={18} className="top-[18%] right-[8%] w-52">
        <div className="flex items-center gap-2 text-emerald-400 mb-1">
          <Activity className="w-4 h-4" />
          <span className="text-xs font-bold uppercase tracking-wider">Live Activity</span>
        </div>
        <LiveChart color="bg-emerald-500" />
      </WidgetCard>

      {/* 5. Energy Level */}
      <WidgetCard delay={2.1} duration={6.5} yRange={15} className="top-[48%] right-[4%] w-44">
        <div className="flex items-center gap-2 text-amber-500 mb-1">
          <Sun className="w-4 h-4" />
          <span className="text-xs font-bold uppercase tracking-wider">Energy State</span>
        </div>
        <div className="text-2xl font-black text-white">Peak</div>
        <div className="text-[10px] text-[#a1a1aa] mt-1">Optimal for coding</div>
      </WidgetCard>

      {/* 6. Habits Consistency */}
      <WidgetCard delay={0.5} duration={7.5} yRange={22} className="bottom-[18%] right-[12%] w-48">
        <div className="flex items-center gap-2 text-purple-400 mb-1">
          <Flame className="w-4 h-4" />
          <span className="text-xs font-bold uppercase tracking-wider">Consistency</span>
        </div>
        <div className="flex items-center gap-3 mt-1">
          <span className="text-3xl font-black text-white">12</span>
          <span className="text-xs text-[#a1a1aa] leading-tight">days<br/>perfect streak</span>
        </div>
      </WidgetCard>

    </div>
  )
}
