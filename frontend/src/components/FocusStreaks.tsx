import { memo, useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CalendarDays, Flame } from 'lucide-react'
import { HeatmapPoint } from '../hooks/useApi'
import { useSound } from '../hooks/useSound'

interface FocusStreaksProps {
  heatmap: HeatmapPoint[]
  xp: number
  level: number
  totalLogs: number
}

const COLS = 30
const ROWS = 5
const GAP = 1
const RADIUS = 3

type Point = { x: number, y: number }
type Particle = { x: number, y: number, vx: number, vy: number, life: number, color: string }
type FloatingText = { x: number, y: number, text: string, life: number }

const FocusStreaks = memo(function FocusStreaks({ heatmap, xp, level }: FocusStreaksProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { playSound } = useSound()
  const [gameScore, setGameScore] = useState(0)
  const [isGameOver, setIsGameOver] = useState(false)

  const snakeRef = useRef<Point[]>([{ x: 2, y: 2 }, { x: 1, y: 2 }, { x: 0, y: 2 }])
  const dirRef = useRef<Point>({ x: 1, y: 0 })
  const nextDirRef = useRef<Point>({ x: 1, y: 0 })
  const foodRef = useRef<Point>({ x: 15, y: 2 })
  const frameCountRef = useRef(0)
  const gameOverRef = useRef(false)
  const manualControlRef = useRef(false)
  const animFrameId = useRef<number>()
  const particlesRef = useRef<Particle[]>([])
  const textsRef = useRef<FloatingText[]>([])

  // Generate background grid lookup
  const countMap = new Map(heatmap.map(p => [p.date, p.count]))
  const today = new Date()
  const days: string[] = []
  for (let i = COLS * ROWS - 1; i >= 0; i--) {
    const d = new Date(today)
    d.setDate(today.getDate() - i)
    days.push(d.toISOString().split('T')[0])
  }

  // Calculate real stats
  let activeDays = 0
  let currentStreakCalc = 0
  let bestStreak = 0
  
  for (let i = 0; i < days.length; i++) {
    if ((countMap.get(days[i]) || 0) > 0) {
      activeDays++
      currentStreakCalc++
      if (currentStreakCalc > bestStreak) {
        bestStreak = currentStreakCalc
      }
    } else {
      currentStreakCalc = 0
    }
  }

  // Calculate actual daily streak leading up to today
  let realCurrentStreak = 0
  const todayIdx = days.length - 1
  const yesterdayIdx = days.length - 2
  
  if ((countMap.get(days[todayIdx]) || 0) > 0) {
    for (let i = todayIdx; i >= 0; i--) {
      if ((countMap.get(days[i]) || 0) > 0) realCurrentStreak++
      else break
    }
  } else if ((countMap.get(days[yesterdayIdx]) || 0) > 0) {
    for (let i = yesterdayIdx; i >= 0; i--) {
      if ((countMap.get(days[i]) || 0) > 0) realCurrentStreak++
      else break
    }
  }

  const completionRate = Math.round((activeDays / days.length) * 100)

  const getManhattan = (p1: Point, p2: Point) => Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y)
  
  const isCollision = (x: number, y: number, snake: Point[]) => {
    if (x < 0 || x >= COLS || y < 0 || y >= ROWS) return true
    for (let i = 0; i < snake.length; i++) {
      if (snake[i].x === x && snake[i].y === y) return true
    }
    return false
  }

  const spawnFood = () => {
    while (true) {
      const x = Math.floor(Math.random() * COLS)
      const y = Math.floor(Math.random() * ROWS)
      if (!isCollision(x, y, snakeRef.current)) {
        foodRef.current = { x, y }
        break
      }
    }
  }

  const getBestMove = () => {
    const head = snakeRef.current[0]
    const food = foodRef.current
    const moves = [
      { x: 0, y: -1 },
      { x: 0, y: 1 },
      { x: -1, y: 0 },
      { x: 1, y: 0 }
    ]

    const validMoves = moves.filter(m => {
       if (m.x === -dirRef.current.x && m.y === -dirRef.current.y) return false
       return !isCollision(head.x + m.x, head.y + m.y, snakeRef.current)
    })

    if (validMoves.length === 0) return dirRef.current

    validMoves.sort((a, b) => {
       const distA = getManhattan({ x: head.x + a.x, y: head.y + a.y }, food)
       const distB = getManhattan({ x: head.x + b.x, y: head.y + b.y }, food)
       return distA - distB
    })

    return validMoves[0]
  }

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const k = e.key
      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(k)) return
      e.preventDefault()
      manualControlRef.current = true
      
      const dir = dirRef.current
      if (k === 'ArrowUp' && dir.y === 0) nextDirRef.current = { x: 0, y: -1 }
      if (k === 'ArrowDown' && dir.y === 0) nextDirRef.current = { x: 0, y: 1 }
      if (k === 'ArrowLeft' && dir.x === 0) nextDirRef.current = { x: -1, y: 0 }
      if (k === 'ArrowRight' && dir.x === 0) nextDirRef.current = { x: 1, y: 0 }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  useEffect(() => {
    const draw = () => {
      const canvas = canvasRef.current
      if (!canvas || !containerRef.current) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      
      const cw = Math.min(640, containerRef.current.clientWidth - 16)
      const cellW = Math.floor(cw / COLS)
      const ch = cellW * ROWS
      canvas.width = cw
      canvas.height = ch
      
      ctx.clearRect(0, 0, cw, ch)
      
      // Draw Grid
      for (let i = 0; i < COLS; i++) {
        for (let j = 0; j < ROWS; j++) {
          const dayIdx = i * ROWS + j
          const dateStr = days[dayIdx]
          const count = dateStr ? countMap.get(dateStr) || 0 : 0
          
          let color = '#0f172a'
          if (count > 0 && count < 3) color = '#14532d'
          else if (count >= 3 && count < 6) color = '#15803d'
          else if (count >= 6 && count < 10) color = '#22c55e'
          else if (count >= 10) color = '#4ade80'
          
          ctx.fillStyle = color
          const cx = i * cellW + GAP
          const cy = j * cellW + GAP
          const w = cellW - GAP * 2
          
          ctx.beginPath()
          if (ctx.roundRect) ctx.roundRect(cx, cy, w, w, RADIUS)
          else ctx.rect(cx, cy, w, w)
          ctx.fill()
          
          if (count >= 10) {
            ctx.shadowColor = 'rgba(74,222,128,0.4)'
            ctx.shadowBlur = 8
            ctx.fill()
            ctx.shadowBlur = 0
          }
        }
      }

      // Draw Food
      const time = Date.now() / 200
      const scale = 1 + Math.sin(time) * 0.15
      const foodW = (cellW - GAP * 2) * scale
      
      ctx.fillStyle = '#ef4444'
      ctx.shadowColor = 'rgba(239, 68, 68, 0.6)'
      ctx.shadowBlur = 10
      ctx.beginPath()
      ctx.arc(
         foodRef.current.x * cellW + cellW/2, 
         foodRef.current.y * cellW + cellW/2, 
         foodW/2, 0, Math.PI * 2
      )
      ctx.fill()
      ctx.shadowBlur = 0
      
      // Draw Snake
      const snake = snakeRef.current
      snake.forEach((p, index) => {
        const opacity = Math.max(0.25, 1 - (index / snake.length))
        ctx.fillStyle = `rgba(74,222,128,${opacity})`
        const cx = p.x * cellW + GAP
        const cy = p.y * cellW + GAP
        const w = cellW - GAP * 2
        
        ctx.beginPath()
        if (ctx.roundRect) ctx.roundRect(cx, cy, w, w, RADIUS)
        else ctx.rect(cx, cy, w, w)
        ctx.fill()
        
        if (index === 0) {
           ctx.fillStyle = '#000000'
           const dir = dirRef.current
           const eyeSize = Math.max(1.5, w * 0.15)
           
           ctx.beginPath()
           if (dir.x !== 0) {
              ctx.arc(cx + w/2 + (dir.x * w*0.2), cy + w*0.25, eyeSize, 0, Math.PI*2)
              ctx.arc(cx + w/2 + (dir.x * w*0.2), cy + w*0.75, eyeSize, 0, Math.PI*2)
           } else {
              ctx.arc(cx + w*0.25, cy + w/2 + (dir.y * w*0.2), eyeSize, 0, Math.PI*2)
              ctx.arc(cx + w*0.75, cy + w/2 + (dir.y * w*0.2), eyeSize, 0, Math.PI*2)
           }
           ctx.fill()
        }
      })

      // Draw Particles
      for (let i = particlesRef.current.length - 1; i >= 0; i--) {
        const p = particlesRef.current[i]
        p.x += p.vx
        p.y += p.vy
        p.life -= 0.03
        if (p.life <= 0) {
          particlesRef.current.splice(i, 1)
          continue
        }
        ctx.fillStyle = p.color
        ctx.globalAlpha = Math.max(0, p.life)
        ctx.beginPath()
        ctx.arc(p.x, p.y, RADIUS, 0, Math.PI * 2)
        ctx.fill()
        ctx.globalAlpha = 1.0
      }

      // Draw Floating Text
      for (let i = textsRef.current.length - 1; i >= 0; i--) {
        const t = textsRef.current[i]
        t.y -= 0.8
        t.life -= 0.02
        if (t.life <= 0) {
          textsRef.current.splice(i, 1)
          continue
        }
        ctx.fillStyle = `rgba(74, 222, 128, ${Math.max(0, t.life)})`
        ctx.font = 'bold 14px system-ui, sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText(t.text, t.x, t.y)
      }
    }

    const loop = () => {
      animFrameId.current = requestAnimationFrame(loop)
      if (gameOverRef.current) {
         draw()
         return
      }

      frameCountRef.current++
      const currentLength = snakeRef.current.length
      const speedThreshold = Math.max(3, 8 - Math.floor((currentLength - 3) / 5))

      if (frameCountRef.current % speedThreshold === 0) {
         if (!manualControlRef.current) {
            nextDirRef.current = getBestMove()
         }
         
         dirRef.current = nextDirRef.current
         const head = snakeRef.current[0]
         const nx = head.x + dirRef.current.x
         const ny = head.y + dirRef.current.y
         
         if (isCollision(nx, ny, snakeRef.current)) {
            gameOverRef.current = true
            setIsGameOver(true)
            try { playSound('hover') } catch(e) {}
            setTimeout(() => {
               snakeRef.current = [{ x: 2, y: 2 }, { x: 1, y: 2 }, { x: 0, y: 2 }]
               dirRef.current = { x: 1, y: 0 }
               nextDirRef.current = { x: 1, y: 0 }
               particlesRef.current = []
               textsRef.current = []
               gameOverRef.current = false
               setIsGameOver(false)
               manualControlRef.current = false
               spawnFood()
               setGameScore(0)
            }, 1800)
         } else {
            const newHead = { x: nx, y: ny }
            const newSnake = [newHead, ...snakeRef.current]
            
            if (nx === foodRef.current.x && ny === foodRef.current.y) {
               setGameScore(s => s + 1)
               try { playSound('click') } catch(e) {}
               
               // Spawn Particles
               const cw = containerRef.current ? Math.min(640, containerRef.current.clientWidth - 16) : 640
               const cellW = Math.floor(cw / COLS)
               const fx = foodRef.current.x * cellW + cellW/2
               const fy = foodRef.current.y * cellW + cellW/2
               
               for (let i = 0; i < 12; i++) {
                 particlesRef.current.push({
                   x: fx, y: fy,
                   vx: (Math.random() - 0.5) * 6,
                   vy: (Math.random() - 0.5) * 6,
                   life: 1.0,
                   color: Math.random() > 0.5 ? '#ef4444' : '#4ade80'
                 })
               }
               textsRef.current.push({ x: fx, y: fy - 5, text: '+1', life: 1.0 })
               
               spawnFood()
            } else {
               newSnake.pop()
            }
            snakeRef.current = newSnake
         }
      }
      draw()
    }
    
    animFrameId.current = requestAnimationFrame(loop)
    return () => {
      if (animFrameId.current) cancelAnimationFrame(animFrameId.current)
    }
  }, [heatmap])

  return (
    <div className="bg-[#0a0a18] rounded-2xl border border-[#ffffff1a] p-5 w-full relative overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <CalendarDays className="w-5 h-5 text-[#a1a1aa]" />
          <div>
            <h3 className="text-sm font-bold text-white">Focus Streaks</h3>
            <p className="text-xs text-[#a1a1aa]">Positive habits · 90 days</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* XP Bar */}
          <div className="flex flex-col items-end gap-1">
            <span className="text-[10px] font-semibold text-[#a1a1aa]">LVL {level} · {xp}/{level * 100} XP</span>
            <div className="w-32 h-1.5 bg-[#ffffff14] rounded-full overflow-hidden relative">
               <motion.div 
                  className="absolute top-0 bottom-0 left-0 bg-[#a855f7] rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(100, (xp / (level * 100)) * 100)}%` }}
               />
               <motion.div 
                  className="absolute top-0 bottom-0 w-8 bg-gradient-to-r from-transparent via-white/40 to-transparent"
                  animate={{ left: ['-80%', '120%'] }}
                  transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
               />
            </div>
          </div>

          {/* Gamified Streak */}
          <div className="flex items-center gap-2 bg-gradient-to-br from-[#1a0b05] to-[#2a1205] border border-orange-500/20 px-3 py-1.5 rounded-xl shadow-[0_0_15px_rgba(249,115,22,0.1)]">
            <Flame className="w-5 h-5 text-orange-500 drop-shadow-[0_0_5px_rgba(249,115,22,0.8)] animate-pulse" />
            <div className="flex flex-col">
              <span className="text-[9px] text-orange-200/80 font-bold uppercase tracking-wider leading-none mb-0.5">Streak</span>
              <span className="text-sm font-black text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-red-500 leading-none">{realCurrentStreak}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Canvas Container */}
      <div ref={containerRef} className="w-full relative flex items-center justify-center rounded-xl bg-black/20 border border-[#ffffff0a] p-2">
         <canvas ref={canvasRef} className="block" style={{ maxWidth: '100%' }} />
         <AnimatePresence>
            {isGameOver && (
               <motion.div 
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 1.1 }}
                  className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm rounded-xl"
               >
                  <span className="text-rose-500 font-bold tracking-widest">GAME OVER · SCORE: {gameScore}</span>
               </motion.div>
            )}
         </AnimatePresence>
      </div>

      {/* Bottom Row: Legend & Stats */}
      <div className="flex items-center justify-between mt-4">
         <div className="flex items-center gap-1.5 text-[10px] font-semibold tracking-wider text-[#a1a1aa] uppercase">
            <span>Less</span>
            <div className="w-2.5 h-2.5 rounded-[2px] bg-[#0f172a]" />
            <div className="w-2.5 h-2.5 rounded-[2px] bg-[#14532d]" />
            <div className="w-2.5 h-2.5 rounded-[2px] bg-[#15803d]" />
            <div className="w-2.5 h-2.5 rounded-[2px] bg-[#22c55e]" />
            <div className="w-2.5 h-2.5 rounded-[2px] bg-[#4ade80] shadow-[0_0_8px_rgba(74,222,128,0.4)]" />
            <span>More</span>
         </div>
         <div className="flex items-center gap-4 text-xs">
            <span className="text-[#4ade80] font-bold">{activeDays} <span className="text-[#a1a1aa] font-medium">days</span></span>
            <span className="text-amber-400 font-bold">{bestStreak} <span className="text-[#a1a1aa] font-medium">best</span></span>
            <span className="text-purple-400 font-bold">{completionRate}% <span className="text-[#a1a1aa] font-medium">rate</span></span>
         </div>
      </div>
    </div>
  )
})

export default FocusStreaks
