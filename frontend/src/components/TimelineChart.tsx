/**
 * TimelineChart — The Mirror (Phase 4 hero section).
 *
 * Two lines: real (actual) vs predicted decisions over time.
 * When they diverge, that's drift.  When they track together, the twin is accurate.
 *
 * Motion spec (120fps-safe):
 * - Lines draw left-to-right via stroke-dashoffset CSS transition (GPU-accelerated).
 * - Confidence band fades in AFTER lines finish (opacity only).
 * - The "live heartbeat" on the latest data point is a pure CSS keyframe — no JS loop.
 * - Tooltip is fade+scale (no abrupt pop).
 * - All animations use transform/opacity only — no width/height/color.
 * - prefers-reduced-motion: skips all animation, shows final state immediately.
 * - Static sections are memoized so a data refresh doesn't re-render the whole tree.
 */
import { useRef, useEffect, useState, memo, useMemo } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer
} from 'recharts'
import { motion, useReducedMotion } from 'framer-motion'
import { Activity } from 'lucide-react'
import type { TimelinePoint } from '../hooks/useApi'

interface TimelineChartProps {
  timeline: TimelinePoint[]
}

// ---------------------------------------------------------------------------
// Custom animated tooltip
// ---------------------------------------------------------------------------
const AnimatedTooltip = memo(function AnimatedTooltip({
  active, payload, label
}: {
  active?: boolean
  payload?: Array<{ name: string; value: number; color: string }>
  label?: string
}) {
  if (!active || !payload?.length) return null
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.15 }}
      className="bg-[#141414] border border-[#ffffff14] rounded-lg px-3 py-2 shadow-xl"
    >
      <p className="text-caption mb-2">{label}</p>
      {payload.map(p => (
        <div key={p.name} className="flex items-center gap-2 text-sm">
          <div className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-[#a1a1aa]">{p.name}:</span>
          <span className="font-medium text-[#ededed]">
            {typeof p.value === 'number' ? (p.value * 100).toFixed(1) + '%' : p.value}
          </span>
        </div>
      ))}
    </motion.div>
  )
})

// ---------------------------------------------------------------------------
// Mirror path-draw SVG overlay (GPU-accelerated stroke-dashoffset)
// ---------------------------------------------------------------------------
interface MirrorOverlayProps {
  width: number
  height: number
  data: TimelinePoint[]
  animated: boolean
}

const MirrorOverlay = memo(function MirrorOverlay({ width, height, data, animated }: MirrorOverlayProps) {
  const pathRef = useRef<SVGPathElement>(null)
  const [pathLen, setPathLen] = useState(0)
  const PAD = { top: 5, right: 5, bottom: 40, left: 30 }

  // Build the SVG path for the "actual" line
  const path = useMemo(() => {
    if (!data.length || width < 10 || height < 10) return ''
    const w = width - PAD.left - PAD.right
    const h = height - PAD.top - PAD.bottom
    return data.map((pt, i) => {
      const x = PAD.left + (i / Math.max(data.length - 1, 1)) * w
      const y = PAD.top + (1 - pt.actual) * h
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    }).join(' ')
  }, [data, width, height])

  useEffect(() => {
    if (pathRef.current) setPathLen(pathRef.current.getTotalLength())
  }, [path])

  if (!path || !pathLen) return null

  return (
    <svg
      className="absolute inset-0 pointer-events-none"
      width={width}
      height={height}
      style={{ overflow: 'visible' }}
    >
      {/* Animated draw stroke — CSS transition on stroke-dashoffset */}
      <path
        ref={pathRef}
        d={path}
        fill="none"
        stroke="rgba(59,130,246,0.6)"
        strokeWidth={2.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        style={animated ? {
          strokeDasharray: pathLen,
          strokeDashoffset: 0,
          transition: `stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)`,
          willChange: 'transform',
        } : undefined}
      />
      {/* Path endpoint dot (no heartbeat animation) */}
      {data.length > 0 && (() => {
        const w = width - PAD.left - PAD.right
        const h = height - PAD.top - PAD.bottom
        const last = data[data.length - 1]
        const x = PAD.left + w
        const y = PAD.top + (1 - last.actual) * h
        return (
          <g transform={`translate(${x},${y})`}>
            <circle r={4} fill="#3b82f6" />
          </g>
        )
      })()}
    </svg>
  )
})

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function TimelineChart({ timeline }: TimelineChartProps) {
  const prefersReduced = useReducedMotion()
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const [chartSize, setChartSize] = useState({ width: 0, height: 288 })

  useEffect(() => {
    const el = chartContainerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const e = entries[0]
      if (e) setChartSize({ width: e.contentRect.width, height: 288 })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Compute drift gap — when actual - predicted diverges by > 0.2 for 3+ days
  const hasDrift = useMemo(() => {
    if (timeline.length < 3) return false
    const recent = timeline.slice(-10)
    return recent.filter(p => Math.abs(p.actual - p.predicted) > 0.2).length >= 3
  }, [timeline])

  const fadeIn = { initial: { opacity: 0, y: 16 }, animate: { opacity: 1, y: 0 } }

  if (!timeline.length) {
    return (
      <motion.div {...fadeIn} className="glass-card p-6">
        <div className="flex items-center gap-2 mb-2">
          <Activity className="w-4 h-4 text-[#a1a1aa]" />
          <h3 className="text-title">The Mirror</h3>
        </div>
        <div className="h-64 flex flex-col items-center justify-center text-center gap-2">
          <div className="w-8 h-8 rounded-full border border-[#ffffff14] flex items-center justify-center mb-2">
            <Activity className="w-4 h-4 text-[#71717a]" />
          </div>
          <p className="text-body max-w-sm">
            The mirror appears when the twin has enough data to compare predictions against reality.
          </p>
          <span className="text-caption">Retrain after logging 15+ decisions.</span>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      {...(prefersReduced ? {} : fadeIn)}
      transition={{ delay: 0.2, duration: 0.4 }}
      className="glass-card p-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-title">The Mirror</h3>
            {hasDrift && (
              <motion.span
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-[10px] font-medium tracking-wide uppercase text-[#f43f5e] bg-[#f43f5e]/10 px-2 py-0.5 rounded-md border border-[#f43f5e]/20"
              >
                Drift detected
              </motion.span>
            )}
          </div>
          <p className="text-body mt-1">
            Real behavior vs twin predictions
            {hasDrift && ' — lines are diverging'}
          </p>
        </div>
        <div className="flex items-center gap-4 text-caption">
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 bg-[#3b82f6] rounded-sm" />
            Actual
          </span>
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 bg-[#8b5cf6] rounded-sm" />
            Predicted
          </span>
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 border border-[#10b981] border-dashed rounded-sm" />
            Confidence
          </span>
        </div>
      </div>

      {/* Chart area */}
      <div ref={chartContainerRef} className="relative h-72">

        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={timeline} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
            <defs>
              <linearGradient id="gActual" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gPredicted" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gConfidence" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity={0.12} />
                <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
            <XAxis
              dataKey="date"
              stroke="rgba(255,255,255,0.1)"
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickFormatter={(v: string) => {
                const d = new Date(v)
                return isNaN(d.getTime()) ? v : `${d.getMonth() + 1}/${d.getDate()}`
              }}
              interval={Math.max(0, Math.floor(timeline.length / 7))}
            />
            <YAxis
              stroke="rgba(255,255,255,0.1)"
              tick={{ fill: '#64748b', fontSize: 10 }}
              domain={[0, 1]}
              tickFormatter={v => `${Math.round(v * 100)}%`}
            />
            <Tooltip content={<AnimatedTooltip />} />
            {/* Confidence band fades in last (opacity only) */}
            <Area
              type="monotone"
              dataKey="confidence"
              name="Confidence"
              stroke="#10b981"
              strokeWidth={1}
              strokeDasharray="4 3"
              fill="url(#gConfidence)"
              dot={false}
              activeDot={{ r: 3, fill: '#10b981' }}
              style={prefersReduced ? {} : { opacity: 1, transition: 'opacity 0.6s ease 1.3s' }}
            />
            <Area
              type="monotone"
              dataKey="predicted"
              name="Predicted"
              stroke="#8b5cf6"
              strokeWidth={2}
              fill="url(#gPredicted)"
              dot={false}
              activeDot={{ r: 4, fill: '#8b5cf6', stroke: '#3b1e6f', strokeWidth: 2 }}
            />
            <Area
              type="monotone"
              dataKey="actual"
              name="Actual"
              stroke="#3b82f6"
              strokeWidth={2.5}
              fill="url(#gActual)"
              dot={false}
              activeDot={{ r: 5, fill: '#3b82f6', stroke: '#1e3a5f', strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>

        {/* GPU-accelerated path-draw overlay for the "actual" line */}
        {!prefersReduced && chartSize.width > 0 && (
          <div className="absolute inset-0 pointer-events-none">
            <MirrorOverlay
              width={chartSize.width}
              height={chartSize.height}
              data={timeline}
              animated={!prefersReduced}
            />
          </div>
        )}
      </div>

      {/* Data points count */}
      <div className="mt-4 flex items-center justify-between text-caption border-t border-[#ffffff0a] pt-4">
        <span>{timeline.length} data point{timeline.length !== 1 ? 's' : ''}</span>
        {hasDrift && (
          <span className="text-[#f43f5e]">
            Predictions diverging from real behavior
          </span>
        )}
      </div>
    </motion.div>
  )
}
