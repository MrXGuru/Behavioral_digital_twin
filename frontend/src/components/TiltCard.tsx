import React, { useRef } from 'react'
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion'

interface TiltCardProps {
  children: React.ReactNode
  className?: string
  intensity?: number
}

export default function TiltCard({ children, className = '', intensity = 15 }: TiltCardProps) {
  const ref = useRef<HTMLDivElement>(null)

  const x = useMotionValue(0)
  const y = useMotionValue(0)

  // Smooth out the mouse movement
  const mouseXSpring = useSpring(x, { stiffness: 300, damping: 40 })
  const mouseYSpring = useSpring(y, { stiffness: 300, damping: 40 })

  // Map normalized mouse position [-0.5, 0.5] to rotation degrees
  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], [`${intensity}deg`, `-${intensity}deg`])
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], [`-${intensity}deg`, `${intensity}deg`])

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return
    const rect = ref.current.getBoundingClientRect()
    
    // Calculate mouse position relative to the card's center
    const width = rect.width
    const height = rect.height
    
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top
    
    // Normalize to [-0.5, 0.5]
    const xPct = (mouseX / width) - 0.5
    const yPct = (mouseY / height) - 0.5
    
    x.set(xPct)
    y.set(yPct)
  }

  const handleMouseLeave = () => {
    // Reset to flat when mouse leaves
    x.set(0)
    y.set(0)
  }

  return (
    <div style={{ perspective: '1200px' }} className={`w-full ${className}`}>
      <motion.div
        ref={ref}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{
          rotateX,
          rotateY,
          transformStyle: "preserve-3d",
        }}
        className="relative w-full h-full transition-shadow duration-300 hover:shadow-[0_20px_40px_rgba(0,0,0,0.4)] rounded-2xl"
      >
        <div 
          className="absolute inset-0 pointer-events-none rounded-[inherit] overflow-hidden z-50 mix-blend-overlay"
        >
          <motion.div
            className="absolute inset-0"
            style={{
              background: `radial-gradient(circle at center, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 60%)`,
              left: useTransform(mouseXSpring, [-0.5, 0.5], ["-50%", "50%"]),
              top: useTransform(mouseYSpring, [-0.5, 0.5], ["-50%", "50%"]),
              width: "200%",
              height: "200%",
              x: "-25%",
              y: "-25%"
            }}
          />
        </div>
        {/* Render actual content wrapped in another div to isolate 3D space */}
        <div style={{ transform: "translateZ(30px)", transformStyle: "preserve-3d" }} className="w-full h-full rounded-[inherit]">
            {children}
        </div>
      </motion.div>
    </div>
  )
}
