import React, { useEffect, useState } from 'react'
import { motion, useSpring } from 'framer-motion'

export default function BackgroundEffects() {
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })

  // Use springs for ultra-smooth cursor tracking
  const mouseX = useSpring(0, { stiffness: 100, damping: 30 })
  const mouseY = useSpring(0, { stiffness: 100, damping: 30 })

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX)
      mouseY.set(e.clientY)
      setMousePos({ x: e.clientX, y: e.clientY })
    }
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [mouseX, mouseY])

  return (
    <div className="fixed inset-0 overflow-hidden bg-[#020205] pointer-events-none z-0">
      
      {/* Cinematic Aurora Mesh Gradient */}
      <div 
        className="absolute top-[-30%] left-[-20%] w-[80%] h-[80%] bg-[#7C5CFF]/20 rounded-[100%] blur-[140px] mix-blend-screen animate-[spin_25s_linear_infinite]"
      />
      <div 
        className="absolute bottom-[-20%] right-[-10%] w-[70%] h-[70%] bg-[#6EE7F9]/15 rounded-[100%] blur-[120px] mix-blend-screen animate-[spin_30s_linear_infinite_reverse]"
      />
      <div 
        className="absolute top-[20%] right-[10%] w-[50%] h-[50%] bg-[#A855F7]/20 rounded-[100%] blur-[130px] mix-blend-screen animate-[spin_40s_linear_infinite]"
      />
      <div 
        className="absolute bottom-[10%] left-[10%] w-[40%] h-[40%] bg-indigo-500/15 rounded-[100%] blur-[100px] mix-blend-screen animate-[spin_35s_linear_infinite_reverse]"
      />

      {/* 3D Neural Perspective Grid */}
      <div className="absolute inset-0 perspective-[1000px] flex items-center justify-center opacity-[0.04]">
        <motion.div 
          className="w-[200vw] h-[200vh]"
          style={{ 
            backgroundImage: 'linear-gradient(rgba(255, 255, 255, 1) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 1) 1px, transparent 1px)',
            backgroundSize: '60px 60px',
            rotateX: 60,
            rotateZ: mouseX.get() * 0.005,
            translateY: -100,
          }}
          animate={{ 
            backgroundPosition: ['0px 0px', '0px 60px'],
          }}
          transition={{ 
            repeat: Infinity, 
            duration: 2, 
            ease: "linear" 
          }}
        />
      </div>

      {/* Moving Cinematic Noise Overlay */}
      <div 
        className="absolute inset-0 opacity-[0.12] mix-blend-overlay"
        style={{ 
          backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=%220 0 400 400%22 xmlns=%22http://www.w3.org/2000/svg%22%3E%3Cfilter id=%22noiseFilter%22%3E%3CfeTurbulence type=%22fractalNoise%22 baseFrequency=%220.8%22 numOctaves=%223%22 stitchTiles=%22stitch%22/%3E%3C/filter%3E%3Crect width=%22100%25%22 height=%22100%25%22 filter=%22url(%23noiseFilter)%22/%3E%3C/svg%3E")',
          backgroundSize: '200px 200px'
        }}
      />

      {/* Vignette Shadow */}
      <div className="absolute inset-0 shadow-[inset_0_0_150px_rgba(0,0,0,0.9)]" />

      {/* Cursor Ambient Glow (Spotlight) */}
      <motion.div
        className="absolute w-[800px] h-[800px] rounded-full mix-blend-screen pointer-events-none"
        style={{
          background: 'radial-gradient(circle at center, rgba(124, 92, 255, 0.15) 0%, rgba(124, 92, 255, 0) 50%)',
          x: mouseX,
          y: mouseY,
          translateX: '-50%',
          translateY: '-50%'
        }}
      />
      
      {/* Inner Core Highlight (Spotlight) */}
      <motion.div
        className="absolute w-[300px] h-[300px] rounded-full mix-blend-screen pointer-events-none"
        style={{
          background: 'radial-gradient(circle at center, rgba(110, 231, 249, 0.2) 0%, rgba(110, 231, 249, 0) 50%)',
          x: mouseX,
          y: mouseY,
          translateX: '-50%',
          translateY: '-50%'
        }}
      />
    </div>
  )
}
