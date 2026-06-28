import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence, useSpring, useTransform, useMotionValue } from 'framer-motion'
import { CheckCircle2, Loader2, Sparkles, ChevronRight, Brain } from 'lucide-react'
import { useGoogleLogin } from '@react-oauth/google'

import { useSound } from '../../hooks/useSound'

type AuthState = 'idle' | 'scanning' | 'success'

interface AuthCardProps {
  onSuccess: (credentialResponse: any) => void
  errorMsg: string
}

const LOADING_TEXTS = [
  "Initializing Digital Twin...",
  "Connecting to Neural Network...",
  "Authenticating Identity...",
  "Loading Behavioral Models...",
  "Synchronizing Memory...",
  "Almost Ready..."
]

export default function AuthCard({ onSuccess, errorMsg }: AuthCardProps) {
  const [authState, setAuthState] = useState<AuthState>('idle')
  const [loadingTextIdx, setLoadingTextIdx] = useState(0)
  const { playSound } = useSound()

  // Tilt Physics
  const cardRef = useRef<HTMLDivElement>(null)
  const x = useMotionValue(0)
  const y = useMotionValue(0)
  const mouseXSpring = useSpring(x, { stiffness: 300, damping: 30 })
  const mouseYSpring = useSpring(y, { stiffness: 300, damping: 30 })
  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ["7deg", "-7deg"])
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ["-7deg", "7deg"])

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    const width = rect.width
    const height = rect.height
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top
    x.set((mouseX / width) - 0.5)
    y.set((mouseY / height) - 0.5)
  }
  const handleMouseLeave = () => {
    x.set(0)
    y.set(0)
  }

  useEffect(() => {
    if (authState === 'scanning') {
      const interval = setInterval(() => {
        setLoadingTextIdx(i => (i + 1) % LOADING_TEXTS.length)
      }, 1500)
      return () => clearInterval(interval)
    }
  }, [authState])

  const login = useGoogleLogin({
    onSuccess: (credentialResponse) => {
      playSound('click')
      setAuthState('scanning')
      // Simulate cinematic delay before triggering actual success
      setTimeout(() => {
        playSound('success')
        setAuthState('success')
        setTimeout(() => {
          onSuccess(credentialResponse)
        }, 1500)
      }, 3000)
    },
    onError: () => {
      console.log('Login Failed')
      setAuthState('idle')
    },
    flow: 'implicit'
  })

  return (
    <div className="perspective-[1200px] z-20">
      <motion.div
        layoutId="app-card"
        ref={cardRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{
          rotateX,
          rotateY,
          transformStyle: "preserve-3d",
        }}
        initial={{ opacity: 0, scale: 0.9, y: 30 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 200, damping: 25, delay: 0.2 }}
        className="relative w-full max-w-[420px] mx-4"
      >
        {/* Glass Card */}
        <div className="relative overflow-hidden rounded-[32px] bg-[#050505]/60 backdrop-blur-[40px] border border-[#ffffff1a] shadow-[0_30px_80px_rgba(0,0,0,0.8),inset_0_0_0_1px_rgba(255,255,255,0.05)] p-8 preserve-3d">
          
          {/* Animated Gradient Border Overlay */}
          <div className="absolute inset-0 rounded-[32px] overflow-hidden pointer-events-none">
            <div className="absolute top-[-50%] left-[-50%] w-[200%] h-[200%] bg-[conic-gradient(from_0deg,transparent_0_340deg,rgba(124,92,255,0.3)_360deg)] animate-[spin_4s_linear_infinite]" />
          </div>

          <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none rounded-[32px]" />
          
          <div className="relative z-10 flex flex-col items-center translate-z-[20px]">
            {/* 3D Core Container */}
            <div className="w-32 h-32 mb-6 relative">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="absolute inset-0 bg-indigo-500/20 rounded-full blur-xl animate-pulse" />
                <Brain className="w-16 h-16 text-indigo-400 drop-shadow-[0_0_15px_rgba(129,140,248,0.5)] animate-breathe relative z-10" />
              </div>
              
              {/* Optional Scanning Ring */}
              <AnimatePresence>
                {authState === 'scanning' && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 1.2 }}
                    className="absolute inset-[-20%] rounded-full border-2 border-dashed border-[#6EE7F9]/40 animate-[spin_3s_linear_infinite]"
                  />
                )}
              </AnimatePresence>
            </div>

            <div className="text-center mb-10 translate-z-[10px]">
              <motion.h2 
                className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-white via-indigo-100 to-[#a1a1aa] tracking-tight mb-2"
              >
                Behavioral Twin
              </motion.h2>
              <p className="text-sm text-[#a1a1aa] font-medium tracking-wide">
                Your Digital Intelligence Layer
              </p>
            </div>

            {errorMsg && (
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full p-3 mb-6 bg-rose-500/10 border border-rose-500/20 rounded-xl text-rose-400 text-sm text-center backdrop-blur-md translate-z-[10px]"
              >
                {errorMsg}
              </motion.div>
            )}

            <div className="w-full translate-z-[30px]">
              {authState === 'idle' && (
                <motion.button
                  onClick={() => login()}
                  onMouseEnter={() => playSound('hover')}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full group relative flex items-center justify-center gap-3 px-6 py-4 bg-white hover:bg-slate-50 text-slate-900 rounded-2xl font-bold transition-all shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.2)] overflow-hidden"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/50 to-transparent -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]" />
                  <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" className="w-5 h-5 relative z-10" alt="Google" />
                  <span className="relative z-10 tracking-wide">Continue with Google</span>
                  <ChevronRight className="w-4 h-4 text-slate-400 group-hover:translate-x-1 transition-transform relative z-10" />
                </motion.button>
              )}

              {authState === 'scanning' && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="w-full flex flex-col items-center gap-4 bg-[#ffffff0a] border border-[#ffffff14] rounded-2xl p-5 backdrop-blur-md"
                >
                  <div className="flex items-center gap-3">
                    <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
                    <AnimatePresence mode="wait">
                      <motion.span
                        key={loadingTextIdx}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        className="text-sm font-semibold text-white tracking-wide"
                      >
                        {LOADING_TEXTS[loadingTextIdx]}
                      </motion.span>
                    </AnimatePresence>
                  </div>
                  
                  {/* Glowing Progress Bar */}
                  <div className="w-full h-1.5 bg-[#ffffff1a] rounded-full overflow-hidden relative">
                    <motion.div
                      className="absolute top-0 left-0 bottom-0 bg-gradient-to-r from-indigo-500 via-purple-500 to-[#6EE7F9] shadow-[0_0_10px_rgba(110,231,249,0.5)]"
                      initial={{ width: "0%" }}
                      animate={{ width: "100%" }}
                      transition={{ duration: 5, ease: "linear" }}
                    />
                  </div>
                </motion.div>
              )}

              {authState === 'success' && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="w-full flex flex-col items-center justify-center p-6 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl backdrop-blur-md shadow-[0_0_30px_rgba(16,185,129,0.2)]"
                >
                  <motion.div 
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", bounce: 0.5 }}
                  >
                    <CheckCircle2 className="w-12 h-12 text-emerald-400 mb-3" />
                  </motion.div>
                  <h3 className="text-emerald-400 font-bold text-lg mb-1">Welcome Back</h3>
                  <p className="text-emerald-400/80 text-xs text-center">Intelligence synchronized successfully.</p>
                </motion.div>
              )}
            </div>
            
            <div className="mt-8 flex items-center justify-center gap-2 translate-z-[5px]">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400/50" />
              <span className="text-[11px] font-medium text-slate-500 uppercase tracking-widest">
                Secure 256-bit AES Encryption
              </span>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
