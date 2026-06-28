import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import BackgroundEffects from './BackgroundEffects'
import FloatingWidgets from './FloatingWidgets'
import AuthCard from './AuthCard'
import { API_BASE } from '../../hooks/useApi'

interface AuthScreenProps {
  onLogin: (email: string) => void
}

export default function AuthScreen({ onLogin }: AuthScreenProps) {
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')

  const handleSuccess = async (credentialResponse: any) => {
    try {
      const res = await fetch(`${API_BASE}/auth/google`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: credentialResponse.access_token })
      })

      if (!res.ok) {
        throw new Error('Authentication failed on server')
      }

      const data = await res.json()
      localStorage.setItem('access_token', data.access_token || credentialResponse.access_token)
      localStorage.setItem('user_profile', JSON.stringify({
        email: data.user.email,
        name: data.user.name,
        picture: data.user.picture
      }))

      setIsTransitioning(true)
      
      setTimeout(() => {
        onLogin(data.user.email)
      }, 1500)
    } catch (err: any) {
      console.error('Login error:', err)
      setErrorMsg(err.message || 'Authentication failed. Please try again.')
    }
  }

  return (
    <div className="min-h-screen relative flex items-center justify-center p-4 overflow-hidden bg-[#020205]">
      <BackgroundEffects />
      <FloatingWidgets />
      
      <AnimatePresence>
        {!isTransitioning && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.1, filter: 'blur(20px)' }}
            transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
            className="relative z-20 w-full flex justify-center"
          >
            <AuthCard onSuccess={handleSuccess} errorMsg={errorMsg} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Cinematic Transition Overlay */}
      <AnimatePresence>
        {isTransitioning && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.8 }}
            className="fixed inset-0 z-50 bg-[#05060A]"
          />
        )}
      </AnimatePresence>
    </div>
  )
}
