import { useCallback, useRef } from 'react'

type SoundType = 'hover' | 'click' | 'success'

export function useSound() {
  const audioCtx = useRef<AudioContext | null>(null)

  const initAudio = () => {
    if (!audioCtx.current) {
      audioCtx.current = new (window.AudioContext || (window as any).webkitAudioContext)()
    }
  }

  const playSound = useCallback((type: SoundType) => {
    initAudio()
    if (!audioCtx.current) return

    const ctx = audioCtx.current
    const osc = ctx.createOscillator()
    const gainNode = ctx.createGain()

    osc.connect(gainNode)
    gainNode.connect(ctx.destination)

    const now = ctx.currentTime

    if (type === 'hover') {
      // Soft electronic tick
      osc.type = 'sine'
      osc.frequency.setValueAtTime(800, now)
      osc.frequency.exponentialRampToValueAtTime(1200, now + 0.05)
      gainNode.gain.setValueAtTime(0, now)
      gainNode.gain.linearRampToValueAtTime(0.05, now + 0.01)
      gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.05)
      osc.start(now)
      osc.stop(now + 0.05)
    } else if (type === 'click') {
      // Premium soft click
      osc.type = 'triangle'
      osc.frequency.setValueAtTime(400, now)
      osc.frequency.exponentialRampToValueAtTime(100, now + 0.1)
      gainNode.gain.setValueAtTime(0, now)
      gainNode.gain.linearRampToValueAtTime(0.1, now + 0.02)
      gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.1)
      osc.start(now)
      osc.stop(now + 0.1)
    } else if (type === 'success') {
      // Ethereal success chime (chord)
      const osc2 = ctx.createOscillator()
      const osc3 = ctx.createOscillator()
      
      osc.type = 'sine'
      osc2.type = 'sine'
      osc3.type = 'sine'
      
      osc.frequency.setValueAtTime(523.25, now) // C5
      osc2.frequency.setValueAtTime(659.25, now) // E5
      osc3.frequency.setValueAtTime(783.99, now) // G5

      osc2.connect(gainNode)
      osc3.connect(gainNode)

      gainNode.gain.setValueAtTime(0, now)
      gainNode.gain.linearRampToValueAtTime(0.15, now + 0.1)
      gainNode.gain.exponentialRampToValueAtTime(0.001, now + 1.5)

      osc.start(now)
      osc2.start(now)
      osc3.start(now)
      
      osc.stop(now + 1.5)
      osc2.stop(now + 1.5)
      osc3.stop(now + 1.5)
    }
  }, [])

  return { playSound }
}
