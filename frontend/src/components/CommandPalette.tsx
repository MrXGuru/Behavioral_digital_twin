import { useState, useEffect, useRef, memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Loader2, Sparkles, User as UserIcon, Send, X } from 'lucide-react'
import { API_BASE } from '../hooks/useApi'

const SUGGESTIONS = [
  'What habits changed this week?',
  'Why did I make that last decision?',
  'When is my twin most accurate?',
  'What patterns do you see in my focus sessions?',
]

async function askTwin(userId: string, question: string): Promise<string> {
  const res = await fetch(`${API_BASE}/chat/${encodeURIComponent(userId)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const body = await res.json()
  return body.answer ?? 'No answer received.'
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

interface CommandPaletteProps {
  userId: string
}

const CommandPalette = memo(function CommandPalette({ userId }: CommandPaletteProps) {
  const [open, setOpen] = useState(false)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [appThinking, setAppThinking] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    { id: 'welcome', role: 'assistant', content: "Hello! I'm your Behavioral Twin's neural core. I continuously analyze your habits, stress, and choices. Ask me anything to understand your patterns." }
  ])
  const inputRef = useRef<HTMLInputElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Toggle with Cmd+K or Ctrl+K or custom event
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setOpen((open) => !open)
      }
      if (e.key === 'Escape') {
        setOpen(false)
      }
    }
    const customOpen = () => setOpen(true)
    document.addEventListener('keydown', down)
    window.addEventListener('open-command-palette', customOpen)
    return () => {
      document.removeEventListener('keydown', down)
      window.removeEventListener('open-command-palette', customOpen)
    }
  }, [])

  // Auto-focus input when opened
  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [open])

  // Scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, loading])

  const submitQuery = async (text: string) => {
    const q = text.trim()
    if (!q || loading) return
    
    setInput('')
    const userMsgId = Date.now().toString()
    setMessages(prev => [...prev, { id: userMsgId, role: 'user', content: q }])
    setLoading(true)
    
    try {
      const answer = await askTwin(userId, q)
      setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), role: 'assistant', content: answer }])
    } catch (err) {
      setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), role: 'assistant', content: 'Cannot reach the explanation service right now.' }])
    } finally {
      setLoading(false)
    }
  }

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    submitQuery(input)
  }

  const handleClear = () => {
    setMessages([{ id: 'welcome', role: 'assistant', content: "Hello! I'm your Behavioral Twin's neural core. Let's explore your behavior." }])
  }

  useEffect(() => {
    const handleStart = () => setAppThinking(true)
    const handleEnd = () => setAppThinking(false)
    window.addEventListener('ai-processing-start', handleStart)
    window.addEventListener('ai-processing-end', handleEnd)
    return () => {
      window.removeEventListener('ai-processing-start', handleStart)
      window.removeEventListener('ai-processing-end', handleEnd)
    }
  }, [])

  return (
    <>
      {/* Ambient AI Companion Orb (Replaces Floating Action Button) */}
      <motion.button
        layoutId={!open ? "command-palette" : undefined}
        onClick={() => setOpen(true)}
        className={`fixed bottom-6 right-6 w-16 h-16 rounded-full flex items-center justify-center z-40 preserve-3d orb-wrapper cursor-pointer border-none bg-transparent ${appThinking || loading ? 'thinking' : ''}`}
        style={{ opacity: open ? 0 : 0.8 }}
        whileHover={{ scale: 1.05, opacity: 1 }}
        whileTap={{ scale: 0.95 }}
      >
        <div className="relative w-full h-full flex items-center justify-center preserve-3d">
          <div className="w-10 h-10 rounded-full orb-core relative z-10" />
          
          {/* Nodes */}
          <div className="orb-ring" style={{ '--orbit-radius': '24px', '--hover-radius': '16px', '--spin-fast': '2s', animationDirection: 'normal', animationDuration: '6s' } as React.CSSProperties}>
            <div className="orb-line" style={{ animationDelay: '0s' }} />
            <div className="orb-node" />
          </div>
          <div className="orb-ring" style={{ '--orbit-radius': '28px', '--hover-radius': '18px', '--spin-fast': '1.5s', animationDirection: 'reverse', animationDuration: '8s' } as React.CSSProperties}>
            <div className="orb-line" style={{ animationDelay: '1s' }} />
            <div className="orb-node" />
          </div>
          <div className="orb-ring" style={{ '--orbit-radius': '32px', '--hover-radius': '20px', '--spin-fast': '2.5s', animationDirection: 'normal', animationDuration: '10s' } as React.CSSProperties}>
            <div className="orb-line" style={{ animationDelay: '2s' }} />
            <div className="orb-node" />
          </div>
          <div className="orb-ring" style={{ '--orbit-radius': '22px', '--hover-radius': '14px', '--spin-fast': '1.8s', animationDirection: 'reverse', animationDuration: '7s' } as React.CSSProperties}>
            <div className="orb-line" style={{ animationDelay: '1.5s' }} />
            <div className="orb-node" />
          </div>
        </div>
      </motion.button>

      <AnimatePresence>
        {open && (
          <div className="fixed inset-0 z-50 flex items-start sm:items-center justify-center p-4 pt-16 sm:p-6 perspective-1000">
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setOpen(false)}
              className="absolute inset-0 bg-black/60 backdrop-blur-md"
            />

            <motion.div
              layoutId="command-palette"
              initial={{ opacity: 0, scale: 0.85, rotateX: -15, y: 40 }}
              animate={{ opacity: 1, scale: 1, rotateX: 0, y: 0 }}
              exit={{ opacity: 0, scale: 0.85, rotateX: 10, y: 20, transition: { duration: 0.25, ease: "easeIn" } }}
              transition={{ type: "spring", stiffness: 250, damping: 20 }}
              className="relative w-full max-w-3xl h-[80vh] sm:h-[600px] flex flex-col bg-[#0a0a0a]/80 backdrop-blur-3xl border border-[#ffffff1a] 
                shadow-[0_0_50px_rgba(99,102,241,0.2)] rounded-2xl overflow-hidden preserve-3d glass-card"
            >
              {/* Header */}
              <div className="flex items-center justify-between px-5 py-4 border-b border-[#ffffff14] bg-[#141414]/50">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center">
                    <Sparkles className="w-4 h-4 text-indigo-400" />
                  </div>
                  <div>
                    <h2 className="text-sm font-semibold tracking-tight text-[#ededed]">Twin Intelligence</h2>
                    <p className="text-xs text-[#a1a1aa]">Neural Insight Chat</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button 
                    onClick={handleClear}
                    className="text-xs px-3 py-1.5 rounded-lg bg-[#ffffff0a] hover:bg-[#ffffff14] border border-[#ffffff0a] transition-colors text-[#a1a1aa]"
                  >
                    Clear Chat
                  </button>
                  <button 
                    onClick={() => setOpen(false)}
                    className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-[#ffffff14] transition-colors"
                  >
                    <X className="w-4 h-4 text-[#a1a1aa]" />
                  </button>
                </div>
              </div>

              {/* Chat Scroll Area */}
              <div ref={scrollRef} className="flex-1 overflow-y-auto p-5 scroll-smooth">
                <div className="flex flex-col gap-5">
                  {messages.map((msg) => (
                    <motion.div 
                      key={msg.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                    >
                      {/* Avatar */}
                      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-md ${
                        msg.role === 'user' 
                          ? 'bg-gradient-to-br from-slate-700 to-slate-900 border border-[#ffffff1a]' 
                          : 'bg-gradient-to-br from-indigo-500 to-purple-600 border border-indigo-400/30'
                      }`}>
                        {msg.role === 'user' ? <UserIcon className="w-4 h-4 text-white" /> : <Sparkles className="w-4 h-4 text-white" />}
                      </div>
                      
                      {/* Bubble */}
                      <div className={`max-w-[80%] rounded-2xl px-5 py-3.5 text-[15px] leading-relaxed shadow-sm ${
                        msg.role === 'user'
                          ? 'bg-[#1f1f1f] text-[#ededed] border border-[#ffffff14] rounded-tr-sm'
                          : 'bg-indigo-500/10 text-[#d4d4d8] border border-indigo-500/20 rounded-tl-sm'
                      }`}>
                        {msg.content}
                      </div>
                    </motion.div>
                  ))}

                  {/* Loading Indicator */}
                  {loading && (
                    <motion.div 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex gap-3"
                    >
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-md">
                        <Loader2 className="w-4 h-4 text-white animate-spin" />
                      </div>
                      <div className="rounded-2xl rounded-tl-sm px-5 py-3.5 bg-indigo-500/5 border border-indigo-500/10 text-[#a1a1aa] flex items-center gap-2">
                        <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"></span>
                        <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                        <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
                      </div>
                    </motion.div>
                  )}
                </div>

                {/* Suggestions if chat is fresh */}
                {messages.length === 1 && !loading && (
                  <div className="mt-8">
                    <p className="text-xs font-medium text-[#71717a] uppercase tracking-wider mb-3 px-1">Suggested Questions</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {SUGGESTIONS.map(q => (
                        <button
                          key={q}
                          onClick={() => submitQuery(q)}
                          className="text-left p-3 rounded-xl bg-[#141414] hover:bg-[#1f1f1f] border border-[#ffffff0a] hover:border-[#ffffff1a] transition-all group"
                        >
                          <p className="text-sm text-[#a1a1aa] group-hover:text-[#ededed]">{q}</p>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Input Area */}
              <div className="p-4 bg-[#141414]/80 border-t border-[#ffffff14] backdrop-blur-md">
                <form onSubmit={onSubmit} className="relative flex items-center">
                  <input
                    ref={inputRef}
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask about your behavioral patterns..."
                    className="w-full bg-[#0a0a0a] border border-[#ffffff1a] focus:border-indigo-500/50 rounded-xl py-4 pl-5 pr-14 text-[15px] text-[#ededed] placeholder-[#71717a] focus:outline-none focus:ring-1 focus:ring-indigo-500/50 transition-all shadow-inner"
                    autoComplete="off"
                    disabled={loading}
                  />
                  <button
                    type="submit"
                    disabled={!input.trim() || loading}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-50 disabled:hover:bg-indigo-600 transition-colors shadow-md"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </form>
                <div className="mt-2 text-center flex items-center justify-center gap-2">
                  <p className="text-[10px] text-[#71717a]">Press <kbd className="px-1.5 py-0.5 rounded bg-[#1f1f1f] border border-[#ffffff14] font-sans">⌘K</kbd> to open/close</p>
                  <span className="text-[#3f3f46]">&bull;</span>
                  <p className="text-[10px] text-[#71717a]">Predictions are powered strictly by the ML engine</p>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  )
})

export default CommandPalette
