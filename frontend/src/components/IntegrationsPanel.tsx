import { useState, useEffect, memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Github, Calendar, Music, Activity, Slack, Plus, CheckCircle, Loader2, X } from 'lucide-react'
import { connectIntegration, syncIntegration } from '../hooks/useApi'

const INTEGRATIONS = [
  { id: 'github', name: 'GitHub', icon: Github, color: '#ededed', bg: '#1f1f1f', desc: 'Sync commits & PRs as focus metrics', url: '/?oauth_mock=true&app=GitHub' },
  { id: 'calendar', name: 'Google Calendar', icon: Calendar, color: '#4285F4', bg: '#4285F415', desc: 'Sync meetings to context', url: '/?oauth_mock=true&app=Google%20Calendar' },
  { id: 'spotify', name: 'Spotify', icon: Music, color: '#1DB954', bg: '#1DB95415', desc: 'Sync music to mood tracking', url: '/?oauth_mock=true&app=Spotify' },
  { id: 'health', name: 'Apple Health', icon: Activity, color: '#FF2D55', bg: '#FF2D5515', desc: 'Sync sleep & stress data', url: '/?oauth_mock=true&app=Apple%20Health' },
  { id: 'slack', name: 'Slack', icon: Slack, color: '#E01E5A', bg: '#E01E5A15', desc: 'Sync communication overhead', url: '/?oauth_mock=true&app=Slack' },
]

interface IntegrationsPanelProps {
  onSynced?: () => void
}

export default memo(function IntegrationsPanel({ onSynced }: IntegrationsPanelProps) {
  const [connectedApps, setConnectedApps] = useState<Record<string, boolean>>({})
  const [syncing, setSyncing] = useState<string | null>(null)
  const [connecting, setConnecting] = useState<string | null>(null)
  
  // Modal state
  const [activeModal, setActiveModal] = useState<typeof INTEGRATIONS[0] | null>(null)
  const [tokenInput, setTokenInput] = useState('')

  // Load saved integrations on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('twin_integrations')
      if (saved) setConnectedApps(JSON.parse(saved))
    } catch (e) {}
  }, [])

  const handleConnectSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!activeModal || !tokenInput.trim()) return

    const appId = activeModal.id
    setConnecting(appId)
    try {
      await connectIntegration(appId, tokenInput)
      
      const newConnected = { ...connectedApps, [appId]: true }
      setConnectedApps(newConnected)
      localStorage.setItem('twin_integrations', JSON.stringify(newConnected))
      
      // Trigger sync animation
      setConnecting(null)
      setActiveModal(null)
      setTokenInput('')
      setSyncing(appId)
      await syncIntegration(appId)
      if (onSynced) onSynced()
    } catch (err) {
      console.error(err)
    } finally {
      setConnecting(null)
      setTimeout(() => setSyncing(null), 3000)
    }
  }

  const handleDisconnect = (id: string) => {
    const newConnected = { ...connectedApps }
    delete newConnected[id]
    setConnectedApps(newConnected)
    localStorage.setItem('twin_integrations', JSON.stringify(newConnected))
  }

  return (
    <div className="glass-card p-6 border border-[#ffffff14]">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-title">Data Sources</h3>
          <p className="text-body">Connect apps to feed your twin real-time data</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {INTEGRATIONS.map((app) => {
          const isConnected = !!connectedApps[app.id]
          const isSyncing = syncing === app.id
          const Icon = app.icon

          return (
            <motion.div
              key={app.id}
              whileHover={{ y: -2 }}
              className={`relative overflow-hidden flex flex-col p-4 rounded-xl border transition-all duration-300 ${
                isConnected 
                  ? 'bg-[#1a1a1a] border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.05)]' 
                  : 'bg-[#141414] border-[#ffffff14] hover:border-[#ffffff2a]'
              }`}
            >
              {isSyncing && (
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-indigo-500 shimmer" />
              )}
              
              <div className="flex items-start justify-between mb-3">
                <div 
                  className="w-10 h-10 rounded-lg flex items-center justify-center border border-[#ffffff0a]"
                  style={{ backgroundColor: app.bg }}
                >
                  <Icon className="w-5 h-5" style={{ color: app.color }} />
                </div>
                
                {isConnected ? (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20">
                    {isSyncing ? (
                      <><Loader2 className="w-3 h-3 text-indigo-400 animate-spin" /><span className="text-[10px] uppercase tracking-wider font-bold text-indigo-400">Syncing</span></>
                    ) : (
                      <><CheckCircle className="w-3 h-3 text-indigo-400" /><span className="text-[10px] uppercase tracking-wider font-bold text-indigo-400">Active</span></>
                    )}
                  </div>
                ) : (
                  <button 
                    onClick={() => setActiveModal(app)}
                    disabled={connecting !== null}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#1f1f1f] border border-[#ffffff14] hover:bg-[#ffffff0a] transition-colors text-xs font-medium text-[#ededed] disabled:opacity-50"
                  >
                    {connecting === app.id ? (
                      <><Loader2 className="w-3 h-3 animate-spin" /> Authorizing...</>
                    ) : (
                      <><Plus className="w-3 h-3" /> Connect</>
                    )}
                  </button>
                )}
              </div>
              
              <h4 className="text-sm font-semibold text-[#ededed] mb-1">{app.name}</h4>
              <p className="text-xs text-[#a1a1aa] mb-4 flex-grow">{app.desc}</p>

              {isConnected && (
                <div className="flex justify-between items-center mt-auto pt-3 border-t border-[#ffffff0a]">
                  <span className="text-[10px] text-[#71717a]">Last sync: just now</span>
                  <button 
                    onClick={() => handleDisconnect(app.id)}
                    className="text-[10px] font-medium text-[#a1a1aa] hover:text-rose-400 transition-colors"
                  >
                    Disconnect
                  </button>
                </div>
              )}
            </motion.div>
          )
        })}
      </div>

      {/* Token Modal */}
      <AnimatePresence>
        {activeModal && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => !connecting && setActiveModal(null)}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.95, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 10 }}
                onClick={e => e.stopPropagation()}
                className="bg-[#141414] border border-[#ffffff14] rounded-2xl w-full max-w-md shadow-2xl overflow-hidden"
              >
                <div className="flex items-center justify-between p-4 border-b border-[#ffffff0a]">
                  <div className="flex items-center gap-2">
                    <activeModal.icon className="w-5 h-5" style={{ color: activeModal.color }} />
                    <h3 className="text-sm font-semibold text-[#ededed]">Connect {activeModal.name}</h3>
                  </div>
                  <button 
                    onClick={() => setActiveModal(null)}
                    disabled={connecting !== null}
                    className="p-1.5 rounded-lg hover:bg-[#ffffff0a] text-[#a1a1aa] transition-colors disabled:opacity-50"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                
                <div className="p-6">
                  <div className="mb-6 flex gap-3 p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs leading-relaxed">
                    <CheckCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    <p>Provide a Personal Access Token. This uses a real connection to fetch your data and securely injects it into your Twin's memory.</p>
                  </div>
                  
                  <form onSubmit={handleConnectSubmit}>
                    <label className="block text-xs font-medium text-[#a1a1aa] mb-2 uppercase tracking-wider">Access Token</label>
                    <input
                      type="password"
                      autoFocus
                      required
                      value={tokenInput}
                      onChange={e => setTokenInput(e.target.value)}
                      placeholder={`ghp_xxxxxxxxxxxx...`}
                      disabled={connecting !== null}
                      className="w-full bg-[#0a0a0a] border border-[#ffffff14] rounded-lg px-4 py-2.5 text-sm text-[#ededed] focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/50 transition-all placeholder:text-[#3f3f46] mb-6"
                    />
                    
                    <div className="flex gap-3 justify-end">
                      <button
                        type="button"
                        onClick={() => setActiveModal(null)}
                        disabled={connecting !== null}
                        className="px-4 py-2 text-sm font-medium text-[#a1a1aa] hover:text-[#ededed] transition-colors disabled:opacity-50"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        disabled={connecting !== null || !tokenInput.trim()}
                        className="btn-primary"
                      >
                        {connecting === activeModal.id ? (
                          <><Loader2 className="w-4 h-4 animate-spin mr-2" /> Connecting...</>
                        ) : (
                          'Connect Account'
                        )}
                      </button>
                    </div>
                  </form>
                </div>
              </motion.div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
})
