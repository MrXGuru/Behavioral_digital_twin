/**
 * BehavioralTwinDashboard — single-file root component.
 *
 * Key fix: `loading` is only true on the very first fetch for a userId.
 * After a decision is logged and `refetch()` is called, `refreshing=true`
 * is set instead — the existing dashboard stays fully visible with just
 * a subtle top-bar indicator, preventing any blank-page flash.
 */
import { useState, useEffect, memo, Component, type ReactNode, type ErrorInfo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Loader2, AlertCircle, Download, LogOut, User, Trash2, Bell, Search, Cloud, CloudLightning, CheckCircle2 } from 'lucide-react'
import AuthScreen from './components/auth/AuthScreen'

import { useDashboard, API_BASE, deleteTwin } from './hooks/useApi'
import { useTwinXP } from './hooks/useTwinXP'

import MaturityBanner from './components/MaturityBanner'
import LogDecisionPanel from './components/LogDecisionPanel'
import StatsGrid from './components/StatsGrid'
import TimelineChart from './components/TimelineChart'
import DecisionsTable from './components/DecisionsTable'
import DriftPanel from './components/DriftPanel'
import RetrainPanel from './components/RetrainPanel'
import PredictPanel from './components/PredictPanel'
import CommandPalette from './components/CommandPalette'
import FocusStreaks from './components/FocusStreaks'
import IntegrationsPanel from './components/IntegrationsPanel'
import AIBriefingPanel from './components/AIBriefingPanel'
import QuickLogFAB from './components/QuickLogFAB'
import TiltCard from './components/TiltCard'
import LiveClock from './components/LiveClock'

// ---------------------------------------------------------------------------
// Error boundary — catches component-level crashes and shows a friendly message
// instead of a blank white screen
// ---------------------------------------------------------------------------

interface EBState { hasError: boolean; message: string }

class ErrorBoundary extends Component<{ children: ReactNode }, EBState> {
  state: EBState = { hasError: false, message: '' }

  static getDerivedStateFromError(err: Error): EBState {
    return { hasError: true, message: err.message }
  }

  componentDidCatch(err: Error, info: ErrorInfo) {
    console.error('Dashboard render error:', err, info)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center py-24 text-center px-4">
          <div className="w-14 h-14 rounded-2xl bg-rose-500/10 flex items-center justify-center mb-4">
            <AlertCircle className="w-7 h-7 text-rose-400" />
          </div>
          <p className="text-lg font-semibold text-slate-200 mb-2">Something went wrong</p>
          <p className="text-sm text-slate-400 max-w-sm mb-6">{this.state.message}</p>
          <button
            onClick={() => this.setState({ hasError: false, message: '' })}
            className="btn-primary text-white"
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

// ---------------------------------------------------------------------------
// Header — memoized
// ---------------------------------------------------------------------------

interface HeaderProps {
  userId: string
  onSignOut: () => void
  connected: boolean
  refreshing: boolean
  totalLogs?: number
  level?: number
}

const Header = memo(function Header({ userId, onSignOut, connected, refreshing, totalLogs, level }: HeaderProps) {
  const [showProfileMenu, setShowProfileMenu] = useState(false)
  const [showNotifications, setShowNotifications] = useState(false)
  const [userProfile, setUserProfile] = useState<{name?: string, email?: string, picture?: string} | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)
  
  useEffect(() => {
    try {
      const p = localStorage.getItem('user_profile')
      if (p) setUserProfile(JSON.parse(p))
    } catch(e){}
  }, [])

  const handleExport = async () => {
    try {
      const res = await fetch(`${API_BASE}/report/${encodeURIComponent(userId)}`)
      if (!res.ok) throw new Error('Export failed')
      const blob = await res.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `weekly_report_${userId}.md`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (e) {
      alert("Failed to export report")
      console.error(e)
    }
  }

  const handleResetTwin = async () => {
    if (!window.confirm("Are you sure you want to wipe all your data and start over? This cannot be undone.")) return
    
    setIsDeleting(true)
    try {
      await deleteTwin(userId)
      localStorage.removeItem(`twin_user_xp_${userId}`)
      localStorage.removeItem('twin_user_xp') // Wipe global legacy key too
      window.location.reload()
    } catch (err) {
      console.error(err)
      alert("Failed to reset twin.")
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <header 
      className="sticky top-0 z-50 bg-[#0a0a0a]/90 backdrop-blur-xl border-b border-[#ffffff14] shadow-[0_4px_30px_rgba(0,0,0,0.5)]"
    >
      {refreshing && (
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-500 shimmer z-50" />
      )}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center h-16 gap-4">
          {/* Logo */}
          <div className="flex items-center gap-3 relative group mr-4">
            <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-600 blur-xl opacity-0 group-hover:opacity-40 transition-opacity duration-700" />
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg relative z-10">
              <Brain className="w-5 h-5 text-white animate-breathe" />
            </div>
            <div className="relative overflow-hidden z-10 hidden sm:block">
              <h1 className="text-lg font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-indigo-600 drop-shadow-md whitespace-nowrap animate-shimmer-sweep">
                Behavioral Twin
              </h1>
            </div>
          </div>

          {/* Search/Command Palette trigger */}
          <div className="hidden md:flex flex-1 max-w-sm">
            <button 
              type="button" 
              onClick={() => window.dispatchEvent(new CustomEvent('open-command-palette'))} 
              className="w-full flex items-center gap-2 px-4 py-2 bg-[#141414] hover:bg-[#1f1f1f] border border-[#ffffff14] hover:border-[#ffffff2a] rounded-xl text-sm text-[#71717a] transition-all shadow-inner group"
            >
              <Search className="w-4 h-4 text-[#a1a1aa] group-hover:text-indigo-400 transition-colors" />
              <span className="flex-1 text-left">Search or ask AI...</span>
              <kbd className="hidden lg:inline-block px-1.5 py-0.5 rounded bg-[#27272a] text-[10px] text-[#a1a1aa] border border-[#3f3f46] shadow-sm font-medium">⌘K</kbd>
            </button>
          </div>

          <div className="flex-1 flex justify-end md:hidden">
             <button 
                type="button" 
                onClick={() => window.dispatchEvent(new CustomEvent('open-command-palette'))}
                className="w-10 h-10 rounded-full bg-[#141414] border border-[#ffffff14] flex items-center justify-center hover:bg-[#1f1f1f] transition-colors"
              >
                <Search className="w-4 h-4 text-[#a1a1aa]" />
              </button>
          </div>

          {/* Actions & Status */}
          <div className="flex items-center gap-3 sm:gap-4 ml-auto">
            {/* Live Clock */}
            <div className="hidden lg:flex items-center justify-center px-4 py-1.5 bg-[#141414] border border-[#ffffff14] rounded-xl shadow-inner">
              <LiveClock className="text-emerald-400 font-mono text-sm tracking-widest drop-shadow-[0_0_8px_rgba(52,211,153,0.5)]" />
            </div>

            {/* Sync Status Animation */}
            <div className="flex items-center justify-center w-10 h-10 rounded-full bg-[#141414] border border-[#ffffff14] shadow-inner" title={refreshing ? "Syncing..." : connected ? "Connected" : "Offline"}>
              <AnimatePresence mode="wait">
                {refreshing ? (
                  <motion.div key="refresh" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.2 }}>
                    <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
                  </motion.div>
                ) : connected ? (
                  <motion.div key="connected" initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0, opacity: 0 }} transition={{ duration: 0.2 }}>
                    <Cloud className="w-4 h-4 text-emerald-400" />
                  </motion.div>
                ) : (
                  <motion.div key="offline" initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0, opacity: 0 }} transition={{ duration: 0.2 }}>
                    <CloudLightning className="w-4 h-4 text-rose-400" />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Notifications */}
            <div className="relative">
              <button 
                type="button" 
                onClick={() => setShowNotifications(!showNotifications)}
                className="relative flex items-center justify-center w-10 h-10 rounded-full bg-[#141414] border border-[#ffffff14] hover:bg-[#1f1f1f] transition-colors shadow-inner"
              >
                <Bell className="w-4 h-4 text-[#a1a1aa]" />
                <span className="absolute top-2 right-2 w-2 h-2 bg-rose-500 rounded-full border-2 border-[#141414] shadow-[0_0_8px_rgba(244,63,94,0.8)]"></span>
              </button>
              
              <AnimatePresence>
                {showNotifications && (
                  <>
                    <div className="fixed inset-0 z-40" onClick={() => setShowNotifications(false)} />
                    <motion.div 
                      initial={{ opacity: 0, scale: 0.95, y: 10 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95, y: 10 }}
                      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
                      className="absolute right-0 mt-3 w-80 rounded-2xl bg-[#0a0a0a]/90 backdrop-blur-2xl border border-[#ffffff1a] shadow-2xl z-50 p-2 flex flex-col gap-1 overflow-hidden"
                    >
                      <div className="px-3 py-3 border-b border-[#ffffff14] mb-2 flex items-center justify-between">
                        <span className="text-sm font-semibold text-[#ededed]">Notifications</span>
                        <span className="text-[10px] font-bold bg-indigo-500 text-white px-2 py-0.5 rounded-full">2 New</span>
                      </div>
                      <div className="flex items-start gap-3 p-2 hover:bg-[#ffffff0a] rounded-xl transition-colors cursor-pointer group">
                        <div className="w-10 h-10 rounded-full bg-emerald-500/10 flex items-center justify-center flex-shrink-0 group-hover:bg-emerald-500/20 transition-colors">
                          <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-[#ededed]">Model Synced</p>
                          <p className="text-xs text-[#a1a1aa] mt-0.5">Your behavioral twin has processed your recent logs.</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-3 p-2 hover:bg-[#ffffff0a] rounded-xl transition-colors cursor-pointer group">
                        <div className="w-10 h-10 rounded-full bg-indigo-500/10 flex items-center justify-center flex-shrink-0 group-hover:bg-indigo-500/20 transition-colors">
                          <Brain className="w-5 h-5 text-indigo-400" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-[#ededed]">New Insight Available</p>
                          <p className="text-xs text-[#a1a1aa] mt-0.5">We found a new focus pattern related to morning work.</p>
                        </div>
                      </div>
                    </motion.div>
                  </>
                )}
              </AnimatePresence>
            </div>

            {/* Profile Dropdown */}
            <div className="relative">
              <button 
                type="button" 
                onClick={() => setShowProfileMenu(!showProfileMenu)}
                className="w-10 h-10 rounded-full overflow-hidden border border-[#ffffff14] hover:border-indigo-500/50 transition-colors focus:outline-none bg-[#141414] flex items-center justify-center active:scale-95"
                title="View Profile"
              >
                {userProfile?.picture ? (
                  <img src={userProfile.picture} alt="Profile" className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                ) : (
                  <span className="flex items-center justify-center w-full h-full">
                    <User className="w-4 h-4 text-[#ededed]" />
                  </span>
                )}
              </button>
              
              <AnimatePresence>
                {showProfileMenu && (
                  <>
                    <div className="fixed inset-0 z-40" onClick={() => setShowProfileMenu(false)} />
                    <motion.div 
                      initial={{ opacity: 0, scale: 0.95, y: 10 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95, y: 10 }}
                      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
                      className="absolute right-0 mt-3 w-64 rounded-2xl bg-[#0a0a0a]/90 backdrop-blur-2xl border border-[#ffffff1a] shadow-2xl z-50 p-2 flex flex-col gap-1 overflow-hidden"
                    >
                      <div className="px-3 py-3 border-b border-[#ffffff14] mb-2">
                        <p className="text-sm font-medium text-[#ededed] truncate">{userProfile?.name || 'User'}</p>
                        <p className="text-xs text-[#a1a1aa] truncate">{userProfile?.email || userId}</p>
                      </div>
                      
                      {totalLogs !== undefined && level !== undefined && (
                        <div className="md:hidden px-3 py-2 mb-1 flex items-center justify-between bg-indigo-500/10 rounded-lg">
                          <span className="text-xs font-medium text-indigo-400">Level {level} Twin</span>
                          <span className="text-[10px] text-indigo-300">{totalLogs} logs</span>
                        </div>
                      )}

                      <button type="button" onClick={handleExport} className="w-full text-left px-3 py-2.5 text-sm text-[#ededed] hover:bg-[#ffffff0a] rounded-xl flex items-center gap-3 transition-colors">
                        <Download className="w-4 h-4 text-[#a1a1aa]" />
                        Export Report
                      </button>
                      <button type="button" onClick={handleResetTwin} disabled={isDeleting} className="w-full text-left px-3 py-2.5 text-sm text-rose-400 hover:bg-[#ffffff0a] rounded-xl flex items-center gap-3 transition-colors">
                        {isDeleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4 text-rose-500" />}
                        Factory Reset
                      </button>
                      <div className="h-[1px] bg-[#ffffff14] my-1" />
                      <button type="button" onClick={onSignOut} className="w-full text-left px-3 py-2.5 text-sm text-[#ededed] hover:bg-[#ffffff0a] rounded-xl flex items-center gap-3 transition-colors">
                        <LogOut className="w-4 h-4 text-[#a1a1aa]" />
                        Sign out
                      </button>
                    </motion.div>
                  </>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
})

// ---------------------------------------------------------------------------
// Main dashboard body — memoized so it doesn't re-render on header state changes
// ---------------------------------------------------------------------------

interface DashboardBodyProps {
  userId: string
  data: ReturnType<typeof useDashboard>['data']
  refetch: () => void
  xp: number
  level: number
  addXp: (amount: number) => void
}

const DashboardBody = memo(function DashboardBody({ userId, data, refetch, xp, level, addXp }: DashboardBodyProps) {
  const maturity = data?.data_maturity ?? null

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >


      {/* Maturity banner */}
      {maturity && maturity.status === 'learning' && (
        <TiltCard intensity={8}>
          <MaturityBanner maturity={maturity} />
        </TiltCard>
      )}

      {/* Activity Gamified Centerpiece */}
      <ErrorBoundary>
        <TiltCard intensity={3}>
          <FocusStreaks 
            heatmap={data?.heatmap || []} 
            xp={xp}
            level={level}
            totalLogs={data?.decisions?.length || 0}
          />
        </TiltCard>
      </ErrorBoundary>

      {/* Top row: Log Decision + Stats/Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="flex flex-col gap-6">
          <ErrorBoundary>
            <LogDecisionPanel 
              userId={userId} 
              onLogged={refetch} 
              totalLogs={data?.decisions?.length || 0}
              xp={xp}
              level={level}
              addXp={addXp}
            />
          </ErrorBoundary>
        </div>

        <div className="lg:col-span-2 space-y-6">
          <ErrorBoundary>
            <>
              <TiltCard intensity={5}>
                <StatsGrid data={data!} />
              </TiltCard>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <TiltCard intensity={10}>
                  <PredictPanel userId={userId} />
                </TiltCard>
                <TiltCard intensity={10}>
                  <RetrainPanel userId={userId} onRetrained={refetch} />
                </TiltCard>
              </div>
            </>
          </ErrorBoundary>
        </div>
      </div>

      {/* Integrations Row */}
      <ErrorBoundary>
        <TiltCard intensity={6}>
          <IntegrationsPanel onSynced={refetch} />
        </TiltCard>
      </ErrorBoundary>

      {/* Autonomous AI Briefing - Moved down here */}
      <ErrorBoundary>
        <TiltCard intensity={8}>
          <AIBriefingPanel userId={userId} />
        </TiltCard>
      </ErrorBoundary>

      {/* Charts — Always visible to show empty states */}
      <ErrorBoundary>
        <>
          <TiltCard intensity={4}>
            <TimelineChart timeline={data!.timeline} />
          </TiltCard>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <TiltCard intensity={5}>
                <DecisionsTable decisions={data!.decisions} />
              </TiltCard>
            </div>
            <div>
              <TiltCard intensity={8}>
                <DriftPanel events={data!.driftEvents} />
              </TiltCard>
            </div>
          </div>
        </>
      </ErrorBoundary>

      {/* Global Command Palette replaces inline ChatPanel */}
      <ErrorBoundary>
        <CommandPalette userId={userId} />
      </ErrorBoundary>
    </motion.div>
  )
})

// ---------------------------------------------------------------------------
// Root component
// ---------------------------------------------------------------------------

export default function BehavioralTwinDashboard() {
  const [userId, setUserId] = useState<string | null>(null)
  
  useEffect(() => {
    const saved = localStorage.getItem('summerr_user_id')
    if (saved) setUserId(saved)
  }, [])

  const handleLogin = (id: string) => {
    localStorage.setItem('summerr_user_id', id)
    setUserId(id)
  }

  const handleSignOut = () => {
    localStorage.removeItem('summerr_user_id')
    localStorage.removeItem('access_token')
    localStorage.removeItem('user_profile')
    setUserId(null)
  }

  return (
    <AnimatePresence mode="wait">
      {!userId ? (
        <AuthScreen key="auth" onLogin={handleLogin} />
      ) : (
        <DashboardContainer key="dash" userId={userId} onSignOut={handleSignOut} />
      )}
    </AnimatePresence>
  )
}

function DashboardContainer({ userId, onSignOut }: { userId: string, onSignOut: () => void }) {
  const { data, loading, refreshing, error, refetch } = useDashboard(userId)
  const totalLogs = data?.decisions?.length || 0
  const { xp, level, addXp } = useTwinXP(userId, totalLogs)

  return (
    <div className="min-h-screen pb-16">
      <Header
        userId={userId}
        onSignOut={onSignOut}
        connected={!error}
        refreshing={refreshing}
        totalLogs={totalLogs}
        level={level}
      />

      <motion.main 
        layoutId="app-card"
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6 relative z-10"
      >
        {/* Full-screen spinner only on initial load for a userId */}
        {loading && !data && (
          <div className="flex flex-col items-center justify-center py-32">
            <Loader2 className="w-6 h-6 text-[#ededed] animate-spin mb-4" />
            <p className="text-body">
              Loading <span className="text-[#ededed] font-medium">{userId}</span>
            </p>
          </div>
        )}

        {/* Error — only shown when we have no data at all */}
        {!loading && error && !data && (
          <div className="flex flex-col items-center justify-center py-32">
            <div className="w-16 h-16 rounded-2xl bg-rose-500/10 flex items-center justify-center mb-4">
              <AlertCircle className="w-8 h-8 text-rose-400" />
            </div>
            <p className="text-lg font-semibold text-slate-200">Cannot reach API</p>
            <p className="text-sm text-slate-400 mt-1 mb-6 max-w-md text-center">{error}</p>
            <button onClick={refetch} className="btn-primary text-white">Retry</button>
          </div>
        )}

        {/* Dashboard — shown whenever data exists, even while refreshing */}
        {data && (
          <ErrorBoundary>
            <DashboardBody 
              userId={userId} 
              data={data} 
              refetch={refetch} 
              xp={xp}
              level={level}
              addXp={addXp}
            />
            <QuickLogFAB userId={userId} onLog={refetch} addXp={addXp} />
          </ErrorBoundary>
        )}
      </motion.main>
    </div>
  )
}
