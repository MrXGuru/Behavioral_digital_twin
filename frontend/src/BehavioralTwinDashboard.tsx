/**
 * BehavioralTwinDashboard — single-file root component.
 *
 * Key fix: `loading` is only true on the very first fetch for a userId.
 * After a decision is logged and `refetch()` is called, `refreshing=true`
 * is set instead — the existing dashboard stays fully visible with just
 * a subtle top-bar indicator, preventing any blank-page flash.
 */
import { useState, useEffect, memo, Component, type ReactNode, type ErrorInfo } from 'react'
import { motion } from 'framer-motion'
import { Brain, Loader2, AlertCircle, Download, LogOut, Share2, User, Trash2 } from 'lucide-react'
import { GoogleLogin } from '@react-oauth/google'

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
import HeatmapCalendar from './components/HeatmapCalendar'
import IntegrationsPanel from './components/IntegrationsPanel'
import AIBriefingPanel from './components/AIBriefingPanel'
import QuickLogFAB from './components/QuickLogFAB'
import TiltCard from './components/TiltCard'

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
  const [userProfile, setUserProfile] = useState<{name?: string, email?: string, picture?: string} | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)
  
  useEffect(() => {
    try {
      const p = localStorage.getItem('user_profile')
      if (p) setUserProfile(JSON.parse(p))
    } catch(e){}
  }, [])

  const handleExport = () => {
    window.location.href = `${API_BASE}/report/${encodeURIComponent(userId)}`
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

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'My Behavioral Twin',
          text: `I'm using the AI Behavioral Twin to track my habits and understand my focus patterns!`,
          url: window.location.href,
        })
      } catch (err) {
        console.log('User canceled share or error occurred.')
      }
    } else {
      alert('Native sharing is not supported on this browser. Copy the URL to share!')
    }
  }

  return (
    <header 
      className="sticky top-0 z-50 bg-[#0a0a0a] relative after:absolute after:bottom-0 after:inset-x-0 after:h-[1px] after:bg-gradient-to-r after:from-orange-500/20 after:to-purple-600/20 shadow-[0_4px_30px_rgba(0,0,0,0.5)]"
      style={{ transform: 'translateZ(10px)' }}
    >
      {refreshing && (
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-[#ededed] shimmer" />
      )}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center h-14">
          {/* Logo */}
          <div className="flex items-center gap-3 relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-orange-500 to-purple-600 blur-xl opacity-0 group-hover:opacity-40 transition-opacity duration-700" />
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-purple-600 flex items-center justify-center shadow-lg shadow-rose-500/20 relative z-10">
              <Brain className="w-4 h-4 text-white animate-breathe" />
            </div>
            <div className="relative overflow-hidden z-10">
              <h1 className="text-lg font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-orange-400 via-rose-500 to-purple-600 drop-shadow-md whitespace-nowrap animate-shimmer-sweep">
                Behavioral Twin
              </h1>
            </div>
          </div>

          {/* Missing Context */}
          {totalLogs !== undefined && totalLogs > 0 && level !== undefined && (
            <div className="hidden md:flex items-center ml-4 px-3 py-1 bg-white/5 rounded-full border border-white/10 shadow-inner">
              <span className="text-xs font-medium text-amber-400 mr-2">🔥 Level {level} Twin</span>
              <span className="text-xs text-slate-400">{totalLogs} logs</span>
            </div>
          )}
          
          <div className="flex-1" />

          {/* Actions */}
          <div className="flex items-center gap-2 sm:gap-3">
            <button type="button" onClick={handleShare} className="btn-secondary active:scale-95 active:shadow-inner flex items-center gap-2" title="Share your twin">
              <Share2 className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Share</span>
            </button>
            <button type="button" onClick={handleExport} className="btn-secondary active:scale-95 active:shadow-inner hidden sm:flex items-center gap-2" title="Export Weekly Report">
              <Download className="w-3.5 h-3.5" />
              <span>Export</span>
            </button>
            
            {/* Profile Dropdown */}
            <div className="relative">
              <button 
                type="button" 
                onClick={() => setShowProfileMenu(!showProfileMenu)}
                className="w-8 h-8 rounded-full overflow-hidden border border-[#ffffff14] hover:border-[#ffffff2a] transition-colors focus:outline-none bg-[#141414] flex items-center justify-center active:scale-95 ring-2 ring-purple-500 ring-offset-2 ring-offset-[#141414] animate-[pulse_2s_ease-in-out_infinite]"
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
              
              {showProfileMenu && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setShowProfileMenu(false)} />
                  <div className="absolute right-0 mt-2 w-56 rounded-xl bg-[#141414] border border-[#ffffff1a] shadow-2xl z-50 p-1 flex flex-col gap-1 overflow-hidden">
                    <div className="px-3 py-3 border-b border-[#ffffff1a] mb-1">
                      <p className="text-sm font-medium text-[#ededed] truncate">{userProfile?.name || 'User'}</p>
                      <p className="text-xs text-[#a1a1aa] truncate">{userProfile?.email || userId}</p>
                    </div>
                    <button type="button" onClick={handleResetTwin} disabled={isDeleting} className="w-full text-left px-3 py-2 text-sm text-rose-400 hover:bg-[#ffffff0a] rounded-lg flex items-center gap-2 transition-colors">
                      {isDeleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                      Factory Reset
                    </button>
                    <button type="button" onClick={onSignOut} className="w-full text-left px-3 py-2 text-sm text-slate-400 hover:bg-[#ffffff0a] rounded-lg flex items-center gap-2 transition-colors">
                      <LogOut className="w-4 h-4" />
                      Sign out
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Status */}
          <div className="hidden sm:flex items-center gap-2 ml-4" title={connected ? "Connected — last synced just now" : "Disconnected"}>
            {refreshing ? (
              <span className="flex items-center gap-1.5 text-xs text-[#a1a1aa]">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-400" /> Syncing...
              </span>
            ) : connected ? (
              <span className="flex items-center gap-1.5 text-xs text-[#a1a1aa]">
                <div className="w-1.5 h-1.5 rounded-full animate-pulse-glow-green" /> Connected
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-xs text-[#71717a]">
                <div className="w-1.5 h-1.5 rounded-full animate-pulse-glow-red" /> Offline
              </span>
            )}
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

      {/* Top row: Heatmap + Log Decision + Stats/Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="flex flex-col gap-6">
          <ErrorBoundary>
            <TiltCard intensity={5}>
              <HeatmapCalendar heatmap={data?.heatmap || []} />
            </TiltCard>
          </ErrorBoundary>
          <ErrorBoundary>
            <TiltCard intensity={8}>
              <LogDecisionPanel 
                userId={userId} 
                onLogged={refetch} 
                totalLogs={data?.decisions?.length || 0}
                xp={xp}
                level={level}
                addXp={addXp}
              />
            </TiltCard>
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
// Auth Screen
// ---------------------------------------------------------------------------

interface AuthScreenProps {
  onLogin: (userId: string) => void
}

function AuthScreen({ onLogin }: AuthScreenProps) {
  const [errorMsg, setErrorMsg] = useState('')

  const handleGoogleSuccess = async (credentialResponse: any) => {
    try {
      const res = await fetch(`http://localhost:8000/auth/google`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: credentialResponse.credential }),
      })
      if (!res.ok) throw new Error('Failed to verify Google token')
      const data = await res.json()
      
      if (data.access_token) {
        localStorage.setItem('access_token', data.access_token)
        localStorage.setItem('user_profile', JSON.stringify(data.user))
      }
      onLogin(data.user.email)
    } catch (err) {
      console.error(err)
      setErrorMsg('Google login was unsuccessful.')
    }
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } }
  }

  return (
    <div className="min-h-screen relative flex items-center justify-center p-4 overflow-hidden bg-[#000000]">
      {/* Dynamic Background Effects */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-600/20 rounded-full blur-[120px] mix-blend-screen animate-pulse" style={{ animationDuration: '4s' }} />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-emerald-600/20 rounded-full blur-[120px] mix-blend-screen animate-pulse" style={{ animationDuration: '6s', animationDelay: '1s' }} />
        <div className="absolute top-[20%] right-[10%] w-[30%] h-[30%] bg-purple-600/20 rounded-full blur-[100px] mix-blend-screen animate-pulse" style={{ animationDuration: '5s', animationDelay: '2s' }} />
      </div>

      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="w-full max-w-md relative z-10"
      >
        <div className="backdrop-blur-2xl bg-[#0a0a0a]/60 border border-[#ffffff1a] rounded-[24px] p-8 shadow-2xl relative overflow-hidden">
          {/* Subtle inner top highlight */}
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#ffffff33] to-transparent" />
          
          <div className="flex flex-col items-center mb-8 relative">
            <motion.div variants={itemVariants} className="relative mb-6 group">
              <div className="absolute inset-0 bg-indigo-500/30 blur-xl rounded-full group-hover:bg-indigo-400/40 transition-colors duration-500" />
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-b from-[#1f1f1f] to-[#0a0a0a] border border-[#ffffff2a] flex items-center justify-center shadow-2xl relative z-10 transform group-hover:scale-105 transition-transform duration-500">
                <Brain className="w-8 h-8 text-indigo-400 drop-shadow-[0_0_8px_rgba(129,140,248,0.5)]" />
              </div>
            </motion.div>
            
            <motion.h1 variants={itemVariants} className="text-3xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-br from-white to-[#a1a1aa] mb-3 text-center">
              Behavioral Twin
            </motion.h1>
            <motion.p variants={itemVariants} className="text-sm text-[#a1a1aa] text-center leading-relaxed max-w-[280px]">
              Sign in with Google to start tracking your digital twin and optimizing your daily habits.
            </motion.p>
          </div>

          <motion.div variants={itemVariants} className="flex flex-col items-center justify-center gap-4 w-full">
            <div className="w-full flex justify-center p-1 rounded-xl bg-gradient-to-b from-[#1a1a1a] to-[#0a0a0a] border border-[#ffffff0f] hover:border-[#ffffff1a] transition-colors shadow-inner">
              <GoogleLogin
                onSuccess={handleGoogleSuccess}
                onError={() => setErrorMsg('Google login was unsuccessful.')}
                theme="filled_black"
                shape="pill"
                size="large"
                text="continue_with"
                width="280"
              />
            </div>
            
            {errorMsg && (
              <motion.p 
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="text-xs text-rose-400 mt-1 bg-rose-500/10 px-3 py-2 rounded-lg border border-rose-500/20 text-center"
              >
                {errorMsg}
              </motion.p>
            )}
          </motion.div>
          
          <motion.p variants={itemVariants} className="mt-8 text-[10px] text-[#71717a] text-center px-4">
            By continuing, you agree to the Terms of Service and Privacy Policy. Secure authentication provided by Google OAuth.
          </motion.p>
        </div>
      </motion.div>
    </div>
  )
}

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

  // If no user is logged in, show Auth
  if (!userId) {
    return <AuthScreen onLogin={handleLogin} />
  }

  return <DashboardContainer userId={userId} onSignOut={handleSignOut} />
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

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
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
      </main>
    </div>
  )
}
