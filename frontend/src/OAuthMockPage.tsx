import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { CheckCircle2, ShieldCheck, Loader2 } from 'lucide-react'

export default function OAuthMockPage() {
  const [appId, setAppId] = useState('')
  const [authorizing, setAuthorizing] = useState(false)
  const [success, setSuccess] = useState(false)

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    setAppId(params.get('app') || 'App')
  }, [])

  const handleAuthorize = () => {
    setAuthorizing(true)
    setTimeout(() => {
      setSuccess(true)
      setTimeout(() => {
        window.close()
      }, 800)
    }, 1500)
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#ededed] flex flex-col items-center justify-center p-6 font-sans">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-md w-full bg-[#141414] border border-[#ffffff14] rounded-2xl p-8 shadow-2xl relative overflow-hidden"
      >
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-indigo-500 to-purple-500" />
        
        <div className="flex justify-center mb-6">
          <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
            <ShieldCheck className="w-8 h-8 text-indigo-400" />
          </div>
        </div>

        <h1 className="text-2xl font-bold text-center mb-2 capitalize text-white">
          Authorize {appId}
        </h1>
        <p className="text-center text-[#a1a1aa] text-sm mb-8">
          Behavioral Twin is requesting access to your {appId} account.
        </p>

        <div className="space-y-4 mb-8">
          <div className="flex items-start gap-3 bg-[#ffffff0a] p-3 rounded-lg border border-[#ffffff14]">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
            <p className="text-sm text-[#ededed]">Read your activity and status</p>
          </div>
          <div className="flex items-start gap-3 bg-[#ffffff0a] p-3 rounded-lg border border-[#ffffff14]">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
            <p className="text-sm text-[#ededed]">Sync contextual data for predictions</p>
          </div>
        </div>

        {success ? (
          <div className="flex flex-col items-center justify-center py-4">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring' }}
            >
              <CheckCircle2 className="w-16 h-16 text-emerald-400 mb-4" />
            </motion.div>
            <p className="text-lg font-bold text-white mb-1">Authorized Successfully!</p>
            <p className="text-sm text-[#a1a1aa]">You can close this window now.</p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <button
              onClick={handleAuthorize}
              disabled={authorizing}
              className="w-full bg-indigo-500 hover:bg-indigo-600 text-white font-semibold py-3 px-4 rounded-xl transition-all shadow-[0_0_15px_rgba(99,102,241,0.3)] disabled:opacity-50 disabled:shadow-none flex justify-center items-center gap-2"
            >
              {authorizing ? (
                <><Loader2 className="w-5 h-5 animate-spin" /> Authorizing...</>
              ) : (
                `Authorize ${appId}`
              )}
            </button>
            <button 
              onClick={() => window.close()}
              disabled={authorizing}
              className="w-full bg-transparent hover:bg-[#ffffff0a] text-[#a1a1aa] font-semibold py-3 px-4 rounded-xl transition-all disabled:opacity-50"
            >
              Cancel
            </button>
          </div>
        )}
      </motion.div>
      <p className="text-[#a1a1aa] text-xs mt-6">
        (This is a simulated OAuth screen for the Behavioral Twin demo)
      </p>
    </div>
  )
}
