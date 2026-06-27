import React, { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, XCircle, ChevronDown, ChevronUp, Filter, Target } from 'lucide-react'
import type { DecisionRow } from '../hooks/useApi'

interface DecisionsTableProps {
  decisions: DecisionRow[]
}

const DOMAIN_COLORS: Record<string, { text: string; dot: string }> = {
  focus:    { text: 'text-[#ededed]', dot: 'bg-blue-500' },
  task:     { text: 'text-[#ededed]', dot: 'bg-purple-500' },
  purchase: { text: 'text-[#ededed]', dot: 'bg-emerald-500' },
}

function getDomainStyle(domain: string) {
  return DOMAIN_COLORS[domain] || { text: 'text-[#ededed]', dot: 'bg-slate-500' }
}

export default function DecisionsTable({ decisions }: DecisionsTableProps) {
  const [activeDomain, setActiveDomain] = useState<string | null>(null)
  const [sortAsc, setSortAsc] = useState(false)
  const [page, setPage] = useState(0)
  const [expandedRow, setExpandedRow] = useState<number | null>(null)
  const pageSize = 15

  const domains = useMemo(() => {
    const set = new Set(decisions.map(d => d.domain))
    return Array.from(set).sort()
  }, [decisions])

  const filtered = useMemo(() => {
    let rows = activeDomain ? decisions.filter(d => d.domain === activeDomain) : decisions
    rows = [...rows].sort((a, b) => {
      const cmp = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      return sortAsc ? cmp : -cmp
    })
    return rows
  }, [decisions, activeDomain, sortAsc])

  const totalPages = Math.ceil(filtered.length / pageSize)
  const pageRows = filtered.slice(page * pageSize, (page + 1) * pageSize)

  // Stats per domain
  const domainStats = useMemo(() => {
    return domains.map(d => {
      const rows = decisions.filter(r => r.domain === d)
      const hits = rows.filter(r => r.hit).length
      return { domain: d, total: rows.length, hits, accuracy: rows.length ? hits / rows.length : 0 }
    })
  }, [decisions, domains])

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
      className="glass-card overflow-hidden"
    >
      {/* Header */}
      <div className="p-4 pb-2">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-title mb-0.5">Decision History</h3>
            <p className="text-body">
              {filtered.length} decision{filtered.length !== 1 ? 's' : ''}
              {activeDomain && <span className="text-[#a1a1aa]"> in {activeDomain}</span>}
            </p>
          </div>
          <button
            onClick={() => setSortAsc(!sortAsc)}
            className="btn-secondary flex items-center gap-1.5"
          >
            {sortAsc ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            {sortAsc ? 'Oldest first' : 'Newest first'}
          </button>
        </div>

        {/* Domain filter pills */}
        <div className="flex items-center gap-2 flex-wrap">
          <Filter className="w-3.5 h-3.5 text-[#71717a]" />
          <button
            onClick={() => { setActiveDomain(null); setPage(0) }}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors border ${
              activeDomain === null
                ? 'bg-[#ededed] text-[#000000] border-transparent'
                : 'bg-[#141414] text-[#a1a1aa] border-[#ffffff14] hover:border-[#ffffff2a] hover:text-[#ededed]'
            }`}
          >
            All
          </button>
          {domainStats.map(ds => {
            const style = getDomainStyle(ds.domain)
            const isActive = activeDomain === ds.domain
            return (
              <button
                key={ds.domain}
                onClick={() => { setActiveDomain(isActive ? null : ds.domain); setPage(0) }}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors flex items-center gap-1.5 border ${
                  isActive
                    ? 'bg-[#ededed] text-[#000000] border-transparent'
                    : 'bg-[#141414] text-[#a1a1aa] border-[#ffffff14] hover:border-[#ffffff2a] hover:text-[#ededed]'
                }`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
                {ds.domain}
                <span className="text-[10px] opacity-70">
                  {(ds.accuracy * 100).toFixed(0)}%
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full decision-table">
          <thead>
            <tr className="border-b border-white/5">
              <th className="text-left">Time</th>
              <th className="text-left">Domain</th>
              <th className="text-left">Predicted</th>
              <th className="text-left">Actual</th>
              <th className="text-center">Result</th>
              <th className="text-right">Confidence</th>
            </tr>
          </thead>
          <tbody>
            <AnimatePresence mode="popLayout">
              {pageRows.map((row, i) => {
                const style = getDomainStyle(row.domain)
                const isExpanded = expandedRow === row.id
                return (
                  <React.Fragment key={row.id}>
                    <motion.tr
                      onClick={() => setExpandedRow(isExpanded ? null : row.id)}
                      className="cursor-pointer hover:bg-[#ffffff05] transition-colors"
                      initial={{ opacity: 0, x: -4 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 4 }}
                      transition={{ delay: i * 0.015, duration: 0.15 }}
                    >
                      <td className="text-[#a1a1aa] font-mono text-[11px] whitespace-nowrap">
                        {new Date(row.timestamp).toLocaleString(undefined, {
                          month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                        })}
                      </td>
                      <td>
                        <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md border border-[#ffffff14] text-xs font-medium ${style.text}`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
                          {row.domain}
                        </span>
                      </td>
                      <td className="text-[#ededed] font-medium">{row.predicted}</td>
                      <td className="text-[#ededed] font-medium">{row.actual}</td>
                      <td className="text-center">
                        {row.hit ? (
                          <span className="inline-flex items-center gap-1 text-[#10b981] text-xs">
                            <CheckCircle className="w-3.5 h-3.5" /> Hit
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[#f43f5e] text-xs">
                            <XCircle className="w-3.5 h-3.5" /> Miss
                          </span>
                        )}
                      </td>
                      <td className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <div className="w-16 h-[3px] bg-[#ffffff14] rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                row.confidence >= 0.7 ? 'bg-[#10b981]' : row.confidence >= 0.5 ? 'bg-[#f59e0b]' : 'bg-[#f43f5e]'
                              }`}
                              style={{ width: `${row.confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-[11px] font-mono text-[#a1a1aa] w-8 text-right">
                            {(row.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </td>
                    </motion.tr>
                    {isExpanded && (
                      <tr className="bg-[#0a0a0a]">
                        <td colSpan={6} className="p-4 border-b border-t border-[#ffffff0a]">
                          <motion.div 
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            className="flex flex-col gap-2"
                          >
                            <h4 className="text-xs font-bold text-[#ededed] uppercase tracking-wider mb-2 flex items-center gap-2">
                              <Target className="w-3.5 h-3.5 text-indigo-400" />
                              Log Details
                            </h4>
                            <div className="grid grid-cols-3 gap-4 text-sm">
                              <div className="p-3 rounded-lg bg-[#141414] border border-[#ffffff0a]">
                                <span className="text-[#a1a1aa] text-[10px] uppercase tracking-wider block mb-1">Time</span>
                                <span className="font-mono text-[#ededed]">{new Date(row.timestamp).toLocaleString()}</span>
                              </div>
                              <div className="p-3 rounded-lg bg-[#141414] border border-[#ffffff0a]">
                                <span className="text-[#a1a1aa] text-[10px] uppercase tracking-wider block mb-1">AI Prediction</span>
                                <span className="font-medium text-indigo-400 capitalize">{row.predicted.replace(/_/g, ' ')}</span>
                              </div>
                              <div className="p-3 rounded-lg bg-[#141414] border border-[#ffffff0a]">
                                <span className="text-[#a1a1aa] text-[10px] uppercase tracking-wider block mb-1">Real Decision</span>
                                <span className="font-medium text-emerald-400 capitalize">{row.actual.replace(/_/g, ' ')}</span>
                              </div>
                            </div>
                            <div className="p-3 mt-1 rounded-lg bg-[#141414] border border-[#ffffff0a] flex items-center justify-between">
                              <span className="text-xs text-[#a1a1aa]">Confidence</span>
                              <div className="flex items-center gap-3 w-1/2">
                                <div className="h-1.5 flex-1 bg-[#1f1f1f] rounded-full overflow-hidden">
                                  <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${row.confidence * 100}%` }} />
                                </div>
                                <span className="font-mono text-xs text-white">{(row.confidence * 100).toFixed(1)}%</span>
                              </div>
                            </div>
                          </motion.div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                )
              })}
            </AnimatePresence>
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-[#ffffff0a]">
          <span className="text-caption">
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              disabled={page === 0}
              onClick={() => setPage(p => p - 1)}
              className="btn-secondary px-2 disabled:opacity-30"
            >
              Previous
            </button>
            <button
              disabled={page >= totalPages - 1}
              onClick={() => setPage(p => p + 1)}
              className="btn-secondary px-2 disabled:opacity-30"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </motion.div>
  )
}
