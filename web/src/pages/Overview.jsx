import { motion } from 'framer-motion'
import { CheckCircle, Globe } from 'lucide-react'
import { useState, useEffect } from 'react'
import { api } from '../api'

const COMPLIANCE = [
  { label: 'GDPR',      note: 'Data-minimization architecture' },
  { label: 'HIPAA',     note: 'No patient data stored by this demo' },
  { label: 'PCI-DSS',   note: 'Aligned handling controls' },
  { label: 'ISO 27001', note: 'Distributed control architecture' },
]

const NODES = [
  { name: 'Hospital', detail: 'Configured federated client', color: '#00f5ff' },
  { name: 'Bank',     detail: 'Configured federated client', color: '#7b2fff' },
  { name: 'Campus',   detail: 'Configured federated client', color: '#00ff88' },
]

const latestMetric = (history, key) => {
  const row = Array.isArray(history) && history.length ? history[history.length - 1] : null
  return typeof row?.[key] === 'number' ? row[key] : null
}

const formatMetric = value => typeof value === 'number' && Number.isFinite(value)
  ? value.toFixed(4)
  : '—'

const comparisonNote = (value, comparison, label) => {
  if (value === null) return 'Unavailable — no verified artifact'
  if (comparison === null) return 'Latest recorded training artifact'
  const delta = value - comparison
  const direction = delta > 0 ? 'above' : delta < 0 ? 'below' : 'matches'
  return `${Math.abs(delta).toFixed(4)} ${direction} ${label}`
}

export default function Overview() {
  const [stats, setStats] = useState(null)
  const [training, setTraining] = useState(null)
  const [statsError, setStatsError] = useState(null)
  const [trainingError, setTrainingError] = useState(null)

  useEffect(() => {
    api.stats()
      .then(setStats)
      .catch(error => setStatsError(error.message))
    api.training()
      .then(data => {
        setTraining(data)
        if (data.errors) setTrainingError('Some training histories are unavailable')
      })
      .catch(error => setTrainingError(error.message))
  }, [])

  const baseline = latestMetric(training?.baseline, 'f1')
  const federated = latestMetric(training?.federated, 'f1')
  const centralizedMulti = latestMetric(training?.multiclass, 'macro_f1')
  const iid = latestMetric(training?.iid, 'macro_f1')
  const noniid = latestMetric(training?.noniid, 'macro_f1')
  const results = [
    { metric: 'Binary F1 — Federated', value: formatMetric(federated), note: comparisonNote(federated, baseline, 'centralized'), color: '#00f5ff' },
    { metric: 'Binary F1 — Centralized', value: formatMetric(baseline), note: 'Latest recorded baseline artifact', color: '#888' },
    { metric: 'Multi-Class Macro F1 — Non-IID Federated', value: formatMetric(noniid), note: comparisonNote(noniid, centralizedMulti, 'centralized'), color: '#00ff88' },
    { metric: 'Multi-Class Macro F1 — IID Federated', value: formatMetric(iid), note: comparisonNote(iid, centralizedMulti, 'centralized'), color: '#a78bff' },
    { metric: 'Multi-Class Macro F1 — Centralized', value: formatMetric(centralizedMulti), note: 'Latest recorded baseline artifact', color: '#888' },
  ]

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      style={{ position: 'relative', zIndex: 10, minHeight: '100vh', padding: '88px 2rem 2rem' }}
    >
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} style={{ marginBottom: '2.5rem' }}>
        <h1 style={{
          fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '2rem',
          background: 'linear-gradient(90deg, #fff, #a78bff)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '0.4rem',
        }}>System Overview</h1>
        <p style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>
          Results and activity sourced from backend artifacts
        </p>
        {(statsError || trainingError) && (
          <div style={{ marginTop: '0.8rem', color: '#ff9500', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
            {statsError ? `Activity unavailable: ${statsError}` : 'Some activity data is unavailable.'}
            {trainingError ? ` ${trainingError}.` : ''}
          </div>
        )}
      </motion.div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '1rem', marginBottom: '2rem' }}>
        {[
          { label: 'Total Packets Logged', value: stats ? stats.total.toLocaleString() : '—',   color: '#00f5ff' },
          { label: 'Attacks Detected',     value: stats ? stats.attacks.toLocaleString() : '—', color: '#ff2d55' },
          { label: 'IPs Auto-Blocked',     value: stats ? stats.blocked.toLocaleString() : '—', color: '#ff9500' },
        ].map((s, i) => (
          <motion.div key={i} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 * i }}
            style={{ padding: '1.2rem', borderRadius: 14, background: 'rgba(255,255,255,0.03)', border: `1px solid ${s.color}20`, backdropFilter: 'blur(20px)' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--muted)', letterSpacing: '0.1em', marginBottom: 6 }}>{s.label}</div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '2rem', fontWeight: 800, color: s.color }}>{s.value}</div>
          </motion.div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
          style={{ padding: '1.4rem', borderRadius: 16, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(20px)' }}>
          <div style={{ fontSize: '0.72rem', letterSpacing: '0.1em', color: 'var(--muted)', marginBottom: '1.2rem' }}>EXPERIMENT RESULTS</div>
          {results.map((r, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 + i * 0.06 }}
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.65rem 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <div>
                <div style={{ fontSize: '0.82rem', fontWeight: 500 }}>{r.metric}</div>
                   <div style={{ fontSize: '0.7rem', color: r.value === '—' ? '#ff9500' : 'var(--muted)', marginTop: 2 }}>{r.note}</div>
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '1.1rem', color: r.color, minWidth: 60, textAlign: 'right' }}>{r.value}</div>
            </motion.div>
          ))}
        </motion.div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
            style={{ padding: '1.4rem', borderRadius: 16, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(20px)' }}>
            <div style={{ fontSize: '0.72rem', letterSpacing: '0.1em', color: 'var(--muted)', marginBottom: '1.2rem' }}>FEDERATED NODES</div>
            {NODES.map((n, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 + i * 0.1 }}
                style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', padding: '0.65rem 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: n.color, boxShadow: `0 0 8px ${n.color}` }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>{n.name}</div>
                   <div style={{ fontSize: '0.7rem', color: 'var(--muted)' }}>{n.detail}</div>
                </div>
                <Globe size={14} color="var(--muted)" />
              </motion.div>
            ))}
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
            style={{ padding: '1.4rem', borderRadius: 16, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(20px)' }}>
            <div style={{ fontSize: '0.72rem', letterSpacing: '0.1em', color: 'var(--muted)', marginBottom: '1.2rem' }}>COMPLIANCE</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.6rem' }}>
              {COMPLIANCE.map((c, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem', borderRadius: 8, background: 'rgba(0,255,136,0.05)', border: '1px solid rgba(0,255,136,0.1)' }}>
                  <CheckCircle size={14} color="#00ff88" />
                  <div>
                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#00ff88' }}>{c.label}</div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--muted)' }}>{c.note}</div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}