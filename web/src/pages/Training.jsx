import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { api } from '../api'

export default function Training() {
  const [data, setData] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.training()
      .then(d => {
        setData(d)
        if (d.errors) setError('Some training history artifacts are unavailable.')
        setLoading(false)
      })
      .catch(e => {
        setError(e.message)
        setLoading(false)
      })
  }, [])

  const binaryData = (() => {
    const baseline = data.baseline || []
    const federated = data.federated || []
    const maxLen = Math.max(baseline.length, federated.length)
    return Array.from({ length: maxLen }, (_, i) => ({
      round: federated[i]?.round ?? baseline[i]?.epoch ?? i + 1,
      Federated: federated[i]?.f1 ?? null,
      Centralized: baseline[i]?.f1 ?? null,
    }))
  })()

  const multiData = (() => {
    const maxLen = Math.max(
      (data.multiclass || []).length,
      (data.iid || []).length,
      (data.noniid || []).length,
    )
    return Array.from({ length: maxLen }, (_, i) => ({
      round: i + 1,
      'Centralized':    data.multiclass?.[i]?.macro_f1 ?? null,
      'IID Federated':  data.iid?.[i]?.macro_f1 ?? null,
      'Non-IID (Best)': data.noniid?.[i]?.macro_f1 ?? null,
    }))
  })()

  const latest = (history, key) => {
    const row = Array.isArray(history) && history.length ? history[history.length - 1] : null
    return typeof row?.[key] === 'number' ? row[key] : null
  }
  const binaryFederated = latest(data.federated, 'f1')
  const binaryCentralized = latest(data.baseline, 'f1')
  const multiCentralized = latest(data.multiclass, 'macro_f1')
  const multiIid = latest(data.iid, 'macro_f1')
  const multiNoniid = latest(data.noniid, 'macro_f1')
  const formatMetric = value => typeof value === 'number' ? value.toFixed(4) : 'unavailable'
  const evaluation = data.noniid_evaluation
  const activeModel = data.active_noniid_model
  const evaluationMatchesActiveModel = Boolean(
    evaluation && activeModel &&
    evaluation.model_version === activeModel.model_version &&
    evaluation.model_sha256 === activeModel.sha256
  )
  const comparison = (value, baseline, label) => {
    if (value === null || baseline === null) return 'Comparison unavailable — required artifact is missing'
    const delta = value - baseline
    const relation = delta > 0 ? 'above' : delta < 0 ? 'below' : 'matches'
    return `${formatMetric(value)} ${relation} ${label} (${Math.abs(delta).toFixed(4)} difference)`
  }

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div style={{ background: '#000', border: '1px solid #333', borderRadius: 8, padding: '0.6rem 0.9rem' }}>
        <div style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: 4 }}>Round / Epoch {label}</div>
        {payload.map((p, i) => (
          <div key={i} style={{ fontSize: '0.8rem', color: p.color }}>{p.name}: {p.value}</div>
        ))}
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      style={{ position: 'relative', zIndex: 10, minHeight: '100vh', padding: '88px 2rem 2rem' }}
    >
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} style={{ marginBottom: '2.5rem' }}>
        <h1 style={{
          fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '2rem',
          background: 'linear-gradient(90deg, #fff, #00f5ff)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '0.4rem',
        }}>Federated Training</h1>
        <p style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Live data from your training history JSON files</p>
      </motion.div>

      {loading ? (
        <div style={{ textAlign: 'center', color: 'var(--muted)', fontFamily: 'var(--font-mono)', paddingTop: '4rem' }}>
          Loading training data...
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {error && (
            <div style={{ padding: '0.8rem 1rem', borderRadius: 8, background: 'rgba(255,149,0,0.06)', border: '1px solid rgba(255,149,0,0.2)', color: '#ff9500', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>
              {error}
            </div>
          )}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
            style={{ padding: '1.4rem', borderRadius: 16, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(20px)' }}>
            <div style={{ fontSize: '0.72rem', letterSpacing: '0.1em', color: 'var(--muted)', marginBottom: '1.2rem' }}>
              BINARY CLASSIFICATION — F1 PER ROUND
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={binaryData}>
                <XAxis dataKey="round" tick={{ fill: '#666', fontSize: 11 }} axisLine={false} tickLine={false} label={{ value: 'Round', position: 'insideBottom', offset: -2, fill: '#666', fontSize: 11 }} />
                <YAxis domain={[0.99, 1.0]} tick={{ fill: '#666', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(3)} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: '0.78rem', paddingTop: '1rem' }} />
                <Line type="monotone" dataKey="Federated"   stroke="#00f5ff" strokeWidth={2.5} dot={{ fill: '#00f5ff', r: 3 }} />
                <Line type="monotone" dataKey="Centralized" stroke="#ff2d55" strokeWidth={1.5} strokeDasharray="6 3" dot={false} />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ marginTop: '0.8rem', padding: '0.7rem 1rem', borderRadius: 8, background: 'rgba(0,245,255,0.05)', border: '1px solid rgba(0,245,255,0.1)', fontSize: '0.8rem', color: '#00f5ff' }}>
              {binaryFederated === null && binaryCentralized === null
                ? 'Federated and centralized binary F1 are unavailable.'
                : `Federated F1: ${formatMetric(binaryFederated)} vs Centralized: ${formatMetric(binaryCentralized)} — ${comparison(binaryFederated, binaryCentralized, 'centralized')}`}
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
            style={{ padding: '1.4rem', borderRadius: 16, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(20px)' }}>
            <div style={{ fontSize: '0.72rem', letterSpacing: '0.1em', color: 'var(--muted)', marginBottom: '1.2rem' }}>
              MULTI-CLASS MACRO F1 — NON-IID vs IID vs CENTRALIZED
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={multiData}>
                <XAxis dataKey="round" tick={{ fill: '#666', fontSize: 11 }} axisLine={false} tickLine={false} label={{ value: 'Epoch / Round', position: 'insideBottom', offset: -2, fill: '#666', fontSize: 11 }} />
                <YAxis domain={[0.6, 0.9]} tick={{ fill: '#666', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(2)} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: '0.78rem', paddingTop: '1rem' }} />
                <Line type="monotone" dataKey="Non-IID (Best)" stroke="#00ff88" strokeWidth={3} dot={{ fill: '#00ff88', r: 3 }} connectNulls />
                <Line type="monotone" dataKey="IID Federated"  stroke="#a78bff" strokeWidth={2} dot={{ fill: '#a78bff', r: 2 }} connectNulls />
                <Line type="monotone" dataKey="Centralized"    stroke="#ff2d55" strokeWidth={1.5} strokeDasharray="6 3" dot={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ marginTop: '0.8rem', padding: '0.7rem 1rem', borderRadius: 8, background: 'rgba(0,255,136,0.05)', border: '1px solid rgba(0,255,136,0.1)', fontSize: '0.8rem', color: '#00ff88' }}>
              {multiNoniid === null
                ? 'Non-IID federated result is unavailable.'
                : `Non-IID Federated: ${formatMetric(multiNoniid)}; IID: ${formatMetric(multiIid)}; Centralized: ${formatMetric(multiCentralized)} — ${comparison(multiNoniid, multiCentralized, 'centralized')}`}
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}
            style={{ padding: '1.4rem', borderRadius: 16, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(20px)' }}>
            <div style={{ fontSize: '0.72rem', letterSpacing: '0.1em', color: 'var(--muted)', marginBottom: '1.2rem' }}>
              NON-IID MODEL — PER-CLASS RESULTS
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    {['Class', 'Precision', 'Recall', 'F1-Score', 'Support'].map(h => (
                      <th key={h} style={{ padding: '0.6rem 0.8rem', textAlign: 'left', fontSize: '0.68rem', color: 'var(--muted)', letterSpacing: '0.08em' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {evaluationMatchesActiveModel ? evaluation.class_order.map(name => {
                    const metric = evaluation.per_class?.[name]
                    return (
                      <tr key={name}>
                        <td style={{ padding: '0.65rem 0.8rem' }}>{name}</td>
                        <td style={{ padding: '0.65rem 0.8rem' }}>{formatMetric(metric?.precision)}</td>
                        <td style={{ padding: '0.65rem 0.8rem' }}>{formatMetric(metric?.recall)}</td>
                        <td style={{ padding: '0.65rem 0.8rem' }}>{formatMetric(metric?.f1)}</td>
                        <td style={{ padding: '0.65rem 0.8rem' }}>{typeof metric?.support === 'number' ? metric.support.toLocaleString() : 'unavailable'}</td>
                      </tr>
                    )
                  }) : (
                    <tr>
                      <td colSpan={5} style={{ padding: '1.2rem 0.8rem', color: '#ff9500', fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>
                        Per-class precision, recall, F1, and support are unavailable: no matching per-class evaluation artifact is stored.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: '0.8rem', fontSize: '0.75rem', color: 'var(--muted)' }}>
              ℹ️ Aggregate macro F1 is available from the training history; per-class evaluation is not reported by the available artifacts.
            </div>
          </motion.div>
        </div>
      )}
    </motion.div>
  )
}
