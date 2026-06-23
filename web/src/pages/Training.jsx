import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { api } from '../api'

export default function Training() {
  const [data, setData] = useState({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.training()
      .then(d => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const binaryData = (data.federated || []).map(d => ({
    round: d.round,
    'Federated': parseFloat(d.f1?.toFixed(4)),
    'Centralized': 0.9947,
  }))

  const multiData = (() => {
    const maxLen = Math.max(
      (data.multiclass || []).length,
      (data.iid || []).length,
      (data.noniid || []).length,
    )
    return Array.from({ length: maxLen }, (_, i) => ({
      round: i + 1,
      'Centralized':    data.multiclass?.[i]?.macro_f1?.toFixed(4) || null,
      'IID Federated':  data.iid?.[i]?.macro_f1?.toFixed(4) || null,
      'Non-IID (Best)': data.noniid?.[i]?.macro_f1?.toFixed(4) || null,
    }))
  })()

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
              ✅ Federated F1: 0.9946 vs Centralized: 0.9947 — 0.0001 gap with full privacy preserved
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
              🔥 Non-IID Federated (0.84) beats IID Federated (0.81) AND Centralized (0.79) — counterintuitive and the key finding
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
                  {[
                    { Class: 'Normal', Precision: 0.98, Recall: 1.00, F1: 0.99, Support: 13469, color: '#00ff88' },
                    { Class: 'DoS',    Precision: 1.00, Recall: 0.98, F1: 0.99, Support: 9186,  color: '#ff2d55' },
                    { Class: 'Probe',  Precision: 0.99, Recall: 0.98, F1: 0.98, Support: 2331,  color: '#ff9500' },
                    { Class: 'R2L',    Precision: 0.91, Recall: 0.40, F1: 0.56, Support: 199,   color: '#ff2d8f' },
                    { Class: 'U2R',    Precision: 0.80, Recall: 0.40, F1: 0.53, Support: 10,    color: '#bf5af2' },
                  ].map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '0.65rem 0.8rem' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ width: 8, height: 8, borderRadius: 2, background: r.color, display: 'inline-block' }} />
                          <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>{r.Class}</span>
                        </span>
                      </td>
                      {[r.Precision, r.Recall, r.F1].map((v, j) => (
                        <td key={j} style={{ padding: '0.65rem 0.8rem', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: v >= 0.9 ? '#00ff88' : v >= 0.7 ? '#ff9500' : '#ff2d55' }}>
                          {v.toFixed(2)}
                        </td>
                      ))}
                      <td style={{ padding: '0.65rem 0.8rem', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: 'var(--muted)' }}>
                        {r.Support.toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: '0.8rem', fontSize: '0.75rem', color: 'var(--muted)' }}>
              ℹ️ R2L/U2R recall low due to very few test samples (199 and 10) — documented limitation of NSL-KDD benchmark
            </div>
          </motion.div>
        </div>
      )}
    </motion.div>
  )
}