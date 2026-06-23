import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from 'recharts'
import { api } from '../api'

const DESCRIPTIONS = {
  dst_host_serror_rate:    'SYN error rate to destination host — primary DDoS indicator',
  logged_in:               'Attackers probe without authentication — key anomaly signal',
  same_srv_rate:           'Port scanners repeatedly hit same service port',
  srv_serror_rate:         'Service-level SYN errors confirm flood attack pattern',
  protocol_type:           'Attack traffic skews toward specific protocols (TCP/ICMP)',
  dst_host_srv_count:      'Unusual service count signals active reconnaissance',
  dst_host_same_srv_rate:  'High same-service rate indicates targeted scanning',
  count:                   'Connection count spike — characteristic of DoS attacks',
  dst_host_count:          'Number of connections to same host in recent window',
  serror_rate:             'SYN error rate in current connection window',
}

export default function SHAP() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.shap()
      .then(d => {
        setData((d.feature_importance || []).slice(0, 15))
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const maxScore = data[0]?.shap_score || 1

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
        }}>SHAP Explainability</h1>
        <p style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>
          Why does the model flag attacks? Real SHAP values from your trained model.
        </p>
      </motion.div>

      {loading ? (
        <div style={{ textAlign: 'center', color: 'var(--muted)', fontFamily: 'var(--font-mono)', paddingTop: '4rem' }}>
          Loading SHAP data...
        </div>
      ) : data.length === 0 ? (
        <div style={{ textAlign: 'center', color: 'var(--muted)', fontFamily: 'var(--font-mono)', paddingTop: '4rem' }}>
          Run explain.py first to generate SHAP results.
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '1.5rem' }}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
            style={{ padding: '1.4rem', borderRadius: 16, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(20px)' }}>
            <div style={{ fontSize: '0.72rem', letterSpacing: '0.1em', color: 'var(--muted)', marginBottom: '1.2rem' }}>
              TOP 15 FEATURES — SHAP IMPORTANCE
            </div>
            <ResponsiveContainer width="100%" height={420}>
              <BarChart data={data} layout="vertical" margin={{ left: 20 }}>
                <XAxis type="number" tick={{ fill: '#666', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(3)} />
                <YAxis type="category" dataKey="feature" tick={{ fill: '#aaa', fontSize: 10 }} axisLine={false} tickLine={false} width={170} />
                <Tooltip contentStyle={{ background: '#000', border: '1px solid #333', borderRadius: 8 }} formatter={(v) => [v.toFixed(4), 'SHAP Score']} />
                <Bar dataKey="shap_score" radius={[0, 4, 4, 0]}>
                  {data.map((_, i) => <Cell key={i} fill={`hsl(${200 + i * 8}, 90%, ${65 - i * 2}%)`} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
            style={{ padding: '1.4rem', borderRadius: 16, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(20px)', overflowY: 'auto', maxHeight: 500 }}>
            <div style={{ fontSize: '0.72rem', letterSpacing: '0.1em', color: 'var(--muted)', marginBottom: '1.2rem' }}>
              WHAT EACH FEATURE MEANS
            </div>
            {data.slice(0, 8).map((f, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 + i * 0.07 }}
                style={{ marginBottom: '1rem', padding: '0.8rem', borderRadius: 8, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: '#a78bff', fontWeight: 600 }}>{f.feature}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: '#00f5ff' }}>{f.shap_score?.toFixed(4)}</span>
                </div>
                <div style={{ height: 3, background: 'rgba(255,255,255,0.06)', borderRadius: 2, marginBottom: 6 }}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(f.shap_score / maxScore) * 100}%` }}
                    transition={{ delay: 0.6 + i * 0.05, duration: 0.8 }}
                    style={{ height: '100%', borderRadius: 2, background: 'linear-gradient(90deg, #7b2fff, #00f5ff)' }}
                  />
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--muted)', lineHeight: 1.5 }}>
                  {DESCRIPTIONS[f.feature] || 'Network traffic feature used in attack classification.'}
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      )}
    </motion.div>
  )
}