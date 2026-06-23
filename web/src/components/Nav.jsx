import { NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Shield, Activity, BarChart3, Zap, Brain } from 'lucide-react'

const links = [
  { to: '/soc',      label: 'SOC Monitor', icon: Activity },
  { to: '/overview', label: 'Overview',    icon: Shield },
  { to: '/training', label: 'Training',    icon: BarChart3 },
  { to: '/shap',     label: 'Explainability', icon: Brain },
]

export default function Nav() {
  return (
    <motion.nav
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 2rem', height: '64px',
        background: 'rgba(0,0,5,0.7)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      {/* Logo */}
      <NavLink to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
        <div style={{
          width: 32, height: 32, borderRadius: 8,
          background: 'linear-gradient(135deg, #7b2fff, #00f5ff)',
          display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
          <Shield size={18} color="white" />
        </div>
        <span style={{
          fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '1.1rem',
          background: 'linear-gradient(90deg, #fff, #00f5ff)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent'
        }}>
          FedShield
        </span>
      </NavLink>

      {/* Links */}
      <div style={{ display: 'flex', gap: '0.25rem' }}>
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} style={{ textDecoration: 'none' }}>
            {({ isActive }) => (
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.97 }}
                style={{
                  display: 'flex', alignItems: 'center', gap: '0.4rem',
                  padding: '0.4rem 0.9rem', borderRadius: 8,
                  background: isActive ? 'rgba(123,47,255,0.2)' : 'transparent',
                  border: `1px solid ${isActive ? 'rgba(123,47,255,0.5)' : 'transparent'}`,
                  color: isActive ? '#00f5ff' : 'rgba(232,232,240,0.6)',
                  fontSize: '0.82rem', fontWeight: 500,
                  transition: 'all 0.2s',
                  cursor: 'pointer',
                }}
              >
                <Icon size={14} />
                {label}
              </motion.div>
            )}
          </NavLink>
        ))}
      </div>

      {/* Live indicator */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem', color: 'var(--muted)' }}>
        <motion.div
          animate={{ opacity: [1, 0.2, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          style={{ width: 6, height: 6, borderRadius: '50%', background: '#00ff88' }}
        />
        LIVE
      </div>
    </motion.nav>
  )
}