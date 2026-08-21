import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ShieldCheck, ArrowRight, Lock, User, AlertCircle } from 'lucide-react'
import { login } from '../api'

export default function Login() {
  const navigate = useNavigate()

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleLogin = async (e) => {
    e.preventDefault()

    setError('')
    setLoading(true)

    try {
      await login(username, password)
      navigate('/soc')
    } catch (err) {
      setError('Authentication failed. Check your credentials.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        width: '100%',
        background:
          'radial-gradient(circle at 50% 40%, rgba(123,47,255,0.14), transparent 35%), var(--void)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        overflow: 'hidden',
        padding: '2rem',
      }}
    >

      {/* Ambient glow */}
      <div
        style={{
          position: 'absolute',
          width: 500,
          height: 500,
          borderRadius: '50%',
          background: 'rgba(0,245,255,0.05)',
          filter: 'blur(100px)',
          top: '10%',
          left: '10%',
          pointerEvents: 'none',
        }}
      />

      <div
        style={{
          position: 'absolute',
          width: 500,
          height: 500,
          borderRadius: '50%',
          background: 'rgba(123,47,255,0.08)',
          filter: 'blur(100px)',
          bottom: '5%',
          right: '10%',
          pointerEvents: 'none',
        }}
      />

      {/* Scan lines */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          backgroundImage:
            'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,245,255,0.012) 2px, rgba(0,245,255,0.012) 4px)',
        }}
      />

      <motion.div
        initial={{ opacity: 0, y: 30, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.7 }}
        style={{
          width: '100%',
          maxWidth: 460,
          position: 'relative',
          zIndex: 2,
        }}
      >

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>

          <motion.div
            animate={{
              boxShadow: [
                '0 0 20px rgba(0,245,255,0.15)',
                '0 0 40px rgba(123,47,255,0.3)',
                '0 0 20px rgba(0,245,255,0.15)',
              ],
            }}
            transition={{ duration: 3, repeat: Infinity }}
            style={{
              width: 64,
              height: 64,
              margin: '0 auto 1.5rem',
              borderRadius: 18,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'rgba(0,245,255,0.06)',
              border: '1px solid rgba(0,245,255,0.3)',
            }}
          >
            <ShieldCheck size={32} color="var(--cyan)" />
          </motion.div>

          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.72rem',
              letterSpacing: '0.18em',
              color: 'var(--cyan)',
              marginBottom: '0.8rem',
            }}
          >
            SECURE ACCESS
          </div>

          <h1
            style={{
              fontFamily: 'var(--font-display)',
              fontSize: '2.7rem',
              fontWeight: 800,
              letterSpacing: '-0.04em',
              background:
                'linear-gradient(135deg, #ffffff 20%, #c4b5fd 50%, #00f5ff 90%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              marginBottom: '0.7rem',
            }}
          >
            FedShield
          </h1>

          <p
            style={{
              color: 'var(--muted)',
              fontSize: '0.9rem',
            }}
          >
            Authenticate to enter the Security Operations Center
          </p>
        </div>

        {/* Login Card */}
        <motion.div
          whileHover={{
            boxShadow:
              '0 0 70px rgba(123,47,255,0.14), 0 0 30px rgba(0,245,255,0.05)',
          }}
          style={{
            background: 'rgba(255,255,255,0.035)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 20,
            padding: '2rem',
            backdropFilter: 'blur(25px)',
            boxShadow: '0 20px 80px rgba(0,0,0,0.35)',
          }}
        >

          <form onSubmit={handleLogin}>

            {/* Username */}
            <div style={{ marginBottom: '1.4rem' }}>
              <label
                style={{
                  display: 'block',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.7rem',
                  letterSpacing: '0.1em',
                  color: 'var(--muted)',
                  marginBottom: '0.6rem',
                }}
              >
                USERNAME
              </label>

              <div style={{ position: 'relative' }}>
                <User
                  size={17}
                  color="var(--cyan)"
                  style={{
                    position: 'absolute',
                    left: 15,
                    top: '50%',
                    transform: 'translateY(-50%)',
                  }}
                />

                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter username"
                  required
                  autoComplete="username"
                  style={{
                    width: '100%',
                    padding: '0.95rem 1rem 0.95rem 2.8rem',
                    background: 'rgba(0,0,10,0.55)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 12,
                    color: 'var(--text)',
                    outline: 'none',
                    fontFamily: 'var(--font-body)',
                    fontSize: '0.95rem',
                  }}
                />
              </div>
            </div>

            {/* Password */}
            <div style={{ marginBottom: '1.5rem' }}>
              <label
                style={{
                  display: 'block',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.7rem',
                  letterSpacing: '0.1em',
                  color: 'var(--muted)',
                  marginBottom: '0.6rem',
                }}
              >
                PASSWORD
              </label>

              <div style={{ position: 'relative' }}>
                <Lock
                  size={17}
                  color="var(--purple)"
                  style={{
                    position: 'absolute',
                    left: 15,
                    top: '50%',
                    transform: 'translateY(-50%)',
                  }}
                />

                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter password"
                  required
                  autoComplete="current-password"
                  style={{
                    width: '100%',
                    padding: '0.95rem 1rem 0.95rem 2.8rem',
                    background: 'rgba(0,0,10,0.55)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 12,
                    color: 'var(--text)',
                    outline: 'none',
                    fontFamily: 'var(--font-body)',
                    fontSize: '0.95rem',
                  }}
                />
              </div>
            </div>

            {/* Error */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.6rem',
                  padding: '0.8rem 1rem',
                  marginBottom: '1.2rem',
                  borderRadius: 10,
                  background: 'rgba(255,45,85,0.08)',
                  border: '1px solid rgba(255,45,85,0.25)',
                  color: '#ff6b88',
                  fontSize: '0.82rem',
                }}
              >
                <AlertCircle size={16} />
                {error}
              </motion.div>
            )}

            {/* Button */}
            <motion.button
              type="submit"
              disabled={loading}
              whileHover={!loading ? {
                scale: 1.02,
                boxShadow:
                  '0 0 40px rgba(123,47,255,0.45), 0 0 80px rgba(0,245,255,0.15)',
              } : {}}
              whileTap={!loading ? { scale: 0.98 } : {}}
              style={{
                width: '100%',
                padding: '1rem',
                borderRadius: 12,
                border: 'none',
                background:
                  'linear-gradient(135deg, var(--purple), var(--cyan))',
                color: 'white',
                cursor: loading ? 'wait' : 'pointer',
                fontFamily: 'var(--font-body)',
                fontSize: '0.95rem',
                fontWeight: 700,
                letterSpacing: '0.08em',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.7rem',
                opacity: loading ? 0.7 : 1,
              }}
            >
              {loading ? 'AUTHENTICATING...' : 'AUTHENTICATE'}
              {!loading && <ArrowRight size={18} />}
            </motion.button>

          </form>
        </motion.div>

        {/* Footer */}
        <div
          style={{
            textAlign: 'center',
            marginTop: '1.5rem',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.65rem',
            letterSpacing: '0.08em',
            color: 'rgba(232,232,240,0.25)',
          }}
        >
          FEDSHIELD // PRIVACY-PRESERVING INTRUSION DETECTION
        </div>

      </motion.div>
    </div>
  )
}