import { BrowserRouter, Routes, Route, useLocation, Navigate } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { isAuthenticated } from './api'
import ParticleUniverse from './components/ParticleUniverse'
import Nav from './components/Nav'
import Landing from './pages/Landing'
import Login from './pages/Login'
import SOCMonitor from './pages/SOCMonitor'
import Overview from './pages/Overview'
import Training from './pages/Training'
import SHAP from './pages/SHAP'

function AnimatedRoutes() {
  const location = useLocation()
  const isLanding = location.pathname === '/'
  const isSOC     = location.pathname === '/soc'
  const isLogin   = location.pathname === '/login'

  return (
    <>
      {/* Subtle particle bg for inner pages only */}
      {!isLanding && !isSOC && !isLogin && <ParticleUniverse intense={false} />}

      {/* Inner page dark overlay */}
      {!isLanding && !isSOC && !isLogin && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 1, pointerEvents: 'none',
          background: 'radial-gradient(ellipse at 50% 0%, rgba(3,0,15,0.6) 0%, rgba(0,0,5,0.92) 60%)',
        }} />
      )}

      {!isLanding && !isLogin && <Nav />}

      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/"         element={<Landing />} />
          <Route path="/login"    element={<Login />} />
          <Route path="/soc"      element={<Protected><SOCMonitor /></Protected>} />
          <Route path="/overview" element={<Protected><Overview /></Protected>} />
          <Route path="/training" element={<Protected><Training /></Protected>} />
          <Route path="/shap"     element={<Protected><SHAP /></Protected>} />
        </Routes>
      </AnimatePresence>
    </>
  )
}

function Protected({ children }) {
  return isAuthenticated() ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <BrowserRouter>
      <AnimatedRoutes />
    </BrowserRouter>
  )
}