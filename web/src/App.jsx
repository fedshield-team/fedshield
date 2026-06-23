import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import ParticleUniverse from './components/ParticleUniverse'
import Nav from './components/Nav'
import Landing from './pages/Landing'
import SOCMonitor from './pages/SOCMonitor'
import Overview from './pages/Overview'
import Training from './pages/Training'
import SHAP from './pages/SHAP'

function AnimatedRoutes() {
  const location = useLocation()
  const isLanding = location.pathname === '/'
  const isSOC     = location.pathname === '/soc'

  return (
    <>
      {/* Subtle particle bg for inner pages only */}
      {!isLanding && !isSOC && <ParticleUniverse intense={false} />}

      {/* Inner page dark overlay */}
      {!isLanding && !isSOC && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 1, pointerEvents: 'none',
          background: 'radial-gradient(ellipse at 50% 0%, rgba(3,0,15,0.6) 0%, rgba(0,0,5,0.92) 60%)',
        }} />
      )}

      {!isLanding && <Nav />}

      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/"         element={<Landing />} />
          <Route path="/soc"      element={<SOCMonitor />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/training" element={<Training />} />
          <Route path="/shap"     element={<SHAP />} />
        </Routes>
      </AnimatePresence>
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AnimatedRoutes />
    </BrowserRouter>
  )
}