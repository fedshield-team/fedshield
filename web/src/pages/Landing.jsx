import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import * as THREE from 'three'
import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'

export default function Landing() {
  const mountRef = useRef(null)
  const navigate = useNavigate()

  useEffect(() => {
    const W = window.innerWidth, H = window.innerHeight
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(W, H)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mountRef.current.appendChild(renderer.domElement)

    const scene  = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000)
    camera.position.set(0, 0, 28)

    // ── Mouse tracking ────────────────────────────────────────────────────
    const mouse = { x: 0, y: 0, tx: 0, ty: 0 }
    const onMouseMove = e => {
      mouse.tx = (e.clientX / W - 0.5) * 2
      mouse.ty = (e.clientY / H - 0.5) * 2
    }
    window.addEventListener('mousemove', onMouseMove)

    // ── Black hole particle ring ──────────────────────────────────────────
    const ringCount = 12000
    const ringPos   = new Float32Array(ringCount * 3)
    const ringCol   = new Float32Array(ringCount * 3)
    const ringSpeeds = new Float32Array(ringCount)

    for (let i = 0; i < ringCount; i++) {
      const theta  = Math.random() * Math.PI * 2
      const r      = 3 + Math.random() * 7
      const spread = (Math.random() - 0.5) * 0.6
      ringPos[i*3]   = Math.cos(theta) * r
      ringPos[i*3+1] = spread
      ringPos[i*3+2] = Math.sin(theta) * r
      const t = (r - 3) / 7
      // Vivid cyan → hot purple → magenta
      ringCol[i*3]   = t * 0.8 + (1-t) * 0.0
      ringCol[i*3+1] = t * 0.0 + (1-t) * 0.9
      ringCol[i*3+2] = 1.0
      ringSpeeds[i]  = 0.003 + Math.random() * 0.012
    }
    const ringGeo = new THREE.BufferGeometry()
    ringGeo.setAttribute('position', new THREE.BufferAttribute(ringPos, 3))
    ringGeo.setAttribute('color',    new THREE.BufferAttribute(ringCol, 3))
    const ringMat = new THREE.PointsMaterial({
      size: 0.04, vertexColors: true, transparent: true,
      opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    })
    const ring = new THREE.Points(ringGeo, ringMat)
    scene.add(ring)

    // ── Core black sphere ─────────────────────────────────────────────────
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(0.8, 32, 32),
      new THREE.MeshBasicMaterial({ color: 0x000000 })
    )
    scene.add(core)

    // ── Glow rings ────────────────────────────────────────────────────────
    ;[1.2, 1.8, 2.5].forEach((r, i) => {
      const g = new THREE.Mesh(
        new THREE.SphereGeometry(r, 32, 32),
        new THREE.MeshBasicMaterial({
          color: i === 0 ? 0x7b2fff : i === 1 ? 0x00f5ff : 0xff2d8f,
          transparent: true, opacity: 0.04 - i * 0.008
        })
      )
      scene.add(g)
    })

    // ── Star field ────────────────────────────────────────────────────────
    const starCount = 5000
    const starPos   = new Float32Array(starCount * 3)
    for (let i = 0; i < starCount; i++) {
      starPos[i*3]   = (Math.random() - 0.5) * 200
      starPos[i*3+1] = (Math.random() - 0.5) * 200
      starPos[i*3+2] = (Math.random() - 0.5) * 200
    }
    const starGeo = new THREE.BufferGeometry()
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3))
    const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({
      size: 0.08, color: 0xffffff, transparent: true, opacity: 0.6, sizeAttenuation: true
    }))
    scene.add(stars)

    // ── Floating data stream particles ────────────────────────────────────
    const streamCount = 1500
    const streamPos   = new Float32Array(streamCount * 3)
    const streamCol   = new Float32Array(streamCount * 3)
    const streamVel   = new Float32Array(streamCount * 3)
    for (let i = 0; i < streamCount; i++) {
      streamPos[i*3]   = (Math.random() - 0.5) * 40
      streamPos[i*3+1] = (Math.random() - 0.5) * 20
      streamPos[i*3+2] = (Math.random() - 0.5) * 20 - 5
      streamVel[i*3]   = (Math.random() - 0.5) * 0.02
      streamVel[i*3+1] = -0.02 - Math.random() * 0.04
      streamVel[i*3+2] = 0
      streamCol[i*3]   = 0.0
      streamCol[i*3+1] = Math.random()
      streamCol[i*3+2] = 0.53
    }
    const streamGeo = new THREE.BufferGeometry()
    streamGeo.setAttribute('position', new THREE.BufferAttribute(streamPos, 3))
    streamGeo.setAttribute('color',    new THREE.BufferAttribute(streamCol, 3))
    const streamMesh = new THREE.Points(streamGeo, new THREE.PointsMaterial({
      size: 0.06, vertexColors: true, transparent: true, opacity: 0.7,
      blending: THREE.AdditiveBlending, depthWrite: false
    }))
    scene.add(streamMesh)

    // ── Holographic floating grid panels ─────────────────────────────────
    const panels = []
    const panelData = [
      { x: -14, y:  4, z: -8,  rx: 0.1,  ry:  0.3,  color: 0x00f5ff },
      { x:  14, y: -3, z: -6,  rx: -0.1, ry: -0.25, color: 0x7b2fff },
      { x:  -9, y: -6, z: -10, rx: 0.2,  ry:  0.15, color: 0xff2d8f },
      { x:  10, y:  7, z: -9,  rx: -0.15,ry: -0.2,  color: 0x00ff88 },
    ]

    panelData.forEach(d => {
      // Grid lines
      const gridGeo = new THREE.EdgesGeometry(new THREE.PlaneGeometry(5, 3.5, 8, 6))
      const gridMat = new THREE.LineBasicMaterial({
        color: d.color, transparent: true, opacity: 0.25
      })
      const grid = new THREE.LineSegments(gridGeo, gridMat)
      grid.position.set(d.x, d.y, d.z)
      grid.rotation.set(d.rx, d.ry, 0)
      scene.add(grid)

      // Glass fill
      const fillGeo = new THREE.PlaneGeometry(5, 3.5)
      const fillMat = new THREE.MeshBasicMaterial({
        color: d.color, transparent: true, opacity: 0.03, side: THREE.DoubleSide
      })
      const fill = new THREE.Mesh(fillGeo, fillMat)
      fill.position.set(d.x, d.y, d.z - 0.01)
      fill.rotation.set(d.rx, d.ry, 0)
      scene.add(fill)

      panels.push({ grid, fill, baseX: d.x, baseY: d.y, phase: Math.random() * Math.PI * 2 })
    })

    // ── Burst particles (attack simulation) ──────────────────────────────
    const burstCount = 400
    const burstPos   = new Float32Array(burstCount * 3)
    const burstVel   = new Float32Array(burstCount * 3)
    const burstLife  = new Float32Array(burstCount)
    let   burstActive = false, burstTimer = 0

    const triggerBurst = (x, y, z) => {
      for (let i = 0; i < burstCount; i++) {
        burstPos[i*3]   = x; burstPos[i*3+1] = y; burstPos[i*3+2] = z
        const speed = 0.08 + Math.random() * 0.18
        const theta = Math.random() * Math.PI * 2
        const phi   = Math.random() * Math.PI
        burstVel[i*3]   = Math.sin(phi) * Math.cos(theta) * speed
        burstVel[i*3+1] = Math.sin(phi) * Math.sin(theta) * speed
        burstVel[i*3+2] = Math.cos(phi) * speed
        burstLife[i]    = 1.0
      }
      burstActive = true
    }

    const burstGeo = new THREE.BufferGeometry()
    burstGeo.setAttribute('position', new THREE.BufferAttribute(burstPos, 3))
    const burstMesh = new THREE.Points(burstGeo, new THREE.PointsMaterial({
      size: 0.08, color: 0xff2d55, transparent: true, opacity: 0.0,
      blending: THREE.AdditiveBlending, depthWrite: false
    }))
    scene.add(burstMesh)

    // Auto-burst every 4s to simulate attacks
    const burstInterval = setInterval(() => {
      triggerBurst((Math.random()-0.5)*8, (Math.random()-0.5)*4, (Math.random()-0.5)*4)
    }, 4000)
    setTimeout(() => triggerBurst(0, 0, 0), 800)

    // ── Animate ───────────────────────────────────────────────────────────
    let frame
    const clock = new THREE.Clock()

    const animate = () => {
      frame = requestAnimationFrame(animate)
      const t = clock.getElapsedTime()
      const dt = clock.getDelta ? 0.016 : 0.016

      // Smooth mouse follow
      mouse.x += (mouse.tx - mouse.x) * 0.04
      mouse.y += (mouse.ty - mouse.y) * 0.04

      // Ring rotation
      ring.rotation.y = t * 0.07
      ring.rotation.x = Math.sin(t * 0.04) * 0.18

      // Stars slow drift
      stars.rotation.y = t * 0.003
      stars.rotation.x = t * 0.001

      // Stream particles fall and reset
      const sp = streamGeo.attributes.position.array
      for (let i = 0; i < streamCount; i++) {
        sp[i*3]   += streamVel[i*3]
        sp[i*3+1] += streamVel[i*3+1]
        if (sp[i*3+1] < -12) {
          sp[i*3]   = (Math.random() - 0.5) * 40
          sp[i*3+1] = 12
          sp[i*3+2] = (Math.random() - 0.5) * 20 - 5
        }
      }
      streamGeo.attributes.position.needsUpdate = true

      // Panels float + parallax
      panels.forEach((p, i) => {
        const floatY = Math.sin(t * 0.4 + p.phase) * 0.4
        p.grid.position.y = p.baseY + floatY - mouse.y * 2
        p.grid.position.x = p.baseX + mouse.x * 1.5
        p.fill.position.y = p.baseY + floatY - mouse.y * 2
        p.fill.position.x = p.baseX + mouse.x * 1.5
        p.grid.material.opacity = 0.15 + Math.sin(t * 0.5 + p.phase) * 0.1
      })

      // Camera parallax
      camera.position.x = mouse.x * 2
      camera.position.y = -mouse.y * 1.5 + 1
      camera.lookAt(0, 0, 0)

      // Burst animation
      if (burstActive) {
        burstTimer += 0.016
        const bp = burstGeo.attributes.position.array
        const alive = burstTimer < 2.5
        burstMesh.material.opacity = alive ? Math.max(0, 1 - burstTimer / 2.5) : 0
        if (alive) {
          for (let i = 0; i < burstCount; i++) {
            bp[i*3]   += burstVel[i*3]
            bp[i*3+1] += burstVel[i*3+1]
            bp[i*3+2] += burstVel[i*3+2]
            burstVel[i*3]   *= 0.97
            burstVel[i*3+1] *= 0.97
            burstVel[i*3+2] *= 0.97
          }
          burstGeo.attributes.position.needsUpdate = true
        } else {
          burstActive = false; burstTimer = 0
        }
      }

      renderer.render(scene, camera)
    }
    animate()

    const onResize = () => {
      const W2 = window.innerWidth, H2 = window.innerHeight
      camera.aspect = W2 / H2; camera.updateProjectionMatrix()
      renderer.setSize(W2, H2)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(frame)
      clearInterval(burstInterval)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('resize', onResize)
      if (mountRef.current) mountRef.current.removeChild(renderer.domElement)
      renderer.dispose()
    }
  }, [])

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 1 }}>
      {/* Three.js canvas */}
      <div ref={mountRef} style={{ position: 'absolute', inset: 0, zIndex: 0 }} />

      {/* Radial vignette */}
      <div style={{
        position: 'absolute', inset: 0, zIndex: 1, pointerEvents: 'none',
        background: 'radial-gradient(ellipse 70% 70% at 50% 50%, transparent 30%, rgba(0,0,5,0.85) 100%)',
      }} />

      {/* Content */}
      <div style={{
        position: 'absolute', inset: 0, zIndex: 2,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        textAlign: 'center', padding: '2rem',
        pointerEvents: 'none',
      }}>
        {/* Eyebrow */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: '0.6rem',
            padding: '0.55rem 1.3rem', borderRadius: 100,
            border: '1px solid rgba(0,245,255,0.4)',
            background: 'rgba(0,245,255,0.07)',
            backdropFilter: 'blur(12px)',
            fontSize: '0.76rem', letterSpacing: '0.15em',
            color: '#00f5ff', marginBottom: '2.5rem',
            fontFamily: 'var(--font-mono)',
          }}
        >
          <motion.div animate={{ opacity: [1,0,1] }} transition={{ duration: 1.4, repeat: Infinity }}
            style={{ width: 6, height: 6, borderRadius: '50%', background: '#00ff88' }} />
          PRIVACY-PRESERVING INTRUSION DETECTION
        </motion.div>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 50, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ delay: 0.7, duration: 1, ease: [0.16, 1, 0.3, 1] }}
          style={{
            fontFamily: 'var(--font-display)', fontWeight: 800,
            fontSize: 'clamp(5rem, 12vw, 11rem)',
            lineHeight: 0.9, letterSpacing: '-0.04em',
            marginBottom: '2rem',
            background: 'linear-gradient(135deg, #ffffff 0%, #c4b5fd 35%, #00f5ff 70%, #ff2d8f 100%)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            filter: 'drop-shadow(0 0 40px rgba(123,47,255,0.4))',
          }}
        >
          FedShield
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0, duration: 0.8 }}
          style={{
            fontSize: 'clamp(1.1rem, 2.5vw, 1.45rem)',
            color: 'rgba(232,232,240,0.72)',
            maxWidth: 600, lineHeight: 1.65,
            marginBottom: '3.5rem', fontWeight: 300,
          }}
        >
          Federated learning meets real-time threat detection.
          <br />Security and privacy — simultaneously.
        </motion.p>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2, duration: 0.8 }}
          style={{
            display: 'flex', gap: '0', marginBottom: '3.5rem',
            background: 'rgba(0,0,10,0.6)', backdropFilter: 'blur(20px)',
            borderRadius: 16, overflow: 'hidden',
            border: '1px solid rgba(255,255,255,0.08)',
            boxShadow: '0 0 60px rgba(123,47,255,0.15)',
          }}
        >
          {[
            { v: '0.9946', l: 'Federated F1' },
            { v: '0.84',   l: 'Non-IID Macro F1' },
            { v: '3',      l: 'Edge Nodes' },
            { v: '100%',   l: 'Privacy' },
          ].map((s, i, arr) => (
            <div key={i} style={{
              padding: '1.5rem 2.8rem', textAlign: 'center',
              borderRight: i < arr.length-1 ? '1px solid rgba(255,255,255,0.07)' : 'none',
            }}>
              <div style={{
                fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '2.4rem',
                background: 'linear-gradient(90deg, #c4b5fd, #00f5ff)',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              }}>{s.v}</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--muted)', letterSpacing: '0.1em', marginTop: 6 }}>{s.l}</div>
            </div>
          ))}
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.4, duration: 0.7 }}
          style={{ display: 'flex', gap: '1rem', pointerEvents: 'all' }}
        >
          <motion.button
            whileHover={{ scale: 1.06, boxShadow: '0 0 60px rgba(123,47,255,0.7), 0 0 120px rgba(0,245,255,0.3)' }}
            whileTap={{ scale: 0.96 }}
            onClick={() => navigate('/soc')}
            style={{
              display: 'flex', alignItems: 'center', gap: '0.8rem',
              padding: '1.1rem 2.6rem', borderRadius: 14,
              background: 'linear-gradient(135deg, #7b2fff 0%, #00f5ff 100%)',
              border: 'none', color: 'white', cursor: 'pointer',
              fontSize: '1.05rem', fontWeight: 700,
              fontFamily: 'var(--font-body)', letterSpacing: '0.06em',
              boxShadow: '0 0 30px rgba(123,47,255,0.4)',
            }}
          >
            ENTER SHIELD <ArrowRight size={18} />
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.04, borderColor: 'rgba(0,245,255,0.4)' }}
            whileTap={{ scale: 0.96 }}
            onClick={() => navigate('/overview')}
            style={{
              padding: '1.1rem 2.6rem', borderRadius: 14,
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.14)',
              backdropFilter: 'blur(10px)',
              color: 'rgba(232,232,240,0.85)', cursor: 'pointer',
              fontSize: '1.05rem', fontWeight: 500,
              fontFamily: 'var(--font-body)',
              transition: 'border-color 0.25s',
            }}
          >
            View Results
          </motion.button>
        </motion.div>
      </div>

      {/* Corner scan lines */}
      <div style={{
        position: 'absolute', inset: 0, zIndex: 1, pointerEvents: 'none',
        backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,245,255,0.012) 2px, rgba(0,245,255,0.012) 4px)',
      }} />

      {/* Scroll hint */}
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.5 }}
        style={{
          position: 'absolute', bottom: '2rem', left: '50%', transform: 'translateX(-50%)',
          zIndex: 3, fontSize: '0.72rem', color: 'rgba(0,245,255,0.5)',
          letterSpacing: '0.14em', fontFamily: 'var(--font-mono)',
        }}
      >
        <motion.div animate={{ y: [0,7,0] }} transition={{ duration: 2.2, repeat: Infinity }}>
          ↓ EXPLORE
        </motion.div>
      </motion.div>
    </div>
  )
}