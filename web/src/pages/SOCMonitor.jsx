import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import * as THREE from 'three'
import { AlertTriangle, Ban, Radio, Activity } from 'lucide-react'
import { PieChart, Pie, Cell, AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { api } from '../api'

const COLORS = { Normal:'#00ff88', DoS:'#ff2d55', Probe:'#ff9500', R2L:'#ff2d8f', U2R:'#bf5af2' }

const CARD = {
  background: 'rgba(0,0,15,0.72)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 18,
  backdropFilter: 'blur(24px)',
  overflow: 'hidden',
}

function AttackGlobe() {
  const mountRef = useRef(null)

  useEffect(() => {
    const el = mountRef.current
    if (!el) return
    const W = el.clientWidth, H = el.clientHeight
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(W, H)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)

    const scene  = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 100)
    camera.position.set(0, 0, 4.5)

    const globeGeo   = new THREE.SphereGeometry(1.8, 32, 32)
    const globeEdges = new THREE.EdgesGeometry(globeGeo)
    const globe = new THREE.LineSegments(globeEdges,
      new THREE.LineBasicMaterial({ color: 0x00f5ff, transparent: true, opacity: 0.12 })
    )
    scene.add(globe)
    scene.add(new THREE.Mesh(globeGeo,
      new THREE.MeshBasicMaterial({ color: 0x000510, transparent: true, opacity: 0.85 })
    ))

    for (let lat = -60; lat <= 60; lat += 30) {
      const r = Math.cos(lat * Math.PI / 180) * 1.8
      const y = Math.sin(lat * Math.PI / 180) * 1.8
      const pts = []
      for (let i = 0; i <= 64; i++) {
        const a = (i / 64) * Math.PI * 2
        pts.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r))
      }
      scene.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0x00f5ff, transparent: true, opacity: 0.06 })
      ))
    }

    for (let lon = 0; lon < 180; lon += 30) {
      const pts = []
      for (let i = 0; i <= 64; i++) {
        const lat = (i / 64) * Math.PI * 2 - Math.PI
        const a   = lon * Math.PI / 180
        pts.push(new THREE.Vector3(
          Math.cos(lat) * Math.cos(a) * 1.8,
          Math.sin(lat) * 1.8,
          Math.cos(lat) * Math.sin(a) * 1.8
        ))
      }
      scene.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0x00f5ff, transparent: true, opacity: 0.06 })
      ))
    }

    const latLon3D = (lat, lon, r) => {
      const phi   = (90 - lat)  * Math.PI / 180
      const theta = (lon + 180) * Math.PI / 180
      return new THREE.Vector3(
        -r * Math.sin(phi) * Math.cos(theta),
         r * Math.cos(phi),
         r * Math.sin(phi) * Math.sin(theta)
      )
    }

    const NODES = [
      { lat: 17.4, lon: 78.5,  color: 0x00f5ff },
      { lat: 19.1, lon: 72.9,  color: 0x7b2fff },
      { lat: 1.35, lon: 103.8, color: 0x00ff88 },
    ]

    const nodeObjs = NODES.map((n, i) => {
      const pos = latLon3D(n.lat, n.lon, 1.8)
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.07, 12, 12),
        new THREE.MeshBasicMaterial({ color: n.color })
      )
      dot.position.copy(pos)
      scene.add(dot)
      const glow = new THREE.Mesh(
        new THREE.SphereGeometry(0.14, 12, 12),
        new THREE.MeshBasicMaterial({ color: n.color, transparent: true, opacity: 0.2 })
      )
      glow.position.copy(pos)
      scene.add(glow)
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.10, 0.16, 24),
        new THREE.MeshBasicMaterial({ color: n.color, transparent: true, opacity: 0.7, side: THREE.DoubleSide })
      )
      ring.position.copy(pos)
      ring.lookAt(0, 0, 0)
      scene.add(ring)
      return { pos, dot, glow, ring, color: n.color, phase: i * 2.1 }
    })

    for (let i = 0; i < NODES.length; i++) {
      const from = nodeObjs[i].pos
      const to   = nodeObjs[(i + 1) % NODES.length].pos
      const count = 80
      const arcPos = new Float32Array(count * 3)
      for (let j = 0; j < count; j++) {
        const t   = j / count
        const mid = from.clone().lerp(to, 0.5).normalize().multiplyScalar(2.6)
        const pt  = new THREE.QuadraticBezierCurve3(from, mid, to).getPoint(t)
        arcPos[j*3]=pt.x; arcPos[j*3+1]=pt.y; arcPos[j*3+2]=pt.z
      }
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(arcPos, 3))
      scene.add(new THREE.Points(geo, new THREE.PointsMaterial({
        size: 0.05, color: nodeObjs[i].color, transparent: true,
        opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false
      })))
    }

    const BURST = 300
    const bPos = new Float32Array(BURST * 3)
    const bVel = new Float32Array(BURST * 3)
    const bGeo = new THREE.BufferGeometry()
    bGeo.setAttribute('position', new THREE.BufferAttribute(bPos, 3))
    const bMesh = new THREE.Points(bGeo, new THREE.PointsMaterial({
      size: 0.08, color: 0xff2d55, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false
    }))
    scene.add(bMesh)

    let bActive = false, bTimer = 0
    const triggerBurst = () => {
      const origin = nodeObjs[Math.floor(Math.random() * 3)].pos
      for (let i = 0; i < BURST; i++) {
        bPos[i*3]=origin.x; bPos[i*3+1]=origin.y; bPos[i*3+2]=origin.z
        const s = 0.03 + Math.random() * 0.08
        const t = Math.random()*Math.PI*2, p = Math.random()*Math.PI
        bVel[i*3]=Math.sin(p)*Math.cos(t)*s; bVel[i*3+1]=Math.sin(p)*Math.sin(t)*s; bVel[i*3+2]=Math.cos(p)*s
      }
      bActive = true; bTimer = 0
      bGeo.attributes.position.needsUpdate = true
    }
    const burstInt = setInterval(triggerBurst, 4000)
    setTimeout(triggerBurst, 600)

    let frame
    const clock = new THREE.Clock()
    const animate = () => {
      frame = requestAnimationFrame(animate)
      const t = clock.getElapsedTime()
      globe.rotation.y = t * 0.07
      nodeObjs.forEach(n => {
        const s = 0.85 + Math.sin(t * 2.2 + n.phase) * 0.4
        n.ring.scale.setScalar(s)
        n.ring.material.opacity = 0.8 - s * 0.35
        n.glow.material.opacity = 0.15 + Math.sin(t * 1.5 + n.phase) * 0.1
      })
      if (bActive) {
        bTimer += 0.016
        const alive = bTimer < 2
        bMesh.material.opacity = alive ? Math.max(0, 0.9 - bTimer / 2) : 0
        if (alive) {
          const bp = bGeo.attributes.position.array
          for (let i = 0; i < BURST; i++) {
            bp[i*3]+=bVel[i*3]; bp[i*3+1]+=bVel[i*3+1]; bp[i*3+2]+=bVel[i*3+2]
            bVel[i*3]*=0.96; bVel[i*3+1]*=0.96; bVel[i*3+2]*=0.96
          }
          bGeo.attributes.position.needsUpdate = true
        } else { bActive = false }
      }
      renderer.render(scene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(frame)
      clearInterval(burstInt)
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement)
      renderer.dispose()
    }
  }, [])

  return <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
}

const TH = { padding:'0.75rem 1rem', fontSize:'0.7rem', color:'var(--muted)', letterSpacing:'0.1em', fontWeight:500, textAlign:'left', borderBottom:'1px solid rgba(255,255,255,0.06)' }
const TD = { padding:'0.7rem 1rem', fontSize:'0.82rem', color:'var(--text)', verticalAlign:'middle', whiteSpace:'nowrap', borderBottom:'1px solid rgba(255,255,255,0.03)' }

export default function SOCMonitor() {
  const [stats,     setStats]     = useState({ total:0, attacks:0, blocked:0, normal:0 })
  const [feed,      setFeed]      = useState([])
  const [breakdown, setBreakdown] = useState([])
  const [timeline,  setTimeline]  = useState([])
  const [blocked,   setBlocked]   = useState([])

  const fetchAll = async () => {
    try {
      const [s,f,b,t,bl] = await Promise.all([
        api.stats(),
        api.feed(40),
        api.breakdown(),
        api.timeline(),
        api.blocked(),
      ])
      setStats(s); setFeed(f); setBreakdown(b); setTimeline(t); setBlocked(bl)
    } catch(e) {}
  }

  useEffect(() => {
    fetchAll()
    const id = setInterval(fetchAll, 3000)
    return () => clearInterval(id)
  }, [])

  const atkTimeline = timeline
    .filter(r => r.tag === 'ATTACK')
    .slice(0, 60).reverse()
    .map(r => ({ time: r.timestamp, v: r.count }))

  return (
    <motion.div
      initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
      style={{ position:'relative', zIndex:10, minHeight:'100vh', padding:'80px 2rem 2rem',
               background:'radial-gradient(ellipse at 50% 0%,rgba(123,47,255,0.07) 0%,transparent 55%)' }}
    >
      <div style={{ display:'flex', alignItems:'center', gap:'1rem', marginBottom:'1.8rem' }}>
        <motion.div animate={{ opacity:[1,0.2,1] }} transition={{ duration:1.2, repeat:Infinity }}
          style={{ width:12, height:12, borderRadius:'50%', background:'#ff2d55', boxShadow:'0 0 16px #ff2d55' }} />
        <h1 style={{
          fontFamily:'var(--font-display)', fontWeight:800, fontSize:'2.4rem',
          background:'linear-gradient(90deg,#fff,#ff2d55)',
          WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent'
        }}>Live SOC Monitor</h1>
        <div style={{ marginLeft:'auto', fontSize:'0.72rem', color:'var(--muted)', fontFamily:'var(--font-mono)' }}>
          AUTO-REFRESH 3s
        </div>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:'1rem', marginBottom:'1.5rem' }}>
        {[
          { icon:Radio,         l:'PACKETS',  v:stats.total,   c:'#00f5ff' },
          { icon:Activity,      l:'NORMAL',   v:stats.normal,  c:'#00ff88' },
          { icon:AlertTriangle, l:'ATTACKS',  v:stats.attacks, c:'#ff2d55' },
          { icon:Ban,           l:'BLOCKED',  v:stats.blocked, c:'#ff9500' },
        ].map(({icon:Icon,l,v,c},i) => (
          <motion.div key={i}
            initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:0.1*i }}
            style={{ ...CARD, padding:'1.5rem', border:`1px solid ${c}22`,
                     background:`linear-gradient(135deg,rgba(0,0,15,0.8),${c}06)` }}>
            <div style={{ fontSize:'0.68rem', color:'var(--muted)', letterSpacing:'0.12em', marginBottom:'0.5rem' }}>{l}</div>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-end' }}>
              <motion.div key={v} initial={{ scale:1.15 }} animate={{ scale:1 }}
                style={{ fontFamily:'var(--font-display)', fontWeight:800, fontSize:'2.8rem', color:c, lineHeight:1 }}>
                {v?.toLocaleString()}
              </motion.div>
              <div style={{ width:40, height:40, borderRadius:10, background:`${c}18`,
                            border:`1px solid ${c}30`, display:'flex', alignItems:'center', justifyContent:'center' }}>
                <Icon size={18} color={c} />
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity:0, scale:0.95 }} animate={{ opacity:1, scale:1 }} transition={{ delay:0.5 }}
        style={{ ...CARD, height:340, marginBottom:'1.5rem', position:'relative',
                 border:'1px solid rgba(0,245,255,0.2)',
                 background:'linear-gradient(135deg,rgba(0,0,20,0.9),rgba(0,10,30,0.8))' }}
      >
        <div style={{ position:'absolute', top:'1rem', left:'1.4rem', zIndex:2 }}>
          <div style={{ fontSize:'0.68rem', color:'var(--muted)', letterSpacing:'0.12em', marginBottom:'0.5rem' }}>
            FEDERATED NODES — LIVE
          </div>
          <div style={{ display:'flex', gap:'1.2rem' }}>
            {[['#00f5ff','Hospital','Hyderabad'],['#7b2fff','Bank','Mumbai'],['#00ff88','Campus','Singapore']].map(([c,n,l],i)=>(
              <div key={i} style={{ display:'flex', alignItems:'center', gap:'0.4rem' }}>
                <div style={{ width:8, height:8, borderRadius:'50%', background:c, boxShadow:`0 0 8px ${c}` }} />
                <div>
                  <div style={{ fontSize:'0.75rem', color:c, fontWeight:600 }}>{n}</div>
                  <div style={{ fontSize:'0.65rem', color:'var(--muted)' }}>{l}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <AttackGlobe />
      </motion.div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 2fr', gap:'1rem', marginBottom:'1.5rem' }}>
        <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:0.6 }}
          style={{ ...CARD, padding:'1.3rem' }}>
          <div style={{ fontSize:'0.68rem', letterSpacing:'0.1em', color:'var(--muted)', marginBottom:'1rem' }}>ATTACK BREAKDOWN</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={breakdown} dataKey="count" nameKey="prediction"
                cx="50%" cy="50%" innerRadius={65} outerRadius={95} paddingAngle={4}>
                {breakdown.map((e,i) => <Cell key={i} fill={COLORS[e.prediction]||'#888'} strokeWidth={0} />)}
              </Pie>
              <Tooltip contentStyle={{ background:'#050510', border:'1px solid #222', borderRadius:8, fontSize:'0.8rem' }} />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display:'flex', flexWrap:'wrap', gap:'0.5rem', marginTop:'0.5rem' }}>
            {breakdown.map((b,i) => (
              <div key={i} style={{ display:'flex', alignItems:'center', gap:'0.35rem', fontSize:'0.75rem', color:'var(--muted)' }}>
                <div style={{ width:8, height:8, borderRadius:2, background:COLORS[b.prediction]||'#888' }} />
                {b.prediction} ({b.count?.toLocaleString()})
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:0.7 }}
          style={{ ...CARD, padding:'1.3rem' }}>
          <div style={{ fontSize:'0.68rem', letterSpacing:'0.1em', color:'var(--muted)', marginBottom:'1rem' }}>ATTACK TIMELINE</div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={atkTimeline}>
              <defs>
                <linearGradient id="ag2" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#ff2d55" stopOpacity={0.5} />
                  <stop offset="100%" stopColor="#ff2d55" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="time" tick={false} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill:'#444', fontSize:11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background:'#050510', border:'1px solid #222', borderRadius:8, fontSize:'0.8rem' }} />
              <Area type="monotone" dataKey="v" stroke="#ff2d55" fill="url(#ag2)" strokeWidth={2.5} />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {blocked.length > 0 && (
        <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:0.8 }}
          style={{ ...CARD, padding:'1.2rem 1.4rem', marginBottom:'1.5rem',
                   border:'1px solid rgba(255,149,0,0.3)', background:'rgba(255,149,0,0.04)' }}>
          <div style={{ fontSize:'0.68rem', letterSpacing:'0.1em', color:'#ff9500', marginBottom:'0.9rem' }}>🛡️ AUTO-BLOCKED</div>
          <div style={{ display:'flex', gap:'0.6rem', flexWrap:'wrap' }}>
            {blocked.map((b,i) => (
              <div key={i} style={{ padding:'0.45rem 0.9rem', borderRadius:8,
                background:'rgba(255,149,0,0.12)', border:'1px solid rgba(255,149,0,0.3)',
                fontFamily:'var(--font-mono)', fontSize:'0.8rem', color:'#ff9500' }}>
                {b.src} <span style={{ opacity:0.6 }}>({b.prediction})</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:0.9 }}
        style={{ ...CARD }}>
        <div style={{ padding:'1rem 1.4rem', borderBottom:'1px solid rgba(255,255,255,0.06)',
                      display:'flex', justifyContent:'space-between', alignItems:'center' }}>
          <div style={{ fontSize:'0.68rem', letterSpacing:'0.1em', color:'var(--muted)' }}>📡 LIVE PACKET FEED</div>
          <motion.div animate={{ opacity:[1,0.3,1] }} transition={{ duration:1.5, repeat:Infinity }}
            style={{ fontSize:'0.68rem', color:'#00ff88', fontFamily:'var(--font-mono)' }}>● LIVE</motion.div>
        </div>
        <div style={{ overflowX:'auto' }}>
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead>
              <tr>{['Time','Source','Destination','Proto','Prediction','Confidence','Status'].map(h => (
                <th key={h} style={TH}>{h}</th>
              ))}</tr>
            </thead>
            <tbody>
              <AnimatePresence>
                {feed.map((row,i) => {
                  const isAtk = row.tag === 'ATTACK'
                  return (
                    <motion.tr key={`${row.timestamp}-${i}`}
                      initial={{ opacity:0, x:-12 }} animate={{ opacity:1, x:0 }} transition={{ delay:i*0.015 }}
                      style={{ background: isAtk ? 'rgba(255,45,85,0.06)' : 'transparent' }}>
                      <td style={TD}>{row.timestamp}</td>
                      <td style={{ ...TD, fontFamily:'var(--font-mono)', fontSize:'0.78rem', color:isAtk?'#ff6b6b':'var(--text)' }}>{row.src}</td>
                      <td style={{ ...TD, fontFamily:'var(--font-mono)', fontSize:'0.78rem' }}>{row.dst}</td>
                      <td style={TD}>{row.proto?.toUpperCase()}</td>
                      <td style={TD}>
                        <span style={{ padding:'0.22rem 0.65rem', borderRadius:6, fontSize:'0.73rem', fontWeight:600,
                          background:`${COLORS[row.prediction]||'#888'}1a`,
                          color:COLORS[row.prediction]||'#888',
                          border:`1px solid ${COLORS[row.prediction]||'#888'}35` }}>
                          {row.prediction}
                        </span>
                      </td>
                      <td style={{ ...TD, fontFamily:'var(--font-mono)', fontSize:'0.78rem' }}>
                        {(row.confidence*100).toFixed(1)}%
                      </td>
                      <td style={TD}>
                        {row.blocked ? <span style={{ color:'#ff9500', fontSize:'0.75rem' }}>🛡️ BLOCKED</span> : null}
                      </td>
                    </motion.tr>
                  )
                })}
              </AnimatePresence>
            </tbody>
          </table>
          {feed.length === 0 && (
            <div style={{ padding:'3rem', textAlign:'center', color:'var(--muted)', fontFamily:'var(--font-mono)', fontSize:'0.85rem' }}>
              Waiting for live_capture.py to start writing data...
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}