import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'

function BlackHole({ intense }) {
  const ringRef = useRef()
  const count = intense ? 6000 : 3000

  const { positions, colors } = useMemo(() => {
    const positions = new Float32Array(count * 3)
    const colors    = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      const theta  = Math.random() * Math.PI * 2
      const radius = 2 + Math.random() * 5
      const spread = (Math.random() - 0.5) * 0.35
      positions[i * 3]     = Math.cos(theta) * radius
      positions[i * 3 + 1] = spread
      positions[i * 3 + 2] = Math.sin(theta) * radius
      const t = (radius - 2) / 5
      colors[i * 3]     = t * 0.48
      colors[i * 3 + 1] = t * 0.18 + (1 - t) * 0.96
      colors[i * 3 + 2] = 1.0
    }
    return { positions, colors }
  }, [count])

  useFrame((state) => {
    const t = state.clock.elapsedTime
    if (ringRef.current) {
      ringRef.current.rotation.y = t * 0.06
      ringRef.current.rotation.x = Math.sin(t * 0.03) * 0.12
    }
  })

  return (
    <group>
      <points ref={ringRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
          <bufferAttribute attach="attributes-color"    args={[colors, 3]} />
        </bufferGeometry>
        <pointsMaterial
          size={0.02} vertexColors transparent opacity={0.75}
          sizeAttenuation blending={THREE.AdditiveBlending} depthWrite={false}
        />
      </points>
      <mesh>
        <sphereGeometry args={[0.4, 32, 32]} />
        <meshBasicMaterial color="#000000" />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.7, 32, 32]} />
        <meshBasicMaterial color="#7b2fff" transparent opacity={0.08} />
      </mesh>
    </group>
  )
}

function StarField() {
  const ref = useRef()
  const positions = useMemo(() => {
    const pos = new Float32Array(3000 * 3)
    for (let i = 0; i < 3000; i++) {
      pos[i * 3]     = (Math.random() - 0.5) * 100
      pos[i * 3 + 1] = (Math.random() - 0.5) * 100
      pos[i * 3 + 2] = (Math.random() - 0.5) * 100
    }
    return pos
  }, [])

  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y = state.clock.elapsedTime * 0.004
    }
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.05} color="#ffffff" transparent opacity={0.45} sizeAttenuation />
    </points>
  )
}

function CameraRig({ intense }) {
  useFrame((state) => {
    const t = state.clock.elapsedTime
    // Camera much further back so content is not obscured
    state.camera.position.x = Math.sin(t * 0.04) * (intense ? 1 : 0.3)
    state.camera.position.y = Math.cos(t * 0.03) * (intense ? 0.5 : 0.2) + (intense ? 1 : 2)
    state.camera.position.z = intense ? 16 : 22
    state.camera.lookAt(0, 0, 0)
  })
  return null
}

export default function ParticleUniverse({ intense = true }) {
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 0,
      background: 'radial-gradient(ellipse at 50% 60%, #03000f 0%, #000005 70%)'
    }}>
      <Canvas
        camera={{ position: [0, 1, 18], fov: 55 }}
        gl={{ antialias: true, alpha: true }}
        style={{ width: '100%', height: '100%' }}
      >
        <StarField />
        <BlackHole intense={intense} />
        <CameraRig intense={intense} />
      </Canvas>
    </div>
  )
}