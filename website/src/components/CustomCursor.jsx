import { useEffect, useRef, useState } from 'react'
import { motion, useMotionValue, useSpring } from 'framer-motion'

export default function CustomCursor() {
  const [hovering, setHovering] = useState(false)
  const [clicking, setClicking] = useState(false)
  const [visible, setVisible] = useState(false)

  const rawX = useMotionValue(-200)
  const rawY = useMotionValue(-200)

  // Dot follows exactly
  const dotX = useMotionValue(-200)
  const dotY = useMotionValue(-200)

  // Ring follows with spring lag
  const ringX = useSpring(rawX, { stiffness: 160, damping: 22, mass: 0.6 })
  const ringY = useSpring(rawY, { stiffness: 160, damping: 22, mass: 0.6 })

  useEffect(() => {
    const onMove = (e) => {
      rawX.set(e.clientX)
      rawY.set(e.clientY)
      dotX.set(e.clientX)
      dotY.set(e.clientY)
      if (!visible) setVisible(true)
    }

    const onOver = (e) => {
      setHovering(!!e.target.closest('a, button, [role="button"]'))
    }

    const onDown = () => setClicking(true)
    const onUp = () => setClicking(false)

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseover', onOver)
    window.addEventListener('mousedown', onDown)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseover', onOver)
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('mouseup', onUp)
    }
  }, [visible])

  const ringSize = clicking ? 28 : hovering ? 52 : 38

  return (
    <div id="custom-cursor" style={{ pointerEvents: 'none' }}>
      {/* Small solid dot: instant follow */}
      <motion.div
        style={{
          position: 'fixed',
          top: 0, left: 0,
          zIndex: 99999,
          width: clicking ? 4 : 6,
          height: clicking ? 4 : 6,
          borderRadius: '50%',
          background: 'white',
          x: dotX,
          y: dotY,
          translateX: '-50%',
          translateY: '-50%',
          opacity: visible ? 1 : 0,
          mixBlendMode: 'difference',
        }}
        transition={{ duration: 0.1 }}
      />

      {/* Outer ring: spring follow */}
      <motion.div
        style={{
          position: 'fixed',
          top: 0, left: 0,
          zIndex: 99998,
          borderRadius: '50%',
          border: '1.5px solid rgba(255,255,255,0.75)',
          x: ringX,
          y: ringY,
          translateX: '-50%',
          translateY: '-50%',
          opacity: visible ? 1 : 0,
          mixBlendMode: 'difference',
        }}
        animate={{
          width: ringSize,
          height: ringSize,
          opacity: visible ? (clicking ? 0.5 : 1) : 0,
        }}
        transition={{ duration: 0.18, ease: 'easeOut' }}
      />
    </div>
  )
}
