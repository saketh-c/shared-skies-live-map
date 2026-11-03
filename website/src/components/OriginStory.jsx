import { useRef } from 'react'
import { motion, useScroll, useTransform, useInView } from 'framer-motion'

export default function OriginStory() {
  const sectionRef = useRef(null)
  const textRef = useRef(null)
  const textInView = useInView(textRef, { once: true, amount: 0.4 })

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start end', 'end start'],
  })
  // Parallax: image moves up slower than scroll
  const imgY = useTransform(scrollYProgress, [0, 1], ['-12%', '12%'])

  return (
    <section
      ref={sectionRef}
      style={{
        position: 'relative',
        minHeight: '100vh',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'flex-start',
        padding: '120px 72px',
      }}
    >
      {/* Parallax background image */}
      <motion.div
        style={{
          position: 'absolute',
          inset: '-15%',
          backgroundImage: 'url(https://images.unsplash.com/photo-1534088568595-a066f410bcda?auto=format&fit=crop&w=1920&q=85)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          y: imgY,
        }}
      />

      {/* Subtle gradient overlay */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'linear-gradient(135deg, rgba(10,20,80,0.22) 0%, rgba(10,20,80,0.08) 60%, rgba(0,0,0,0) 100%)',
        pointerEvents: 'none',
      }} />

      {/* Text: top-left, matching original */}
      <motion.div
        ref={textRef}
        initial={{ opacity: 0, y: 32 }}
        animate={textInView ? { opacity: 1, y: 0 } : {}}
        transition={{ type: 'spring', stiffness: 50, damping: 18 }}
        style={{ position: 'relative', zIndex: 1, maxWidth: '560px' }}
      >
        <p style={{
          fontFamily: 'Lora, Georgia, serif',
          fontSize: 'clamp(24px, 3.2vw, 44px)',
          fontWeight: 400,
          color: 'white',
          lineHeight: 1.32,
          textShadow: '0 2px 20px rgba(0,0,0,0.18)',
        }}>
          It all began in a small school nestled in the heart of Dallas
        </p>
      </motion.div>
    </section>
  )
}
