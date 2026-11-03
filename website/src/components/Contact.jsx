import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

const ease = [0.22, 1, 0.36, 1]

export default function Contact() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.25 })

  return (
    <section id="contact" style={{
      position: 'relative',
      minHeight: '480px',
      overflow: 'hidden',
      background: '#152060',
    }}>
      {/* Background image */}
      <motion.div
        initial={{ scale: 1.06 }}
        animate={{ scale: 1 }}
        transition={{ duration: 1.4, ease }}
        style={{
          position: 'absolute', inset: 0,
          backgroundImage: 'url(https://images.unsplash.com/photo-1518992648583-fc5e79c7a9c4?auto=format&fit=crop&w=1920&q=80)',
          backgroundSize: 'cover',
          backgroundPosition: 'center 65%',
          opacity: 0.55,
        }}
      />
      <div style={{
        position: 'absolute', inset: 0,
        background: 'linear-gradient(to right, rgba(12,24,90,0.65) 0%, rgba(12,24,90,0.3) 100%)',
      }} />

      <motion.div
        ref={ref}
        initial={{ opacity: 0, y: 28 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.9, ease }}
        style={{ position: 'relative', zIndex: 1, padding: '72px 72px' }}
      >
        <h2 style={{
          fontFamily: 'Inter, sans-serif',
          fontSize: '34px', fontWeight: 400,
          color: 'white', marginBottom: '52px',
        }}>
          Contact
        </h2>

        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '48px', maxWidth: '680px',
        }}>
          <div>
            <p style={label}>Phone</p>
            <p style={value}>+1 469 465 3300</p>
            <p style={{ ...label, marginTop: '32px' }}>Email</p>
            <a
              href="mailto:sharedskiesinitiative@gmail.com"
              style={{ ...value, display: 'block' }}
              onMouseEnter={e => e.currentTarget.style.textDecoration = 'underline'}
              onMouseLeave={e => e.currentTarget.style.textDecoration = 'none'}
            >
              sharedskiesinitiative@gmail.com
            </a>
          </div>
          <div>
            <p style={label}>Address</p>
            <p style={value}>5800 Furneaux Dr, Plano TX</p>
            <p style={{ ...label, marginTop: '32px' }}>Hours</p>
            <p style={value}>8:00 AM – 8:00 PM</p>
          </div>
        </div>
      </motion.div>
    </section>
  )
}

const label = {
  fontFamily: 'Inter, sans-serif',
  fontSize: '11px', fontWeight: 500,
  color: 'rgba(255,255,255,0.5)',
  textTransform: 'uppercase',
  letterSpacing: '0.1em',
  marginBottom: '6px',
}

const value = {
  fontFamily: 'Inter, sans-serif',
  fontSize: '14px', fontWeight: 400,
  color: 'rgba(255,255,255,0.92)',
  lineHeight: 1.55,
}
