import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

function FadeUp({ children, delay = 0, style = {} }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.2 })
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 36 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ type: 'spring', stiffness: 55, damping: 20, delay }}
      style={style}
    >
      {children}
    </motion.div>
  )
}

export default function Mission() {
  return (
    <section style={{ background: '#1e3d6e' }}>

      {/* Two-column mission text */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '72px',
        padding: '120px 80px 110px',
        borderBottom: '1px solid rgba(255,255,255,0.1)',
      }}>
        <FadeUp>
          <p style={body}>
            At the Shared Skies Initiative, we're more than an environmental organization; we're a movement of young leaders, scientists, and community advocates working together to bring clean air to every neighborhood. With a mission rooted in justice, data, and action, we empower frontline communities with the tools and voice they need to fight pollution and reclaim their right to breathe freely. Through a blend of grassroots engagement, youth-driven advocacy, and cutting-edge air quality monitoring, we transform data into meaningful change that uplifts lives and landscapes alike.
          </p>
        </FadeUp>
        <FadeUp delay={0.14}>
          <p style={{ ...body, marginBottom: '26px' }}>
            Guided by collaboration and transparency, we build bridges between schools, local leaders, and residents to ensure every voice is heard and every effort is grounded in community needs. Our work goes beyond measuring air. We're also restoring trust in data, fostering youth leadership, and shaping a shared, healthier future for all.
          </p>
          <p style={{ ...body, color: 'rgba(255,255,255,0.95)', fontWeight: 500 }}>
            Clean air shouldn't be a privilege. It's a promise.
          </p>
        </FadeUp>
      </div>

      {/* Equity quote: Lora, regular (not italic) */}
      <div style={{ padding: '120px 80px 130px' }}>
        <FadeUp>
          <p style={{
            fontFamily: 'Lora, Georgia, serif',
            fontSize: 'clamp(18px, 2.6vw, 36px)',
            fontWeight: 500,
            color: 'rgba(255,255,255,0.94)',
            maxWidth: '980px',
            lineHeight: 1.44,
            letterSpacing: '-0.015em',
          }}>
            Equity is at the heart of everything we do. We're identifying disparities and shifting power from institutions to community that lives and breathes the air.
          </p>
        </FadeUp>
      </div>
    </section>
  )
}

const body = {
  fontFamily: 'Inter, sans-serif',
  fontSize: '14px',
  fontWeight: 400,
  color: 'rgba(255,255,255,0.72)',
  lineHeight: 1.85,
}
