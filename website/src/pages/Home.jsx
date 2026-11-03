import { Suspense, useRef, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import Earth from '../components/Earth'
import Footer from '../components/Footer'

/* ── Reusable scroll-reveal wrapper ─────────────────────────── */
function Reveal({ children, delay = 0, y = 32, style = {} }) {
  return (
    <motion.div
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{ type: 'spring', stiffness: 55, damping: 20, delay }}
      style={style}
    >
      {children}
    </motion.div>
  )
}

/* ── Animated stat counter ───────────────────────────────────── */
function Counter({ target, suffix, active }) {
  const [val, setVal] = useState(0)
  useEffect(() => {
    if (!active) return
    const dur = 1800, start = Date.now()
    const id = setInterval(() => {
      const p = Math.min((Date.now() - start) / dur, 1)
      setVal(Math.round((1 - Math.pow(1 - p, 3)) * target))
      if (p >= 1) clearInterval(id)
    }, 16)
    return () => clearInterval(id)
  }, [active, target])
  return <>{val.toLocaleString()}{suffix}</>
}

const WORDS = 'We turn air quality data into community change.'.split(' ')
const PILLARS = [
  {
    n: '01', title: 'Research',
    body: 'We install and maintain accurate, low-cost air quality sensors in frontline neighborhoods, creating a real-time picture of local pollution accessible to every resident.',
    icon: '◎',
  },
  {
    n: '02', title: 'Education',
    body: 'We influence daily decisions through education. Students of any age can understand environmental justice topics through our curriculum, workshops, and digital tools.',
    icon: '◈',
  },
  {
    n: '03', title: 'Advocacy',
    body: 'We advocate for local policy change using data-driven insights. Youth ambassadors and community members gain the skills to interpret air data and drive meaningful change.',
    icon: '◉',
  },
]
const STATS = [
  { target: 6500, suffix: '+', label: 'Dollars Raised' },
  { target: 10,   suffix: '+', label: 'Partnerships' },
  { target: 20,   suffix: '+', label: 'Volunteers' },
]

export default function Home() {
  return (
    <div className="page-enter">
      <HeroSection />
      <MissionSection />
      <PillarsSection />
      <StatsSection />
      <OriginSection />
      <CTASection />
      <Footer />
    </div>
  )
}

/* ─────────────────────────────── HERO ─────────────────────── */
function HeroSection() {
  return (
    <section style={{
      position: 'relative',
      minHeight: '100vh',
      display: 'grid',
      gridTemplateColumns: '58% 42%',
      alignItems: 'center',
      background: `linear-gradient(
        140deg,
        #1c3880 0%,
        #2952a8 16%,
        #3868c0 32%,
        #4c80d0 48%,
        #6098dc 63%,
        #7db2e6 76%,
        #9fcbf0 87%,
        #c2dff8 95%,
        #deeefb 100%
      )`,
      overflow: 'hidden',
    }}>
      {/* Radial depth glow behind Earth */}
      <div style={{
        position: 'absolute', right: '-5%', top: '50%',
        transform: 'translateY(-50%)',
        width: '58vw', height: '100vh',
        background: 'radial-gradient(ellipse at center, rgba(20,60,160,0.35) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />

      {/* ── Text ── */}
      <div style={{ padding: '120px 72px 80px', position: 'relative', zIndex: 1 }}>
        {/* Eyebrow */}
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          style={{
            fontFamily: 'Inter, sans-serif', fontSize: '12px', fontWeight: 600,
            color: 'rgba(255,255,255,0.65)',
            letterSpacing: '0.14em', textTransform: 'uppercase',
            marginBottom: '28px',
          }}
        >
          Dallas, Texas · Environmental Justice · Since 2024
        </motion.p>

        {/* Headline, word by word */}
        <h1 style={{
          fontFamily: 'Lora, Georgia, serif',
          fontSize: 'clamp(32px, 4.2vw, 62px)',
          fontWeight: 600,
          color: 'white',
          lineHeight: 1.18,
          letterSpacing: '-0.025em',
          marginBottom: '28px',
        }}>
          {WORDS.map((w, i) => (
            <motion.span
              key={i}
              initial={{ opacity: 0, y: 24, filter: 'blur(5px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              transition={{ delay: 0.25 + i * 0.065, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
              style={{ display: 'inline-block', marginRight: '0.24em' }}
            >
              {w}
            </motion.span>
          ))}
        </h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          style={{
            fontFamily: 'Inter, sans-serif', fontSize: 'clamp(15px, 1.4vw, 18px)',
            fontWeight: 400, color: 'rgba(255,255,255,0.75)',
            lineHeight: 1.7, maxWidth: '500px', marginBottom: '44px',
          }}
        >
          Shared Skies deploys real-time air quality monitors in frontline neighborhoods across Texas, making pollution data accessible to the communities that need it most.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}
        >
          <motion.div whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}>
            <Link to="/join" style={{
              display: 'inline-block',
              background: 'rgba(255,255,255,0.95)',
              color: '#1a3278',
              padding: '14px 30px', borderRadius: '8px',
              fontSize: '14px', fontWeight: 700,
              fontFamily: 'Inter, sans-serif', letterSpacing: '0.01em',
            }}>
              Get Involved
            </Link>
          </motion.div>
          <motion.div whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}>
            <Link to="/map" style={{
              display: 'inline-block',
              background: 'rgba(255,255,255,0.12)',
              color: 'white',
              padding: '14px 30px', borderRadius: '8px',
              fontSize: '14px', fontWeight: 500,
              fontFamily: 'Inter, sans-serif',
              border: '1.5px solid rgba(255,255,255,0.4)',
              letterSpacing: '0.01em',
            }}>
              View Real-Time Data →
            </Link>
          </motion.div>
        </motion.div>
      </div>

      {/* ── Earth ── */}
      <motion.div
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.4, duration: 1.4, ease: [0.22, 1, 0.36, 1] }}
        style={{ position: 'relative', height: '100vh', zIndex: 0 }}
      >
        <Suspense fallback={null}>
          <Earth />
        </Suspense>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        transition={{ delay: 2.2, duration: 1 }}
        style={{
          position: 'absolute', bottom: '32px', left: '72px',
          display: 'flex', alignItems: 'center', gap: '10px',
        }}
      >
        <motion.div
          animate={{ y: [0, 6, 0] }}
          transition={{ repeat: Infinity, duration: 2.4, ease: 'easeInOut' }}
          style={{ width: '1.5px', height: '26px', background: 'linear-gradient(180deg, rgba(255,255,255,0.6), transparent)', borderRadius: '1px' }}
        />
        <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '10px', fontWeight: 500, color: 'rgba(255,255,255,0.4)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>
          Scroll
        </span>
      </motion.div>
    </section>
  )
}

/* ─────────────────────────────── MISSION ───────────────────── */
function MissionSection() {
  return (
    <section style={{ background: '#050d1f', padding: '120px 80px' }}>
      <Reveal>
        <div style={{ maxWidth: '820px', margin: '0 auto', textAlign: 'center' }}>
          <div style={{ width: '48px', height: '1.5px', background: 'rgba(100,150,240,0.5)', margin: '0 auto 40px' }} />
          <p style={{
            fontFamily: 'Lora, Georgia, serif',
            fontSize: 'clamp(20px, 2.4vw, 32px)', fontWeight: 500,
            color: 'rgba(255,255,255,0.9)', lineHeight: 1.5,
            letterSpacing: '-0.01em',
          }}>
            Environmental justice is the promise that no one's health is determined by their ZIP code.
          </p>
          <div style={{ width: '48px', height: '1.5px', background: 'rgba(100,150,240,0.5)', margin: '40px auto 0' }} />
        </div>
      </Reveal>
    </section>
  )
}

/* ─────────────────────────────── PILLARS ───────────────────── */
function PillarsSection() {
  return (
    <section style={{
      background: 'linear-gradient(180deg, #080f2a 0%, #0c1840 100%)',
      padding: '100px 72px',
    }}>
      <Reveal>
        <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '11px', fontWeight: 600, color: 'rgba(100,150,240,0.8)', textTransform: 'uppercase', letterSpacing: '0.15em', marginBottom: '16px' }}>
          How We Work
        </p>
        <h2 style={{ fontFamily: 'Lora, Georgia, serif', fontSize: 'clamp(26px, 3vw, 42px)', fontWeight: 600, color: 'white', letterSpacing: '-0.02em', marginBottom: '64px', maxWidth: '520px', lineHeight: 1.25 }}>
          Three pillars driving real change.
        </h2>
      </Reveal>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
        {PILLARS.map((p, i) => (
          <Reveal key={p.n} delay={i * 0.1}>
            <motion.div
              whileHover={{ y: -6, borderColor: 'rgba(100,150,240,0.35)' }}
              transition={{ type: 'spring', stiffness: 280, damping: 22 }}
              style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '16px', padding: '40px 32px',
                backdropFilter: 'blur(8px)',
              }}
            >
              <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '11px', color: 'rgba(100,150,240,0.7)', letterSpacing: '0.1em' }}>{p.n}</span>
              <h3 style={{ fontFamily: 'Lora, Georgia, serif', fontSize: '24px', fontWeight: 600, color: 'white', margin: '14px 0 16px', letterSpacing: '-0.01em' }}>{p.title}</h3>
              <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '14px', color: 'rgba(255,255,255,0.62)', lineHeight: 1.78 }}>{p.body}</p>
            </motion.div>
          </Reveal>
        ))}
      </div>
    </section>
  )
}

/* ─────────────────────────────── STATS ─────────────────────── */
function StatsSection() {
  const ref = useRef(null)
  const [active, setActive] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setActive(true) }, { threshold: 0.4 })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  return (
    <section ref={ref} style={{ background: '#050d1f', padding: '100px 72px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '40px', maxWidth: '900px', margin: '0 auto' }}>
        {STATS.map((s, i) => (
          <Reveal key={s.label} delay={i * 0.1} style={{ textAlign: 'center' }}>
            <p style={{ fontFamily: 'Lora, Georgia, serif', fontSize: 'clamp(44px, 5vw, 64px)', fontWeight: 700, color: 'white', lineHeight: 1, marginBottom: '10px' }}>
              <Counter target={s.target} suffix={s.suffix} active={active} />
            </p>
            <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '14px', fontWeight: 500, color: 'rgba(100,150,240,0.85)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              {s.label}
            </p>
          </Reveal>
        ))}
      </div>
    </section>
  )
}

/* ─────────────────────────────── ORIGIN ────────────────────── */
function OriginSection() {
  return (
    <section style={{
      position: 'relative', minHeight: '70vh', overflow: 'hidden',
      display: 'grid', gridTemplateColumns: '1fr 1fr',
    }}>
      {/* Sky image */}
      <div style={{
        backgroundImage: 'url(https://images.unsplash.com/photo-1534088568595-a066f410bcda?auto=format&fit=crop&w=1200&q=85)',
        backgroundSize: 'cover', backgroundPosition: 'center',
      }} />
      {/* Text side */}
      <div style={{ background: '#080f2a', padding: '80px 64px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <Reveal>
          <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '11px', fontWeight: 600, color: 'rgba(100,150,240,0.8)', textTransform: 'uppercase', letterSpacing: '0.15em', marginBottom: '20px' }}>
            Our Story
          </p>
          <h2 style={{ fontFamily: 'Lora, Georgia, serif', fontSize: 'clamp(22px, 2.6vw, 36px)', fontWeight: 600, color: 'white', lineHeight: 1.3, letterSpacing: '-0.015em', marginBottom: '24px' }}>
            It all began in a small school nestled in the heart of Dallas.
          </h2>
          <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '14px', color: 'rgba(255,255,255,0.65)', lineHeight: 1.82, marginBottom: '32px' }}>
            Founded by a group of student scientists and community advocates, Shared Skies began with a single air quality sensor and a conviction: that frontline communities deserve the same clean air as anyone else. Today, we operate across the Dallas-Fort Worth metroplex with a growing network of monitors, partnerships, and passionate volunteers.
          </p>
          <Link to="/about" style={{
            fontFamily: 'Inter, sans-serif', fontSize: '13px', fontWeight: 600,
            color: 'rgba(120,165,245,0.9)', letterSpacing: '0.04em',
          }}>
            Read our full story →
          </Link>
        </Reveal>
      </div>
    </section>
  )
}

/* ─────────────────────────────── CTA ───────────────────────── */
function CTASection() {
  return (
    <section style={{
      background: 'linear-gradient(135deg, #2550c0 0%, #3568d8 40%, #4878e8 70%, #3260d0 100%)',
      padding: '120px 80px',
      position: 'relative', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse 70% 60% at 30% 50%, rgba(255,255,255,0.08) 0%, transparent 100%)',
        pointerEvents: 'none',
      }} />
      <Reveal style={{ position: 'relative', zIndex: 1 }}>
        <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '11px', fontWeight: 600, color: 'rgba(255,255,255,0.55)', textTransform: 'uppercase', letterSpacing: '0.15em', marginBottom: '20px' }}>
          Make a Difference
        </p>
        <h2 style={{ fontFamily: 'Lora, Georgia, serif', fontSize: 'clamp(28px, 4vw, 52px)', fontWeight: 600, color: 'white', lineHeight: 1.2, letterSpacing: '-0.02em', maxWidth: '680px', marginBottom: '48px' }}>
          Ready to help every neighborhood breathe easier?
        </h2>
        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
          <motion.div whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}>
            <Link to="/join" style={{
              display: 'inline-block', background: 'white', color: '#1a3278',
              padding: '15px 32px', borderRadius: '8px',
              fontSize: '14px', fontWeight: 700, fontFamily: 'Inter, sans-serif',
            }}>
              Join the Movement
            </Link>
          </motion.div>
          <motion.div whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}>
            <Link to="/donate" style={{
              display: 'inline-block', background: 'rgba(255,255,255,0.12)',
              color: 'white', padding: '15px 32px', borderRadius: '8px',
              fontSize: '14px', fontWeight: 500, fontFamily: 'Inter, sans-serif',
              border: '1.5px solid rgba(255,255,255,0.4)',
            }}>
              Donate Now
            </Link>
          </motion.div>
        </div>
      </Reveal>
    </section>
  )
}
