import { motion } from 'framer-motion'
import Footer from '../components/Footer'

function Reveal({ children, delay = 0, y = 28, style = {} }) {
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

const ROLES = [
  {
    title: 'Sensor Technician',
    desc: 'Help install and maintain our air quality monitoring devices in neighborhoods across the DFW area.',
  },
  {
    title: 'Community Outreach',
    desc: 'Connect with residents, attend local events, and share our mission with frontline communities.',
  },
  {
    title: 'Data Analyst',
    desc: 'Analyze real-time pollution data and help translate it into actionable insights for community members.',
  },
  {
    title: 'Education Lead',
    desc: 'Develop and deliver workshops on environmental justice topics to schools and youth organizations.',
  },
]

export default function Join() {
  return (
    <div className="page-enter">
      {/* Header */}
      <section style={{
        background: 'linear-gradient(140deg, #1c3880 0%, #2952a8 30%, #3868c0 60%, #4c80d0 100%)',
        paddingTop: '130px',
        paddingBottom: '80px',
        paddingLeft: '80px',
        paddingRight: '80px',
      }}>
        <Reveal>
          <p style={{
            fontFamily: 'Inter, sans-serif', fontSize: '11px', fontWeight: 600,
            color: 'rgba(255,255,255,0.6)', textTransform: 'uppercase',
            letterSpacing: '0.16em', marginBottom: '20px',
          }}>
            Get Involved
          </p>
          <h1 style={{
            fontFamily: 'Lora, Georgia, serif',
            fontSize: 'clamp(32px, 4vw, 56px)', fontWeight: 600,
            color: 'white', lineHeight: 1.18,
            letterSpacing: '-0.025em', maxWidth: '680px',
            marginBottom: '24px',
          }}>
            Join the movement for clean air.
          </h1>
          <p style={{
            fontFamily: 'Inter, sans-serif', fontSize: 'clamp(15px, 1.3vw, 17px)',
            color: 'rgba(255,255,255,0.72)', lineHeight: 1.72,
            maxWidth: '540px',
          }}>
            Whether you're a student, scientist, educator, or advocate, there's a place for you in Shared Skies. Fill out the form below to get started.
          </p>
        </Reveal>
      </section>

      {/* Roles grid */}
      <section style={{ background: '#080f2a', padding: '80px 80px 60px' }}>
        <Reveal>
          <h2 style={{
            fontFamily: 'Lora, Georgia, serif', fontSize: 'clamp(20px, 2.2vw, 30px)',
            fontWeight: 600, color: 'white', letterSpacing: '-0.015em',
            marginBottom: '40px',
          }}>
            Ways to contribute
          </h2>
        </Reveal>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', maxWidth: '900px' }}>
          {ROLES.map((r, i) => (
            <Reveal key={r.title} delay={i * 0.07}>
              <div style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '12px', padding: '28px 28px',
                backdropFilter: 'blur(8px)',
              }}>
                <h3 style={{ fontFamily: 'Lora, Georgia, serif', fontSize: '17px', fontWeight: 600, color: 'white', marginBottom: '10px', letterSpacing: '-0.01em' }}>
                  {r.title}
                </h3>
                <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '13.5px', color: 'rgba(255,255,255,0.6)', lineHeight: 1.72 }}>
                  {r.desc}
                </p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* Typeform embed */}
      <section style={{ background: '#080f2a', padding: '0 80px 100px' }}>
        <Reveal>
          <div style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '16px',
            overflow: 'hidden',
            maxWidth: '900px',
          }}>
            <iframe
              src="https://form.typeform.com/to/FeqteoZD"
              title="Join Shared Skies Initiative"
              width="100%"
              height="640"
              frameBorder="0"
              allow="camera; microphone; autoplay; encrypted-media;"
              style={{ display: 'block', border: 'none' }}
            />
          </div>
        </Reveal>
      </section>

      <Footer />
    </div>
  )
}
