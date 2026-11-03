import { motion } from 'framer-motion'

const TITLE = 'We turn air quality data into community change.'
const words = TITLE.split(' ')

export default function Hero() {
  return (
    <section style={{
      position: 'relative',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'flex-start',
      paddingTop: '26vh',
      paddingLeft: '48px',
      paddingRight: '48px',
      paddingBottom: '80px',
      textAlign: 'center',
      overflow: 'hidden',
      background: `linear-gradient(
        180deg,
        #030b22 0%,
        #060e34 4%,
        #0a1858 10%,
        #102478 18%,
        #183898 28%,
        #2252b8 40%,
        #3068cc 52%,
        #4882d8 63%,
        #6aa0e2 73%,
        #92beed 81%,
        #bbd4f5 88%,
        #d8e9fb 93%,
        #ecf5ff 97%,
        #f5faff 100%
      )`,
    }}>

      {/* Subtle radial glow at center adds depth */}
      <div style={{
        position: 'absolute',
        top: '15%',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '900px',
        height: '500px',
        background: 'radial-gradient(ellipse at center, rgba(80,140,255,0.12) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />

      {/* Headline */}
      <h1 style={{
        fontFamily: 'Lora, Georgia, serif',
        fontSize: 'clamp(30px, 4.8vw, 68px)',
        fontWeight: 500,
        color: 'rgba(255,255,255,0.97)',
        lineHeight: 1.22,
        letterSpacing: '-0.02em',
        maxWidth: '860px',
        position: 'relative',
        zIndex: 1,
      }}>
        {words.map((word, i) => (
          <motion.span
            key={i}
            initial={{ opacity: 0, y: 28, filter: 'blur(4px)' }}
            animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
            transition={{
              delay: 0.2 + i * 0.07,
              duration: 0.9,
              ease: [0.22, 1, 0.36, 1],
            }}
            style={{ display: 'inline-block', marginRight: '0.26em' }}
          >
            {word}
          </motion.span>
        ))}
      </h1>

      {/* Tagline */}
      <motion.p
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.1, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        style={{
          fontFamily: 'Inter, sans-serif',
          fontSize: 'clamp(14px, 1.4vw, 18px)',
          fontWeight: 400,
          color: 'rgba(255,255,255,0.55)',
          marginTop: '28px',
          letterSpacing: '0.01em',
          position: 'relative',
          zIndex: 1,
        }}
      >
        Dallas, Texas · Environmental Justice · Real-Time Air Quality
      </motion.p>

      {/* CTA buttons */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.3, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        style={{
          display: 'flex',
          gap: '14px',
          marginTop: '44px',
          flexWrap: 'wrap',
          justifyContent: 'center',
          position: 'relative',
          zIndex: 1,
        }}
      >
        <motion.a
          href="https://shared-skies-initiative.vercel.app/"
          target="_blank"
          rel="noopener noreferrer"
          whileHover={{ scale: 1.04, background: 'rgba(255,255,255,0.98)' }}
          whileTap={{ scale: 0.97 }}
          transition={{ duration: 0.18 }}
          style={{
            background: 'rgba(255,255,255,0.92)',
            color: '#0d2478',
            padding: '13px 28px',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: 600,
            fontFamily: 'Inter, sans-serif',
            letterSpacing: '0.01em',
          }}
        >
          View Real-Time Map
        </motion.a>
        <motion.a
          href="https://form.typeform.com/to/FeqteoZD?typeform-source=sharedskiesinitiative.org"
          target="_blank"
          rel="noopener noreferrer"
          whileHover={{ scale: 1.04, background: 'rgba(255,255,255,0.12)' }}
          whileTap={{ scale: 0.97 }}
          transition={{ duration: 0.18 }}
          style={{
            background: 'transparent',
            color: 'rgba(255,255,255,0.88)',
            padding: '13px 28px',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: 500,
            fontFamily: 'Inter, sans-serif',
            border: '1.5px solid rgba(255,255,255,0.3)',
            letterSpacing: '0.01em',
          }}
        >
          Join the Movement
        </motion.a>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2, duration: 1 }}
        style={{
          position: 'absolute',
          bottom: '36px',
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '6px',
        }}
      >
        <span style={{
          fontFamily: 'Inter, sans-serif',
          fontSize: '10px',
          fontWeight: 500,
          color: 'rgba(255,255,255,0.35)',
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
        }}>
          Scroll
        </span>
        <motion.div
          animate={{ y: [0, 7, 0] }}
          transition={{ repeat: Infinity, duration: 2.2, ease: 'easeInOut' }}
          style={{
            width: '1.5px',
            height: '28px',
            background: 'linear-gradient(180deg, rgba(255,255,255,0.5) 0%, transparent 100%)',
            borderRadius: '1px',
          }}
        />
      </motion.div>
    </section>
  )
}
