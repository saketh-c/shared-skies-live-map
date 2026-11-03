import { useState, useRef } from 'react'
import { motion, AnimatePresence, useInView } from 'framer-motion'

const FAQS = [
  {
    q: 'How is Shared Skies different from other environmental organizations?',
    a: 'Unlike traditional environmental organizations, Shared Skies is youth-led and hyper-local. We deploy real-time air quality monitors directly in frontline neighborhoods, making data accessible to the very communities breathing that air. We combine scientific rigor with grassroots advocacy, empowering residents, not just researchers, to drive change.',
  },
  {
    q: 'What communities does Shared Skies help?',
    a: 'We focus on frontline communities: neighborhoods disproportionately burdened by pollution due to proximity to highways, industrial facilities, and other pollution sources. We primarily work in the Dallas-Fort Worth metroplex, with plans to expand to other Texas cities and communities across the country.',
  },
  {
    q: 'Is this problem real, and does Shared Skies really work?',
    a: 'Air pollution causes over 7 million deaths per year globally, and frontline communities bear a disproportionate share of that burden. Our real-time monitoring has already identified pollution hotspots that official EPA monitoring missed. Our data has been used in community organizing, local policy discussions, and educational programs reaching hundreds of students.',
  },
  {
    q: 'How do you decide where to put sensors and focus your work?',
    a: 'We use EPA EJScreen data, census tract demographics, and community input to identify areas with the highest pollution burden and lowest monitoring coverage. We prioritize places where residents are most likely to be harmed but least likely to have access to real-time air quality information.',
  },
]

function FAQItem({ q, a, index }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.3 })

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 18 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.65, delay: index * 0.08, ease: [0.22, 1, 0.36, 1] }}
      style={{
        borderRadius: '8px',
        marginBottom: '8px',
        overflow: 'hidden',
        background: 'rgba(255,255,255,0.08)',
        border: '1px solid rgba(255,255,255,0.1)',
      }}
    >
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', background: 'none', border: 'none',
          padding: '20px 24px',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          fontFamily: 'Inter, sans-serif', fontSize: '15px', fontWeight: 400,
          color: 'rgba(255,255,255,0.9)', textAlign: 'left', gap: '16px',
        }}
      >
        <span>{q}</span>
        <motion.span
          animate={{ rotate: open ? 90 : 0 }}
          transition={{ duration: 0.25, ease: 'easeInOut' }}
          style={{ fontSize: '18px', flexShrink: 0, color: 'rgba(255,255,255,0.45)', display: 'inline-block' }}
        >
          ›
        </motion.span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
            style={{ overflow: 'hidden' }}
          >
            <p style={{
              padding: '0 24px 22px',
              fontFamily: 'Inter, sans-serif', fontSize: '14px', fontWeight: 400,
              color: 'rgba(255,255,255,0.6)', lineHeight: 1.8,
            }}>
              {a}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default function FAQ() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.2 })

  return (
    <section style={{
      background: `linear-gradient(180deg,
        #0b1458 0%,
        #0e1c78 28%,
        #1530a0 55%,
        #1e44b8 75%,
        #2858d0 90%,
        #3060d8 100%
      )`,
      padding: '80px 68px 120px',
    }}>
      <motion.h2
        ref={ref}
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
        style={{ fontFamily: 'Lora, Georgia, serif', fontSize: '30px', fontWeight: 500, color: 'white', marginBottom: '32px', letterSpacing: '-0.01em' }}
      >
        FAQ
      </motion.h2>
      {FAQS.map((item, i) => <FAQItem key={item.q} q={item.q} a={item.a} index={i} />)}
    </section>
  )
}
