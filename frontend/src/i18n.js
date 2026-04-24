const TRANSLATIONS = {
  en: {
    brand_tagline_all: "All of Texas · Census Tract Level",
    back_to: (name) => `Back to ${name}`,
    current_pm25: "Current PM2.5",
    current_conditions: "Current Conditions",
    temperature: "Temperature",
    humidity: "Humidity",
    pressure: "Pressure",
    wind: "Wind",
    use_search: "Use the search bar to find any Texas address and view its air quality. Tap any tract on the map for details with real-time weather.",
    trusted_by: (n) => `Trusted by ${n} users across Texas`,
    updated: (when) => `Updated ${when} · refreshes every 30 min`,
    ej_context: "Environmental Justice Context",
    tract_distribution: "Tract Distribution",
    loading_predictions: "Generating predictions for all Texas tracts...",
    // Air Quality Guide
    guide_intro: "PM2.5 refers to fine particulate matter — tiny particles 2.5 micrometers or smaller. They can penetrate deep into the lungs and enter the bloodstream.",
    guide: {
      why_title: "Why PM2.5 Matters",
      why_body: "Particles this small bypass the nose and throat and lodge deep in lung tissue. Short-term exposure causes coughing and irritation. Long-term exposure is linked to heart disease, stroke, lung cancer, and reduced life expectancy.",
      about_title: "About These Predictions",
      about_body: "Values combine real-time weather data with EPA environmental justice indicators. Predictions refresh every 30 minutes and reflect estimated ground-level concentrations at the census tract level.",
      levels: {
        good: {
          name: "Good",
          range: "0 – 8.9 µg/m³",
          description: "Air quality is excellent. No health concerns for anyone. Ideal for outdoor activities and extended time outside.",
          who: "Safe for everyone",
        },
        moderate: {
          name: "Moderate",
          range: "9 – 12.9 µg/m³",
          description: "Air quality is acceptable. The vast majority of people will not experience effects. A very small number of unusually sensitive individuals may notice minor symptoms.",
          who: "Unusually sensitive people: consider reducing prolonged outdoor exertion",
        },
        unhealthy: {
          name: "Unhealthy",
          range: "13 – 17.9 µg/m³",
          description: "Everyone may begin to experience health effects. Sensitive groups — people with asthma, heart disease, the elderly, and children — are at greater risk.",
          who: "Sensitive groups: limit prolonged outdoor exertion. Everyone: reduce extended heavy outdoor activity.",
        },
        hazardous: {
          name: "Hazardous",
          range: "18+ µg/m³",
          description: "Health warnings of emergency conditions. Serious aggravation of heart and lung disease, premature mortality in sensitive groups, and respiratory effects in the general population.",
          who: "Everyone: avoid all outdoor physical activity. Sensitive groups: remain indoors.",
        },
      },
    },
    // Search / UI
    search: {
      address_tab: "📍 Address",
      coordinates_tab: "🧭 Coordinates",
      placeholder_address: "Enter address (e.g., Austin, TX)",
      placeholder_coords: "Latitude, Longitude (e.g., 30.2672, -97.7431)",
      search_button: "Search",
      searching: "Searching...",
      loading: "Loading...",
      errors: {
        address_outside: "Address is outside Texas. Please enter a Texas address.",
        enter_address: "Please enter an address",
        address_not_found: "Address not found. Try a more specific Texas address.",
        search_failed: "Search failed. Please try again.",
        coords_format: "Please enter coordinates as: latitude, longitude",
        coords_invalid: "Please enter valid numbers for latitude and longitude",
        coords_out_of_bounds: "Coordinates must be within Texas bounds",
      },
      hint_coords: "Enter latitude and longitude separated by comma",
    },
    tooltip: {
      county_suffix: "County",
      updating: "Updating...",
    },
    legend_title: "PM2.5 µg/m³",
    census_tract_prefix: "Census Tract",
    // Quantum Sensor Placement
    quantum: {
      tab: "Sensor Placement",
      title: "Quantum-Optimized Sensor Placement",
      subtitle: "Finding optimal locations for new air quality sensors using quantum annealing",
      loading: "Running quantum optimization...",
      loading_detail: "Solving QUBO with simulated quantum annealing across 6,900+ tracts",
      error: "Could not load quantum results. Make sure the backend is running.",
      num_sensors: "Sensors to Place",
      coverage: "Coverage",
      avg_distance: "Avg Distance to Sensor",
      max_gap: "Largest Gap",
      ej_equity: "EJ Equity Score",
      method_comparison: "Method Comparison",
      quantum_annealing: "Quantum Annealing",
      greedy: "Greedy Algorithm",
      classical_sa: "Classical Sim. Annealing",
      tracts_covered: "Tracts Covered",
      avg_ej: "Avg EJ Score",
      runtime: "Runtime",
      coverage_by_ej: "Coverage by EJ Quartile",
      q1_low: "Q1 (Low Burden)",
      q2: "Q2",
      q3: "Q3",
      q4_high: "Q4 (High Burden)",
      recommended_locations: "Recommended Sensor Locations",
      rank: "Rank",
      tract: "Tract",
      county: "County",
      composite: "Score",
      view_on_map: "View on Map",
      what_is: "What is Quantum Annealing?",
      what_is_body: "Quantum annealing uses principles from quantum mechanics — superposition and tunneling — to explore vast solution spaces for combinatorial optimization problems. Sensor placement is NP-hard: choosing 25 locations from 6,900+ tracts yields more combinations than atoms in the universe. The quantum approach formulates this as a QUBO (Quadratic Unconstrained Binary Optimization) and explores solutions via simulated quantum tunneling, escaping local optima that trap classical methods.",
      why_ej: "Why EJ-Weighted Placement?",
      why_ej_body: "Environmental justice communities bear disproportionate pollution burdens. By weighting sensor placement toward high-EJ tracts (45% of the objective), we ensure monitoring infrastructure prioritizes the communities that need it most — not just the areas easiest to cover.",
      show_quantum: "Quantum",
      show_greedy: "Greedy",
      show_classical: "Classical",
      generated: (when) => `Generated ${when}`,
    },
  },
  es: {
    brand_tagline_all: "Todo Texas · Nivel de tracto censal",
    back_to: (name) => `Volver a ${name}`,
    current_pm25: "PM2.5 actual",
    current_conditions: "Condiciones actuales",
    temperature: "Temperatura",
    humidity: "Humedad",
    pressure: "Presión",
    wind: "Viento",
    use_search: "Usa la barra de búsqueda para encontrar cualquier dirección en Texas y ver su calidad del aire. Toca cualquier tracto en el mapa para detalles con el clima en tiempo real.",
    trusted_by: (n) => `Con la confianza de ${n} usuarios en Texas`,
    updated: (when) => `Actualizado ${when} · se actualiza cada 30 min`,
    ej_context: "Contexto de Justicia Ambiental",
    tract_distribution: "Distribución de tractos",
    loading_predictions: "Generando predicciones para todos los tractos de Texas...",
    // Air Quality Guide (Spanish)
    guide_intro: "PM2.5 se refiere a las partículas finas — diminutas partículas de 2.5 micrómetros o menos. Pueden penetrar profundamente en los pulmones y entrar al torrente sanguíneo.",
    guide: {
      why_title: "Por qué importa PM2.5",
      why_body: "Partículas tan pequeñas evitan la nariz y la garganta y se alojan en el tejido pulmonar. La exposición a corto plazo causa tos e irritación. La exposición a largo plazo se relaciona con enfermedades cardíacas, accidente cerebrovascular, cáncer de pulmón y menor esperanza de vida.",
      about_title: "Sobre estas predicciones",
      about_body: "Los valores combinan datos meteorológicos en tiempo real con indicadores de justicia ambiental de la EPA. Las predicciones se actualizan cada 30 minutos y reflejan concentraciones estimadas a nivel de tracto censal.",
      levels: {
        good: {
          name: "Bueno",
          range: "0 – 8.9 µg/m³",
          description: "La calidad del aire es excelente. No hay preocupaciones de salud para nadie. Ideal para actividades al aire libre.",
          who: "Seguro para todos",
        },
        moderate: {
          name: "Moderado",
          range: "9 – 12.9 µg/m³",
          description: "La calidad del aire es aceptable. La gran mayoría de las personas no experimentará efectos. Un pequeño número de individuos inusualmente sensibles puede notar síntomas menores.",
          who: "Personas sensibles: considere reducir el esfuerzo prolongado al aire libre",
        },
        unhealthy: {
          name: "Insalubre",
          range: "13 – 17.9 µg/m³",
          description: "Todas las personas pueden comenzar a experimentar efectos en la salud. Grupos sensibles — personas con asma, enfermedades cardíacas, ancianos y niños — tienen mayor riesgo.",
          who: "Grupos sensibles: limitar el esfuerzo prolongado al aire libre. Todos: reducir actividad física intensa y prolongada.",
        },
        hazardous: {
          name: "Peligroso",
          range: "18+ µg/m³",
          description: "Avisos de salud por condiciones de emergencia. Agravamiento serio de enfermedades cardíacas y pulmonares, mortalidad prematura en grupos sensibles y efectos respiratorios en la población general.",
          who: "Todos: eviten actividad física al aire libre. Grupos sensibles: permanezcan en interiores.",
        },
      },
    },
    // Search / UI (Spanish)
    search: {
      address_tab: "📍 Dirección",
      coordinates_tab: "🧭 Coordenadas",
      placeholder_address: "Introduce dirección (p. ej., Austin, TX)",
      placeholder_coords: "Latitud, Longitud (p. ej., 30.2672, -97.7431)",
      search_button: "Buscar",
      searching: "Buscando...",
      loading: "Cargando...",
      errors: {
        address_outside: "La dirección está fuera de Texas. Por favor ingresa una dirección en Texas.",
        enter_address: "Por favor ingresa una dirección",
        address_not_found: "Dirección no encontrada. Intenta una dirección más específica en Texas.",
        search_failed: "La búsqueda falló. Por favor inténtalo de nuevo.",
        coords_format: "Por favor ingresa coordenadas como: latitud, longitud",
        coords_invalid: "Por favor ingresa números válidos para latitud y longitud",
        coords_out_of_bounds: "Las coordenadas deben estar dentro de Texas",
      },
      hint_coords: "Introduce latitud y longitud separadas por coma",
    },
    tooltip: {
      county_suffix: "Condado",
      updating: "Actualizando...",
    },
    legend_title: "PM2.5 µg/m³",
    census_tract_prefix: "Tracto censal",
    // Quantum Sensor Placement (Spanish)
    quantum: {
      tab: "Ubicación de Sensores",
      title: "Ubicación de Sensores Optimizada por Quantum",
      subtitle: "Encontrando ubicaciones óptimas para nuevos sensores de calidad del aire usando recocido cuántico",
      loading: "Ejecutando optimización cuántica...",
      loading_detail: "Resolviendo QUBO con recocido cuántico simulado en más de 6,900 tractos",
      error: "No se pudieron cargar los resultados cuánticos. Asegúrate de que el backend esté ejecutándose.",
      num_sensors: "Sensores a Colocar",
      coverage: "Cobertura",
      avg_distance: "Distancia Prom. al Sensor",
      max_gap: "Mayor Brecha",
      ej_equity: "Puntuación de Equidad EJ",
      method_comparison: "Comparación de Métodos",
      quantum_annealing: "Recocido Cuántico",
      greedy: "Algoritmo Voraz",
      classical_sa: "Recocido Simulado Clásico",
      tracts_covered: "Tractos Cubiertos",
      avg_ej: "Puntaje EJ Prom.",
      runtime: "Tiempo de Ejecución",
      coverage_by_ej: "Cobertura por Cuartil EJ",
      q1_low: "Q1 (Baja Carga)",
      q2: "Q2",
      q3: "Q3",
      q4_high: "Q4 (Alta Carga)",
      recommended_locations: "Ubicaciones Recomendadas",
      rank: "Rango",
      tract: "Tracto",
      county: "Condado",
      composite: "Puntaje",
      view_on_map: "Ver en Mapa",
      what_is: "¿Qué es el Recocido Cuántico?",
      what_is_body: "El recocido cuántico utiliza principios de la mecánica cuántica — superposición y efecto túnel — para explorar vastos espacios de soluciones en problemas de optimización combinatoria. La ubicación de sensores es NP-difícil: elegir 25 ubicaciones entre más de 6,900 tractos produce más combinaciones que átomos en el universo. El enfoque cuántico formula esto como un QUBO (Optimización Binaria Cuadrática Sin Restricciones) y explora soluciones mediante efecto túnel cuántico simulado.",
      why_ej: "¿Por Qué Ubicación Ponderada por EJ?",
      why_ej_body: "Las comunidades de justicia ambiental soportan cargas desproporcionadas de contaminación. Al ponderar la ubicación de sensores hacia tractos de alto EJ (45% del objetivo), aseguramos que la infraestructura de monitoreo priorice las comunidades que más lo necesitan.",
      show_quantum: "Cuántico",
      show_greedy: "Voraz",
      show_classical: "Clásico",
      generated: (when) => `Generado ${when}`,
    },
  }
};

const CATEGORY_MAP = {
  en: {
    Good: "Good",
    Moderate: "Moderate",
    Unhealthy: "Unhealthy",
    Hazardous: "Hazardous",
  },
  es: {
    Good: "Bueno",
    Moderate: "Moderado",
    Unhealthy: "Insalubre",
    Hazardous: "Peligroso",
  }
};

const HEALTH_MSG = {
  en: {
    Good: "Air quality is good. Enjoy outdoor activities.",
    Moderate: "Air quality is acceptable. Sensitive individuals should take precautions.",
    Unhealthy: "Air quality is unhealthy. Everyone should limit outdoor exposure.",
    Hazardous: "⚠️ Air quality is hazardous. Avoid all outdoor activities.",
  },
  es: {
    Good: "La calidad del aire es buena. Disfruta de actividades al aire libre.",
    Moderate: "La calidad del aire es aceptable. Las personas sensibles deben tomar precauciones.",
    Unhealthy: "La calidad del aire es insalubre. Todos deberían limitar la exposición al aire libre.",
    Hazardous: "⚠️ La calidad del aire es peligrosa. Evita todas las actividades al aire libre.",
  }
};

export function t(lang = 'en', key, ...args) {
  const group = TRANSLATIONS[lang] || TRANSLATIONS.en;
  const val = group[key];
  if (typeof val === 'function') return val(...args);
  // support nested keys like 'guide.levels.good.name'
  if (val == null && key.includes('.')) {
    const parts = key.split('.');
    let cur = group;
    for (const p of parts) {
      cur = cur?.[p];
      if (cur == null) break;
    }
    if (typeof cur === 'function') return cur(...args);
    if (cur != null) return cur;
    // fallback to english
    cur = TRANSLATIONS.en;
    for (const p of parts) {
      cur = cur?.[p];
      if (cur == null) break;
    }
    if (typeof cur === 'function') return cur(...args);
    if (cur != null) return cur;
  }
  if (typeof val === 'function') return val(...args);
  return val ?? TRANSLATIONS.en[key] ?? key;
}

export function translateCategory(lang = 'en', cat) {
  return CATEGORY_MAP[lang]?.[cat] ?? cat;
}

export function translateHealth(lang = 'en', cat) {
  return HEALTH_MSG[lang]?.[cat] ?? cat;
}

export default { t, translateCategory, translateHealth };
