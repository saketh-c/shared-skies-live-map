const TRANSLATIONS = {
  en: {
    brand_tagline_all: "All of Texas · Census Tract Level",
    back_to: (name) => `Back to ${name}`,
    current_pm25: "Predicted PM2.5 · 24-hr avg",
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
    guide_intro: "PM2.5 refers to fine particulate matter, tiny particles 2.5 micrometers or smaller. They can penetrate deep into the lungs and enter the bloodstream.",
    guide: {
      why_title: "Why PM2.5 Matters",
      why_body: "Particles this small bypass the nose and throat and lodge deep in lung tissue. Short-term exposure causes coughing and irritation. Long-term exposure is linked to heart disease, stroke, lung cancer, and reduced life expectancy.",
      about_title: "About These Predictions",
      about_body: "Values combine real-time weather data with EPA environmental justice indicators. Predictions refresh every 30 minutes and reflect estimated ground-level concentrations at the census tract level.",
      levels: {
        good: {
          name: "Good",
          range: "0 – 5 µg/m³",
          description: "At or below the WHO annual air-quality guideline (5 µg/m³). No health concerns for anyone — ideal for outdoor activities.",
          who: "Safe for everyone",
        },
        moderate: {
          name: "Moderate",
          range: "5 – 9 µg/m³",
          description: "Above the WHO annual guideline but within the U.S. EPA annual standard (9 µg/m³). Most people experience no effects; a few unusually sensitive individuals may notice minor symptoms.",
          who: "Unusually sensitive people: consider reducing prolonged outdoor exertion",
        },
        elevated: {
          name: "Elevated",
          range: "9 – 15 µg/m³",
          description: "Above the U.S. EPA annual PM2.5 standard (9 µg/m³). Long-term exposure at these levels is linked to cardiovascular and respiratory risk. Common in dense urban and industrial areas.",
          who: "Sensitive groups (asthma, heart disease, elderly, children): limit prolonged outdoor exertion.",
        },
        high: {
          name: "High",
          range: "15+ µg/m³",
          description: "Above the WHO 24-hour guideline (15 µg/m³). Everyone may begin to experience effects; sensitive groups are at greater risk. Often driven by wildfire smoke or dust events.",
          who: "Everyone: limit prolonged or heavy outdoor activity. Sensitive groups: stay indoors when possible.",
        },
      },
    },
    // Search / UI
    search: {
      address_tab: "Address",
      coordinates_tab: "Coordinates",
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
      aqi_label: "AQI",
      modeled_note: "Model prediction · 24-hr average",
    },
    aqi_equiv: (aqi) => `≈ US EPA AQI ${aqi}`,
    fallback_badge: "⚠ Live sensors unavailable — showing historical-average estimates",
    live_sensors_note: (n) => `${n} live PurpleAir sensors`,
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
      tab_intro_title: "About this tab",
      tab_intro_body: "This tab proposes 25 strategic locations for new PM2.5 sensors across Texas. The optimizer balances three signals: distance to the existing 240-sensor PurpleAir network (gray dots), predicted pollution at each of 6,896 census tracts, and the EPA EJSCREEN environmental-justice burden index. The result is a deployment plan that maximizes new monitoring coverage while concentrating sensors in the communities carrying the heaviest pollution load. Each blue dot is a recommended placement. Click one to inspect the underlying tract data.",
      what_is: "What is Quantum Annealing?",
      what_is_body: "Sensor placement is an NP-hard combinatorial problem. Selecting 25 sites from 6,896 candidate tracts produces roughly 10^77 possible arrangements, far beyond what brute-force enumeration can reach. We encode the task as a QUBO (Quadratic Unconstrained Binary Optimization) and solve it with simulated quantum annealing. The solver evolves a population of candidate placements across a quantum-inspired energy landscape and uses tunneling dynamics to escape local minima that trap classical greedy and gradient methods. Each run draws thousands of low-energy samples and consensus-ranks the most consistently optimal sites.",
      why_ej: "Why EJ-Weighted Placement?",
      why_ej_body: "PM2.5 exposure is not evenly distributed. Communities of color and low-income neighborhoods systematically face higher pollution levels with fewer monitoring resources. Our objective function reserves 45% of its weight for EJSCREEN burden indicators, steering new infrastructure toward the populations that need it most instead of only filling the spatial gaps that happen to be easiest to cover.",
      show_quantum: "Quantum",
      show_greedy: "Greedy",
      show_classical: "Classical",
      generated: (when) => `Generated ${when}`,
    },
  },
  es: {
    brand_tagline_all: "Todo Texas · Nivel de tracto censal",
    back_to: (name) => `Volver a ${name}`,
    current_pm25: "PM2.5 previsto · media 24 h",
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
    guide_intro: "PM2.5 se refiere a las partículas finas, diminutas partículas de 2.5 micrómetros o menos. Pueden penetrar profundamente en los pulmones y entrar al torrente sanguíneo.",
    guide: {
      why_title: "Por qué importa PM2.5",
      why_body: "Partículas tan pequeñas evitan la nariz y la garganta y se alojan en el tejido pulmonar. La exposición a corto plazo causa tos e irritación. La exposición a largo plazo se relaciona con enfermedades cardíacas, accidente cerebrovascular, cáncer de pulmón y menor esperanza de vida.",
      about_title: "Sobre estas predicciones",
      about_body: "Los valores combinan datos meteorológicos en tiempo real con indicadores de justicia ambiental de la EPA. Las predicciones se actualizan cada 30 minutos y reflejan concentraciones estimadas a nivel de tracto censal.",
      levels: {
        good: {
          name: "Bueno",
          range: "0 – 5 µg/m³",
          description: "Igual o por debajo de la guía anual de calidad del aire de la OMS (5 µg/m³). Sin preocupaciones de salud para nadie — ideal para actividades al aire libre.",
          who: "Seguro para todos",
        },
        moderate: {
          name: "Moderado",
          range: "5 – 9 µg/m³",
          description: "Por encima de la guía anual de la OMS pero dentro del estándar anual de la EPA de EE. UU. (9 µg/m³). La mayoría no experimenta efectos; algunos individuos inusualmente sensibles pueden notar síntomas menores.",
          who: "Personas sensibles: considere reducir el esfuerzo prolongado al aire libre",
        },
        elevated: {
          name: "Elevado",
          range: "9 – 15 µg/m³",
          description: "Por encima del estándar anual de PM2.5 de la EPA (9 µg/m³). La exposición prolongada a estos niveles se asocia con riesgo cardiovascular y respiratorio. Común en zonas urbanas densas e industriales.",
          who: "Grupos sensibles (asma, enfermedades cardíacas, ancianos, niños): limitar el esfuerzo prolongado al aire libre.",
        },
        high: {
          name: "Alto",
          range: "15+ µg/m³",
          description: "Por encima de la guía de 24 horas de la OMS (15 µg/m³). Todos pueden comenzar a experimentar efectos; los grupos sensibles tienen mayor riesgo. A menudo causado por humo de incendios o eventos de polvo.",
          who: "Todos: limitar la actividad al aire libre prolongada o intensa. Grupos sensibles: permanecer en interiores cuando sea posible.",
        },
      },
    },
    // Search / UI (Spanish)
    search: {
      address_tab: "Dirección",
      coordinates_tab: "Coordenadas",
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
      aqi_label: "AQI",
      modeled_note: "Predicción del modelo · media de 24 h",
    },
    aqi_equiv: (aqi) => `≈ AQI ${aqi} (EPA de EE. UU.)`,
    fallback_badge: "⚠ Sensores en vivo no disponibles — mostrando estimaciones históricas",
    live_sensors_note: (n) => `${n} sensores PurpleAir en vivo`,
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
      tab_intro_title: "Sobre esta pestaña",
      tab_intro_body: "Esta pestaña propone 25 ubicaciones estratégicas para nuevos sensores PM2.5 en Texas. El optimizador equilibra tres señales: la distancia a la red existente de 240 sensores PurpleAir (puntos grises), la contaminación pronosticada en cada uno de los 6,896 tractos censales, y el índice de carga ambiental EPA EJSCREEN del tracto. El resultado es un plan de despliegue que maximiza la cobertura de monitoreo y concentra sensores en las comunidades que cargan con la mayor contaminación. Cada punto azul es una ubicación recomendada. Haz clic en uno para inspeccionar los datos del tracto.",
      what_is: "¿Qué es el Recocido Cuántico?",
      what_is_body: "La ubicación de sensores es un problema combinatorio NP-difícil. Seleccionar 25 sitios entre 6,896 tractos candidatos produce aproximadamente 10^77 configuraciones posibles, mucho más allá de lo que la enumeración exhaustiva puede alcanzar. Codificamos la tarea como un QUBO (Optimización Binaria Cuadrática Sin Restricciones) y la resolvemos con recocido cuántico simulado. El solver hace evolucionar una población de configuraciones candidatas a través de un paisaje de energía inspirado en la mecánica cuántica, y usa dinámicas de efecto túnel para escapar de mínimos locales donde los métodos voraces y de gradiente clásicos se quedan atascados. Cada ejecución extrae miles de muestras de baja energía y clasifica por consenso los sitios más consistentemente óptimos.",
      why_ej: "¿Por Qué Ubicación Ponderada por EJ?",
      why_ej_body: "La exposición a PM2.5 no se distribuye uniformemente. Las comunidades de color y los vecindarios de bajos ingresos enfrentan sistemáticamente niveles más altos de contaminación con menos recursos de monitoreo. Nuestra función objetivo reserva el 45% de su peso para los indicadores de carga de EJSCREEN, dirigiendo la nueva infraestructura hacia las poblaciones que más la necesitan, en lugar de solo llenar los vacíos espaciales que resultan más fáciles de cubrir.",
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
    Elevated: "Elevated",
    High: "High",
  },
  es: {
    Good: "Bueno",
    Moderate: "Moderado",
    Elevated: "Elevado",
    High: "Alto",
  }
};

const HEALTH_MSG = {
  en: {
    Good: "At or below the WHO annual guideline (5 µg/m³). Air quality is good.",
    Moderate: "Above the WHO annual guideline; within the U.S. EPA annual standard (9 µg/m³).",
    Elevated: "Above the U.S. EPA annual PM2.5 standard (9 µg/m³). Sensitive groups should take care.",
    High: "⚠️ Above the WHO 24-hour guideline (15 µg/m³). Everyone should limit prolonged outdoor exposure.",
  },
  es: {
    Good: "Igual o por debajo de la guía anual de la OMS (5 µg/m³). La calidad del aire es buena.",
    Moderate: "Por encima de la guía anual de la OMS; dentro del estándar anual de la EPA de EE. UU. (9 µg/m³).",
    Elevated: "Por encima del estándar anual de PM2.5 de la EPA (9 µg/m³). Los grupos sensibles deben tener cuidado.",
    High: "⚠️ Por encima de la guía de 24 horas de la OMS (15 µg/m³). Todos deberían limitar la exposición prolongada al aire libre.",
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
