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
  return val ?? TRANSLATIONS.en[key] ?? key;
}

export function translateCategory(lang = 'en', cat) {
  return CATEGORY_MAP[lang]?.[cat] ?? cat;
}

export function translateHealth(lang = 'en', cat) {
  return HEALTH_MSG[lang]?.[cat] ?? cat;
}

export default { t, translateCategory, translateHealth };
