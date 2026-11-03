/**
 * PM2.5 → category utilities with a gradient color scale.
 *
 * Bands (9 µg/m³ = U.S. EPA annual NAAQS 2024; 15 = WHO 24-hr guideline):
 * 0.0-9.0:   Green  → darker green       "Good"
 * 9.0-13.0:  Yellow → gold               "Moderate"
 * 13.0-17.0: Light orange → burnt orange "Elevated"
 * 17.0+:     Red → dark red (keeps darkening with concentration) "High"
 * Each band darkens toward its upper boundary.
 * MUST stay in sync with backend/main.py pm25_color_gradient/pm25_info.
 */

// Color gradients with smooth transitions
const COLOR_SCALE = {
  goodRange: {
    min: 0.0,
    max: 9.0,
    colorMin: "#90EE90",  // Light green
    colorMax: "#00b894",  // Darker green
    category: "Good",
    label: "0–9 µg/m³"
  },
  moderateRange: {
    min: 9.0,
    max: 13.0,
    colorMin: "#FFFF99",  // Light yellow
    colorMax: "#FFD700",  // Darker yellow/gold
    category: "Moderate",
    label: "9–13 µg/m³"
  },
  elevatedRange: {
    min: 13.0,
    max: 17.0,
    colorMin: "#FFB347",  // Light orange
    colorMax: "#E8590C",  // Burnt orange
    category: "Elevated",
    label: "13–17 µg/m³"
  },
  highRange: {
    min: 17.0,
    max: Infinity,
    colorMin: "#FF6B6B",   // Red
    colorMax: "#800000",   // Dark red (darkens as pollution rises; saturates ~55)
    category: "High",
    label: "17+ µg/m³"
  }
};

/**
 * Converts hex color to RGB
 */
function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

/**
 * Converts RGB to hex
 */
function rgbToHex(r, g, b) {
  return "#" + [r, g, b].map(x => {
    const hex = x.toString(16);
    return hex.length === 1 ? "0" + hex : hex;
  }).join('');
}

/**
 * Interpolates between two colors based on a value between 0 and 1
 */
function interpolateColor(color1, color2, factor) {
  factor = Math.max(0, Math.min(1, factor));
  const rgb1 = hexToRgb(color1);
  const rgb2 = hexToRgb(color2);

  const r = Math.round(rgb1.r + (rgb2.r - rgb1.r) * factor);
  const g = Math.round(rgb1.g + (rgb2.g - rgb1.g) * factor);
  const b = Math.round(rgb1.b + (rgb2.b - rgb1.b) * factor);

  return rgbToHex(r, g, b);
}

/**
 * Gets color with gradient interpolation based on PM2.5 value
 */
export function pm25Color(pm25) {
  if (pm25 < COLOR_SCALE.goodRange.min) {
    return COLOR_SCALE.goodRange.colorMin;
  }

  if (pm25 <= COLOR_SCALE.goodRange.max) {
    // Interpolate within green range
    const factor = (pm25 - COLOR_SCALE.goodRange.min) /
                   (COLOR_SCALE.goodRange.max - COLOR_SCALE.goodRange.min);
    return interpolateColor(
      COLOR_SCALE.goodRange.colorMin,
      COLOR_SCALE.goodRange.colorMax,
      factor
    );
  }

  if (pm25 <= COLOR_SCALE.moderateRange.max) {
    // Interpolate within yellow range
    const factor = (pm25 - COLOR_SCALE.moderateRange.min) /
                   (COLOR_SCALE.moderateRange.max - COLOR_SCALE.moderateRange.min);
    return interpolateColor(
      COLOR_SCALE.moderateRange.colorMin,
      COLOR_SCALE.moderateRange.colorMax,
      factor
    );
  }

  if (pm25 <= COLOR_SCALE.elevatedRange.max) {
    // Interpolate within orange range (13–17)
    const factor = (pm25 - COLOR_SCALE.elevatedRange.min) /
                   (COLOR_SCALE.elevatedRange.max - COLOR_SCALE.elevatedRange.min);
    return interpolateColor(
      COLOR_SCALE.elevatedRange.colorMin,
      COLOR_SCALE.elevatedRange.colorMax,
      factor
    );
  }

  // High (17+) - red that keeps darkening with concentration (saturates ~55,
  // so wildfire-smoke/dust days read dramatically dark).
  const highFactor = Math.min(1.0, (pm25 - 17.0) / 38.0);
  return interpolateColor(
    COLOR_SCALE.highRange.colorMin,
    COLOR_SCALE.highRange.colorMax,
    highFactor
  );
}

/**
 * Gets AQI info for display
 */
export function getAQIInfo(pm25) {
  if (pm25 <= COLOR_SCALE.goodRange.max) {
    return {
      category: COLOR_SCALE.goodRange.category,
      color: pm25Color(pm25),
      bg: "rgba(144, 238, 144, 0.12)",
      label: COLOR_SCALE.goodRange.label,
      aqi_range: "Good",
      health_msg: "Air quality is good: within the U.S. EPA annual PM2.5 standard (9 µg/m³)."
    };
  }

  if (pm25 <= COLOR_SCALE.moderateRange.max) {
    return {
      category: COLOR_SCALE.moderateRange.category,
      color: pm25Color(pm25),
      bg: "rgba(255, 255, 153, 0.12)",
      label: COLOR_SCALE.moderateRange.label,
      aqi_range: "Moderate",
      health_msg: "Moderate: above the EPA annual standard. Unusually sensitive people may want to limit prolonged outdoor exertion."
    };
  }

  if (pm25 <= COLOR_SCALE.elevatedRange.max) {
    return {
      category: COLOR_SCALE.elevatedRange.category,
      color: pm25Color(pm25),
      bg: "rgba(232, 89, 12, 0.12)",
      label: COLOR_SCALE.elevatedRange.label,
      aqi_range: "Elevated",
      health_msg: "Elevated: approaching or above the WHO 24-hour guideline (15 µg/m³). Sensitive groups should limit prolonged outdoor activity."
    };
  }

  // High
  return {
    category: COLOR_SCALE.highRange.category,
    color: pm25Color(pm25),
    bg: "rgba(128, 0, 0, 0.12)",
    label: COLOR_SCALE.highRange.label,
    aqi_range: "High",
    health_msg: "⚠️ High: everyone may begin to feel effects; sensitive groups are at greater risk. Often driven by wildfire smoke or dust."
  };
}

/**
 * Returns 0-100 gauge fill for a given PM2.5 value (using 20 as max for better scale)
 */
export function pm25ToGaugePct(pm25) {
  return Math.min(100, (pm25 / 20) * 100);
}

/**
 * Returns a short health icon based on category
 */
export function healthIcon(category) {
  const icons = {
    "Good": "✓",
    "Moderate": "~",
    "Elevated": "!",
    "High": "✕",
  };
  return icons[category] ?? "?";
}

/**
 * Convert PM2.5 (µg/m³) to the U.S. EPA AQI (May 2024 breakpoints).
 * Shown as an equivalent next to our µg/m³ so users can compare directly with
 * AQI-displaying apps (PurpleAir map, AirNow). Mirrors backend pm25_to_epa_aqi.
 */
const EPA_AQI_BREAKPOINTS = [
  [0.0,   9.0,   0,   50],
  [9.1,   35.4,  51,  100],
  [35.5,  55.4,  101, 150],
  [55.5,  125.4, 151, 200],
  [125.5, 225.4, 201, 300],
  [225.5, 325.4, 301, 500],
];

export function pm25ToEpaAqi(pm25) {
  const c = Math.max(0, Math.floor(Number(pm25) * 10) / 10); // EPA truncates to 0.1
  for (const [cLo, cHi, aLo, aHi] of EPA_AQI_BREAKPOINTS) {
    if (c <= cHi) {
      const lo = c >= cLo ? cLo : 0.0;
      return Math.round((aHi - aLo) / (cHi - lo) * (c - lo) + aLo);
    }
  }
  return 500;
}

/**
 * Export breakpoints for legend display.
 *
 * Swatch colors are sampled straight from the gradient (pm25Color) so the map
 * legend, the SidePanel distribution dots, and the choropleth always agree.
 * Good/Moderate/Elevated use each band's upper-edge color; High samples a
 * mid-band red since that band is open-ended. (Previously the High swatch was a
 * hardcoded #b30000 that the gradient never actually produces at any value.)
 */
export const BREAKPOINTS = [
  { max: 9.0,      category: "Good",     color: pm25Color(9),  label: "0–9" },
  { max: 13.0,     category: "Moderate", color: pm25Color(13), label: "9–13" },
  { max: 17.0,     category: "Elevated", color: pm25Color(17), label: "13–17" },
  { max: Infinity, category: "High",     color: pm25Color(35), label: "17+" },
];
