/**
 * PM2.5 → AQI category utilities with custom gradient color scale.
 * Safety threshold: 9.0 µg/m³ (federal limit for good air quality)
 *
 * Scale:
 * 0.0-3.9:   Green gradient (light → dark)
 * 4.0-8.9:   Yellow gradient (light → dark)
 * 9.0-12.9:  Red gradient (light → dark)
 * 13.0+:     Dark red (hazardous)
 */

// Color gradients with smooth transitions
const COLOR_SCALE = {
  goodRange: {
    min: 0.0,
    max: 3.9,
    colorMin: "#90EE90",  // Light green
    colorMax: "#00b894",  // Darker green
    category: "Good",
    label: "0–3.9 µg/m³"
  },
  moderateRange: {
    min: 4.0,
    max: 8.9,
    colorMin: "#FFFF99",  // Light yellow
    colorMax: "#FFD700",  // Darker yellow/gold
    category: "Moderate",
    label: "4–8.9 µg/m³"
  },
  unhealthyRange: {
    min: 9.0,
    max: 12.9,
    colorMin: "#FF6B6B",  // Light red
    colorMax: "#d63031",  // Darker red
    category: "Unhealthy",
    label: "9–12.9 µg/m³"
  },
  hazardousRange: {
    min: 13.0,
    max: Infinity,
    colorMin: "#9d4edd",   // Light purple
    colorMax: "#3c096c",   // Dark purple
    category: "Hazardous",
    label: "13+ µg/m³"
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

  if (pm25 <= COLOR_SCALE.unhealthyRange.max) {
    // Interpolate within red range
    const factor = (pm25 - COLOR_SCALE.unhealthyRange.min) /
                   (COLOR_SCALE.unhealthyRange.max - COLOR_SCALE.unhealthyRange.min);
    return interpolateColor(
      COLOR_SCALE.unhealthyRange.colorMin,
      COLOR_SCALE.unhealthyRange.colorMax,
      factor
    );
  }

  // Hazardous (13+) - gradient from light purple to dark purple
  const hazardousFactor = Math.min(1.0, (pm25 - 13.0) / 12.0);
  return interpolateColor(
    COLOR_SCALE.hazardousRange.colorMin,
    COLOR_SCALE.hazardousRange.colorMax,
    hazardousFactor
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
      health_msg: "Air quality is good. Enjoy outdoor activities."
    };
  }

  if (pm25 <= COLOR_SCALE.moderateRange.max) {
    return {
      category: COLOR_SCALE.moderateRange.category,
      color: pm25Color(pm25),
      bg: "rgba(255, 255, 153, 0.12)",
      label: COLOR_SCALE.moderateRange.label,
      aqi_range: "Moderate",
      health_msg: "Air quality is acceptable. Sensitive individuals should take precautions."
    };
  }

  if (pm25 <= COLOR_SCALE.unhealthyRange.max) {
    return {
      category: COLOR_SCALE.unhealthyRange.category,
      color: pm25Color(pm25),
      bg: "rgba(255, 107, 107, 0.12)",
      label: COLOR_SCALE.unhealthyRange.label,
      aqi_range: "Unhealthy",
      health_msg: "Air quality is unhealthy. Everyone should limit outdoor exposure."
    };
  }

  // Hazardous
  return {
    category: COLOR_SCALE.hazardousRange.category,
    color: pm25Color(pm25),
    bg: "rgba(157, 78, 221, 0.12)",
    label: COLOR_SCALE.hazardousRange.label,
    aqi_range: "Hazardous",
    health_msg: "⚠️ Air quality is hazardous. Avoid all outdoor activities."
  };
}

/**
 * Returns 0-100 gauge fill for a given PM2.5 value (using 15 as max for better scale)
 */
export function pm25ToGaugePct(pm25) {
  return Math.min(100, (pm25 / 15) * 100);
}

/**
 * Returns a short health icon based on category
 */
export function healthIcon(category) {
  const icons = {
    "Good": "✓",
    "Moderate": "~",
    "Unhealthy": "!",
    "Hazardous": "✕",
  };
  return icons[category] ?? "?";
}

/**
 * Export breakpoints for legend display
 */
export const BREAKPOINTS = [
  { max: 3.9,   category: "Good",      color: "#00b894", label: "0–3.9" },
  { max: 8.9,   category: "Moderate",  color: "#FFD700", label: "4–8.9" },
  { max: 12.9,  category: "Unhealthy", color: "#d63031", label: "9–12.9" },
  { max: Infinity, category: "Hazardous", color: "#9d4edd", label: "13+" },
];
