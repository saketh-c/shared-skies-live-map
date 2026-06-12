/**
 * PM2.5 → category utilities with a STANDARDS-ANCHORED gradient color scale.
 *
 * Breakpoints map to published health standards so the map is paper-defensible:
 *   5  = WHO annual guideline
 *   9  = U.S. EPA annual NAAQS (2024)
 *   15 = WHO 24-hour guideline
 *
 * Scale:
 * 0.0-6.0:  Green  (clean, at/near WHO annual of 5)  "Good"
 * 6.0-9.0:  Yellow (above WHO annual, within EPA 9)   "Moderate"
 * 9.0-15.0: Orange (above EPA annual standard)      "Elevated"
 * 15.0+:    Red → dark red (above WHO 24-hr)        "High"
 * MUST stay in sync with backend/main.py pm25_color_gradient/pm25_info.
 */

// Color gradients with smooth transitions
const COLOR_SCALE = {
  goodRange: {
    min: 0.0,
    max: 6.0,
    colorMin: "#90EE90",  // Light green
    colorMax: "#00b894",  // Darker green
    category: "Good",
    label: "0–6 µg/m³"
  },
  moderateRange: {
    min: 6.0,
    max: 9.0,
    colorMin: "#FFFF99",  // Light yellow
    colorMax: "#FFD700",  // Darker yellow/gold
    category: "Moderate",
    label: "6–9 µg/m³"
  },
  elevatedRange: {
    min: 9.0,
    max: 15.0,
    colorMin: "#FFB347",  // Light orange
    colorMax: "#E8590C",  // Burnt orange
    category: "Elevated",
    label: "9–15 µg/m³"
  },
  highRange: {
    min: 15.0,
    max: Infinity,
    colorMin: "#FF6B6B",   // Red
    colorMax: "#800000",   // Dark red (darkens as pollution rises; saturates ~55)
    category: "High",
    label: "15+ µg/m³"
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
    // Interpolate within orange range (9–15, above EPA annual standard)
    const factor = (pm25 - COLOR_SCALE.elevatedRange.min) /
                   (COLOR_SCALE.elevatedRange.max - COLOR_SCALE.elevatedRange.min);
    return interpolateColor(
      COLOR_SCALE.elevatedRange.colorMin,
      COLOR_SCALE.elevatedRange.colorMax,
      factor
    );
  }

  // High (15+) - red that darkens as pollution rises (saturates ~55, so
  // wildfire-smoke days read dramatically dark).
  const highFactor = Math.min(1.0, (pm25 - 15.0) / 40.0);
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
      health_msg: "Air quality is good (at or near the WHO annual guideline of 5 µg/m³)."
    };
  }

  if (pm25 <= COLOR_SCALE.moderateRange.max) {
    return {
      category: COLOR_SCALE.moderateRange.category,
      color: pm25Color(pm25),
      bg: "rgba(255, 255, 153, 0.12)",
      label: COLOR_SCALE.moderateRange.label,
      aqi_range: "Moderate",
      health_msg: "Above the WHO annual guideline; within the U.S. EPA annual standard (9 µg/m³)."
    };
  }

  if (pm25 <= COLOR_SCALE.elevatedRange.max) {
    return {
      category: COLOR_SCALE.elevatedRange.category,
      color: pm25Color(pm25),
      bg: "rgba(232, 89, 12, 0.12)",
      label: COLOR_SCALE.elevatedRange.label,
      aqi_range: "Elevated",
      health_msg: "Above the U.S. EPA annual PM2.5 standard (9 µg/m³). Sensitive groups should take care."
    };
  }

  // High (above WHO 24-hr guideline)
  return {
    category: COLOR_SCALE.highRange.category,
    color: pm25Color(pm25),
    bg: "rgba(128, 0, 0, 0.12)",
    label: COLOR_SCALE.highRange.label,
    aqi_range: "High",
    health_msg: "⚠️ Above the WHO 24-hour guideline (15 µg/m³). Everyone should limit prolonged outdoor exposure."
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
 * Export breakpoints for legend display
 */
export const BREAKPOINTS = [
  { max: 6.0,   category: "Good",     color: "#00b894", label: "0–6" },
  { max: 9.0,   category: "Moderate", color: "#FFD700", label: "6–9" },
  { max: 15.0,  category: "Elevated", color: "#E8590C", label: "9–15" },
  { max: Infinity, category: "High",  color: "#b30000", label: "15+" },
];
