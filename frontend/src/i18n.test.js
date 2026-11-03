import { describe, expect, it } from "vitest";
import { getAQIInfo } from "./utils/aqi.js";
import { formatPercentile, t, translateHealth } from "./i18n.js";

const CATEGORY_PM25 = { Good: 5, Moderate: 11, Elevated: 14, High: 30 };

describe("health messages", () => {
  it("English matches getAQIInfo for every category", () => {
    for (const [cat, pm25] of Object.entries(CATEGORY_PM25)) {
      expect(translateHealth("en", cat)).toBe(getAQIInfo(pm25).health_msg);
    }
  });

  it("Spanish has its own text for every category, without em dashes", () => {
    for (const cat of Object.keys(CATEGORY_PM25)) {
      const es = translateHealth("es", cat);
      expect(es).not.toBe(translateHealth("en", cat));
      expect(es).not.toContain("\u2014");
    }
  });
});

describe("calibration and live-layer strings", () => {
  it.each(["en", "es"])("%s strings render and avoid em dashes", (lang) => {
    const strings = [
      t(lang, "calib_applied", 46),
      t(lang, "calib_accuracy", "1.6"),
      t(lang, "calib_unavailable"),
      t(lang, "calib_reused", "Nov 2, 3:15 PM"),
      t(lang, "before_calibration", "8.70"),
      t(lang, "weather_source", "KAUS", "2:53 PM"),
      t(lang, "sensor_tip_value", "5.9", "8.4"),
      t(lang, "monitor_tip_title", "Austin Webberville"),
      t(lang, "monitor_tip_value", "5.9", 22),
      t(lang, "monitor_tip_map", "6.1"),
      t(lang, "guide.about_body"),
      t(lang, "guide.about_calib"),
    ];
    for (const s of strings) {
      expect(typeof s).toBe("string");
      expect(s).not.toContain("\u2014");
      expect(s).not.toMatch(/undefined|null/);
    }
    expect(t(lang, "calib_applied", 46)).toContain("46");
    expect(t(lang, "calib_accuracy", "1.6")).toContain("±1.6");
  });

  it("omits the raw value and hours when they are missing", () => {
    expect(t("en", "sensor_tip_value", "5.9", null)).toBe("24-hr avg 5.9 µg/m³");
    expect(t("en", "sensor_tip_value", "5.9", "8.4")).toBe("24-hr avg 5.9 µg/m³ (raw 8.4)");
    expect(t("en", "monitor_tip_value", "5.9", null)).toBe("24-hr avg 5.9 µg/m³");
    expect(t("en", "weather_source", "KAUS", null)).toBe("NWS · KAUS");
    expect(t("en", "calib_reused", null)).toBe(
      "Using the last EPA calibration; current monitor data unavailable");
  });

  it("credits EPA AirNow and the National Weather Service in both languages", () => {
    for (const lang of ["en", "es"]) {
      const text = JSON.stringify(t(lang, "credits"));
      expect(text).toContain("https://www.airnow.gov/");
      expect(text).toContain("https://www.weather.gov/");
      expect(text).not.toContain("\u2014");
    }
  });

  it("uses N/A, not a dash, for a missing percentile", () => {
    expect(formatPercentile("en", null)).toBe("N/A");
  });
});
