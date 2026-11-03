import { useContext } from "react";
import { LanguageContext } from "../App";
import { t } from "../i18n";
import { pm25Color } from "../utils/aqi";

// Swatch colors are sampled from the choropleth gradient (pm25Color) at each
// band's UPPER edge so the guide matches the legend swatches (BREAKPOINTS)
// exactly. Bands: Good 0–9, Moderate 9–13, Elevated 13–17, High 17+.
const COLOR_MAP = {
  good:     pm25Color(9),    // darkest green  (#00b894)
  moderate: pm25Color(13),   // gold           (#FFD700)
  elevated: pm25Color(17),   // burnt orange   (#E8590C)
  high:     pm25Color(35),   // mid-High red   (gradient #FF6B6B→#800000)
};

// `calibrated`: the backend calibrates to EPA monitors (it sends a
// calibration block), so the About text may say so.
export default function AirQualityGuide({ calibrated = false }) {
  const { lang } = useContext(LanguageContext);

  const levels = ["good", "moderate", "elevated", "high"].map((k) => ({
    key: k,
    name: t(lang, `guide.levels.${k}.name`),
    range: t(lang, `guide.levels.${k}.range`),
    color: COLOR_MAP[k],
    description: t(lang, `guide.levels.${k}.description`),
    who: t(lang, `guide.levels.${k}.who`),
  }));

  return (
    <div className="guide-content">
      <p className="guide-intro">{t(lang, "guide_intro")}</p>

      {levels.map((level) => (
        <div className="guide-level" key={level.key}>
          <div className="guide-level-header">
            <div className="guide-level-swatch" style={{ background: level.color, color: level.color }} />
            <span className="guide-level-name" style={{ color: level.color }}>
              {level.name}
            </span>
            <span className="guide-level-range" data-num>{level.range}</span>
          </div>
          <div className="guide-level-body">
            {level.description}
            <div className="guide-level-who">{level.who}</div>
          </div>
        </div>
      ))}

      <div className="guide-explainer">
        <h4>{t(lang, "guide.why_title")}</h4>
        {t(lang, "guide.why_body")}
      </div>

      <div className="guide-explainer">
        <h4>{t(lang, "guide.about_title")}</h4>
        {t(lang, "guide.about_body")}
        {calibrated && ` ${t(lang, "guide.about_calib")}`}
      </div>
    </div>
  );
}
