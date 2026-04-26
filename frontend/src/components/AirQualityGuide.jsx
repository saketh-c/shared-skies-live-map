import { useContext } from "react";
import { LanguageContext } from "../App";
import { t } from "../i18n";

const COLOR_MAP = {
  good:      "#10b981",
  moderate:  "#f59e0b",
  unhealthy: "#ef4444",
  hazardous: "#7f1d1d",
};

export default function AirQualityGuide() {
  const { lang } = useContext(LanguageContext);

  const levels = ["good", "moderate", "unhealthy", "hazardous"].map((k) => ({
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
      </div>
    </div>
  );
}
