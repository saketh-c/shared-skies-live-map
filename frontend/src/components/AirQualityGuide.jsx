import { useContext } from 'react';
import { LanguageContext } from '../App';
import { t } from '../i18n';

export default function AirQualityGuide() {
  const { lang } = useContext(LanguageContext);

  const colorMap = {
    good: '#00b894',
    moderate: '#FFD700',
    unhealthy: '#d63031',
    hazardous: '#8b0000',
  };

  const levels = ['good', 'moderate', 'unhealthy', 'hazardous'].map((k) => ({
    key: k,
    name: t(lang, `guide.levels.${k}.name`),
    range: t(lang, `guide.levels.${k}.range`),
    color: colorMap[k],
    description: t(lang, `guide.levels.${k}.description`),
    who: t(lang, `guide.levels.${k}.who`),
  }));

  return (
    <div className="guide-content">
      <p className="guide-intro">{t(lang, 'guide_intro')}</p>

      {levels.map((level) => (
        <div className="guide-level" key={level.key}>
          <div className="guide-level-header">
            <div className="guide-level-swatch" style={{ background: level.color }} />
            <span className="guide-level-name" style={{ color: level.color }}>
              {level.name}
            </span>
            <span className="guide-level-range">{level.range}</span>
          </div>
          <div className="guide-level-body">
            {level.description}
            <div className="guide-level-who">{level.who}</div>
          </div>
        </div>
      ))}

      <div className="guide-explainer">
        <h4>{t(lang, 'guide.why_title')}</h4>
        {t(lang, 'guide.why_body')}
      </div>

      <div className="guide-explainer">
        <h4>{t(lang, 'guide.about_title')}</h4>
        {t(lang, 'guide.about_body')}
      </div>
    </div>
  );
}
