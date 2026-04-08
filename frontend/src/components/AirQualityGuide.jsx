export default function AirQualityGuide() {
  const levels = [
    {
      name: "Good",
      range: "0 – 3.9 µg/m³",
      color: "#00b894",
      description:
        "Air quality is excellent. No health concerns for anyone. Ideal for outdoor activities and extended time outside.",
      who: "Safe for everyone",
    },
    {
      name: "Moderate",
      range: "4 – 8.9 µg/m³",
      color: "#FFD700",
      description:
        "Air quality is acceptable. The vast majority of people will not experience effects. A very small number of unusually sensitive individuals may notice minor symptoms.",
      who: "Unusually sensitive people: consider reducing prolonged outdoor exertion",
    },
    {
      name: "Unhealthy",
      range: "9 – 12.9 µg/m³",
      color: "#d63031",
      description:
        "Everyone may begin to experience health effects. Sensitive groups — people with asthma, heart disease, the elderly, and children — are at greater risk.",
      who: "Sensitive groups: limit prolonged outdoor exertion. Everyone: reduce extended heavy outdoor activity.",
    },
    {
      name: "Hazardous",
      range: "13+ µg/m³",
      color: "#9d4edd",
      description:
        "Health warnings of emergency conditions. Serious aggravation of heart and lung disease, premature mortality in sensitive groups, and respiratory effects in the general population. Color intensifies (light purple to dark purple) as pollution increases.",
      who: "Everyone: avoid all outdoor physical activity. Sensitive groups: remain indoors.",
    },
  ];

  return (
    <div className="guide-content">
      <p className="guide-intro">
        PM2.5 refers to fine particulate matter — tiny particles 2.5 micrometers or smaller.
        They can penetrate deep into the lungs and enter the bloodstream, making them one of the
        most harmful air pollutants.
      </p>

      {levels.map((level) => (
        <div className="guide-level" key={level.name}>
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
        <h4>Why PM2.5 Matters</h4>
        Particles this small bypass the nose and throat and lodge deep in lung tissue. Short-term
        exposure causes coughing and irritation. Long-term exposure is linked to heart disease,
        stroke, lung cancer, and reduced life expectancy.
      </div>

      <div className="guide-explainer">
        <h4>About These Predictions</h4>
        Values combine real-time weather data with EPA environmental justice indicators.
        Predictions refresh every 30 minutes and reflect estimated ground-level concentrations
        at the census tract level.
      </div>
    </div>
  );
}
