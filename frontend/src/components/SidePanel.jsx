import { BREAKPOINTS, pm25ToGaugePct, getAQIInfo } from "../utils/aqi.js";
import { useContext } from "react";
import { LanguageContext } from "../App";
import { t, translateCategory, translateHealth } from "../i18n";

function fmt(val, decimals = 1) {
  if (val == null || isNaN(val)) return "—";
  return Number(val).toFixed(decimals);
}

function fmtPct(val) {
  if (val == null || isNaN(val)) return "—";
  return `${Math.round(val)}th %ile`;
}

function timeAgo(date, lang = 'en') {
  if (!date) return "";
  const sec = Math.floor((Date.now() - date.getTime()) / 1000);
  if (sec < 60) return lang === 'es' ? "ahora" : "just now";
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
}

function PM25Hero({ pm25, color, category, lang }) {
  const gaugePct = pm25ToGaugePct(pm25);
  return (
    <>
      <div className="pm25-hero">
        <div className="pm25-label">{t(lang, 'current_pm25')}</div>
        <div className="pm25-number-row">
          <div className="pm25-number" style={{ color, textShadow: `0 0 40px ${color}55` }}>
            {fmt(pm25, 2)}
          </div>
          <div className="pm25-unit">µg/m³</div>
        </div>
        <div className="aqi-badge" style={{ background: `${color}22`, color }}>
          {translateCategory(lang, category)}
        </div>
      </div>
      <div className="gauge-wrap">
        <div className="gauge-bar-bg">
          <div className="gauge-bar-fill" style={{ width: `${gaugePct}%`, background: color }} />
        </div>
        <div className="gauge-labels">
          <span>{translateCategory(lang, 'Good')}</span>
          <span>{translateCategory(lang, 'Moderate')}</span>
          <span>{translateCategory(lang, 'Unhealthy')}</span>
          <span>{translateCategory(lang, 'Hazardous')}</span>
        </div>
      </div>
    </>
  );
}

function WeatherWidget({ weather, isLoading, lang }) {
  return (
    <>
      <div className="section-header">
        {t(lang, 'current_conditions')}
        {isLoading && <span className="weather-loading-badge">{t(lang, 'tooltip.updating')}</span>}
      </div>
      <div className="weather-grid">
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, 'temperature')}</div>
          <div className="weather-item-value">
            {!weather ? "..." : `${fmt(weather.temperature, 0)}°`}
          </div>
          <div className="weather-item-sub">Fahrenheit</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, 'humidity')}</div>
          <div className="weather-item-value">
            {!weather ? "..." : `${fmt(weather.humidity, 0)}%`}
          </div>
          <div className="weather-item-sub">Relative</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, 'pressure')}</div>
          <div className="weather-item-value">
            {!weather ? "..." : fmt(weather.pressure, 0)}
          </div>
          <div className="weather-item-sub">hPa</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, 'wind')}</div>
          <div className="weather-item-value">
            {!weather ? "..." : fmt(weather.wind_speed, 0)}
          </div>
          <div className="weather-item-sub">mph</div>
        </div>
      </div>
    </>
  );
}

function EJContext({ tract, lang }) {
  const ejFields = [
    { key: "ejf_score",           label: lang === 'es' ? "Puntaje EJ" : "EJ Score",          desc: lang === 'es' ? "Percentil de Justicia Ambiental" : "Environmental Justice percentile" },
    { key: "pct_people_of_color", label: lang === 'es' ? "Personas de Color" : "People of Color",   desc: lang === 'es' ? "% de la población del tracto censal" : "% of census tract population" },
    { key: "pct_low_income",      label: lang === 'es' ? "Bajos ingresos" : "Low Income",        desc: lang === 'es' ? "% hogares por debajo del umbral de pobreza" : "% households below poverty threshold" },
    { key: "traffic_proximity",   label: lang === 'es' ? "Proximidad al tráfico" : "Traffic Proximity",  desc: lang === 'es' ? "Percentil de exposición al volumen de tráfico" : "Traffic volume exposure percentile" },
    { key: "diesel_pm_proximity", label: lang === 'es' ? "PM diésel" : "Diesel PM",          desc: lang === 'es' ? "Percentil de exposición a partículas diésel" : "Diesel particulate exposure percentile" },
    { key: "superfund_proximity", label: lang === 'es' ? "Sitios Superfund" : "Superfund Sites",    desc: lang === 'es' ? "Proximidad a sitios Superfund de la EPA" : "Proximity to EPA Superfund sites" },
  ];
  const hasAny = ejFields.some((f) => tract[f.key] != null);
  if (!hasAny) return null;

  return (
    <>
      <div className="section-header">{t(lang, 'ej_context')}</div>
      <div className="ej-list">
        {ejFields.map(({ key, label, desc }) => {
          const val = tract[key];
          if (val == null) return null;
          const pct = Math.min(100, Math.max(0, val));
          return (
            <div className="ej-item" key={key}>
              <div className="ej-item-header">
                <span className="ej-item-name">{label}</span>
                <span className="ej-item-value">{fmtPct(val)}</span>
              </div>
              <div className="ej-bar-bg">
                <div className="ej-bar-fill" style={{ width: `${pct}%` }} />
              </div>
              <div className="ej-context">{desc}</div>
            </div>
          );
        })}
      </div>
    </>
  );
}

function DistributionSummary({ tracts, lang }) {
  if (!tracts?.length) return null;
  const counts = {};
  BREAKPOINTS.forEach((b) => { counts[b.category] = 0; });
  tracts.forEach((t) => { if (t.category) counts[t.category] = (counts[t.category] ?? 0) + 1; });
  const total = tracts.length;
  return (
    <>
      <div className="section-header">{t(lang, 'tract_distribution')}</div>
      <div className="distribution">
        {BREAKPOINTS.map((b) => {
          const n = counts[b.category] ?? 0;
          if (n === 0) return null;
          return (
            <div className="dist-row" key={b.category}>
              <div className="dist-dot" style={{ background: b.color }} />
              <span className="dist-label">{translateCategory(lang, b.category).split(" ")[0]}</span>
              <span className="dist-count">{n}</span>
              <span className="dist-pct">{Math.round((n / total) * 100)}%</span>
            </div>
          );
        })}
      </div>
    </>
  );
}

export default function SidePanel({
  predictions,
  selectedTract,
  localWeather,
  onDeselect,
  loading,
  weatherLoading = false,
  error,
  lastUpdated,
  statewide = false,
  visitCount,
}) {
  const { lang, setLang } = useContext(LanguageContext);
  const displayName = statewide ? (lang === 'es' ? 'Todo Texas' : 'All of Texas') : (lang === 'es' ? 'Resumen de la región' : 'Region Overview');
  // Clean up county name — remove duplicate "County" if present
  const countyName = selectedTract?.county
    ? selectedTract.county.replace(/ County$/i, "")
    : "";

  function toggleLang(newLang) {
    setLang(newLang);
    try { localStorage.setItem('ssi_lang', newLang); } catch (e) {}
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        {selectedTract && (
          <button className="back-btn" onClick={onDeselect}>← {t(lang, 'back_to', displayName)}</button>
        )}
        <div className="brand">
          <div className="brand-left">
            <div className="brand-icon">✦</div>
            <span className="brand-name">Shared Skies Initiative</span>
          </div>
          {visitCount != null && (
            <div className="visit-counter-header">
              <span className="visit-counter-dot" />
              <span><strong>{visitCount.toLocaleString()}</strong> {lang === 'es' ? 'usuarios' : 'users'}</span>
            </div>
          )}
        </div>
        <div className="brand-tagline">
          {selectedTract
            ? `Tract ${selectedTract.geoid?.slice(-6)}${countyName ? ` · ${countyName} County` : ''}`
            : t(lang, 'brand_tagline_all')}
        </div>

        {/* Language toggle next to brand */}
        <div style={{ paddingTop: 8, paddingLeft: 24 }}>
          <div className="lang-toggle" role="tablist" aria-label="Language selector">
            <button
              className={`lang-btn ${lang === 'en' ? 'active' : ''}`}
              onClick={() => toggleLang('en')}
            >EN</button>
            <span className="sep">|</span>
            <button
              className={`lang-btn ${lang === 'es' ? 'active' : ''}`}
              onClick={() => toggleLang('es')}
            >ES</button>
          </div>
        </div>
      </div>

      {error && !loading && (
        <div className="error-wrap">
          <strong>{lang === 'es' ? 'No se pudieron cargar las predicciones.' : 'Could not load predictions.'}</strong><br />
          {lang === 'es' ? 'Asegúrate de que el backend esté ejecutándose en el puerto 8000 y que el modelo haya sido entrenado.' : 'Make sure the backend is running on port 8000 and the model has been trained.'}<br />
          <code style={{ fontSize: 10, opacity: 0.7 }}>{error}</code>
        </div>
      )}

      {loading && !predictions && (
        <div className="loading-wrap">
          <div className="spinner" />
          <div className="loading-text">{t(lang, 'loading_predictions')}</div>
        </div>
      )}

      {/* Selected tract — PM2.5 is from bulk predictions (matches tooltip exactly) */}
      {!loading && selectedTract && (
        <>
          <PM25Hero
            pm25={selectedTract.pm25}
            color={selectedTract.color}
            category={selectedTract.category}
            lang={lang}
          />
          <div className="health-card">{translateHealth(lang, selectedTract.category) || selectedTract.health_msg}</div>
          {/* Weather: show local real-time weather, or fallback to bulk weather */}
          <WeatherWidget
            weather={localWeather || predictions?.weather}
            isLoading={weatherLoading}
            lang={lang}
          />
          {localWeather && (
            <div className="weather-source-note">
              {lang === 'es' ? "Clima obtenido en tiempo real para las coordenadas de este tracto" : "Weather fetched in real-time for this tract's coordinates"}
            </div>
          )}
          <EJContext tract={selectedTract} lang={lang} />
        </>
      )}

      {!loading && !selectedTract && predictions && (
        <div className="overview-body">
          <PM25Hero
            pm25={predictions.avg_pm25}
            color={predictions.avg_info?.color ?? "#00b894"}
            category={predictions.avg_info?.category ?? "Good"}
            lang={lang}
          />
          <div className="health-card">{translateHealth(lang, predictions.avg_info?.category) || predictions.avg_info?.health_msg}</div>
          <div className="prompt-hint">
            {t(lang, 'use_search')}
          </div>
          <WeatherWidget weather={predictions.weather} isLoading={false} lang={lang} />
          <DistributionSummary tracts={predictions.tracts} lang={lang} />
        </div>
      )}

      <div className="sidebar-footer">
        {lastUpdated && (
          <div className="last-updated">{t(lang, 'updated', timeAgo(lastUpdated, lang))}</div>
        )}
        {lang === 'es' ? 'Datos: EPA EJScreen · Open-Meteo · Modelo de ensamblado ML' : 'Data: EPA EJScreen · Open-Meteo · ML ensemble model'}<br />
        {lang === 'es' ? 'Creado por' : 'Built by'} <a href="#" target="_blank" rel="noreferrer">Shared Skies Initiative</a> · {lang === 'es' ? 'Cobertura en todo Texas' : 'Texas-wide Coverage'}
      </div>
    </aside>
  );
}
