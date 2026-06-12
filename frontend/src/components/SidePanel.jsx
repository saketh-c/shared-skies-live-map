import { BREAKPOINTS, pm25ToGaugePct, pm25ToEpaAqi } from "../utils/aqi.js";
import { useContext, useMemo, memo } from "react";
import { LanguageContext } from "../App";
import { t, translateCategory, translateHealth } from "../i18n";
import SidebarHeader from "./SidebarHeader.jsx";

function fmt(val, decimals = 1) {
  if (val == null || isNaN(val)) return "—";
  return Number(val).toFixed(decimals);
}

function fmtPct(val) {
  if (val == null || isNaN(val)) return "—";
  return `${Math.round(val)}th %ile`;
}

function fmtPctEs(val) {
  if (val == null || isNaN(val)) return "—";
  return `Pct. ${Math.round(val)}`;
}

function timeAgo(date, lang = "en") {
  if (!date) return "";
  const sec = Math.floor((Date.now() - date.getTime()) / 1000);
  if (lang === "es") {
    if (sec < 60) return "ahora";
    if (sec < 3600) return `hace ${Math.floor(sec / 60)} min`;
    return `hace ${Math.floor(sec / 3600)} h`;
  }
  if (sec < 60) return "just now";
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
}

const PM25Hero = memo(function PM25Hero({ pm25, epaAqi, color, category, lang }) {
  const gaugePct = pm25ToGaugePct(pm25);
  const aqi = epaAqi ?? pm25ToEpaAqi(pm25);
  return (
    <>
      <div className="pm25-hero">
        <div className="pm25-label">{t(lang, "current_pm25")}</div>
        <div className="pm25-number-row">
          <div
            className="pm25-number"
            style={{ color, textShadow: `0 0 40px ${color}55` }}
            data-num
          >
            {fmt(pm25, 2)}
          </div>
          <div className="pm25-unit">µg/m³</div>
        </div>
        <div className="pm25-aqi-equiv" data-num>{t(lang, "aqi_equiv", aqi)}</div>
        <div
          className="aqi-badge"
          style={{
            background: `${color}1f`,
            color,
            borderColor: `${color}55`,
          }}
        >
          {translateCategory(lang, category)}
        </div>
      </div>
      <div className="gauge-wrap">
        <div className="gauge-bar-bg">
          <div
            className="gauge-bar-fill"
            style={{ width: `${gaugePct}%`, background: color, color }}
          />
        </div>
        <div className="gauge-labels">
          <span>{translateCategory(lang, "Good")}</span>
          <span>{translateCategory(lang, "Moderate")}</span>
          <span>{translateCategory(lang, "Elevated")}</span>
          <span>{translateCategory(lang, "High")}</span>
        </div>
      </div>
    </>
  );
});

const WeatherWidget = memo(function WeatherWidget({ weather, isLoading, lang }) {
  return (
    <>
      <div className="section-header">
        <span>{t(lang, "current_conditions")}</span>
        {isLoading && (
          <span className="weather-loading-badge">{t(lang, "tooltip.updating")}</span>
        )}
      </div>
      <div className="weather-grid">
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, "temperature")}</div>
          <div className="weather-item-value" data-num>
            {!weather ? "—" : `${fmt(weather.temperature, 0)}°`}
          </div>
          <div className="weather-item-sub">Fahrenheit</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, "humidity")}</div>
          <div className="weather-item-value" data-num>
            {!weather ? "—" : `${fmt(weather.humidity, 0)}%`}
          </div>
          <div className="weather-item-sub">{lang === "es" ? "Relativa" : "Relative"}</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, "pressure")}</div>
          <div className="weather-item-value" data-num>
            {!weather ? "—" : fmt(weather.pressure, 0)}
          </div>
          <div className="weather-item-sub">hPa</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">{t(lang, "wind")}</div>
          <div className="weather-item-value" data-num>
            {!weather ? "—" : fmt(weather.wind_speed, 0)}
          </div>
          <div className="weather-item-sub">mph</div>
        </div>
      </div>
    </>
  );
});

function EJContext({ tract, lang }) {
  const ejFields = useMemo(() => ([
    { key: "ejf_score",           label: lang === "es" ? "Puntaje EJ" : "EJ Score",          desc: lang === "es" ? "Percentil de Justicia Ambiental" : "Environmental Justice percentile" },
    { key: "pct_people_of_color", label: lang === "es" ? "Personas de Color" : "People of Color", desc: lang === "es" ? "% de la población del tracto" : "% of census tract population" },
    { key: "pct_low_income",      label: lang === "es" ? "Bajos ingresos" : "Low Income",        desc: lang === "es" ? "% bajo el umbral de pobreza" : "% households below poverty threshold" },
    { key: "traffic_proximity",   label: lang === "es" ? "Tráfico" : "Traffic Proximity",        desc: lang === "es" ? "Exposición al volumen de tráfico" : "Traffic volume exposure percentile" },
    { key: "diesel_pm_proximity", label: lang === "es" ? "PM diésel" : "Diesel PM",              desc: lang === "es" ? "Exposición a partículas diésel" : "Diesel particulate exposure percentile" },
    { key: "superfund_proximity", label: lang === "es" ? "Superfund" : "Superfund Sites",        desc: lang === "es" ? "Cercanía a sitios Superfund" : "Proximity to EPA Superfund sites" },
  ]), [lang]);

  const hasAny = ejFields.some((f) => tract[f.key] != null);
  if (!hasAny) return null;

  return (
    <>
      <div className="section-header">{t(lang, "ej_context")}</div>
      <div className="ej-list">
        {ejFields.map(({ key, label, desc }) => {
          const val = tract[key];
          if (val == null) return null;
          const pct = Math.min(100, Math.max(0, val));
          return (
            <div className="ej-item" key={key}>
              <div className="ej-item-header">
                <span className="ej-item-name">{label}</span>
                <span className="ej-item-value" data-num>
                  {lang === "es" ? fmtPctEs(val) : fmtPct(val)}
                </span>
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
  const stats = useMemo(() => {
    if (!tracts?.length) return null;
    const counts = {};
    BREAKPOINTS.forEach((b) => { counts[b.category] = 0; });
    tracts.forEach((tt) => {
      if (tt.category) counts[tt.category] = (counts[tt.category] ?? 0) + 1;
    });
    return { counts, total: tracts.length };
  }, [tracts]);

  if (!stats) return null;
  const { counts, total } = stats;

  return (
    <>
      <div className="section-header">{t(lang, "tract_distribution")}</div>
      <div className="distribution">
        {BREAKPOINTS.map((b) => {
          const n = counts[b.category] ?? 0;
          if (n === 0) return null;
          return (
            <div className="dist-row" key={b.category}>
              <div className="dist-dot" style={{ background: b.color, color: b.color }} />
              <span className="dist-label">{translateCategory(lang, b.category)}</span>
              <span className="dist-count" data-num>{n.toLocaleString()}</span>
              <span className="dist-pct" data-num>{Math.round((n / total) * 100)}%</span>
            </div>
          );
        })}
      </div>
    </>
  );
}

function LoadingSkeleton() {
  return (
    <div className="loading-wrap" aria-busy="true" aria-live="polite">
      <div className="sk-row">
        <div className="skeleton sk-h" style={{ width: "40%" }} />
        <div className="skeleton sk-num" />
      </div>
      <div className="skeleton sk-bar" style={{ marginTop: 6 }} />
      <div className="sk-grid" style={{ marginTop: 12 }}>
        <div className="skeleton sk-card" />
        <div className="skeleton sk-card" />
        <div className="skeleton sk-card" />
        <div className="skeleton sk-card" />
      </div>
      <div className="loading-text">{/* fallback narrative */}</div>
    </div>
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
  const { lang } = useContext(LanguageContext);

  return (
    <aside className="sidebar-content">
      <SidebarHeader
        selectedTract={selectedTract}
        visitCount={visitCount}
        onDeselect={onDeselect}
        statewide={statewide}
      />

      {error && !loading && (
        <div className="error-wrap">
          <strong>
            {lang === "es" ? "No se pudieron cargar las predicciones." : "Could not load predictions."}
          </strong>
          <br />
          {lang === "es"
            ? "Asegúrate de que el backend esté ejecutándose en el puerto 8000."
            : "Make sure the backend is running on port 8000."}
          <br />
          <code style={{ fontSize: 10, opacity: 0.7 }}>{error}</code>
        </div>
      )}

      {loading && !predictions && <LoadingSkeleton />}

      {!loading && selectedTract && (
        <>
          <PM25Hero
            pm25={selectedTract.pm25}
            epaAqi={selectedTract.epa_aqi}
            color={selectedTract.color}
            category={selectedTract.category}
            lang={lang}
          />
          <div className="health-card">
            {translateHealth(lang, selectedTract.category) || selectedTract.health_msg}
          </div>
          <WeatherWidget
            weather={localWeather || predictions?.weather}
            isLoading={weatherLoading}
            lang={lang}
          />
          {localWeather && (
            <div className="weather-source-note">
              {lang === "es"
                ? "Clima en tiempo real para las coordenadas de este tracto"
                : "Live weather fetched for this tract's coordinates"}
            </div>
          )}
          <EJContext tract={selectedTract} lang={lang} />
        </>
      )}

      {!loading && !selectedTract && predictions && (
        <div className="overview-body">
          <PM25Hero
            pm25={predictions.avg_pm25}
            epaAqi={predictions.avg_epa_aqi}
            color={predictions.avg_info?.color ?? "#10b981"}
            category={predictions.avg_info?.category ?? "Good"}
            lang={lang}
          />
          <div className="health-card">
            {translateHealth(lang, predictions.avg_info?.category) || predictions.avg_info?.health_msg}
          </div>
          <div className="prompt-hint">{t(lang, "use_search")}</div>
          <WeatherWidget weather={predictions.weather} isLoading={false} lang={lang} />
          <DistributionSummary tracts={predictions.tracts} lang={lang} />
        </div>
      )}

      <div className="sidebar-footer">
        {predictions?.data_sources && !predictions.data_sources.using_live_neighbors && (
          <div className="footer-row fallback-badge">
            {t(lang, "fallback_badge")}
          </div>
        )}
        {lastUpdated && (
          <div className="footer-row">
            <span className="last-updated">
              {t(lang, "updated", timeAgo(lastUpdated, lang))}
              {predictions?.data_sources?.using_live_neighbors && (
                <> · {t(lang, "live_sensors_note", predictions.data_sources.live_purpleair_sensors)}</>
              )}
            </span>
          </div>
        )}
        <div className="footer-row footer-data">
          {lang === "es"
            ? "Datos: EPA EJScreen · Open-Meteo · Modelo de ensamblado ML"
            : "Data: EPA EJScreen · Open-Meteo · ML ensemble model"}
        </div>
        <div className="footer-row footer-built">
          {lang === "es" ? "Creado por" : "Built by"}{" "}
          <a
            href="https://sharedskiesinitiative.org/real-time-map"
            target="_blank"
            rel="noreferrer"
          >
            Shared Skies Initiative
          </a>{" "}
          · {lang === "es" ? "Cobertura en todo Texas" : "Texas-wide Coverage"}
        </div>
      </div>
    </aside>
  );
}
