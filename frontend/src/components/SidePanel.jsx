import { BREAKPOINTS, pm25ToGaugePct, getAQIInfo } from "../utils/aqi.js";

function fmt(val, decimals = 1) {
  if (val == null || isNaN(val)) return "—";
  return Number(val).toFixed(decimals);
}

function fmtPct(val) {
  if (val == null || isNaN(val)) return "—";
  return `${Math.round(val)}th %ile`;
}

function timeAgo(date) {
  if (!date) return "";
  const sec = Math.floor((Date.now() - date.getTime()) / 1000);
  if (sec < 60) return "just now";
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
}

function PM25Hero({ pm25, color, category }) {
  const gaugePct = pm25ToGaugePct(pm25);
  return (
    <>
      <div className="pm25-hero">
        <div className="pm25-label">Current PM2.5</div>
        <div className="pm25-number-row">
          <div className="pm25-number" style={{ color, textShadow: `0 0 40px ${color}55` }}>
            {fmt(pm25, 2)}
          </div>
          <div className="pm25-unit">µg/m³</div>
        </div>
        <div className="aqi-badge" style={{ background: `${color}22`, color }}>
          {category}
        </div>
      </div>
      <div className="gauge-wrap">
        <div className="gauge-bar-bg">
          <div className="gauge-bar-fill" style={{ width: `${gaugePct}%`, background: color }} />
        </div>
        <div className="gauge-labels">
          <span>Good</span><span>Moderate</span><span>Unhealthy</span><span>Hazardous</span>
        </div>
      </div>
    </>
  );
}

function WeatherWidget({ weather, isLoading }) {
  return (
    <>
      <div className="section-header">
        Current Conditions
        {isLoading && <span className="weather-loading-badge">Updating...</span>}
      </div>
      <div className="weather-grid">
        <div className="weather-item">
          <div className="weather-item-label">Temperature</div>
          <div className="weather-item-value">
            {!weather ? "..." : `${fmt(weather.temperature, 0)}°`}
          </div>
          <div className="weather-item-sub">Fahrenheit</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">Humidity</div>
          <div className="weather-item-value">
            {!weather ? "..." : `${fmt(weather.humidity, 0)}%`}
          </div>
          <div className="weather-item-sub">Relative</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">Pressure</div>
          <div className="weather-item-value">
            {!weather ? "..." : fmt(weather.pressure, 0)}
          </div>
          <div className="weather-item-sub">hPa</div>
        </div>
        <div className="weather-item">
          <div className="weather-item-label">Wind</div>
          <div className="weather-item-value">
            {!weather ? "..." : fmt(weather.wind_speed, 0)}
          </div>
          <div className="weather-item-sub">mph</div>
        </div>
      </div>
    </>
  );
}

function EJContext({ tract }) {
  const ejFields = [
    { key: "ejf_score",           label: "EJ Score",          desc: "Environmental Justice percentile" },
    { key: "pct_people_of_color", label: "People of Color",   desc: "% of census tract population" },
    { key: "pct_low_income",      label: "Low Income",        desc: "% households below poverty threshold" },
    { key: "traffic_proximity",   label: "Traffic Proximity",  desc: "Traffic volume exposure percentile" },
    { key: "diesel_pm_proximity", label: "Diesel PM",          desc: "Diesel particulate exposure percentile" },
    { key: "superfund_proximity", label: "Superfund Sites",    desc: "Proximity to EPA Superfund sites" },
  ];
  const hasAny = ejFields.some((f) => tract[f.key] != null);
  if (!hasAny) return null;

  return (
    <>
      <div className="section-header">Environmental Justice Context</div>
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

function DistributionSummary({ tracts }) {
  if (!tracts?.length) return null;
  const counts = {};
  BREAKPOINTS.forEach((b) => { counts[b.category] = 0; });
  tracts.forEach((t) => { if (t.category) counts[t.category] = (counts[t.category] ?? 0) + 1; });
  const total = tracts.length;
  return (
    <>
      <div className="section-header">Tract Distribution</div>
      <div className="distribution">
        {BREAKPOINTS.map((b) => {
          const n = counts[b.category] ?? 0;
          if (n === 0) return null;
          return (
            <div className="dist-row" key={b.category}>
              <div className="dist-dot" style={{ background: b.color }} />
              <span className="dist-label">{b.category.split(" ")[0]}</span>
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
  const displayName = statewide ? "All of Texas" : "Region Overview";
  // Clean up county name — remove duplicate "County" if present
  const countyName = selectedTract?.county
    ? selectedTract.county.replace(/ County$/i, "")
    : "";

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        {selectedTract && (
          <button className="back-btn" onClick={onDeselect}>← Back to {displayName}</button>
        )}
        <div className="brand">
          <div className="brand-left">
            <div className="brand-icon">✦</div>
            <span className="brand-name">Shared Skies Initiative</span>
          </div>
          {visitCount != null && (
            <div className="visit-counter-header">
              <span className="visit-counter-dot" />
              <span><strong>{visitCount.toLocaleString()}</strong> users</span>
            </div>
          )}
        </div>
        <div className="brand-tagline">
          {selectedTract
            ? `Tract ${selectedTract.geoid?.slice(-6)}${countyName ? ` · ${countyName} County` : ''}`
            : `${displayName} · Census Tract Level`}
        </div>
      </div>

      {error && !loading && (
        <div className="error-wrap">
          <strong>Could not load predictions.</strong><br />
          Make sure the backend is running on port 8000 and the model has been trained.<br />
          <code style={{ fontSize: 10, opacity: 0.7 }}>{error}</code>
        </div>
      )}

      {loading && !predictions && (
        <div className="loading-wrap">
          <div className="spinner" />
          <div className="loading-text">Generating predictions for all Texas tracts...</div>
        </div>
      )}

      {/* Selected tract — PM2.5 is from bulk predictions (matches tooltip exactly) */}
      {!loading && selectedTract && (
        <>
          <PM25Hero
            pm25={selectedTract.pm25}
            color={selectedTract.color}
            category={selectedTract.category}
          />
          <div className="health-card">{selectedTract.health_msg}</div>
          {/* Weather: show local real-time weather, or fallback to bulk weather */}
          <WeatherWidget
            weather={localWeather || predictions?.weather}
            isLoading={weatherLoading}
          />
          {localWeather && (
            <div className="weather-source-note">
              Weather fetched in real-time for this tract's coordinates
            </div>
          )}
          <EJContext tract={selectedTract} />
        </>
      )}

      {!loading && !selectedTract && predictions && (
        <div className="overview-body">
          <PM25Hero
            pm25={predictions.avg_pm25}
            color={predictions.avg_info?.color ?? "#00b894"}
            category={predictions.avg_info?.category ?? "Good"}
          />
          <div className="health-card">{predictions.avg_info?.health_msg}</div>
          <div className="prompt-hint">
            Use the search bar to find any Texas address and view its air quality.
            Tap any tract on the map for details with real-time weather.
          </div>
          <WeatherWidget weather={predictions.weather} isLoading={false} />
          <DistributionSummary tracts={predictions.tracts} />
        </div>
      )}

      <div className="sidebar-footer">
        {lastUpdated && (
          <div className="last-updated">Updated {timeAgo(lastUpdated)} · refreshes every 30 min</div>
        )}
        Data: EPA EJScreen · Open-Meteo · ML ensemble model<br />
        Built by <a href="#" target="_blank" rel="noreferrer">Shared Skies Initiative</a> · Texas-wide Coverage
      </div>
    </aside>
  );
}
