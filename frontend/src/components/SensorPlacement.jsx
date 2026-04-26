import { useContext, memo } from "react";
import { LanguageContext } from "../App";
import { t } from "../i18n";
import SidebarHeader from "./SidebarHeader.jsx";

const MetricCard = memo(function MetricCard({ label, value, sub, accent }) {
  return (
    <div className="qm-metric-card" style={accent ? { color: accent } : {}}>
      <div className="qm-metric-label">{label}</div>
      <div className="qm-metric-value" style={accent ? { color: accent } : {}} data-num>
        {value}
      </div>
      {sub && <div className="qm-metric-sub">{sub}</div>}
    </div>
  );
});

function EJQuartileBar({ label, data }) {
  if (!data) return null;
  return (
    <div className="qm-ej-bar-row">
      <span className="qm-ej-bar-label">{label}</span>
      <div className="qm-ej-bar-track">
        <div className="qm-ej-bar-fill" style={{ width: `${data.pct}%` }} />
      </div>
      <span className="qm-ej-bar-pct" data-num>{data.pct}%</span>
      <span className="qm-ej-bar-detail" data-num>
        {data.covered}/{data.total}
      </span>
    </div>
  );
}

export default function SensorPlacement({
  quantumData,
  loading,
  error,
  onViewSensor,
  selectedTract,
  onDeselect,
  visitCount,
}) {
  const { lang } = useContext(LanguageContext);

  const sharedHeader = (
    <SidebarHeader
      selectedTract={selectedTract}
      visitCount={visitCount}
      onDeselect={onDeselect}
      statewide={true}
    />
  );

  if (loading) {
    return (
      <>
        {sharedHeader}
        <div className="qm-loading" aria-busy="true" aria-live="polite">
          <div className="qm-loading-icon">
            <div className="qm-atom">
              <div className="qm-orbit qm-orbit-1" />
              <div className="qm-orbit qm-orbit-2" />
              <div className="qm-orbit qm-orbit-3" />
              <div className="qm-nucleus" />
            </div>
          </div>
          <div className="qm-loading-text">{t(lang, "quantum.loading")}</div>
          <div className="qm-loading-detail">{t(lang, "quantum.loading_detail")}</div>
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        {sharedHeader}
        <div className="qm-error">
          <strong>{t(lang, "quantum.error")}</strong>
          <code>{error}</code>
        </div>
      </>
    );
  }

  if (!quantumData?.methods) return sharedHeader;

  const quantum = quantumData.methods.quantum_annealing;
  const quantumCov = quantum.coverage;
  const quantumTracts = quantum.selected_tracts;
  const ejQuartiles = quantumCov.coverage_by_ej_quartile || {};

  return (
    <div className="qm-container">
      {sharedHeader}

      <div className="qm-header">
        <div className="qm-title">{t(lang, "quantum.title")}</div>
        <div className="qm-subtitle">{t(lang, "quantum.subtitle")}</div>
      </div>

      {quantumData.num_existing_sensors > 0 && (
        <div className="qm-existing-info">
          <span className="qm-existing-dot" />
          <span data-num>
            {lang === "es"
              ? `${quantumData.num_existing_sensors} sensores PurpleAir existentes`
              : `${quantumData.num_existing_sensors} existing PurpleAir sensors`}
          </span>
          <span className="qm-existing-hint">
            {lang === "es" ? "puntos grises en el mapa" : "grey dots on the map"}
          </span>
        </div>
      )}

      <div className="qm-metrics-grid">
        <MetricCard
          label={t(lang, "quantum.num_sensors")}
          value={quantumData.num_sensors}
          sub={lang === "es" ? "nuevos a colocar" : "new placements"}
          accent="#38bdf8"
        />
        <MetricCard
          label={t(lang, "quantum.coverage")}
          value={`${quantumCov.pct_covered}%`}
          sub={
            quantumCov.new_covered != null
              ? `+${quantumCov.new_covered} ${lang === "es" ? "nuevos" : "new"}`
              : `${quantumCov.covered_count}/${quantumCov.total_tracts}`
          }
          accent="#10b981"
        />
        <MetricCard
          label={t(lang, "quantum.avg_distance")}
          value={`${quantumCov.avg_distance_miles} mi`}
          accent="#fbbf24"
        />
        <MetricCard
          label={t(lang, "quantum.max_gap")}
          value={`${quantumCov.max_distance_miles} mi`}
          accent="#ef4444"
        />
      </div>

      <div className="section-header">{t(lang, "quantum.coverage_by_ej")}</div>
      <div className="qm-ej-bars">
        <EJQuartileBar label={t(lang, "quantum.q1_low")}  data={ejQuartiles["Q1 (Low EJ)"]} />
        <EJQuartileBar label={t(lang, "quantum.q2")}      data={ejQuartiles["Q2"]} />
        <EJQuartileBar label={t(lang, "quantum.q3")}      data={ejQuartiles["Q3"]} />
        <EJQuartileBar label={t(lang, "quantum.q4_high")} data={ejQuartiles["Q4 (High EJ)"]} />
      </div>

      <div className="section-header">{t(lang, "quantum.recommended_locations")}</div>
      <div className="qm-sensor-list">
        {quantumTracts.slice(0, 25).map((s) => {
          const county = s.county ? s.county.replace(/ County$/i, "") : "";
          return (
            <div
              className="qm-sensor-item"
              key={s.geoid}
              onClick={() => onViewSensor?.(s)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && onViewSensor?.(s)}
              aria-label={`${lang === "es" ? "Ver tracto" : "View tract"} ${s.geoid?.slice(-6)} ${county}`}
            >
              <div className="qm-sensor-rank" data-num>#{s.placement_rank}</div>
              <div className="qm-sensor-info">
                <div className="qm-sensor-geoid" data-num>
                  {t(lang, "census_tract_prefix")} {s.geoid?.slice(-6)}
                </div>
                <div className="qm-sensor-county">{county} {t(lang, "tooltip.county_suffix")}</div>
              </div>
              <div className="qm-sensor-scores">
                <div className="qm-sensor-score-row">
                  <span className="qm-score-label">EJ</span>
                  <div className="qm-score-bar-bg">
                    <div
                      className="qm-score-bar-fill ej"
                      style={{ width: `${Math.min(100, (s.ej_priority || 0) * 100)}%` }}
                    />
                  </div>
                </div>
                <div className="qm-sensor-score-row">
                  <span className="qm-score-label">{lang === "es" ? "Cob" : "Cov"}</span>
                  <div className="qm-score-bar-bg">
                    <div
                      className="qm-score-bar-fill cov"
                      style={{ width: `${Math.min(100, (s.coverage_need || 0) * 100)}%` }}
                    />
                  </div>
                </div>
              </div>
              <div className="qm-sensor-composite" data-num>
                {(s.composite_score * 100).toFixed(0)}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ padding: "0 22px", display: "flex", flexDirection: "column", gap: 12 }}>
        <div className="guide-explainer">
          <h4>{t(lang, "quantum.tab_intro_title")}</h4>
          {t(lang, "quantum.tab_intro_body")}
        </div>

        <div className="guide-explainer">
          <h4>{t(lang, "quantum.what_is")}</h4>
          {t(lang, "quantum.what_is_body")}
        </div>

        <div className="guide-explainer">
          <h4>{t(lang, "quantum.why_ej")}</h4>
          {t(lang, "quantum.why_ej_body")}
        </div>
      </div>
    </div>
  );
}
