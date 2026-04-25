import { useContext } from "react";
import { LanguageContext } from "../App";
import { t } from "../i18n";

/**
 * Shared sidebar header used by both the Map tab (SidePanel) and the Sensors
 * tab (SensorPlacement). Renders the brand row, the visit counter, the tagline
 * (which morphs into the selected-tract identifier), and the EN | ES language
 * toggle. When a tract is selected, a back button appears to deselect.
 */
export default function SidebarHeader({ selectedTract, visitCount, onDeselect, statewide = true }) {
  const { lang, setLang } = useContext(LanguageContext);
  const displayName = statewide
    ? (lang === "es" ? "Todo Texas" : "All of Texas")
    : (lang === "es" ? "Resumen de la región" : "Region Overview");
  const countyName = selectedTract?.county
    ? selectedTract.county.replace(/ County$/i, "")
    : "";

  function toggleLang(newLang) {
    setLang(newLang);
    try { localStorage.setItem("ssi_lang", newLang); } catch (e) {}
  }

  return (
    <div className="sidebar-header">
      {selectedTract && onDeselect && (
        <button className="back-btn" onClick={onDeselect}>← {t(lang, "back_to", displayName)}</button>
      )}
      <div className="brand">
        <div className="brand-left">
          <div className="brand-icon">✦</div>
          <span className="brand-name">Shared Skies Initiative</span>
        </div>
        {visitCount != null && (
          <div className="visit-counter-header">
            <span className="visit-counter-dot" />
            <span><strong>{visitCount.toLocaleString()}</strong> {lang === "es" ? "usuarios" : "users"}</span>
          </div>
        )}
      </div>
      <div className="brand-tagline">
        {selectedTract
          ? `Tract ${selectedTract.geoid?.slice(-6)}${countyName ? ` · ${countyName} County` : ""}`
          : t(lang, "brand_tagline_all")}
      </div>

      <div className="lang-toggle-wrap">
        <div className="lang-toggle" role="tablist" aria-label="Language selector">
          <button
            className={`lang-btn ${lang === "en" ? "active" : ""}`}
            onClick={() => toggleLang("en")}
          >EN</button>
          <span className="sep">|</span>
          <button
            className={`lang-btn ${lang === "es" ? "active" : ""}`}
            onClick={() => toggleLang("es")}
          >ES</button>
        </div>
      </div>
    </div>
  );
}
