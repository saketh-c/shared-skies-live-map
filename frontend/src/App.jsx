import { useEffect, useState, useCallback, useRef, useMemo, createContext, lazy, Suspense } from "react";
import MapView from "./components/MapView.jsx";
import SidePanel from "./components/SidePanel.jsx";
import SearchBar from "./components/SearchBar.jsx";
import { findNearestTract } from "./utils/geo.js";

// Code-split the heavy / less-used tabs so the initial bundle stays small.
const SensorPlacement = lazy(() => import("./components/SensorPlacement.jsx"));
const AirQualityGuide = lazy(() => import("./components/AirQualityGuide.jsx"));

// API base URL: in production set VITE_API_URL=https://your-backend.onrender.com
// In dev, leave empty so /api/... uses the Vite proxy to localhost:8000
const API_BASE = import.meta.env.VITE_API_URL || "";
const REFRESH_MS = 30 * 60 * 1000; // 30 min

export const LanguageContext = createContext({ lang: "en", setLang: () => {} });

function TabFallback() {
  return (
    <div className="loading-wrap" aria-busy="true" aria-live="polite">
      <div className="sk-row">
        <div className="skeleton sk-h" style={{ width: "60%" }} />
        <div className="skeleton sk-h" style={{ width: "40%", height: 10 }} />
      </div>
      <div className="skeleton sk-card" style={{ marginTop: 14 }} />
      <div className="skeleton sk-card" />
      <div className="skeleton sk-card" />
    </div>
  );
}

export default function App() {
  const [predictions, setPredictions] = useState(null);
  const [geojson, setGeojson] = useState(null);
  const [selectedTract, setSelectedTract] = useState(null);
  const [localWeather, setLocalWeather] = useState(null);
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [activeTab, setActiveTab] = useState("map");
  const [searchMarker, setSearchMarker] = useState(null);
  const [searchExpanded, setSearchExpanded] = useState(false);
  const mapRef = useRef(null);
  const [visitCount, setVisitCount] = useState(null);

  // Quantum sensor placement state
  const [quantumData, setQuantumData] = useState(null);
  const [quantumLoading, setQuantumLoading] = useState(false);
  const [quantumError, setQuantumError] = useState(null);
  const [quantumSensors, setQuantumSensors] = useState(null);
  const [existingSensors, setExistingSensors] = useState(null);

  // Language (persisted)
  const [lang, setLang] = useState(() => {
    try { return localStorage.getItem("ssi_lang") || "en"; } catch (e) { return "en"; }
  });
  const langCtx = useMemo(() => ({ lang, setLang }), [lang]);

  const fetchPredictions = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/api/texas/predictions`);
      if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
      const data = await res.json();
      setPredictions(data);
      setLastUpdated(new Date());
      setError(null);
    } catch (e) {
      console.error("Failed to fetch predictions:", e);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchGeojson = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/texas/tracts/geojson`);
      if (!res.ok) throw new Error(`GeoJSON error ${res.status}`);
      const data = await res.json();
      if (data.features?.length > 0 && data.features[0].geometry) {
        setGeojson(data);
      }
    } catch (e) {
      console.warn("GeoJSON not available, using markers:", e.message);
    }
  }, []);

  useEffect(() => {
    fetchPredictions();
    fetchGeojson();
    const timer = setInterval(fetchPredictions, REFRESH_MS);
    return () => clearInterval(timer);
  }, [fetchPredictions, fetchGeojson]);

  // Record a page visit
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/visit`, { method: "POST" });
        if (!cancelled && res.ok) {
          const data = await res.json();
          setVisitCount(data.visits);
        }
      } catch (e) {
        console.warn("Failed to record visit:", e.message);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const handleTractSelect = useCallback(async (geoid) => {
    if (!predictions) return;
    const tract = predictions.tracts?.find((t) => t.geoid === geoid);
    if (!tract) return;

    setSelectedTract({ ...tract });
    setLocalWeather(null);
    setWeatherLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/tract/${geoid}`);
      if (res.ok) {
        const data = await res.json();
        setLocalWeather(data.weather);
      }
    } catch (e) {
      console.error("Failed to fetch local weather:", e);
    } finally {
      setWeatherLoading(false);
    }
  }, [predictions]);

  const handleDeselect = useCallback(() => {
    setSelectedTract(null);
    setLocalWeather(null);
  }, []);

  const fetchQuantumData = useCallback(async () => {
    try {
      setQuantumLoading(true);
      setQuantumError(null);
      const res = await fetch(`${API_BASE}/api/quantum/sensor-placement`);
      if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
      const data = await res.json();
      setQuantumData(data);
      if (data?.methods?.quantum_annealing?.selected_tracts) {
        setQuantumSensors(data.methods.quantum_annealing.selected_tracts);
      }
      if (data?.existing_sensors) setExistingSensors(data.existing_sensors);
    } catch (e) {
      console.error("Failed to fetch quantum data:", e);
      setQuantumError(e.message);
    } finally {
      setQuantumLoading(false);
    }
  }, []);

  const handleTabChange = useCallback((tab) => {
    setActiveTab(tab);
    if (tab === "quantum" && !quantumData && !quantumLoading) fetchQuantumData();
  }, [quantumData, quantumLoading, fetchQuantumData]);

  const handleViewSensor = useCallback((sensor) => {
    if (sensor?.lat && sensor?.lon) {
      setSearchMarker({ lat: sensor.lat, lon: sensor.lon });
      handleTractSelect(sensor.geoid);
    }
  }, [handleTractSelect]);

  const handleSearch = useCallback((coords) => {
    if (!predictions?.tracts) return;
    const nearestTract = findNearestTract(coords.lat, coords.lon, predictions.tracts);
    if (nearestTract) {
      handleTractSelect(nearestTract.geoid);
      setSearchMarker({ lat: coords.lat, lon: coords.lon });
      setSearchExpanded(false);
    }
  }, [predictions, handleTractSelect]);

  const sensorMarkersForMap = activeTab === "quantum" ? quantumSensors : null;
  const existingSensorsForMap = activeTab === "quantum" ? existingSensors : null;

  return (
    <LanguageContext.Provider value={langCtx}>
      <div className="app-layout">
        <div className="sidebar">
          <div className="sidebar-tabs" role="tablist" aria-label={lang === "es" ? "Pestañas" : "Tabs"}>
            <button
              role="tab"
              aria-selected={activeTab === "map"}
              className={`sidebar-tab${activeTab === "map" ? " active" : ""}`}
              onClick={() => handleTabChange("map")}
            >
              {lang === "es" ? "Mapa" : "Map"}
            </button>
            <button
              role="tab"
              aria-selected={activeTab === "quantum"}
              className={`sidebar-tab${activeTab === "quantum" ? " active" : ""}`}
              onClick={() => handleTabChange("quantum")}
            >
              {lang === "es" ? "Sensores" : "Sensors"}
            </button>
            <button
              role="tab"
              aria-selected={activeTab === "guide"}
              className={`sidebar-tab${activeTab === "guide" ? " active" : ""}`}
              onClick={() => handleTabChange("guide")}
            >
              {lang === "es" ? "Guía" : "Guide"}
            </button>
          </div>

          {activeTab === "map" && (
            <SidePanel
              predictions={predictions}
              selectedTract={selectedTract}
              localWeather={localWeather}
              onDeselect={handleDeselect}
              loading={loading}
              weatherLoading={weatherLoading}
              error={error}
              lastUpdated={lastUpdated}
              statewide={true}
              visitCount={visitCount}
            />
          )}

          {activeTab === "quantum" && (
            <Suspense fallback={<TabFallback />}>
              <SensorPlacement
                quantumData={quantumData}
                loading={quantumLoading}
                error={quantumError}
                onViewSensor={handleViewSensor}
                selectedTract={selectedTract}
                onDeselect={handleDeselect}
                visitCount={visitCount}
              />
            </Suspense>
          )}

          {activeTab === "guide" && (
            <Suspense fallback={<TabFallback />}>
              <AirQualityGuide />
            </Suspense>
          )}
        </div>

        <div className="map-wrapper">
          <button
            type="button"
            className="mobile-search-toggle"
            aria-label={searchExpanded
              ? (lang === "es" ? "Cerrar búsqueda" : "Close search")
              : (lang === "es" ? "Abrir búsqueda" : "Open search")}
            aria-expanded={searchExpanded}
            onClick={() => setSearchExpanded((v) => !v)}
          >
            {searchExpanded ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="11" cy="11" r="7" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
            )}
          </button>
          <div className={`map-search-overlay${searchExpanded ? " expanded" : ""}`}>
            <SearchBar onSearch={handleSearch} loading={loading} />
          </div>
          <MapView
            ref={mapRef}
            geojson={geojson}
            predictions={predictions?.tracts ?? []}
            onTractSelect={handleTractSelect}
            onBackgroundClick={handleDeselect}
            selectedGeoid={selectedTract?.geoid ?? null}
            searchMarker={searchMarker}
            statewide={true}
            sensorMarkers={sensorMarkersForMap}
            existingSensors={existingSensorsForMap}
            activeTab={activeTab}
          />
        </div>
      </div>
    </LanguageContext.Provider>
  );
}
