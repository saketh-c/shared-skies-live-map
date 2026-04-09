import { useEffect, useState, useCallback, useRef } from "react";
import MapView from "./components/MapView.jsx";
import SidePanel from "./components/SidePanel.jsx";
import SearchBar from "./components/SearchBar.jsx";
import AirQualityGuide from "./components/AirQualityGuide.jsx";
import { findNearestTract } from "./utils/geo.js";

// API base URL: in production set VITE_API_URL=https://your-backend.onrender.com
// In dev, leave empty so /api/... uses the Vite proxy to localhost:8000
const API_BASE = import.meta.env.VITE_API_URL || "";

const REFRESH_MS = 30 * 60 * 1000; // 30 min

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
  const mapRef = useRef(null);
  const [visitCount, setVisitCount] = useState(null);

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

  // Record a page visit (increments on every page load)
  useEffect(() => {
    const p = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/visit`, { method: "POST" });
        if (res.ok) {
          const data = await res.json();
          setVisitCount(data.visits);
        } else {
          console.warn("Visit endpoint failed", res.status);
        }
      } catch (e) {
        console.warn("Failed to record visit:", e.message);
      }
    };
    p();
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

  const handleSearch = useCallback((coords) => {
    if (!predictions?.tracts) return;
    const nearestTract = findNearestTract(coords.lat, coords.lon, predictions.tracts);
    if (nearestTract) {
      handleTractSelect(nearestTract.geoid);
      setSearchMarker({ lat: coords.lat, lon: coords.lon });
    }
  }, [predictions, handleTractSelect]);

  return (
    <div className="app-layout">
      <div className="sidebar">
        {/* Tab bar */}
        <div className="sidebar-tabs">
          <button
            className={`sidebar-tab${activeTab === "map" ? " active" : ""}`}
            onClick={() => setActiveTab("map")}
          >
            Map
          </button>
          <button
            className={`sidebar-tab${activeTab === "guide" ? " active" : ""}`}
            onClick={() => setActiveTab("guide")}
          >
            Air Quality Guide
          </button>
        </div>

        {activeTab === "map" ? (
          <>
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
          </>
        ) : (
          <AirQualityGuide />
        )}
      </div>

      <div className="map-wrapper">
        <div className="map-search-overlay">
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
        />
      </div>
    </div>
  );
}
