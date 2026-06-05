import { useMemo, useRef, useEffect, useState, forwardRef, useCallback, useContext } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap, Circle, CircleMarker, Tooltip, Pane } from "react-leaflet";
import L from "leaflet";
import { BREAKPOINTS } from "../utils/aqi.js";
import { LanguageContext } from '../App';
import { t, translateCategory } from '../i18n';

const CARTO_LIGHT_NOLABELS =
  "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png";
const CARTO_LIGHT_LABELS =
  "https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png";
const ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>';

const TEXAS_CENTER = [31.5, -99.0];
const TEXAS_ZOOM = 6;

// API base — matches App.jsx convention.
const API_BASE = import.meta.env.VITE_API_URL || "";

// NOAA HMS smoke polygon styling by density tier.
const HMS_STYLE = {
  Light:   { fillColor: "#ffeb99", color: "#caa84a", fillOpacity: 0.32, weight: 1 },
  Medium:  { fillColor: "#ff9933", color: "#cc6600", fillOpacity: 0.42, weight: 1.2 },
  Heavy:   { fillColor: "#cc3300", color: "#80190f", fillOpacity: 0.55, weight: 1.4 },
  Unknown: { fillColor: "#888888", color: "#555555", fillOpacity: 0.30, weight: 1 },
};
// Refresh HMS once an hour to match the cache TTL.
const HMS_REFRESH_MS = 60 * 60 * 1000;

function normGeoid(val) {
  if (!val) return "";
  return String(val).padStart(11, "0");
}

function FlyToHandler({ target }) {
  const map = useMap();
  const lastTargetRef = useRef(null);
  useEffect(() => {
    if (!target || isNaN(target.lat) || isNaN(target.lon)) return;
    const last = lastTargetRef.current;
    if (last && last.lat === target.lat && last.lon === target.lon) return;
    lastTargetRef.current = { lat: target.lat, lon: target.lon };
    map.flyTo([target.lat, target.lon], 13, { duration: 1.8, easeLinearity: 0.2 });
  }, [target, map]);
  return null;
}

function BackgroundClickHandler({ onBackgroundClick, justClickedRef }) {
  const map = useMap();
  useEffect(() => {
    const handleMapClick = () => {
      if (justClickedRef.current) {
        justClickedRef.current = false;
        return;
      }
      onBackgroundClick?.();
    };
    map.on("click", handleMapClick);
    return () => map.off("click", handleMapClick);
  }, [map, onBackgroundClick, justClickedRef]);
  return null;
}

function SmoothWheelZoom() {
  const map = useMap();
  useEffect(() => {
    map.scrollWheelZoom.disable();
    let accDelta = 0;
    let lastMousePoint = null;
    let lastMouseLatLng = null;
    let timer = null;

    const onWheel = (e) => {
      e.preventDefault();
      accDelta += e.deltaY < 0 ? 0.3 : -0.3;
      accDelta = Math.max(-2.5, Math.min(2.5, accDelta));
      lastMousePoint = map.mouseEventToContainerPoint(e);
      lastMouseLatLng = map.containerPointToLatLng(lastMousePoint);
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        const targetZoom = Math.round(map.getZoom() + accDelta);
        const newZoom = Math.max(1, Math.min(18, targetZoom));
        const mouseNewPx = map.project(lastMouseLatLng, newZoom);
        const newCenterPx = mouseNewPx
          .subtract(lastMousePoint)
          .add(map.getSize().divideBy(2));
        const newCenter = map.unproject(newCenterPx, newZoom);
        map.flyTo(newCenter, newZoom, { animate: true, duration: 0.45, easeLinearity: 0.25 });
        accDelta = 0;
        timer = null;
      }, 35);
    };

    map.getContainer().addEventListener("wheel", onWheel, { passive: false });
    return () => {
      map.getContainer().removeEventListener("wheel", onWheel);
      if (timer) clearTimeout(timer);
    };
  }, [map]);
  return null;
}

function MapLifecycle() {
  const map = useMap();
  useEffect(() => {
    if (!map) return;
    const onZoomStart = () => document.body.classList.add("disable-transitions");
    const onZoomEnd = () => {
      // Strip the no-transitions guard immediately so labels/polys snap to the new
      // zoom in the same frame the tiles arrive — no lingering "stale text" pause.
      document.body.classList.remove("disable-transitions");
    };
    map.on("zoomstart", onZoomStart);
    map.on("zoomend", onZoomEnd);
    return () => {
      map.off("zoomstart", onZoomStart);
      map.off("zoomend", onZoomEnd);
    };
  }, [map]);
  return null;
}

const MapViewContent = forwardRef(
  ({ geojson, predictions, onTractSelect, onBackgroundClick, selectedGeoid, searchMarker, sensorMarkers, existingSensors, activeTab = "map" }, _ref) => {
    const { lang } = useContext(LanguageContext);
    const justClickedRef = useRef(false);
    const onTractSelectRef = useRef(onTractSelect);
    const selectedGeoidRef = useRef(selectedGeoid);
    const predMapRef = useRef({});
    const tooltipDomRef = useRef(null);
    const hoveredLayerRef = useRef(null);
    const hoveredGeoidRef = useRef(null);
    const lastMouseLatLngRef = useRef(null);
    // Polygon hover/preview/click is only enabled on the Map tab. We use a ref
    // so the long-lived layer.on() handlers always read the *current* tab
    // (their useCallback has empty deps to keep handler identity stable).
    const activeTabRef = useRef(activeTab);
    const [tooltipData, setTooltipData] = useState(null);
    const [legendExpanded, setLegendExpanded] = useState(false);

    // NOAA HMS smoke polygon layer state. Hidden by default; toggle via legend.
    const [hmsData, setHmsData] = useState(null);
    const [showHmsSmoke, setShowHmsSmoke] = useState(false);

    // Fetch the HMS smoke layer once on mount and refresh every hour. Survives
    // when toggled off because the cached feature collection is cheap to keep.
    useEffect(() => {
      let alive = true;
      const fetchHms = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/live/hms-smoke`);
          if (!res.ok) return;
          const data = await res.json();
          if (alive) setHmsData(data);
        } catch (e) {
          // Network failures are non-fatal — the layer just won't render.
          console.warn("HMS fetch failed:", e?.message);
        }
      };
      fetchHms();
      const id = setInterval(fetchHms, HMS_REFRESH_MS);
      return () => { alive = false; clearInterval(id); };
    }, []);

    useEffect(() => { onTractSelectRef.current = onTractSelect; }, [onTractSelect]);
    useEffect(() => { selectedGeoidRef.current = selectedGeoid; }, [selectedGeoid]);
    useEffect(() => { activeTabRef.current = activeTab; }, [activeTab]);

    // When the user leaves the Map tab, kill any in-flight hover state so it
    // doesn't visually persist (mouseout won't fire because the cursor isn't
    // moving off the polygon — the tab is just being switched).
    useEffect(() => {
      if (activeTab === "map") return;
      setTooltipData(null);
      const layer = hoveredLayerRef.current;
      const geoid = hoveredGeoidRef.current;
      if (layer && geoid) {
        const isSelected = geoid === selectedGeoidRef.current;
        const pred = predMapRef.current[geoid];
        try {
          layer.setStyle({
            fillColor: pred ? pred.color : "#2d3436",
            fillOpacity: isSelected ? 0.95 : 0.80,
            color: isSelected ? "#ffffff" : (pred ? pred.color : "rgba(0,0,0,0.08)"),
            weight: isSelected ? 2.5 : 1.0,
          });
        } catch (_) {}
      }
      hoveredLayerRef.current = null;
      hoveredGeoidRef.current = null;
      lastMouseLatLngRef.current = null;
    }, [activeTab]);

    // Dedicated SVG renderer for the quantum-sensor markers so they live in
    // their own high-z-index pane and never get hidden by polygon
    // bringToFront(). SVG (not canvas) is critical here — a canvas in a
    // top-of-stack pane would swallow clicks across the whole map area.
    const sensorSvgRenderer = useMemo(
      () => L.svg({ pane: "quantumSensorsPane" }),
      []
    );

    const predMap = useMemo(() => {
      const m = {};
      predictions.forEach((p) => { m[normGeoid(p.geoid)] = p; });
      return m;
    }, [predictions]);

    useEffect(() => { predMapRef.current = predMap; }, [predMap]);

    const styleFeature = useCallback((feature) => {
      const geoid = normGeoid(feature.properties?.GEOID);
      const pred = predMap[geoid];
      const isSelected = geoid === selectedGeoid;
      return {
        fillColor: pred ? pred.color : "#2d3436",
        fillOpacity: isSelected ? 0.95 : 0.80,
        color: isSelected ? "#ffffff" : (pred ? pred.color : "rgba(0,0,0,0.08)"),
        weight: isSelected ? 2.5 : 1.0,
        lineCap: "round",
        lineJoin: "round",
      };
    }, [predMap, selectedGeoid]);

    // onEachFeature runs once per feature — uses refs to always access latest state.
    // Tooltip is a custom React overlay (not Leaflet bindTooltip) so it never glitches.
    const onEachFeature = useCallback((feature, layer) => {
      const geoid = normGeoid(feature.properties?.GEOID);
      layer.options.bubblingMouseEvents = false;

      const rawName = feature.properties?.NAME ?? geoid;
      const displayName = rawName.startsWith(t(lang, 'census_tract_prefix')) ? rawName : `${t(lang, 'census_tract_prefix')} ${rawName}`;

      layer.on({
        click: () => {
          // Polygon clicks are only meaningful on the Map tab. On the Sensors
          // tab the polygons are a passive backdrop — only blue dots are
          // interactive. Skipping this also prevents handleDeselect from
          // running (justClickedRef stays false but we don't want any
          // selection-mutating side effects either).
          if (activeTabRef.current !== "map") return;
          justClickedRef.current = true;
          onTractSelectRef.current?.(geoid);
        },

        mouseover: (e) => {
          if (activeTabRef.current !== "map") return;
          hoveredLayerRef.current = e.target;
          hoveredGeoidRef.current = geoid;
          lastMouseLatLngRef.current = e.latlng;
          const pred = predMapRef.current[geoid];
          const isSelected = geoid === selectedGeoidRef.current;
          e.target.setStyle({
            fillOpacity: 0.92,
            weight: isSelected ? 2.5 : 1.8,
            color: isSelected ? "#ffffff" : "rgba(255,255,255,0.8)",
          });
          e.target.bringToFront();

          if (!pred) return;
          const county = pred.county ? pred.county.replace(/ County$/i, "") : "";
          const pt = e.containerPoint;
          setTooltipData({ name: displayName, pm25: pred.pm25, color: pred.color, category: pred.category, county, x: pt.x, y: pt.y });
        },

        // Update tooltip position directly on DOM — no React re-render on every mouse move
        mousemove: (e) => {
          if (activeTabRef.current !== "map") return;
          lastMouseLatLngRef.current = e.latlng;
          const el = tooltipDomRef.current;
          if (!el) return;
          const pt = e.containerPoint;
          const mapSize = e.target._map?.getSize();
          const ttW = el.offsetWidth || 180;
          const ttH = el.offsetHeight || 110;
          const flipX = mapSize && pt.x + ttW + 20 > mapSize.x;
          const flipY = mapSize && pt.y + ttH + 20 > mapSize.y;
          el.style.left = `${pt.x + (flipX ? -(ttW + 10) : 14)}px`;
          el.style.top  = `${pt.y + (flipY ? -(ttH + 10) : -10)}px`;
        },

        mouseout: (e) => {
          // mouseout always runs to keep style state clean (it's a no-op if
          // mouseover was suppressed — the polygon is already in default style).
          hoveredLayerRef.current = null;
          hoveredGeoidRef.current = null;
          lastMouseLatLngRef.current = null;
          const currentPred = predMapRef.current[geoid];
          const isSelected = geoid === selectedGeoidRef.current;
          e.target.setStyle({
            fillColor: currentPred ? currentPred.color : "#2d3436",
            fillOpacity: isSelected ? 0.95 : 0.80,
            color: isSelected ? "#ffffff" : (currentPred ? currentPred.color : "rgba(0,0,0,0.08)"),
            weight: isSelected ? 2.5 : 1.0,
          });
          setTooltipData(null);
        },
      });
    }, []); // empty deps — all state accessed via refs

    // Only render GeoJSON once BOTH geojson and predictions are ready,
    // so onEachFeature always has a populated predMapRef.
    const hasPolygons =
      geojson?.features?.length > 0 &&
      geojson.features[0]?.geometry &&
      predictions.length > 0;

    function StyleUpdater() {
      const map = useMap();
      useEffect(() => {
        if (!map) return;
        let raf = null;
        const applyStyles = () => {
          map.eachLayer((layer) => {
            if (layer?.feature && typeof layer.setStyle === "function") {
              try { layer.setStyle(styleFeature(layer.feature)); } catch (_) {}
            }
          });
          // Bring the SELECTED polygon to the front so its full white border
          // renders cleanly above neighbour polygons. Without this, neighbours
          // that happen to be drawn later in the canvas can clip the border
          // and leave the visible "gap" the user sees when a sensor is
          // clicked on the Sensors tab (where mouseover.bringToFront() is
          // gated off).
          const sg = selectedGeoidRef.current;
          if (sg) {
            map.eachLayer((layer) => {
              if (
                layer?.feature &&
                normGeoid(layer.feature.properties?.GEOID) === sg &&
                typeof layer.bringToFront === "function"
              ) {
                try { layer.bringToFront(); } catch (_) {}
              }
            });
          }
          // After resetting all layers, re-apply hover style so it is never wiped.
          // (StyleUpdater re-mounts on every render because it's defined inside the
          // render function — its useEffect always fires, which would otherwise clear
          // any style set by mouseover.)
          const layer = hoveredLayerRef.current;
          const hGeoid = hoveredGeoidRef.current;
          if (layer && hGeoid) {
            const isSelected = hGeoid === selectedGeoidRef.current;
            try {
              layer.setStyle({
                fillOpacity: 0.92,
                weight: isSelected ? 2.5 : 1.8,
                color: isSelected ? "#ffffff" : "rgba(255,255,255,0.8)",
              });
              layer.bringToFront();
            } catch (_) {}
          }
        };
        raf = requestAnimationFrame(applyStyles);
        return () => { if (raf) cancelAnimationFrame(raf); };
      }, [predMap, selectedGeoid, map]);
      return null;
    }

    // Fires on every map move frame: re-projects last hover lat/lng → new container px
    // so the tooltip follows the map during drag, and re-applies hover outline style
    // so canvas redraws during pan don't wipe the highlight.
    function MapMoveHandler() {
      const map = useMap();
      useEffect(() => {
        const onMove = () => {
          // --- tooltip: re-project geographic point to current screen position ---
          const el = tooltipDomRef.current;
          const latLng = lastMouseLatLngRef.current;
          if (el && latLng) {
            const pt = map.latLngToContainerPoint(latLng);
            const mapSize = map.getSize();
            const ttW = el.offsetWidth || 180;
            const ttH = el.offsetHeight || 110;
            const flipX = pt.x + ttW + 20 > mapSize.x;
            const flipY = pt.y + ttH + 20 > mapSize.y;
            el.style.left = `${pt.x + (flipX ? -(ttW + 10) : 14)}px`;
            el.style.top  = `${pt.y + (flipY ? -(ttH + 10) : -10)}px`;
          }
          // --- hover outline: re-apply after canvas redraw ---
          const layer = hoveredLayerRef.current;
          const hGeoid = hoveredGeoidRef.current;
          if (layer && hGeoid) {
            const isSelected = hGeoid === selectedGeoidRef.current;
            try {
              layer.setStyle({
                fillOpacity: 0.92,
                weight: isSelected ? 2.5 : 1.8,
                color: isSelected ? "#ffffff" : "rgba(255,255,255,0.8)",
              });
              layer.bringToFront();
            } catch (_) {}
          }
        };
        map.on("move", onMove);
        return () => map.off("move", onMove);
      }, [map]);
      return null;
    }

    const onTileLoadStart = useCallback(() => {}, []);
    const onTileLoadEnd = useCallback(() => {}, []);

    return (
      <div style={{ position: "relative", height: "100%", width: "100%" }}>
        <MapContainer
          center={TEXAS_CENTER}
          zoom={TEXAS_ZOOM}
          style={{ height: "100%", width: "100%" }}
          zoomControl={true}
          zoomSnap={0}
          preferCanvas={true}
          touchZoom={true}
          tap={true}
          zoomAnimation={true}
          fadeAnimation={true}
          inertia={true}
          inertiaDeceleration={3000}
          inertiaMaxSpeed={1500}
          maxZoom={13}
          minZoom={4}
        >
          <TileLayer
            url={CARTO_LIGHT_NOLABELS}
            attribution={ATTRIBUTION}
            zIndex={1}
            keepBuffer={64}
            updateWhenZooming={true}
            updateWhenIdle={false}
            updateInterval={40}
            crossOrigin="anonymous"
            noWrap={true}
            maxNativeZoom={19}
            eventHandlers={{ tileloadstart: onTileLoadStart, tileload: onTileLoadEnd, tileerror: onTileLoadEnd }}
          />

          <FlyToHandler target={searchMarker} />
          <SmoothWheelZoom />
          <MapLifecycle />
          <BackgroundClickHandler onBackgroundClick={onBackgroundClick} justClickedRef={justClickedRef} />
          <StyleUpdater />
          <MapMoveHandler />

          {hasPolygons && (
            <GeoJSON
              data={geojson}
              style={styleFeature}
              onEachFeature={onEachFeature}
            />
          )}

          {/* NOAA HMS smoke polygons — own pane between choropleth (z=400) and
              sensor markers (z=600). Pointer-events:none so the smoke overlay
              never steals clicks from the choropleth underneath. */}
          {showHmsSmoke && hmsData?.features?.length > 0 && (
            <Pane name="hmsSmokePane" style={{ zIndex: 450, pointerEvents: "none" }}>
              <GeoJSON
                key={`hms-${hmsData.data_date}-${hmsData.count}`}
                data={hmsData}
                style={(feat) => {
                  const d = feat?.properties?.density || "Unknown";
                  return HMS_STYLE[d] || HMS_STYLE.Unknown;
                }}
              />
            </Pane>
          )}

          {searchMarker && (
            <Circle
              center={[searchMarker.lat, searchMarker.lon]}
              radius={120}
              interactive={false}
              bubblingMouseEvents={false}
              className="search-marker-circle"
              pathOptions={{ color: "#fff", weight: 2, fillColor: "#0077b6", fillOpacity: 1 }}
            />
          )}

          {/* Existing PurpleAir sensors (grey dots) */}
          {existingSensors && existingSensors.map((s) => (
            <CircleMarker
              key={`existing-${s.sensor_id}`}
              center={[s.lat, s.lon]}
              radius={4}
              pathOptions={{
                color: "rgba(255,255,255,0.3)",
                weight: 1,
                fillColor: "rgba(255,255,255,0.25)",
                fillOpacity: 0.7,
              }}
              interactive={false}
              bubblingMouseEvents={false}
            />
          ))}

          {/* Quantum sensor placement markers — rendered as SVG (not canvas)
              in their OWN pane with z-index above overlayPane so polygon
              bringToFront() calls can never push them behind. The pane DIV
              itself has pointer-events:none so empty space passes clicks
              through; only the SVG paths (with their default
              pointer-events:visiblePainted) capture clicks on the actual dots. */}
          <Pane name="quantumSensorsPane" style={{ zIndex: 600, pointerEvents: "none" }}>
            {sensorMarkers && sensorMarkers.map((s) => {
              const isTop3 = (s.placement_rank ?? 99) <= 3;
              return (
                <CircleMarker
                  key={`sensor-${s.geoid}`}
                  center={[s.lat, s.lon]}
                  radius={isTop3 ? 8 : 7}
                  pathOptions={{
                    renderer: sensorSvgRenderer,
                    color: "#ffffff",
                    weight: isTop3 ? 2.5 : 2,
                    fillColor: isTop3 ? "#fbbf24" : "#38bdf8",
                    fillOpacity: 0.96,
                    className: isTop3 ? "ssi-sensor-top" : "ssi-sensor",
                  }}
                  eventHandlers={{
                    click: () => {
                      justClickedRef.current = true;
                      onTractSelectRef.current?.(s.geoid);
                    },
                  }}
                >
                  <Tooltip
                    direction="top"
                    offset={[0, -10]}
                    opacity={0.95}
                    className="sensor-tooltip"
                  >
                    <span style={{ fontWeight: 700 }}>#{s.placement_rank}</span>
                    {" "}{lang === "es" ? "Sensor Recomendado" : "Recommended Sensor"}
                    <br />
                    <span style={{ fontSize: 10, opacity: 0.7 }}>
                      EJ: {((s.ej_priority || 0) * 100).toFixed(0)}%
                      {" | "}
                      {lang === "es" ? "Cob" : "Cov"}: {((s.coverage_need || 0) * 100).toFixed(0)}%
                    </span>
                  </Tooltip>
                </CircleMarker>
              );
            })}
          </Pane>

          <TileLayer
            url={CARTO_LIGHT_LABELS}
            zIndex={650}
            pane="shadowPane"
            keepBuffer={48}
            updateWhenZooming={true}
            updateWhenIdle={false}
            updateInterval={40}
            crossOrigin="anonymous"
            noWrap={true}
            maxNativeZoom={19}
          />

          <button
            type="button"
            className="mobile-legend-toggle"
            aria-label={legendExpanded
              ? (lang === 'es' ? 'Cerrar leyenda' : 'Close legend')
              : (lang === 'es' ? 'Abrir leyenda' : 'Open legend')}
            aria-expanded={legendExpanded}
            onClick={() => setLegendExpanded(v => !v)}
          >
            {legendExpanded ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="9" />
                <line x1="12" y1="11" x2="12" y2="17" />
                <circle cx="12" cy="7.5" r="0.9" fill="currentColor" stroke="none" />
              </svg>
            )}
          </button>
          <div className={`map-legend${legendExpanded ? ' expanded' : ''}`}>
            <div className="legend-title">PM2.5 µg/m³</div>
            {BREAKPOINTS.map((b) => (
              <div className="legend-row" key={b.category}>
                <div className="legend-swatch" style={{ background: b.color }} />
                <span className="legend-category">{translateCategory(lang, b.category)}</span>
                <span className="legend-range">{b.label}</span>
              </div>
            ))}

            {/* NOAA HMS smoke polygon layer toggle + density legend */}
            <div className="legend-divider" />
            <label className="legend-row legend-toggle" style={{ cursor: hmsData?.features?.length ? "pointer" : "not-allowed", opacity: hmsData?.features?.length ? 1 : 0.55 }}>
              <input
                type="checkbox"
                checked={showHmsSmoke}
                disabled={!hmsData?.features?.length}
                onChange={(e) => setShowHmsSmoke(e.target.checked)}
                style={{ marginRight: 8 }}
              />
              <span className="legend-category">
                {lang === "es" ? "Humo (NOAA HMS)" : "Smoke (NOAA HMS)"}
              </span>
              <span className="legend-range">
                {hmsData?.count ? `${hmsData.count} ${hmsData.count === 1 ? "plume" : "plumes"}` : (lang === "es" ? "ninguno" : "none")}
              </span>
            </label>
            {showHmsSmoke && hmsData?.features?.length > 0 && (
              <>
                {["Light", "Medium", "Heavy"].map((d) => (
                  <div className="legend-row legend-sub" key={`hms-${d}`}>
                    <div className="legend-swatch" style={{ background: HMS_STYLE[d].fillColor, opacity: HMS_STYLE[d].fillOpacity + 0.2 }} />
                    <span className="legend-category">{d}</span>
                    <span className="legend-range">{hmsData?.density_counts?.[d] || 0}</span>
                  </div>
                ))}
                <div className="legend-row legend-source">
                  <span className="legend-range" style={{ fontSize: "0.72em", opacity: 0.75 }}>
                    {hmsData?.data_date && `${lang === "es" ? "Fecha" : "Date"}: ${hmsData.data_date}`}
                  </span>
                </div>
              </>
            )}
          </div>
        </MapContainer>

        {/* Custom tooltip — positioned absolutely over the map, never glitches */}
        {tooltipData && (
          <div
            ref={tooltipDomRef}
            className="custom-tract-tooltip"
            style={{ left: tooltipData.x + 14, top: tooltipData.y - 10 }}
          >
            <div className="ctt-name">{tooltipData.name}</div>
            <div className="ctt-pm25" style={{ color: tooltipData.color }}>
              {tooltipData.pm25} <span className="ctt-unit">µg/m³</span>
            </div>
            <div className="ctt-cat">{translateCategory(lang, tooltipData.category)}</div>
            {tooltipData.county && (
              <div className="ctt-county">{tooltipData.county} {t(lang, 'tooltip.county_suffix')}</div>
            )}
          </div>
        )}
      </div>
    );
  }
);

MapViewContent.displayName = "MapViewContent";

export default forwardRef((props, ref) => <MapViewContent {...props} ref={ref} />);
