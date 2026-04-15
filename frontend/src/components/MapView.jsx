import { useMemo, useRef, useEffect, useState, forwardRef, useCallback, useContext } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap, Circle } from "react-leaflet";
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
      accDelta += e.deltaY < 0 ? 0.25 : -0.25;
      accDelta = Math.max(-2, Math.min(2, accDelta));
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
        map.flyTo(newCenter, newZoom, { animate: true, duration: 0.9, easeLinearity: 0.2 });
        accDelta = 0;
        timer = null;
      }, 60);
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
      setTimeout(() => document.body.classList.remove("disable-transitions"), 80);
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
  ({ geojson, predictions, onTractSelect, onBackgroundClick, selectedGeoid, searchMarker }, _ref) => {
    const { lang } = useContext(LanguageContext);
    const justClickedRef = useRef(false);
    const onTractSelectRef = useRef(onTractSelect);
    const selectedGeoidRef = useRef(selectedGeoid);
    const predMapRef = useRef({});
    const tooltipDomRef = useRef(null);
    const hoveredLayerRef = useRef(null);
    const hoveredGeoidRef = useRef(null);
    const lastMouseLatLngRef = useRef(null);
    const [tooltipData, setTooltipData] = useState(null);

    useEffect(() => { onTractSelectRef.current = onTractSelect; }, [onTractSelect]);
    useEffect(() => { selectedGeoidRef.current = selectedGeoid; }, [selectedGeoid]);

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
          justClickedRef.current = true;
          onTractSelectRef.current?.(geoid);
        },

        mouseover: (e) => {
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
            keepBuffer={32}
            updateWhenZooming={false}
            updateWhenIdle={true}
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

          <TileLayer
            url={CARTO_LIGHT_LABELS}
            zIndex={650}
            pane="shadowPane"
            keepBuffer={16}
            updateWhenZooming={false}
            updateWhenIdle={true}
          />

          <div className="map-legend">
            <div className="legend-title">PM2.5 µg/m³</div>
            {BREAKPOINTS.map((b) => (
              <div className="legend-row" key={b.category}>
                <div className="legend-swatch" style={{ background: b.color }} />
                <span>{translateCategory(lang, b.category)}</span>
              </div>
            ))}
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
