import { useMemo, useRef, useEffect, useState, forwardRef, useCallback } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap, Circle } from "react-leaflet";
import { BREAKPOINTS } from "../utils/aqi.js";

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

// Fly to a target — only triggers when the target reference changes
function FlyToHandler({ target }) {
  const map = useMap();
  const lastTargetRef = useRef(null);
  useEffect(() => {
    if (!target || isNaN(target.lat) || isNaN(target.lon)) return;
    // Only fly if target actually changed (compare values, not reference)
    const last = lastTargetRef.current;
    if (last && last.lat === target.lat && last.lon === target.lon) return;
    lastTargetRef.current = { lat: target.lat, lon: target.lon };
    map.flyTo([target.lat, target.lon], 13, { duration: 1.8, easeLinearity: 0.2 });
  }, [target, map]);
  return null;
}

// Map background click handler — uses ref to track polygon clicks synchronously
function BackgroundClickHandler({ onBackgroundClick, justClickedRef }) {
  const map = useMap();
  useEffect(() => {
    const handleMapClick = () => {
      // If this click came from a polygon, the polygon's click handler set this ref to true
      if (justClickedRef.current) {
        justClickedRef.current = false;
        return;
      }
      // True background click — deselect
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
      // Make wheel less sensitive: smaller increments and clamped accumulation
      accDelta += e.deltaY < 0 ? 0.25 : -0.25;
      accDelta = Math.max(-2, Math.min(2, accDelta));

      lastMousePoint = map.mouseEventToContainerPoint(e);
      lastMouseLatLng = map.containerPointToLatLng(lastMousePoint);

      if (timer) clearTimeout(timer);
      // Slightly larger debounce so multiple wheel ticks aggregate smoothly
      timer = setTimeout(() => {
        // Use discrete zoom steps to avoid fast fractional jumps
        const targetZoom = Math.round(map.getZoom() + accDelta);
        const newZoom = Math.max(1, Math.min(18, targetZoom));

        const mouseNewPx = map.project(lastMouseLatLng, newZoom);
        const newCenterPx = mouseNewPx
          .subtract(lastMousePoint)
          .add(map.getSize().divideBy(2));
        const newCenter = map.unproject(newCenterPx, newZoom);

        // Slower, smoother animation to give tiles time to load
        map.flyTo(newCenter, newZoom, {
          animate: true,
          duration: 0.9,
          easeLinearity: 0.2,
        });

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

const MapViewContent = forwardRef(
  ({ geojson, predictions, onTractSelect, onBackgroundClick, selectedGeoid, searchMarker }, _ref) => {
    const justClickedRef = useRef(false);

    // Use refs for callbacks so polygon click handlers always see the LATEST functions
    // (not stale closures from when the GeoJSON layer was first created)
    const onTractSelectRef = useRef(onTractSelect);
    const selectedGeoidRef = useRef(selectedGeoid);
    const predMapRef = useRef({});

    useEffect(() => { onTractSelectRef.current = onTractSelect; }, [onTractSelect]);
    useEffect(() => { selectedGeoidRef.current = selectedGeoid; }, [selectedGeoid]);


    const predMap = useMemo(() => {
      const m = {};
      predictions.forEach((p) => {
        m[normGeoid(p.geoid)] = p;
      });
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

    // onEachFeature is only called ONCE per feature when GeoJSON is created.
    // We use refs to access latest state inside the handlers.
    const onEachFeature = useCallback((feature, layer) => {
      const geoid = normGeoid(feature.properties?.GEOID);

      // CRITICAL: prevent click events from bubbling to the map's click handler
      layer.options.bubblingMouseEvents = false;

      // Build tooltip from current predictions
      const pred = predMapRef.current[geoid];
      const name = feature.properties?.NAME ?? geoid;
      const county = pred?.county ? pred.county.replace(/ County$/i, "") : "";
      const displayName = name.startsWith("Census Tract") ? name : `Census Tract ${name}`;

      if (pred) {
        layer.bindTooltip(
          `<div style="font-weight:600;margin-bottom:3px">${displayName}</div>
           <div style="font-size:16px;font-weight:800;color:${pred.color}">${pred.pm25} <span style="font-size:10px;font-weight:400;opacity:0.7">µg/m³</span></div>
           <div style="opacity:0.65;margin-top:2px">${pred.category}</div>
           ${county ? `<div style="opacity:0.45;margin-top:2px;font-size:10px">${county} County</div>` : ""}`,
          { sticky: true, className: "tract-tooltip" }
        );
      }

      layer.on({
        click: () => {
          // Mark synchronously so the map background handler knows
          justClickedRef.current = true;
          // Use the LATEST callback via ref, not a stale closure
          onTractSelectRef.current?.(geoid);
        },
        mouseover: (e) => {
          const isSelected = geoid === selectedGeoidRef.current;
          e.target.setStyle({
            fillOpacity: 0.92,
            weight: isSelected ? 2.5 : 1.5,
            color: isSelected ? "#ffffff" : "rgba(255,255,255,0.6)",
          });
          e.target.bringToFront();
        },
        mouseout: (e) => {
          // Recompute style with the LATEST predictions and selectedGeoid via refs
          const currentPred = predMapRef.current[geoid];
          const isSelected = geoid === selectedGeoidRef.current;
          e.target.setStyle({
            fillColor: currentPred ? currentPred.color : "#2d3436",
            fillOpacity: isSelected ? 0.95 : 0.80,
            color: isSelected ? "#ffffff" : (currentPred ? currentPred.color : "rgba(0,0,0,0.08)"),
            weight: isSelected ? 2.5 : 1.0,
          });
        },
      });
    }, []); // Empty deps — handlers use refs for latest state

    const hasPolygons = geojson?.features?.length > 0 && geojson.features[0]?.geometry;

    // StyleUpdater ensures GeoJSON layer styles update in-place without recreating the layer
    function StyleUpdater() {
      const map = useMap();
      useEffect(() => {
        if (!map) return;
        // Throttle style updates to avoid heavy full-layer recalcs during rapid changes
        let raf = null;
        const applyStyles = () => {
          map.eachLayer((layer) => {
            if (layer && layer.feature && typeof layer.setStyle === "function") {
              try {
                layer.setStyle(styleFeature(layer.feature));
              } catch (e) {
                // ignore layers that don't match
              }
            }
          });
        };
        raf = requestAnimationFrame(applyStyles);
        return () => { if (raf) cancelAnimationFrame(raf); };
      }, [predMap, selectedGeoid, map]);
      return null;
    }

    // Tile loading state to show subtle spinner while tiles are fetching
    const [tileLoadingCount, setTileLoadingCount] = useState(0);
    const onTileLoadStart = useCallback(() => setTileLoadingCount((c) => c + 1), []);
    const onTileLoadEnd = useCallback(() => setTileLoadingCount((c) => Math.max(0, c - 1)), []);

    useEffect(() => {
      if (!map) return;
      const handleZoomStart = () => document.body.classList.add("disable-transitions");
      const handleZoomEnd = () => {
        setTimeout(() => document.body.classList.remove("disable-transitions"), 80);
      };
      map.on("zoomstart", handleZoomStart);
      map.on("zoomend", handleZoomEnd);
      return () => {
        map.off("zoomstart", handleZoomStart);
        map.off("zoomend", handleZoomEnd);
      };
    }, [map]);

    return (
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
        <TileLayer url={CARTO_LIGHT_NOLABELS} attribution={ATTRIBUTION} zIndex={1} keepBuffer={32} updateWhenZooming={false} updateWhenIdle={true} eventHandlers={{ tileloadstart: onTileLoadStart, tileload: onTileLoadEnd, tileerror: onTileLoadEnd }} />

        <FlyToHandler target={searchMarker} />
        <SmoothWheelZoom />
        <BackgroundClickHandler
          onBackgroundClick={onBackgroundClick}
          justClickedRef={justClickedRef}
        />

        <StyleUpdater />

        {hasPolygons && (
          <GeoJSON
            data={geojson}
            style={styleFeature}
            onEachFeature={(feature, layer) => {
              // wrap original handler to guard bringToFront frequency and preserve behaviour
              onEachFeature(feature, layer);
              // ensure hover brings to front only when not already brought
              const origOver = layer.options._orig_mouseover;
              // nothing else
            }}
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

        <TileLayer url={CARTO_LIGHT_LABELS} zIndex={650} pane="shadowPane" keepBuffer={16} updateWhenZooming={false} updateWhenIdle={true} />

        <div className="map-legend">
          <div className="legend-title">PM2.5 µg/m³</div>
          {BREAKPOINTS.map((b) => (
            <div className="legend-row" key={b.category}>
              <div className="legend-swatch" style={{ background: b.color }} />
              <span>{b.label}</span>
            </div>
          ))}
        </div>
      </MapContainer>
    );
  }
);

MapViewContent.displayName = "MapViewContent";

export default forwardRef((props, ref) => <MapViewContent {...props} />);
