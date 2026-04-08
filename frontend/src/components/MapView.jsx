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

function FlyToHandler({ target }) {
  const map = useMap();
  useEffect(() => {
    if (target && !isNaN(target.lat) && !isNaN(target.lon)) {
      map.flyTo([target.lat, target.lon], 13, { duration: 1.8, easeLinearity: 0.2 });
    }
  }, [target]);
  return null;
}

function BackgroundClickHandler({ onBackgroundClick, lastClickedFeature, onResetFeatureClick }) {
  const map = useMap();
  useEffect(() => {
    const handleMapClick = () => {
      if (!lastClickedFeature) {
        onBackgroundClick?.();
      }
      onResetFeatureClick?.();
    };

    map.on("click", handleMapClick);
    return () => map.off("click", handleMapClick);
  }, [map, onBackgroundClick, lastClickedFeature, onResetFeatureClick]);
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
      accDelta += e.deltaY < 0 ? 0.5 : -0.5;
      lastMousePoint = map.mouseEventToContainerPoint(e);
      lastMouseLatLng = map.containerPointToLatLng(lastMousePoint);

      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        const newZoom = Math.max(1, Math.min(18, map.getZoom() + accDelta));
        const mouseNewPx = map.project(lastMouseLatLng, newZoom);
        const newCenterPx = mouseNewPx
          .subtract(lastMousePoint)
          .add(map.getSize().divideBy(2));
        const newCenter = map.unproject(newCenterPx, newZoom);

        map.flyTo(newCenter, newZoom, {
          animate: true,
          duration: 0.6,
          easeLinearity: 0.15,
        });

        accDelta = 0;
        timer = null;
      }, 40);
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
  ({ geojson, predictions, onTractSelect, onBackgroundClick, selectedGeoid, searchMarker, statewide }, _ref) => {
    const [mapKey, setMapKey] = useState(0);
    const [lastClickedFeature, setLastClickedFeature] = useState(false);
    const prevPredLen = useRef(0);
    const mapRef = useRef(null);

    useEffect(() => {
      if (predictions.length > 0 && prevPredLen.current === 0) {
        setMapKey((k) => k + 1);
      }
      prevPredLen.current = predictions.length;
    }, [predictions.length]);

    const predMap = useMemo(() => {
      const m = {};
      predictions.forEach((p) => {
        m[normGeoid(p.geoid)] = p;
      });
      return m;
    }, [predictions]);

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

    const onEachFeature = useCallback((feature, layer) => {
      const geoid = normGeoid(feature.properties?.GEOID);
      const pred = predMap[geoid];
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
        click: (e) => {
          setLastClickedFeature(true);
          onTractSelect(geoid);
          e.stopPropagation();
        },
        mouseover: (e) => {
          e.target.setStyle({
            fillOpacity: 0.9,
            weight: geoid === selectedGeoid ? 2.5 : 1.5,
            color: geoid === selectedGeoid ? "#ffffff" : "rgba(255,255,255,0.6)",
          });
          e.target.bringToFront();
        },
        mouseout: (e) => {
          e.target.setStyle(styleFeature(feature));
        },
      });
    }, [predMap, selectedGeoid, onTractSelect]);

    const hasPolygons = geojson?.features?.length > 0 && geojson.features[0]?.geometry;

    return (
      <MapContainer
        center={TEXAS_CENTER}
        zoom={TEXAS_ZOOM}
        style={{ height: "100%", width: "100%" }}
        zoomControl={true}
        zoomSnap={0}
        preferCanvas={true}
        maxBounds={[[25.5, -106.6], [36.5, -93.5]]}
        maxZoom={13}
        minZoom={4}
      >
        <TileLayer url={CARTO_LIGHT_NOLABELS} attribution={ATTRIBUTION} zIndex={1} keepBuffer={8} />

        <FlyToHandler target={searchMarker} />
        <SmoothWheelZoom />
        <BackgroundClickHandler 
          onBackgroundClick={onBackgroundClick}
          lastClickedFeature={lastClickedFeature}
          onResetFeatureClick={() => setLastClickedFeature(false)}
        />

        {hasPolygons && (
          <GeoJSON
            key={`geojson-${mapKey}`}
            data={geojson}
            style={styleFeature}
            onEachFeature={onEachFeature}
          />
        )}

        {searchMarker && (
          <Circle
            center={[searchMarker.lat, searchMarker.lon]}
            radius={120}
            pane="markerPane"
            pathOptions={{ color: "#fff", weight: 2, fillColor: "#0077b6", fillOpacity: 1 }}
          />
        )}

        <TileLayer url={CARTO_LIGHT_LABELS} zIndex={650} pane="shadowPane" keepBuffer={8} />

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
