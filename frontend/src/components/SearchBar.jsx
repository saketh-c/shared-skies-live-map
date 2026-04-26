import React, { useState, useRef, useEffect, useContext } from "react";
import "./SearchBar.css";
import { LanguageContext } from "../App";
import { t } from "../i18n";

const PinIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
    <circle cx="12" cy="10" r="3" />
  </svg>
);

const CompassIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="10" />
    <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
  </svg>
);

const SearchIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="search-input-icon">
    <circle cx="11" cy="11" r="7" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const AlertIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>
);

export default function SearchBar({ onSearch, loading }) {
  const { lang } = useContext(LanguageContext);
  const [searchInput, setSearchInput] = useState("");
  const [searchType, setSearchType] = useState("address");
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [error, setError] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const searchRef = useRef(null);
  const timeoutRef = useRef(null);

  // Debounced Nominatim suggestions
  useEffect(() => {
    if (searchType !== "address" || !searchInput.trim() || searchInput.length < 3) {
      setSuggestions([]);
      return;
    }
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(async () => {
      try {
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchInput + ", Texas")}&limit=5`
        );
        const results = await response.json();
        setSuggestions(results);
        setShowSuggestions(true);
      } catch (err) {
        console.error("Autocomplete error:", err);
        setSuggestions([]);
      }
    }, 300);
    return () => clearTimeout(timeoutRef.current);
  }, [searchInput, searchType]);

  // Click-outside
  useEffect(() => {
    const handle = (e) => {
      if (searchRef.current && !searchRef.current.contains(e.target)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, []);

  const geocodeAddress = async (address, lat, lon) => {
    try {
      setIsSearching(true);
      setError("");
      if (lat < 25.8 || lat > 36.5 || lon < -106.6 || lon > -93.5) {
        setError(t(lang, "search.errors.address_outside"));
        setIsSearching(false);
        return;
      }
      onSearch({ lat, lon, address });
      setSuggestions([]);
      setShowSuggestions(false);
    } catch (err) {
      setError(t(lang, "search.errors.search_failed"));
      console.error(err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    geocodeAddress(suggestion.display_name, parseFloat(suggestion.lat), parseFloat(suggestion.lon));
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    setError("");

    if (searchType === "address") {
      if (!searchInput.trim()) {
        setError(t(lang, "search.errors.enter_address"));
        return;
      }
      if (suggestions.length > 0) {
        handleSuggestionClick(suggestions[0]);
      } else {
        setIsSearching(true);
        try {
          const response = await fetch(
            `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchInput + ", Texas")}&limit=1`
          );
          const results = await response.json();
          if (results.length > 0) {
            const r = results[0];
            geocodeAddress(r.display_name, parseFloat(r.lat), parseFloat(r.lon));
          } else {
            setError(t(lang, "search.errors.address_not_found"));
          }
        } catch (err) {
          setError(t(lang, "search.errors.search_failed"));
        } finally {
          setIsSearching(false);
        }
      }
    } else {
      const coords = searchInput.trim().split(",");
      if (coords.length !== 2) {
        setError(t(lang, "search.errors.coords_format"));
        return;
      }
      const lat = parseFloat(coords[0].trim());
      const lon = parseFloat(coords[1].trim());
      if (isNaN(lat) || isNaN(lon)) {
        setError(t(lang, "search.errors.coords_invalid"));
        return;
      }
      if (lat < 25.8 || lat > 36.5 || lon < -106.6 || lon > -93.5) {
        setError(t(lang, "search.errors.coords_out_of_bounds"));
        return;
      }
      onSearch({ lat, lon });
      setError("");
    }
  };

  return (
    <div className="search-bar" ref={searchRef}>
      <form onSubmit={handleSearch}>
        <div className="search-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={searchType === "address"}
            className={`search-tab ${searchType === "address" ? "active" : ""}`}
            onClick={() => { setSearchType("address"); setSearchInput(""); setSuggestions([]); setError(""); }}
          >
            <PinIcon />
            <span>{t(lang, "search.address_tab")}</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={searchType === "coordinates"}
            className={`search-tab ${searchType === "coordinates" ? "active" : ""}`}
            onClick={() => { setSearchType("coordinates"); setSearchInput(""); setSuggestions([]); setError(""); }}
          >
            <CompassIcon />
            <span>{t(lang, "search.coordinates_tab")}</span>
          </button>
        </div>

        <div className="search-input-group">
          <div className="search-input-wrapper">
            <SearchIcon />
            <input
              type="text"
              placeholder={searchType === "address"
                ? t(lang, "search.placeholder_address")
                : t(lang, "search.placeholder_coords")}
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              onFocus={() => searchType === "address" && setShowSuggestions(true)}
              required
              aria-label={searchType === "address"
                ? t(lang, "search.placeholder_address")
                : t(lang, "search.placeholder_coords")}
            />

            {searchType === "address" && showSuggestions && suggestions.length > 0 && (
              <div className="suggestions-dropdown" role="listbox">
                {suggestions.map((s, idx) => (
                  <div
                    key={idx}
                    className="suggestion-item"
                    role="option"
                    tabIndex={0}
                    onClick={() => handleSuggestionClick(s)}
                    onKeyDown={(e) => e.key === "Enter" && handleSuggestionClick(s)}
                  >
                    <div className="suggestion-name">{s.name || s.display_name.split(",")[0]}</div>
                    <div className="suggestion-address">{s.display_name}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <button type="submit" disabled={loading || isSearching}>
            {isSearching ? t(lang, "search.searching") : loading ? t(lang, "search.loading") : t(lang, "search.search_button")}
          </button>
        </div>

        {error && (
          <div className="search-error">
            <AlertIcon />
            <span>{error}</span>
          </div>
        )}
      </form>

      <div className="search-hint">
        {searchType === "coordinates" ? t(lang, "search.hint_coords") : ""}
      </div>
    </div>
  );
}
