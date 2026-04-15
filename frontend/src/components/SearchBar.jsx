import React, { useState, useRef, useEffect, useContext } from 'react';
import './SearchBar.css';
import { LanguageContext } from '../App';
import { t } from '../i18n';

export default function SearchBar({ onSearch, loading }) {
  const { lang } = useContext(LanguageContext);
  const [searchInput, setSearchInput] = useState('');
  const [searchType, setSearchType] = useState('address');
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [error, setError] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const searchRef = useRef(null);
  const timeoutRef = useRef(null);

  // Fetch suggestions from Nominatim
  useEffect(() => {
    if (searchType !== 'address' || !searchInput.trim() || searchInput.length < 3) {
      setSuggestions([]);
      return;
    }

    // Clear previous timeout
    if (timeoutRef.current) clearTimeout(timeoutRef.current);

    // Debounce the search
    timeoutRef.current = setTimeout(async () => {
      try {
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
            searchInput + ', Texas'
          )}&limit=5`
        );
        const results = await response.json();
        setSuggestions(results);
        setShowSuggestions(true);
      } catch (err) {
        console.error('Autocomplete error:', err);
        setSuggestions([]);
      }
    }, 300);

    return () => clearTimeout(timeoutRef.current);
  }, [searchInput, searchType]);

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (searchRef.current && !searchRef.current.contains(e.target)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const geocodeAddress = async (address, lat, lon) => {
    try {
      setIsSearching(true);
      setError('');

      // Validate Texas bounds
      if (lat < 25.8 || lat > 36.5 || lon < -106.6 || lon > -93.5) {
        setError(t(lang, 'search.errors.address_outside'));
        setIsSearching(false);
        return;
      }

      onSearch({ lat, lon, address });
      setSuggestions([]);
      setShowSuggestions(false);
    } catch (err) {
      setError(t(lang, 'search.errors.search_failed'));
      console.error(err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    geocodeAddress(
      suggestion.display_name,
      parseFloat(suggestion.lat),
      parseFloat(suggestion.lon)
    );
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    setError('');

    if (searchType === 'address') {
      if (!searchInput.trim()) {
        setError(t(lang, 'search.errors.enter_address'));
        return;
      }
      if (suggestions.length > 0) {
        handleSuggestionClick(suggestions[0]);
      } else {
        // No suggestions loaded yet — do a fresh Nominatim lookup
        setIsSearching(true);
        try {
          const response = await fetch(
            `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
              searchInput + ', Texas'
            )}&limit=1`
          );
          const results = await response.json();
          if (results.length > 0) {
            const r = results[0];
            geocodeAddress(r.display_name, parseFloat(r.lat), parseFloat(r.lon));
          } else {
            setError(t(lang, 'search.errors.address_not_found'));
          }
        } catch (err) {
          setError(t(lang, 'search.errors.search_failed'));
        } finally {
          setIsSearching(false);
        }
      }
    } else {
      // Coordinates mode
      const coords = searchInput.trim().split(',');
      if (coords.length !== 2) {
        setError(t(lang, 'search.errors.coords_format'));
        return;
      }

      const lat = parseFloat(coords[0].trim());
      const lon = parseFloat(coords[1].trim());

      if (isNaN(lat) || isNaN(lon)) {
        setError(t(lang, 'search.errors.coords_invalid'));
        return;
      }

      if (lat < 25.8 || lat > 36.5 || lon < -106.6 || lon > -93.5) {
        setError(t(lang, 'search.errors.coords_out_of_bounds'));
        return;
      }

      onSearch({ lat, lon });
      setError('');
    }
  };

  return (
    <div className="search-bar" ref={searchRef}>
      <form onSubmit={handleSearch}>
        <div className="search-tabs">
          <button
            type="button"
            className={`search-tab ${searchType === 'address' ? 'active' : ''}`}
            onClick={() => {
              setSearchType('address');
              setSearchInput('');
              setSuggestions([]);
              setError('');
            }}
          >
            {t(lang, 'search.address_tab')}
          </button>
          <button
            type="button"
            className={`search-tab ${searchType === 'coordinates' ? 'active' : ''}`}
            onClick={() => {
              setSearchType('coordinates');
              setSearchInput('');
              setSuggestions([]);
              setError('');
            }}
          >
            {t(lang, 'search.coordinates_tab')}
          </button>
        </div>

        <div className="search-input-group">
          <div className="search-input-wrapper">
            <input
              type="text"
              placeholder={
                searchType === 'address'
                  ? t(lang, 'search.placeholder_address')
                  : t(lang, 'search.placeholder_coords')
              }
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              onFocus={() => searchType === 'address' && setShowSuggestions(true)}
              required
            />

            {/* Autocomplete Dropdown */}
            {searchType === 'address' && showSuggestions && suggestions.length > 0 && (
              <div className="suggestions-dropdown">
                {suggestions.map((suggestion, idx) => (
                  <div
                    key={idx}
                    className="suggestion-item"
                    onClick={() => handleSuggestionClick(suggestion)}
                  >
                    <div className="suggestion-name">{suggestion.name || suggestion.display_name.split(',')[0]}</div>
                    <div className="suggestion-address">{suggestion.display_name}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <button type="submit" disabled={loading || isSearching}>
            {isSearching ? t(lang, 'search.searching') : loading ? t(lang, 'search.loading') : t(lang, 'search.search_button')}
          </button>
        </div>

        {error && <div className="search-error">{error}</div>}
      </form>

      <div className="search-hint">
        {searchType === 'coordinates' ? t(lang, 'search.hint_coords') : ''}
      </div>
    </div>
  );
}
