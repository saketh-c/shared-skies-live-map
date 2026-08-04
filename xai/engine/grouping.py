"""Concept groups: map the 30 model features onto 5 policy-legible categories.

SHAP values are additive, so summing per-row contributions within a group is an
exact decomposition of the prediction: base + sum(group sums) = raw prediction.
This is the layer that turns "nbr_pm25_50km = +3.1" into "Regional PM signal
= +5.4 ug/m3" - the sentence a policy or classroom audience can actually use.

The partition is validated against the deployed feature list at import-call
time; adding a feature to the model without assigning it a group is an error.
"""
import numpy as np
import pandas as pd

# Ordered: roughly "most physically causal" -> "most contextual".
GROUPS = {
    "Wildfire smoke": ["hms_smoke"],
    "Regional PM signal": [
        "nbr_pm25_25km", "nbr_count_25km",
        "nbr_pm25_50km", "nbr_count_50km", "nbr_std_50km",
        "nbr_pm25_100km", "nbr_count_100km",
        "aod", "cams_pm25",
    ],
    "Meteorology": [
        "humidity", "temperature", "pressure", "wind_speed", "precipitation",
        "temp_x_humidity", "wind_x_temp",
    ],
    "Geography": [
        "latitude", "longitude", "dist_to_nearest_sensor", "dist_to_coast",
    ],
    "Season & calendar": [
        "month", "dow", "day_of_year",
        "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    ],
}

FEATURE_TO_GROUP = {f: g for g, feats in GROUPS.items() for f in feats}

GROUP_COLORS = {
    "Wildfire smoke": "#DD8452",
    "Regional PM signal": "#4C72B0",
    "Meteorology": "#55A868",
    "Geography": "#937860",
    "Season & calendar": "#64B5CD",
}

# One-line plain-language meaning per group. Seed for the phase-3 narration
# layer; also used in figure captions.
GROUP_BLURBS = {
    "Wildfire smoke": "Satellite-detected smoke plumes overhead (NOAA HMS).",
    "Regional PM signal": "What nearby sensors and satellites say the regional air already looks like today.",
    "Meteorology": "Weather that traps, disperses, or washes out particles.",
    "Geography": "Where this location sits in the state and relative to the sensor network.",
    "Season & calendar": "Time-of-year and day-of-week patterns.",
}


def validate(feature_names):
    """Assert the groups exactly partition the deployed feature list."""
    grouped = [f for feats in GROUPS.values() for f in feats]
    dupes = {f for f in grouped if grouped.count(f) > 1}
    missing = [f for f in feature_names if f not in FEATURE_TO_GROUP]
    extra = [f for f in grouped if f not in feature_names]
    problems = []
    if dupes:
        problems.append(f"features in >1 group: {sorted(dupes)}")
    if missing:
        problems.append(f"model features with no group: {missing}")
    if extra:
        problems.append(f"grouped features not in model: {extra}")
    if problems:
        raise ValueError("grouping.GROUPS is out of sync - " + "; ".join(problems))


def group_sums(shap_df):
    """Per-row group contributions: DataFrame [n_rows x n_groups], ug/m3.

    shap_df: DataFrame of per-feature SHAP values (columns = feature names).
    """
    validate(list(shap_df.columns))
    out = {}
    for group, feats in GROUPS.items():
        out[group] = shap_df[feats].sum(axis=1).to_numpy()
    return pd.DataFrame(out, index=shap_df.index)


def group_importance(shap_df):
    """Global ranking: mean |per-row group sum|, descending. Series in ug/m3."""
    g = group_sums(shap_df)
    return g.abs().mean(axis=0).sort_values(ascending=False)
