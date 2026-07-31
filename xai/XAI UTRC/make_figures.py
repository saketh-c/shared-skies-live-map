"""Publication figures for the URTC 2026 paper, regenerated from primary
artifacts at IEEE conference sizes (column 3.45 in, full width 7.0 in).

Outputs PDF (for LaTeX) + PNG (for preview) into figures/.
Run from repo root:  python "xai/XAI UTRC/make_figures.py"
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
try:
    from scipy.stats import gaussian_kde  # noqa: F401  (unused, keep deps obvious)
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
XAI = ROOT / "xai"
CACHE = XAI / "outputs" / "cache"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(XAI))

from engine import geo, grouping, loader  # noqa: E402
from engine.explain_shap import explain_rows  # noqa: E402

# ---------------------------------------------------------------- style
ACCENT = "#4C72B0"     # bars, trend lines (validated)
POS = "#C44E52"        # positive contributions (validated pair vs ACCENT)
NEG = "#4C72B0"        # negative contributions
SCATTER = "#9aa5b1"    # recessive scatter ink
INK = "#1a1a1a"
MUTED = "#555555"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.edgecolor": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "pdf.fonttype": 42,
    "savefig.dpi": 400,
})

COL_W = 3.45
FULL_W = 7.0
UGM3 = r"$\mu$g/m$^3$"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[fig] {name}")


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- data
group_imp = pd.read_csv(XAI / "outputs" / "group_importance.csv")
feat_imp = pd.read_csv(XAI / "outputs" / "feature_importance.csv")
S = pd.read_parquet(CACHE / "shap_ensemble.parquet").reset_index(drop=True)
F = pd.read_parquet(CACHE / "features_sample.parquet").reset_index(drop=True)
extra = json.loads((HERE / "verified_numbers_extra.json").read_text()) \
    if (HERE / "verified_numbers_extra.json").exists() else {}

gcol, vcol = group_imp.columns[0], group_imp.columns[1]
fcol, fvcol = feat_imp.columns[0], feat_imp.columns[1]

# ================================================================ FIG 1
# system + explanation-layer schematic
fig, ax = plt.subplots(figsize=(COL_W, 2.45))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")


def box(x, y, w, h, text, fc="#eef1f6", ec=MUTED, fs=6.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.18",
                                linewidth=0.6, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED,
                                shrinkA=1, shrinkB=1))


srcs = ["PurpleAir\nlive sensors", "Open-Meteo\nweather", "NOAA HMS\nsmoke", "CAMS\nAOD, PM2.5", "EJScreen,\ncensus tracts"]
for i, s in enumerate(srcs):
    box(0.1 + i * 2.0, 8.3, 1.75, 1.5, s, fs=6.0)
    arrow(0.975 + i * 2.0, 8.15, 4.0 + (i - 2) * 0.55, 7.05)

box(2.0, 5.6, 6.0, 1.35, "38-feature pipeline\n(BallTree neighbor PM2.5 at 25/50/100 km)", fs=6.6)
arrow(5.0, 5.45, 5.0, 4.75)
box(2.0, 3.3, 6.0, 1.35, "RF + LightGBM + CatBoost simplex ensemble\n(LOSO $R^2$ = 0.714, 310 sensors)", fs=6.6, bold=True)
arrow(3.4, 3.15, 2.6, 2.35)
arrow(6.6, 3.15, 7.4, 2.35)
box(0.35, 0.85, 4.35, 1.45, "live tract map\n6,896 TX census tracts", fc="#e8efe9", fs=6.6)
box(5.35, 0.85, 4.35, 1.45, "exact SHAP blend\n7 concept groups (this paper)", fc="#f5e9e7", fs=6.6, bold=True)
save(fig, "fig1_system")

# ================================================================ FIG 2
# (a) concept-group importance  (b) top-8 features
fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.9),
                         gridspec_kw={"height_ratios": [7, 8], "hspace": 0.55})
g = group_imp.sort_values(vcol, ascending=True)
ax = axes[0]
ax.barh(g[gcol], g[vcol], color=ACCENT, height=0.62)
for i, v in enumerate(g[vcol]):
    ax.text(v + 0.07, i, f"{v:.2f}", va="center", fontsize=6.5, color=INK)
ax.set_xlim(0, 5.6)
ax.set_xlabel(f"mean |contribution| ({UGM3})", labelpad=1.5)
ax.set_title("(a) concept groups", loc="left", fontsize=8)
despine(ax)
ax.tick_params(axis="y", length=0)

f8 = feat_imp.head(8).sort_values(fvcol, ascending=True)
ax = axes[1]
ax.barh(f8[fcol], f8[fvcol], color=ACCENT, height=0.62)
for i, v in enumerate(f8[fvcol]):
    ax.text(v + 0.025, i, f"{v:.2f}", va="center", fontsize=6.5, color=INK)
ax.set_xlim(0, 2.15)
ax.set_xlabel(f"mean |SHAP| ({UGM3})", labelpad=1.5)
ax.set_title("(b) top 8 of 38 features", loc="left", fontsize=8)
despine(ax)
ax.tick_params(axis="y", length=0)
save(fig, "fig2_global")

# ================================================================ FIG 3
# dependence: neighbor response, smoke tiers, traffic, linguistic isolation
fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 1.78), gridspec_kw={"wspace": 0.42})


def dep_panel(ax, feat, xlabel, vline_q=None, vline_label=None):
    x, y = F[feat], S[feat]
    ax.scatter(x, y, s=3, color=SCATTER, alpha=0.35, linewidths=0, rasterized=True)
    bins = pd.qcut(x, 12, duplicates="drop")
    bx = x.groupby(bins, observed=True).median()
    by = y.groupby(bins, observed=True).median()
    ax.plot(bx, by, color=ACCENT, lw=1.6)
    if vline_q is not None:
        xq = x.quantile(vline_q)
        ax.axvline(xq, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.text(xq, ax.get_ylim()[1], vline_label, fontsize=6.0, color=MUTED,
                ha="left", va="top", rotation=0)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_xlabel(xlabel, labelpad=1.5)
    despine(ax)


dep_panel(axes[0], "nbr_pm25_50km", f"50 km neighbor mean ({UGM3})")
axes[0].set_ylabel(f"contribution ({UGM3})", labelpad=1.5)
axes[0].set_title("(a) regional signal", loc="left")

ax = axes[1]
rng = np.random.default_rng(0)
xj = F["hms_smoke"] + rng.uniform(-0.16, 0.16, len(F))
ax.scatter(xj, S["hms_smoke"], s=3, color=SCATTER, alpha=0.3, linewidths=0, rasterized=True)
tier_stats = []
samp_means = S.groupby(F["hms_smoke"])["hms_smoke"].mean()
for tier in (0, 1, 2, 3):
    if tier <= 1:
        m = float(samp_means.get(tier, np.nan))
        lo = hi = None
    else:
        key = "tier2" if tier == 2 else "tier3"
        m = extra.get(f"{key}_hms_mean_shap", float(samp_means.get(tier, np.nan)))
        ci = extra.get(f"{key}_hms_ci95")
        lo, hi = (ci if ci else (None, None))
    tier_stats.append((tier, m, lo, hi))
for tier, m, lo, hi in tier_stats:
    if lo is not None:
        ax.errorbar(tier, m, yerr=[[m - lo], [hi - m]], fmt="o", color=POS,
                    ms=3.5, lw=1.0, capsize=2, zorder=5)
    else:
        ax.plot(tier, m, "o", color=POS, ms=3.5, zorder=5)
ax.plot([t for t, m, *_ in tier_stats], [m for t, m, *_ in tier_stats],
        color=POS, lw=1.0, zorder=4)
ax.axhline(0, color=MUTED, lw=0.5)
ax.set_xticks([0, 1, 2, 3], ["none", "light", "med.", "heavy"])
ax.set_xlabel("HMS smoke tier", labelpad=1.5)
ax.set_title("(b) smoke tier", loc="left")
despine(ax)

dep_panel(axes[2], "traffic_proximity", "traffic proximity (EJScreen pctile)",
          vline_q=0.85, vline_label=" p85")
axes[2].set_title("(c) traffic proximity", loc="left")

dep_panel(axes[3], "pct_ling_isolated", "linguistically isolated (%)",
          vline_q=0.70, vline_label=" p70")
axes[3].set_title("(d) linguistic isolation", loc="left")
save(fig, "fig3_dependence")

# ================================================================ FIG 4
# smoke event vs clean day maps (column width, 2x2)
days = [("2024-05-27 smoke event", CACHE / "day_shap_20240527.parquet"),
        ("2024-04-03 clean day", CACHE / "day_shap_20240403.parquet")]
feat_cols = None
panels = []
for label, path in days:
    d = pd.read_parquet(path)
    if feat_cols is None:
        feat_cols = [c for c in d.columns if c in grouping.FEATURE_TO_GROUP]
    gsum = grouping.group_sums(d[feat_cols])
    panels.append((label, d, gsum))

fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.0),
                         gridspec_kw={"wspace": 0.04, "hspace": 0.02})
cbars = []
for r, (label, d, gsum) in enumerate(panels):
    lon, lat = d["id_longitude"], d["id_latitude"]
    specs = [
        (gsum["Regional PM signal"], plt.cm.RdBu_r, -30.0, 30.0, f"regional signal ({UGM3})"),
        (d["pred"], plt.cm.viridis, 0.0, 50.0, f"predicted PM2.5 ({UGM3})"),
    ]
    for c, (vals, cmap, vmin, vmax, cl) in enumerate(specs):
        ax = axes[r, c]
        geo.draw_outline(ax, color="0.6", lw=0.5)
        sc = ax.scatter(lon, lat, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=5, linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        if r == 1:
            cbars.append((sc, cl, c))
    axes[r, 0].text(-0.02, 0.5, label, transform=axes[r, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=6.5)
for sc, cl, c in cbars:
    cb = fig.colorbar(sc, ax=axes[:, c], orientation="horizontal",
                      fraction=0.04, pad=0.015, aspect=22)
    cb.set_label(cl, fontsize=6.5, labelpad=1.5)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
save(fig, "fig4_event_maps")

# ================================================================ FIG 5
# concept-group decomposition: accurate smoke hit vs structural miss
bundle = loader.load_bundle()
feats = bundle["feature_names"]
frame = pd.read_parquet(CACHE / "training_frame.parquet")
frame["date"] = pd.to_datetime(frame["date"])
cases = [
    ("(a) smoke event, explained", 217461, "2024-05-27"),
    ("(b) hyper-local event, missed", 242357, "2024-11-19"),
]
rows = []
sid_str = frame["sensor_id"].astype(str)
for title, sid, date in cases:
    row = frame[(sid_str == str(sid)) & (frame["date"] == date)]
    assert len(row) == 1, f"case row not found: {sid} {date}"
    rows.append(row)
X = pd.concat(rows)[feats].to_numpy(dtype=np.float64)
phi, base = explain_rows(bundle, X)
preds = np.maximum(base + phi.sum(axis=1), 0.0)

order = None
fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.05), gridspec_kw={"wspace": 0.12})
for k, ((title, sid, date), row) in enumerate(zip(cases, rows)):
    gvals = grouping.group_sums(pd.DataFrame([phi[k]], columns=feats)).iloc[0]
    if order is None:
        order = gvals.sort_values().index  # fix group order from case (a)
    gv = gvals[order]
    ax = axes[k]
    colors = [POS if v > 0 else NEG for v in gv.values]
    ax.barh(np.arange(len(gv)), gv.values, color=colors, height=0.62)
    span = max(abs(gv.values).max() * 1.45, 2.0)
    for i, v in enumerate(gv.values):
        ax.text(v + (0.04 * span if v >= 0 else -0.04 * span), i, f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=6.0, color=INK)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlim(-span, span)
    ax.set_yticks(np.arange(len(gv)))
    if k == 0:
        short = {"Regional PM signal": "regional signal", "Wildfire smoke": "smoke tier",
                 "Meteorology": "meteorology", "Local sources": "local sources",
                 "Community & EJ context": "community/EJ", "Geography": "geography",
                 "Season & calendar": "season"}
        ax.set_yticklabels([short[gname] for gname in order], fontsize=7)
    else:
        ax.set_yticklabels([])
    actual = float(row["pm25"].iloc[0])
    ax.set_title(f"{title}\nactual {actual:.1f}, predicted {preds[k]:.1f} {UGM3}",
                 loc="left", fontsize=7)
    ax.set_xlabel(f"contribution ({UGM3})", labelpad=1.5)
    despine(ax)
    ax.tick_params(axis="y", length=0)
save(fig, "fig5_cases")

print("done:", sorted(p.name for p in FIGS.glob("*.pdf")))
