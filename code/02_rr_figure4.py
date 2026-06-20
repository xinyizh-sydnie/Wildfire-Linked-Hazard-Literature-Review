# %% [markdown]
# # ERL Revision — Consolidated Figure 4 & Figure 5 Notebook (v4)
#
# **Purpose:** Single reproducible pipeline for the sub-figures in **Figure 4**
# (Global compound and fire hotspots, global RR violin, SREX forest plots by
# fire regime) and **Figure 5** (four separate mismatch maps).
#
# ### Reviewer comments addressed
#
# 1. **SREX polygon consistency (Fig 4 vs Fig 5).** Both figures now draw region
#    boundaries from a single source of truth: `regionmask.defined_regions.srex`.
#
# 2. **Hard-to-differentiate colors on the global RR map.** Bolder hues + distinct
#    marker shapes per hazard category (hollow circles → single hazards,
#    filled squares → 2-way compounds, filled triangle → H+D+W,
#    filled diamond → windy+flood).
#
# 3. **SREX region forest plot by fire regime** rendered in Section 10.
#
# 4. **Figure 5 mismatch maps — separate files, original colors preserved.**
#
# 5. **(v3) Transparent research-attention assignment.** Section 7b replaces the
#    primary-centroid literature count with simple areal weighting following
#    Goodchild & Lam (1980), with binary dasymetric refinement via a 50 m land
#    mask (Eicher & Brewer, 2001).
#
# 6. **(NEW in v4) Percentile-aware colorbars on Figure 5.** The Figure 5
#    colorbars now show tick marks at the p25, p50, and p75 quantiles of each
#    index distribution (computed over nonzero land cells), plus tier labels
#    ("low" / "elevated" / "severe") above each segment. This makes the
#    threshold convention used in the text (p25 = elevated tier, p75 = severe
#    tier) directly legible from the figure itself, addressing the concern
#    that a low-to-high gradient could be misread as implying that p25 is
#    already "high."
#
# ### Running order
#
# Run cells top to bottom. Paths in Section 3 assume the same Google Drive layout.

# %% [markdown]
# ## 1. Setup and installation

# %%
# Colab: install geospatial deps. For local runs use the repo's requirements.txt instead.
# !pip install -q cartopy regionmask rasterio rioxarray geopandas   # notebook shell/magic - install via requirements.txt
# !pip install -q matplotlib seaborn scipy pandas numpy shapely pyproj   # notebook shell/magic - install via requirements.txt
# !pip install -q plotly kaleido   # notebook shell/magic - install via requirements.txt

# Mount Google Drive (Colab only). Skips automatically when not on Colab.
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ModuleNotFoundError:
    print("Not on Colab; skipping Drive mount. Using local DATA_DIR / OUTPUT_DIR instead.")

# %% [markdown]
# ## 2. Imports

# %%
import warnings
warnings.filterwarnings('ignore')

import os
import re
import json
import math
import logging
from pathlib import Path
from collections import Counter

# Make both `glob(...)` (mismatch pipeline) AND `glob.glob(...)` (RR helpers)
# resolve to the same file-matching function.
import glob as _glob_module
from glob import glob
glob.glob = _glob_module.glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.path as mpath
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon as MplPolygon
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import rasterio
from rasterio.transform import rowcol
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling as RResampling

from scipy import stats                     # needed by RR CI helpers
from scipy.ndimage import gaussian_filter, zoom
from scipy.interpolate import griddata

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader

from shapely.ops import unary_union
from shapely.geometry import Polygon as ShapelyPolygon, Point, box
try:
    from shapely import contains_xy
    def _shape_contains_xy(geom, xs, ys):
        return contains_xy(geom, xs, ys)
except ImportError:
    from shapely.vectorized import contains as _vcontains
    def _shape_contains_xy(geom, xs, ys):
        return _vcontains(geom, xs, ys)

try:
    import regionmask
    HAS_REGIONMASK = True
except ImportError:
    HAS_REGIONMASK = False
    raise RuntimeError("regionmask is required for this notebook.")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('erl_revision')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

print("Imports complete.")

# %% [markdown]
# ## 3. File paths
#
# Update these if your Drive layout differs.

# %%
# ---------------------------------------------------------------------------
# File paths. Default to a local ./data and ./outputs tree (see README).
# To use Google Drive instead, set DATA_DIR / OUTPUT_DIR env vars, e.g.
#   os.environ["DATA_DIR"] = "/content/drive/MyDrive/ERL/data"
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

DATA_YEARLY_DIR       = DATA_DIR / "hotspot_50km"        # output of 00_preprocess_hotspots
PHYSICAL_REALITY_PATH = DATA_YEARLY_DIR

DATA_10KM_DIR    = DATA_DIR / "land_10km"                # 2020 land/WUI fraction rasters (10 km)
LITERATURE_PATH  = DATA_DIR / "study_table.csv"          # trimmed LLM classification output
LLM_DATA_PATH    = LITERATURE_PATH
GRDI_PATH        = DATA_DIR / "grdi" / "povmap-grdi-v1.tif"
WUI_DIR          = DATA_DIR / "wui"

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs")) / "figure4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'axes.linewidth': 1.0,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.default': 'regular',
})

SINGLE_COLORS = {
    "HOT":   "#7b3294", "DRY":   "#fdae61",
    "WINDY": "#66c2a5", "FLOOD": "#3288bd",
}
COMPOUND_COLORS = {
    'HOT_DRY': '#d73027', 'DRY_WINDY': '#fc8d59',
    'HOT_WINDY': '#fee090', 'HOT_DRY_WINDY': '#91003f',
    'WINDY_FLOOD': '#4575b4',
}
CLASS_NAMES_3  = {0: "NoData", 1: "Wildland", 2: "WUI Belt", 3: "Urban"}
CLASS_COLORS_3 = {0: "#ffffff", 1: "#006400", 2: "#87CEEB", 3: "#4169E1"}
WF_RGB_MIN_HEX = "#5a0012"
WF_RGB_MAX_HEX = "#d4002a"
DEFAULT_COLOR  = "#333333"
IPCC_MILESTONES = {2007: 'AR4', 2012: 'SREX', 2014: 'AR5',
                   2018: 'SR1.5', 2021: 'AR6'}

print("Path checks:")
for lbl, p in [('DATA_YEARLY_DIR', DATA_YEARLY_DIR),
               ('LITERATURE_PATH', LITERATURE_PATH),
               ('GRDI_PATH',       GRDI_PATH),
               ('WUI_DIR',         str(WUI_DIR)),
               ('OUTPUT_DIR',      str(OUTPUT_DIR))]:
    print(f"  {'OK ' if os.path.exists(p) else 'MISS'} {lbl}: {p}")

# %% [markdown]
# ## 4. Unified SREX polygon definition (core revision fix)
#
# Both Figure 4 and Figure 5 draw region boundaries from this single
# `regionmask.defined_regions.srex` object. The Amazon slanted pentagon, West
# South America polygon, etc. will render identically in every map.

# %%
# SREX reference regions + utilities now live in code/srex_regions.py
# (shared across all figure notebooks; keep that file alongside this one).
from srex_regions import (
    srex_regions,
    SREX_NAME_TO_ABBREV, SREX_ABBREV_TO_NAME,
    SREX_NUM_TO_ABBREV, SREX_NUM_TO_NAME, SREX_ABBREV_TO_NUM,
    SREX_POLYGONS,
    srex_abbrev, get_srex_mask_and_names,
    assign_srex_by_point, assign_srex_vectorized, draw_srex_polygons,
)
print(f"SREX utilities loaded from srex_regions.py ({len(SREX_POLYGONS)} polygons).")

# %% [markdown]
# ## 5. Shared helper functions
#
# Unchanged from the original RR notebook. Provides raster IO, yearly RR,
# confidence-interval calculations, and SREX mask utilities. Some definitions
# (e.g. `SREX_NAME_TO_ABBREV`) are re-declared here with identical content; the
# unified `SREX_POLYGONS` from Section 4 is NOT redefined.

# %%
import warnings
warnings.filterwarnings('ignore')

import os
import re
import json
import math
import logging
from pathlib import Path
from collections import Counter

# Make both `glob(...)` (mismatch pipeline) AND `glob.glob(...)` (RR helpers)
# resolve to the same file-matching function.
import glob as _glob_module
from glob import glob
glob.glob = _glob_module.glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.path as mpath
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon as MplPolygon
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import rasterio
from rasterio.transform import rowcol
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling as RResampling

from scipy import stats                     # needed by RR CI helpers
from scipy.ndimage import gaussian_filter, zoom
from scipy.interpolate import griddata

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader

from shapely.ops import unary_union
from shapely.geometry import Polygon as ShapelyPolygon, Point, box
try:
    from shapely import contains_xy
    def _shape_contains_xy(geom, xs, ys):
        return contains_xy(geom, xs, ys)
except ImportError:
    from shapely.vectorized import contains as _vcontains
    def _shape_contains_xy(geom, xs, ys):
        return _vcontains(geom, xs, ys)

try:
    import regionmask
    HAS_REGIONMASK = True
except ImportError:
    HAS_REGIONMASK = False
    raise RuntimeError("regionmask is required for this notebook.")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('erl_revision')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

print("Imports complete.")
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_yearly_data(yearly_dir):
    """Load and parse yearly compound and fire data."""
    tifs = sorted(glob.glob(os.path.join(yearly_dir, "*.tif")))

    compounds = {}  # compound_type -> {year: filepath}
    fires = {}      # year -> filepath

    for p in tifs:
        bn = os.path.basename(p)
        m = re.search(r"(19|20)\d{2}", bn)
        if not m:
            continue
        year = int(m.group())

        if "fire" in bn.lower():
            fires[year] = p
        elif "days" in bn.lower(): # Assuming compound files have 'days' in their name and fire files have 'fire'
            # Example: hotspot_2001_HOT_DRY_days.tif
            # Extract compound type from filename: HOT_DRY
            # Remove year and 'days' from filename
            core = bn.replace(".tif", "") # hotspot_2001_HOT_DRY_days
            parts = core.split("_") # ['hotspot', '2001', 'HOT', 'DRY', 'days']
            # The compound type is typically the parts between year and 'days'
            # If the format is hotspot_YEAR_COMPOUNDTYPE_days
            if len(parts) >= 4 and parts[0].lower() == 'hotspot' and parts[-1].lower() == 'days':
                # Reconstruct compound name, excluding the year and 'days' parts
                ctype = "_".join(parts[2:-1])
                compounds.setdefault(ctype, {})[year] = p
            else:
                # Fallback for unexpected naming, or log a warning
                # print(f"Warning: Could not parse compound type from: {bn}") # Commented out to reduce verbose output
                pass

    return compounds, fires


def calculate_relative_risk_with_ci(a, b, c, d, alpha=0.05):
    """
    Calculate Relative Risk with confidence interval.

    Contingency table:
                    Fire+    Fire-
    Compound+       a        b
    Compound-       c        d
    """
    if (a + b) == 0 or (c + d) == 0:
        return np.nan, np.nan, np.nan, False

    risk_exposed = a / (a + b)
    risk_unexposed = c / (c + d)

    if risk_unexposed == 0:
        return np.nan, np.nan, np.nan, False

    rr = risk_exposed / risk_unexposed

    # Log-transformed confidence interval
    if a > 0 and b > 0 and c > 0 and d > 0:
        log_rr = np.log(rr)
        se_log_rr = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
        z = stats.norm.ppf(1 - alpha/2)

        ci_lower = np.exp(log_rr - z * se_log_rr)
        ci_upper = np.exp(log_rr + z * se_log_rr)

        significant = ci_lower > 1.0 or ci_upper < 1.0
    else:
        ci_lower, ci_upper = np.nan, np.nan
        significant = False

    return rr, ci_lower, ci_upper, significant


# Canonical SREX name → abbreviation mapping (from regionmask)
SREX_NAME_TO_ABBREV = {
    "Alaska/N.W. Canada": "ALA",
    "Canada/Greenl./Icel.": "CGI",
    "W. North America": "WNA",
    "C. North America": "CNA",
    "E. North America": "ENA",
    "Central America/Mexico": "CAM",
    "Amazon": "AMZ",
    "N.E. Brazil": "NEB",
    "Coast South America": "WSA",
    "S.E. South America": "SSA",
    "N. Europe": "NEU",
    "C. Europe": "CEU",
    "S. Europe/Mediterranean": "MED",
    "Sahara": "SAH",
    "W. Africa": "WAF",
    "E. Africa": "EAF",
    "S. Africa": "SAF",
    "N. Asia": "NAS",
    "W. Asia": "WAS",
    "C. Asia": "CAS",
    "Tibetan Plateau": "TIB",
    "E. Asia": "EAS",
    "S. Asia": "SAS",
    "S.E. Asia": "SEA",
    "N. Australia": "NAU",
    "S. Australia/New Zealand": "SAU",
}

def srex_abbrev(name):
    """Get canonical SREX abbreviation from full region name."""
    if name in SREX_NAME_TO_ABBREV:
        return SREX_NAME_TO_ABBREV[name]
    # Fuzzy fallback
    name_lower = name.lower().strip()
    for k, v in SREX_NAME_TO_ABBREV.items():
        if k.lower() == name_lower:
            return v
    # Last resort: first 3 alpha chars
    letters = "".join([c for c in name if c.isalpha()])
    return letters[:3].upper() if len(letters) >= 3 else name[:3].upper()


def get_srex_mask_and_names(lons, lats):
    """Build SREX region mask from coordinate arrays.

    CRITICAL FIX: Returns region_names as a DICT {mask_number: name}
    because regionmask SREX numbers are 1-26 but .names is 0-indexed.
    Using names[ridx] with ridx from mask gives off-by-one errors
    (e.g., mask=10=SSA but names[10]=NEU).
    """
    if not HAS_REGIONMASK:
        return None, None, None

    lon2d, lat2d = np.meshgrid(lons, lats)

    try:
        srex = regionmask.defined_regions.srex
        srex_mask = srex.mask(lon2d, lat2d)
        # FIX: Build dict {mask_number → name} to avoid off-by-one
        region_names = dict(zip(srex.numbers, srex.names))
        # Also store abbreviations globally for direct lookup
        global SREX_NUM_TO_ABBREV
        SREX_NUM_TO_ABBREV = dict(zip(srex.numbers, srex.abbrevs))
        print(f"  ✓ Using SREX regions (N={len(region_names)})")
    except:
        ar6 = regionmask.defined_regions.ar6.land
        srex_mask = ar6.mask(lon2d, lat2d)
        region_names = dict(zip(ar6.numbers, ar6.names))
        SREX_NUM_TO_ABBREV = dict(zip(ar6.numbers, ar6.abbrevs))
        print(f"  ✓ Using AR6 regions (N={len(region_names)})")

    srex_arr = np.array(srex_mask)
    valid_region = ~np.isnan(srex_arr)
    srex_int = np.where(valid_region, srex_arr.astype(int), -1)

    return srex_int, valid_region, region_names


def compute_regional_relative_risk(compounds, fires, compound_type='HOT_DRY',
                                   compound_pct=75, land_mask=None):
    """
    Compute Relative Risk by SREX region.

    Fire (outcome) is always BINARY (>0 = fire occurred).
    Compound (exposure) is thresholded at compound_pct on the multi-year sum.

    Parameters
    ----------
    compound_pct : int
        Percentile on compound multi-year sum defining "high-frequency" exposure.
        75 = upper quartile (recommended). 50 = above-median. 0 = >0 presence/absence.
    land_mask : np.ndarray (bool) or None
        If provided, intersected with SREX valid_region to exclude ocean pixels.
    """
    years = sorted(set(fires.keys()) & set(compounds.get(compound_type, {}).keys()))

    if not years:
        return None

    print(f"  Computing RR for {compound_type} across {len(years)} years...")

    # Load template
    template_path = list(fires.values())[0]
    with rasterio.open(template_path) as src:
        transform = src.transform
        height, width = src.height, src.width

    lon0 = transform.c + transform.a / 2
    lat0 = transform.f + transform.e / 2
    lons = lon0 + np.arange(width) * transform.a
    lats = lat0 + np.arange(height) * transform.e

    # Build SREX mask
    srex_int, valid_region, region_names = get_srex_mask_and_names(lons, lats)

    if srex_int is None:
        valid_region = np.ones((height, width), dtype=bool)
        srex_int = np.zeros((height, width), dtype=int)
        region_names = {0: 'Global'}

    # FIX: Intersect SREX with actual land mask to exclude ocean pixels
    if land_mask is not None:
        n_before = int(valid_region.sum())
        valid_region = valid_region & land_mask
        n_after = int(valid_region.sum())
        print(f"  Land mask applied: {n_before:,} -> {n_after:,} pixels "
              f"(removed {n_before - n_after:,} ocean/nodata)")

    # Accumulate multi-year data
    fire_sum = np.zeros((height, width), dtype=np.float32)
    comp_sum = np.zeros((height, width), dtype=np.float32)

    for year in years:
        with rasterio.open(fires[year]) as src:
            fire_arr = src.read(1).astype(np.float32)
        with rasterio.open(compounds[compound_type][year]) as src:
            comp_arr = src.read(1).astype(np.float32)

        fire_arr = np.where(np.isnan(fire_arr), 0, fire_arr)
        comp_arr = np.where(np.isnan(comp_arr), 0, comp_arr)

        fire_sum += fire_arr
        comp_sum += comp_arr

    # ── Domain: use land_mask for Global, SREX ∩ land for regional ──
    global_domain = land_mask if land_mask is not None else valid_region

    # ── FIRE (outcome): always BINARY ──
    fire_hot = (fire_sum > 0)

    # ── COMPOUND (exposure): percentile on multi-year sum ──
    if compound_pct is not None and compound_pct > 0:
        comp_valid = comp_sum[global_domain & (comp_sum > 0)]
        comp_threshold = np.percentile(comp_valid, compound_pct) if len(comp_valid) > 0 else 0
        comp_hot = (comp_sum >= comp_threshold) & (comp_sum > 0)
        n_exposed = int((comp_hot & global_domain).sum())
        n_total = int(global_domain.sum())
        print(f"  Fire: >0 (binary) | Compound: {compound_pct}th pct (>={comp_threshold:.1f})")
        print(f"  Exposed: {n_exposed:,}/{n_total:,} ({100*n_exposed/max(n_total,1):.1f}%)")
    else:
        comp_hot = (comp_sum > 0)
        n_exposed = int((comp_hot & global_domain).sum())
        n_total = int(global_domain.sum())
        print(f"  Fire: >0 (binary) | Compound: >0 (presence/absence)")
        print(f"  Exposed: {n_exposed:,}/{n_total:,} ({100*n_exposed/max(n_total,1):.1f}%)")
        if n_exposed > 0.95 * n_total:
            print(f"  ⚠ WARNING: >95% exposed — compound is ubiquitous, RR unreliable")

    # Compute RR
    results = []

    # ── Global RR: full classified land domain ──
    a = int(np.sum(fire_hot & comp_hot & global_domain))
    b = int(np.sum(~fire_hot & comp_hot & global_domain))
    c = int(np.sum(fire_hot & ~comp_hot & global_domain))
    d = int(np.sum(~fire_hot & ~comp_hot & global_domain))

    rr, ci_lo, ci_hi, sig = calculate_relative_risk_with_ci(a, b, c, d)

    results.append({
        'region': 'GLOBAL',
        'region_idx': -1,
        'RR': rr,
        'CI_lower': ci_lo,
        'CI_upper': ci_hi,
        'significant': sig,
        'a': a, 'b': b, 'c': c, 'd': d,
        'n_cells': int(global_domain.sum()),
    })

    # ── Per-region RR: SREX ∩ land_mask ──
    if HAS_REGIONMASK:
        unique_regions = np.unique(srex_int[valid_region])
        unique_regions = unique_regions[unique_regions >= 0]

        for ridx in unique_regions:
            reg_mask = (srex_int == ridx) & valid_region

            a = int(np.sum(fire_hot & comp_hot & reg_mask))
            b = int(np.sum(~fire_hot & comp_hot & reg_mask))
            c = int(np.sum(fire_hot & ~comp_hot & reg_mask))
            d = int(np.sum(~fire_hot & ~comp_hot & reg_mask))

            rr, ci_lo, ci_hi, sig = calculate_relative_risk_with_ci(a, b, c, d)

            # FIX: region_names is now a dict {mask_number: name}
            region_name = region_names.get(ridx, f'Region_{ridx}')

            results.append({
                'region': region_name,
                'region_idx': ridx,
                'RR': rr,
                'CI_lower': ci_lo,
                'CI_upper': ci_hi,
                'significant': sig,
                'a': a, 'b': b, 'c': c, 'd': d,
                'n_cells': int(reg_mask.sum()),
            })

    return pd.DataFrame(results)



# =============================================================================
# v6 UPDATE: YEARLY RR + AVERAGE ACROSS YEARS (applies to all downstream figures)
# =============================================================================
# The previous implementation computed RR from 24-year summed fields (ever-burned vs multi-year exposure).
# This v6 mode computes RR for EACH YEAR (fire outcome per year) using a FIXED exposure mask
# (defined from the multi-year compound sum percentile), then averages RR across years.
#
# Averaging uses a geometric mean on RR (mean of log(RR)), with uncertainty from between-year variability:
#   log(RR̄) ± z * sd(logRR)/sqrt(n_years)
#
# The original multi-year function is preserved as:
#   compute_regional_relative_risk_multiyear(...)
# =============================================================================

# Keep original multi-year implementation available
compute_regional_relative_risk_multiyear = compute_regional_relative_risk


def _read_raster_float32(path):
    """Read raster band 1 as float32; nodata/NaN -> 0 (counting convention)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def _within_year_se_logrr(a, b, c, d):
    """
    Within-year standard error of log(RR) from 2x2 contingency table.
    SE(logRR) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    Returns np.nan if any cell is zero (RR undefined).
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        return np.nan
    return np.sqrt(1.0/a - 1.0/(a+b) + 1.0/c - 1.0/(c+d))


def _dersimonian_laird_tau2(logrr_arr, se_arr):
    """
    Estimate between-study heterogeneity variance (tau^2) via DerSimonian-Laird.

    Parameters
    ----------
    logrr_arr : array of log(RR) per year
    se_arr    : array of within-year SE(logRR) per year

    Returns
    -------
    tau2 : float (>= 0)
    """
    w = 1.0 / (se_arr ** 2)                       # fixed-effect weights
    w_sum = np.sum(w)
    mu_fe = np.sum(w * logrr_arr) / w_sum          # fixed-effect pooled mean
    Q = np.sum(w * (logrr_arr - mu_fe) ** 2)       # Cochran's Q
    k = len(logrr_arr)
    c = w_sum - np.sum(w ** 2) / w_sum
    tau2 = max((Q - (k - 1)) / c, 0.0)             # truncate at 0
    return tau2


def _aggregate_yearly_rr(df_yearly, group_cols, alpha=0.05):
    """
    Aggregate per-year RR via random-effects meta-analysis (DerSimonian-Laird).

    For each group (region or setting):
      1. Compute within-year SE(logRR) from each year's contingency table (a,b,c,d).
      2. Estimate between-year heterogeneity tau^2 via DerSimonian-Laird.
      3. Pool using inverse-variance weights: w_i = 1/(SE_i^2 + tau^2).
      4. Pooled logRR = sum(w_i * logRR_i) / sum(w_i); SE_pooled = 1/sqrt(sum(w_i)).

    This propagates BOTH within-year sampling uncertainty AND between-year variability
    into the final CI — critical for strata with small cell counts (urban, rare compounds).

    Falls back to simple geometric mean if within-year SE cannot be computed
    (e.g., zero cells in some years).
    """
    z = stats.norm.ppf(1 - alpha / 2)
    out = []

    for keys, g in df_yearly.groupby(group_cols, dropna=False):
        g = g.copy()
        g = g[np.isfinite(g["RR"]) & (g["RR"] > 0)]
        n = int(len(g))

        # --- Build key dict ---
        row = {}
        if isinstance(keys, tuple):
            for i, col in enumerate(group_cols):
                row[col] = keys[i]
        else:
            row[group_cols[0]] = keys

        if n == 0:
            row.update({"RR": np.nan, "CI_lower": np.nan, "CI_upper": np.nan,
                        "significant": False, "n_years": 0,
                        "tau2": np.nan, "I2": np.nan,
                        "a": 0, "b": 0, "c": 0, "d": 0})
            out.append(row)
            continue

        # --- Carry stable metadata ---
        for stable in ["region_idx", "n_cells", "n_exposed"]:
            if stable in g.columns:
                row[stable] = int(g[stable].iloc[0]) if stable != "region_idx" else g[stable].iloc[0]

        # Sum contingency table counts
        for cc in ["a", "b", "c", "d"]:
            row[cc] = int(g[cc].sum()) if cc in g.columns else 0

        logrr = np.log(g["RR"].values)

        # --- Compute within-year SE for each year ---
        has_counts = all(c in g.columns for c in ["a", "b", "c", "d"])
        se_within = np.full(n, np.nan)
        if has_counts:
            for j in range(n):
                se_within[j] = _within_year_se_logrr(
                    g["a"].iloc[j], g["b"].iloc[j],
                    g["c"].iloc[j], g["d"].iloc[j]
                )

        valid_se = np.isfinite(se_within) & (se_within > 0)

        # --- Meta-analytic pooling (random effects) ---
        if np.sum(valid_se) >= 2:
            lr_v = logrr[valid_se]
            se_v = se_within[valid_se]

            tau2 = _dersimonian_laird_tau2(lr_v, se_v)
            w_re = 1.0 / (se_v ** 2 + tau2)
            w_sum = np.sum(w_re)

            mu_re = float(np.sum(w_re * lr_v) / w_sum)
            se_re = float(1.0 / np.sqrt(w_sum))

            rr_bar = float(np.exp(mu_re))
            ci_lo = float(np.exp(mu_re - z * se_re))
            ci_hi = float(np.exp(mu_re + z * se_re))

            # I^2 statistic (proportion of variance due to heterogeneity)
            Q = float(np.sum((1.0 / se_v**2) * (lr_v - mu_re)**2))
            k = len(lr_v)
            I2 = max((Q - (k - 1)) / Q, 0.0) * 100 if Q > 0 else 0.0

            row["tau2"] = float(tau2)
            row["I2"] = float(I2)

        elif np.sum(valid_se) == 1:
            # Single usable year: use that year's point estimate + within-year CI
            idx_ok = np.where(valid_se)[0][0]
            mu_re = float(logrr[idx_ok])
            se_re = float(se_within[idx_ok])
            rr_bar = float(np.exp(mu_re))
            ci_lo = float(np.exp(mu_re - z * se_re))
            ci_hi = float(np.exp(mu_re + z * se_re))
            row["tau2"] = 0.0
            row["I2"] = 0.0

        else:
            # Fallback: geometric mean with between-year variance only
            mu = float(np.mean(logrr))
            rr_bar = float(np.exp(mu))
            if n >= 2:
                sd = float(np.std(logrr, ddof=1))
                se_fb = sd / np.sqrt(n)
                ci_lo = float(np.exp(mu - z * se_fb))
                ci_hi = float(np.exp(mu + z * se_fb))
            else:
                ci_lo = float(g["CI_lower"].iloc[0]) if np.isfinite(g["CI_lower"].iloc[0]) else np.nan
                ci_hi = float(g["CI_upper"].iloc[0]) if np.isfinite(g["CI_upper"].iloc[0]) else np.nan
            row["tau2"] = np.nan
            row["I2"] = np.nan

        sig = (np.isfinite(ci_lo) and ci_lo > 1.0) or (np.isfinite(ci_hi) and ci_hi < 1.0)

        row.update({
            "RR": rr_bar,
            "CI_lower": ci_lo,
            "CI_upper": ci_hi,
            "significant": bool(sig),
            "n_years": n
        })
        out.append(row)

    return pd.DataFrame(out)


def compute_regional_relative_risk_yearly(compounds, fires, compound_type='HOT_DRY',
                                         compound_pct=75, land_mask=None, alpha=0.05):
    """
    Compute per-YEAR Relative Risk by SREX region.

    - Outcome: fire is BINARY per year (fire_year > 0)
    - Exposure: FIXED mask from compound_pct percentile on the MULTI-YEAR compound sum
    - Returns one row per (year, region) including GLOBAL.
    """
    years = sorted(set(fires.keys()) & set(compounds.get(compound_type, {}).keys()))
    if not years:
        return None

    # Template + grid centers
    template_path = fires[years[0]]
    with rasterio.open(template_path) as src:
        transform = src.transform
        height, width = src.height, src.width

    lon0 = transform.c + transform.a / 2
    lat0 = transform.f + transform.e / 2
    lons = lon0 + np.arange(width) * transform.a
    lats = lat0 + np.arange(height) * transform.e

    srex_int, valid_region, region_names = get_srex_mask_and_names(lons, lats)

    if srex_int is None:
        valid_region = np.ones((height, width), dtype=bool)
        srex_int = np.zeros((height, width), dtype=int)
        region_names = {0: 'Global'}

    if land_mask is not None:
        valid_region = valid_region & land_mask

    # Domains
    global_domain = land_mask if land_mask is not None else valid_region

    # Exposure (fixed): percentile on multi-year compound sum
    comp_sum = np.zeros((height, width), dtype=np.float32)
    for y in years:
        comp_sum += _read_raster_float32(compounds[compound_type][y])

    # Global threshold for the GLOBAL row in the output
    if compound_pct is not None and compound_pct > 0:
        comp_valid_global = comp_sum[global_domain & (comp_sum > 0)]
        comp_threshold_global = np.percentile(comp_valid_global, compound_pct) if len(comp_valid_global) > 0 else 0.0
        comp_hot_global = (comp_sum >= comp_threshold_global) & (comp_sum > 0)
    else:
        comp_hot_global = (comp_sum > 0)

    # Precompute region ids (MOVED UP to fix UnboundLocalError)
    if HAS_REGIONMASK:
        region_ids = np.unique(srex_int[valid_region])
        region_ids = region_ids[region_ids >= 0]
    else:
        region_ids = np.array([], dtype=int)

    # Per-region threshold for REGIONAL rows (NEW)
    comp_hot_regional = np.zeros_like(comp_sum, dtype=bool)
    if HAS_REGIONMASK and compound_pct is not None and compound_pct > 0:
        for ridx in region_ids:
            reg_mask = (srex_int == ridx) & valid_region
            comp_valid_reg = comp_sum[reg_mask & (comp_sum > 0)]
            if len(comp_valid_reg) > 0:
                thr_reg = np.percentile(comp_valid_reg, compound_pct)
                comp_hot_regional |= ((comp_sum >= thr_reg) & (comp_sum > 0) & reg_mask)

    n_exposed_global = int((comp_hot_global & global_domain).sum()) # FIX: use comp_hot_global

    rows = []

    for y in years:
        fire_arr = _read_raster_float32(fires[y])
        fire_hot = (fire_arr > 0)

        # GLOBAL
        a = int(np.sum(fire_hot & comp_hot_global & global_domain)) # FIX: use comp_hot_global
        b = int(np.sum((~fire_hot) & comp_hot_global & global_domain)) # FIX: use comp_hot_global
        c = int(np.sum(fire_hot & (~comp_hot_global) & global_domain)) # FIX: use comp_hot_global
        d = int(np.sum((~fire_hot) & (~comp_hot_global) & global_domain)) # FIX: use comp_hot_global
        rr, ci_lo, ci_hi, sig = calculate_relative_risk_with_ci(a, b, c, d, alpha=alpha)

        rows.append({
            "year": y,
            "region": "GLOBAL",
            "region_idx": -1,
            "RR": rr,
            "CI_lower": ci_lo,
            "CI_upper": ci_hi,
            "significant": sig,
            "n_cells": int(global_domain.sum()),
            "n_exposed": n_exposed_global,
            'a': a, 'b': b, 'c': c, 'd': d, # Add counts here
        })

        # REGIONS
        if HAS_REGIONMASK:
            for ridx in region_ids:
                reg_mask = (srex_int == ridx) & valid_region
                if not np.any(reg_mask):
                    continue

                # Use comp_hot_regional for regional RR calculations
                a = int(np.sum(fire_hot & comp_hot_regional & reg_mask))
                b = int(np.sum((~fire_hot) & comp_hot_regional & reg_mask))
                c = int(np.sum(fire_hot & (~comp_hot_regional) & reg_mask))
                d = int(np.sum((~fire_hot) & (~comp_hot_regional) & reg_mask))

                rr, ci_lo, ci_hi, sig = calculate_relative_risk_with_ci(a, b, c, d, alpha=alpha)
                region_name = region_names.get(ridx, f"Region_{ridx}")

                rows.append({
                    "year": y,
                    "region": region_name,
                    "region_idx": ridx,
                    "RR": rr,
                    "CI_lower": ci_lo,
                    "CI_upper": ci_hi,
                    "significant": sig,
                    "n_cells": int(reg_mask.sum()),
                    "n_exposed": int((comp_hot_regional & reg_mask).sum()), # FIX: use comp_hot_regional
                    'a': a, 'b': b, 'c': c, 'd': d, # Add counts here
                })

    return pd.DataFrame(rows)


def compute_regional_relative_risk(compounds, fires, compound_type='HOT_DRY',
                                  compound_pct=75, land_mask=None, alpha=0.05,
                                  return_yearly=False):
    """
    v6 default: compute yearly RR (fire outcome per year) and return the AVERAGE RR across years.
    Downstream figures will now use this averaged RR consistently.
    """
    df_yearly = compute_regional_relative_risk_yearly(
        compounds, fires, compound_type=compound_type,
        compound_pct=compound_pct, land_mask=land_mask, alpha=alpha
    )
    if df_yearly is None or len(df_yearly) == 0:
        return None

    df_avg = _aggregate_yearly_rr(df_yearly, group_cols=["region", "region_idx"], alpha=alpha)

    # Standardize GLOBAL label and keep expected columns
    # (downstream code expects: region, region_idx, RR, CI_lower, CI_upper, significant, n_cells)
    if return_yearly:
        return df_avg, df_yearly
    return df_avg


# -----------------------------------------------------------------------------
# Area helper: convert exposed pixel masks to acres (used in Figure B2 count axis)
# -----------------------------------------------------------------------------
def pixel_area_acres_map_from_raster(raster_path):
    """
    Return per-pixel area in acres for a raster.
    - If CRS is projected (meters), uses |a*e| constant area.
    - If CRS is geographic (degrees), approximates spherical Earth cell areas by latitude.
    """
    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs
        h, w = src.height, src.width

    # Projected CRS (meters): constant area
    if crs is not None and getattr(crs, "is_projected", False):
        px_area_m2 = abs(transform.a * transform.e)
        return np.full((h, w), px_area_m2 / 4046.8564224, dtype=np.float64)

    # Geographic degrees: area varies by latitude
    lon_res_deg = abs(transform.a)
    lat_res_deg = abs(transform.e)
    lon_res_rad = np.deg2rad(lon_res_deg)

    # row center latitudes
    lat0 = transform.f + transform.e / 2
    lats = lat0 + np.arange(h) * transform.e
    # bounds per row
    lat1 = np.deg2rad(lats - lat_res_deg/2)
    lat2 = np.deg2rad(lats + lat_res_deg/2)

    R = 6371008.8  # mean Earth radius (m)
    band_area_m2 = (R**2) * lon_res_rad * (np.sin(lat2) - np.sin(lat1))  # shape (h,)
    band_area_m2 = np.abs(band_area_m2)

    area_m2 = np.repeat(band_area_m2[:, None], w, axis=1)
    return area_m2 / 4046.8564224


def mask_area_acres(area_acres_map, mask_bool):
    """Sum acres over a boolean mask."""
    return float(np.sum(area_acres_map[mask_bool]))


print("✓ Helper functions defined")


# ─── Additional helpers from combined_setting_hotspot_overlay ───

def _hex_to_rgb01(h):
    """Convert hex color to RGB (0-1 scale)."""
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def nan_gaussian(arr, sigma, min_weight=0.3):
    """NaN-aware Gaussian smoothing."""
    m = np.isfinite(arr)
    v = np.where(m, arr, 0.0)
    v_s = gaussian_filter(v, sigma=sigma, mode='nearest')
    w_s = gaussian_filter(m.astype(float), sigma=sigma, mode='nearest')
    out = v_s / np.maximum(w_s, 1e-12)
    out[w_s < min_weight] = np.nan
    return out, w_s


def build_land_path(land_union_geom):
    """Create a Matplotlib Path from land geometry."""
    try:
        from cartopy.mpl.path import shapely_to_path
        land_path = shapely_to_path(land_union_geom)
        if isinstance(land_path, list):
            land_path = mpath.Path.make_compound_path(*land_path)
        return land_path
    except Exception:
        from cartopy.mpl.patch import geos_to_path
        paths = geos_to_path(land_union_geom)
        return mpath.Path.make_compound_path(*paths)


def get_contour_artists(contourset):
    """Get contour artists for clipping (version-safe)."""
    if hasattr(contourset, "collections"):
        return contourset.collections
    if hasattr(contourset, "artists"):
        return contourset.artists
    return [ch for ch in contourset.get_children() if hasattr(ch, "set_clip_path")]


def _sum_rasters(file_dict, years):
    """Sum rasters across years."""
    with rasterio.open(file_dict[years[0]]) as src:
        transform = src.transform
        bounds = src.bounds
        height, width = src.height, src.width
    s = np.zeros((height, width), dtype=np.float32)
    for y in years:
        with rasterio.open(file_dict[y]) as src:
            a = src.read(1).astype(np.float32)
        a = np.where(np.isfinite(a), a, 0.0)
        s += a
    return s, transform, bounds


def _grid_centers(transform, height, width):
    """Get grid centers from transform."""
    lon0 = transform.c + transform.a / 2
    lat0 = transform.f + transform.e / 2
    lons = lon0 + np.arange(width) * transform.a
    lats = lat0 + np.arange(height) * transform.e
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lon2d, lat2d, lons, lats


def _build_hotspot_points(arr, lon2d, lat2d, hazard, pixel_p=98.5, agg_deg=1.0, keep_top_p=60.0, max_points=20000):
    """Build hotspot points from raster array."""
    mask = np.isfinite(arr) & (arr > 0)
    vals = arr[mask]
    if len(vals) == 0:
        return pd.DataFrame(columns=["hazard","lon","lat","freq","s"])
    thresh = np.percentile(vals, pixel_p)
    hot = arr >= thresh
    hot &= mask
    lons_hot = lon2d[hot]
    lats_hot = lat2d[hot]
    vals_hot = arr[hot]
    df = pd.DataFrame({"lon": lons_hot, "lat": lats_hot, "freq": vals_hot})
    df["lon_bin"] = (df["lon"] / agg_deg).round() * agg_deg
    df["lat_bin"] = (df["lat"] / agg_deg).round() * agg_deg
    agg = df.groupby(["lon_bin","lat_bin"]).agg(freq=("freq","sum")).reset_index()
    agg.rename(columns={"lon_bin":"lon","lat_bin":"lat"}, inplace=True)
    top_thresh = np.percentile(agg["freq"], keep_top_p)
    agg = agg[agg["freq"] >= top_thresh]
    if len(agg) > max_points:
        agg = agg.nlargest(max_points, "freq")
    agg["hazard"] = hazard
    return agg


def _sizes_from_freq(df, r_min, r_max, power):
    """Map frequency to marker sizes."""
    if len(df) == 0:
        df["s"] = []
        return df
    fmin = df["freq"].min()
    fmax = df["freq"].max()
    if fmax > fmin:
        norm = ((df["freq"] - fmin) / (fmax - fmin)) ** power
    else:
        norm = pd.Series(0.5, index=df.index)
    df["s"] = r_min + (r_max - r_min) * norm
    return df


print("✓ All helper functions defined")

# %% [markdown]
# ## 6. Data loading functions (Figure 4 pipeline)

# %% [markdown]
# ### 6a. Load data for Figure 4

# %%
# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def parse_geo_regions(geo_json_str):
    """Parse geo_regions_json to extract lat/lon coordinates."""
    if pd.isna(geo_json_str) or geo_json_str == '' or geo_json_str == '[]':
        return None, None

    try:
        regions = json.loads(geo_json_str)
    except (json.JSONDecodeError, TypeError):
        return None, None

    if not regions or not isinstance(regions, list):
        return None, None

    all_lats = []
    all_lons = []

    for region in regions:
        if not isinstance(region, dict):
            continue

        cp = region.get('CP')
        if cp is not None and isinstance(cp, (list, tuple)) and len(cp) >= 2:
            try:
                lat, lon = float(cp[0]), float(cp[1])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    all_lats.append(lat)
                    all_lons.append(lon)
                    continue
            except (TypeError, ValueError):
                pass

        mob = region.get('MOB')
        if mob is not None and isinstance(mob, dict):
            try:
                top_left = mob.get('top_left')
                bottom_right = mob.get('bottom_right')

                if top_left and bottom_right:
                    if (top_left[0] is not None and top_left[1] is not None and
                        bottom_right[0] is not None and bottom_right[1] is not None):

                        lat1, lon1 = float(top_left[0]), float(top_left[1])
                        lat2, lon2 = float(bottom_right[0]), float(bottom_right[1])

                        center_lat = (lat1 + lat2) / 2
                        center_lon = (lon1 + lon2) / 2

                        if -90 <= center_lat <= 90 and -180 <= center_lon <= 180:
                            all_lats.append(center_lat)
                            all_lons.append(center_lon)
            except (TypeError, ValueError, IndexError):
                pass

    if all_lats and all_lons:
        return np.mean(all_lats), np.mean(all_lons)

    return None, None


def load_literature_data(path):
    """Load and filter literature data."""
    df = pd.read_csv(path)
    print(f"Total papers loaded: {len(df)}")

    linked = df[df['linked_label'] == 'linked'].copy()
    print(f"Linked-hazard papers: {len(linked)}")

    # Parse coordinates
    if 'geo_regions_json' in linked.columns:
        lats, lons = [], []
        for _, row in linked.iterrows():
            lat, lon = parse_geo_regions(row['geo_regions_json'])
            lats.append(lat)
            lons.append(lon)
        linked['lat'] = lats
        linked['lon'] = lons
        valid_coords = linked['lat'].notna().sum()
        print(f"Papers with valid coordinates: {valid_coords}")

    return df, linked


def load_wui_urban_data(wui_dir):
    """Load WUI and urban fraction rasters."""
    wui_path = list(wui_dir.glob('*WUI*.tif')) + list(wui_dir.glob('*wui*.tif'))
    urban_path = list(wui_dir.glob('*urban*.tif')) + list(wui_dir.glob('*Urban*.tif'))

    wui_fraction, urban_fraction = None, None
    wui_data, wui_bounds, wui_shape, wui_transform = None, None, None, None

    if wui_path:
        with rasterio.open(wui_path[0]) as src:
            wui_fraction = src.read(1).astype(np.float32)
            wui_bounds = src.bounds
            wui_shape = wui_fraction.shape
            wui_transform = src.transform
            wui_data = wui_fraction.copy()
            print(f"WUI data loaded: {wui_shape}")

    if urban_path:
        with rasterio.open(urban_path[0]) as src:
            urban_fraction = src.read(1).astype(np.float32)
            print(f"Urban data loaded: {urban_fraction.shape}")

    return wui_fraction, urban_fraction, wui_data, wui_bounds, wui_shape, wui_transform


def classify_setting_3class(urban_frac, wui_frac):
    """Classify into 3 setting classes."""
    S = np.where(np.isfinite(urban_frac), urban_frac, np.nan)
    S = np.where(np.isfinite(wui_frac), np.maximum(S, wui_frac * 0.5), S)

    cls = np.zeros(S.shape, dtype=np.uint8)
    cls[S >= 0.5] = 3              # Urban
    cls[(S >= 0.1) & (S < 0.5)] = 2  # WUI Belt
    cls[(S >= 0.0) & (S < 0.1)] = 1  # Wildland
    cls[np.isnan(S)] = 0           # NoData/Ocean

    return cls, S


print("Data loading functions defined!")

# %%
# =============================================================================
# LOAD ALL DATA
# =============================================================================

print("="*60)
print("LOADING DATA")
print("="*60)

# Load literature data
print("\n--- Loading Literature Data ---")
if os.path.exists(LITERATURE_PATH):
    df_all, linked = load_literature_data(LITERATURE_PATH)
else:
    print("Literature data not found - creating dummy data")
    linked = pd.DataFrame()

# Load WUI data
print("\n--- Loading WUI/Urban Data ---")
if WUI_DIR.exists():
    wui_fraction, urban_fraction, wui_data, wui_bounds, wui_shape, wui_transform = load_wui_urban_data(WUI_DIR)
    if wui_fraction is not None and urban_fraction is not None: # Added check for urban_fraction
        setting_class_3, setting_score = classify_setting_3class(urban_fraction, wui_fraction)
        print(f"Setting classes: {setting_class_3.shape}")
    else:
        print("Warning: WUI or Urban data not fully loaded, skipping setting classification.")
        setting_class_3, setting_score = None, None
else:
    print("WUI data not found!")
    wui_fraction, urban_fraction = None, None
    setting_class_3, setting_score = None, None

# Load hotspot data
print("\n--- Loading Hotspot Data ---")
if os.path.exists(DATA_YEARLY_DIR):
    compounds, fires = load_yearly_data(DATA_YEARLY_DIR)
    print(f"Compound types available: {list(compounds.keys())}")
    print(f"Fire years: {sorted(fires.keys())[:5]}...{sorted(fires.keys())[-5:]}")
else:
    print("Yearly data not found!")
    compounds, fires = {}, {}

print("\n" + "="*60)
print("DATA LOADING COMPLETE")
print("="*60)

# %% [markdown]
# ### 6b. Build hotspot points and wildfire base layer

# %%
# =============================================================================
# BUILD HOTSPOT POINTS AND WILDFIRE RGBA
# =============================================================================

# Settings
SINGLE_HAZARDS_TO_PLOT = ["HOT", "DRY", "WINDY", "FLOOD"]
COMPOUND_PREFERRED_ORDER = ["HOT_DRY", "DRY_WINDY", "HOT_WINDY", "HOT_DRY_WINDY", "WINDY_FLOOD"]

# Opacity controls
OPACITY_WILDFIRE = 0.6
OPACITY_SINGLES = 0.5
OPACITY_COMPOUND = 0.6

# Size scaling
R_MIN, R_MAX = 2.0, 10.0
R2_MIN, R2_MAX = 6.0, 28.0
P_SINGLE = 0.55
P_COMP = 0.60

# Initialize
df_single = pd.DataFrame(columns=["hazard", "lon", "lat", "freq", "s"])
df_comp = pd.DataFrame(columns=["hazard", "lon", "lat", "freq", "s"])
wf_rgba = None
extent = None

if fires:
    fire_years = sorted(fires.keys())
    fire_sum, fire_transform, fire_bounds = _sum_rasters(fires, fire_years)
    H, W = fire_sum.shape
    lon2d, lat2d, lons_1d, lats_1d = _grid_centers(fire_transform, H, W)
    extent = [fire_bounds.left, fire_bounds.right, fire_bounds.bottom, fire_bounds.top]

    # Build wildfire RGBA
    fire_mask = (fire_sum > 0) & np.isfinite(fire_sum)
    fire_log = np.log1p(fire_sum).astype(np.float32)
    vals = fire_log[fire_mask]

    if vals.size > 0:
        vmin = np.percentile(vals, 5.0)
        vmax = np.percentile(vals, 99.5)
    else:
        vmin, vmax = 0.0, 1.0

    den = (vmax - vmin) if vmax > vmin else 1.0
    norm = np.zeros_like(fire_log, dtype=np.float32)
    norm[fire_mask] = np.clip((fire_log[fire_mask] - vmin) / den, 0, 1)

    inten = np.zeros_like(norm, dtype=np.float32)
    inten[fire_mask] = norm[fire_mask] ** 0.70

    rgb_min = _hex_to_rgb01(WF_RGB_MIN_HEX)
    rgb_max = _hex_to_rgb01(WF_RGB_MAX_HEX)

    rgb = (rgb_min[None, None, :] * (1 - inten[..., None]) +
           rgb_max[None, None, :] * inten[..., None])

    alpha_map = np.zeros_like(norm, dtype=np.float32)
    alpha_map[fire_mask] = 0.85 + (0.98 - 0.85) * inten[fire_mask]
    alpha_map *= OPACITY_WILDFIRE
    alpha_map = np.clip(alpha_map, 0, 1)

    wf_rgba = np.zeros((H, W, 4), dtype=np.float32)
    wf_rgba[..., :3] = rgb
    wf_rgba[..., 3] = alpha_map
    print(f"Wildfire layer created: {wf_rgba.shape}")

    # Build single hazard points
    single_pts = []
    for hz in SINGLE_HAZARDS_TO_PLOT:
        hz_upper = hz.upper()
        if hz_upper not in compounds:
            continue
        years = sorted(set(compounds[hz_upper].keys()) & set(fires.keys()))
        if not years:
            continue
        hz_sum, _, _ = _sum_rasters(compounds[hz_upper], years)
        df = _build_hotspot_points(hz_sum, lon2d, lat2d, hz_upper,
                                   pixel_p=98.5, agg_deg=1.0, keep_top_p=60.0, max_points=20000)
        df = _sizes_from_freq(df, R_MIN, R_MAX, P_SINGLE)
        single_pts.append(df)

    if single_pts:
        df_single = pd.concat(single_pts, ignore_index=True)
    print(f"Single hazard points: {len(df_single)}")

    # Build compound hazard points
    comp_pts = []
    for ct in COMPOUND_PREFERRED_ORDER:
        if ct not in compounds:
            continue
        years = sorted(set(compounds[ct].keys()) & set(fires.keys()))
        if not years:
            continue
        ct_sum, _, _ = _sum_rasters(compounds[ct], years)
        df = _build_hotspot_points(ct_sum, lon2d, lat2d, ct,
                                   pixel_p=99.0, agg_deg=2.0, keep_top_p=65.0, max_points=9000)
        df = _sizes_from_freq(df, R2_MIN, R2_MAX, P_COMP)
        comp_pts.append(df)

    if comp_pts:
        df_comp = pd.concat(comp_pts, ignore_index=True)
    print(f"Compound hazard points: {len(df_comp)}")

print("Hotspot data prepared!")

# %% [markdown]
# ## 7. Data loading functions (Figure 5 pipeline)
#
# Loads literature, extracts coordinates, detects equity dimensions, samples
# GRDI at study points, and builds the fire/GRDI/research grid for mismatch
# analysis.

# %% [markdown]
# ### 7b. Areal-weighted research-attention surface (NEW, addresses reviewer comment on geocoding transparency)
#
# Replaces the primary-centroid literature count from cell 22 with **simple
# areal weighting** following Goodchild & Lam (1980), refined dasymetrically
# with a **binary land mask** (Eicher & Brewer, 2001; Mennis, 2003).
#
# **Protocol.** For each paper `p` with `K_p` geoparsed regions:
#
# 1. Each region receives total weight `1 / K_p`.
# 2. Within a region's bounding box, the region weight is distributed across
#    overlapping 2° cells proportional to the cos(latitude)-corrected area of
#    intersection, restricted to land cells.
# 3. Every paper contributes total weight 1.0 across the grid (the
#    volume-preserving property of simple areal weighting).
#
# **Examples.** A single-site study with a small MOB contributes weight ≈ 1.0 to
# one cell. A state-scale study (California, ~15 land cells) contributes ~0.07
# per cell. A global-scope paper covering all land (~6000 cells) contributes
# ~0.00017 per cell and cannot meaningfully shift any single cell's attention
# score — the intended behavior for weak spatial evidence.
#
# The original centroid-based counts are preserved as `n_papers_centroid` and
# `n_equity_centroid` for the sensitivity analysis in SM Section C5.1a.
#
# **References.**
# Goodchild, M. F., & Lam, N. S. N. (1980). Areal interpolation: A variant of
# the traditional spatial problem. *Geo-Processing*, 1, 297–312.
# Eicher, C. L., & Brewer, C. A. (2001). Dasymetric mapping and areal
# interpolation: Implementation and evaluation. *Cartography and Geographic
# Information Science*, 28(2), 125–138.
# Mennis, J. (2003). Generating surface models of population using dasymetric
# mapping. *The Professional Geographer*, 55(1), 31–42.
# Comber, A., & Zeng, W. (2019). Spatial interpolation using areal features:
# A review of methods and opportunities. *Geography Compass*, 13(10), e12465.

# %% [markdown]
# ### 7a. Build SREX assignment on the mismatch grid (unified)
#
# Replaces the original mismatch notebook's rectangular-box SREX assignment with
# the unified polygon-based one from Section 4.

# %% [markdown]
# ## 8. Figure 4 — Panel b: Global RR violin + top/bottom forest panels
#
# Uses the original RR-notebook forest-plot cell that produces nine hazard panels
# (top-8 / bottom-2 regions per hazard) and the global RR violin at the bottom.
# Required for panel b of Figure 4 and to populate `global_rr_baselines`.

# %%
# =============================================================================
# FIGURE 4 (FULL CELL) — wider left/right spacing + all tunable params at top
# - 4 panels per row (no panel i: keep first 8 hazards)
# - SREX labels -> 3-letter codes
# - RR text OUTSIDE axes (right side)
# - Slightly larger horizontal spacing between panels
# - Violin lighter + prominent scatter
# =============================================================================

import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------------------------
# ✅ TUNABLE PARAMETERS (adjust here)
# -----------------------------------------------------------------------------
# A) Compound exposure threshold (percentile on multi-year sum)
# ═══════════════════════════════════════════════════════════════════
# Fire is always binary (>0). This controls the compound threshold.
# 75 = upper quartile of extreme-day frequency (recommended)
# 50 = above-median
# 0  = >0 presence/absence (not recommended for single hazards)
COMPOUND_PERCENTILE = 75
# ═══════════════════════════════════════════════════════════════════

# B) What hazards to draw (in order)
REQUESTED_COMPOUNDS = [
    "HOT", "DRY", "WINDY", "FLOOD",
    "HOT_DRY", "DRY_WINDY", "HOT_WINDY", "HOT_DRY_WINDY", "WINDY_FLOOD"
]

# C) Selection rule per hazard
TOP_HIGH = 8                 # top N highest RR SREX regions
TOP_LOW  = 2                 # top N lowest RR SREX regions

# D) Layout
NCOLS = 4                    # panels per row
KEEP_N_PANELS = 9            # ✅ keep first N hazards (8 => no panel i)
INCLUDE_VIOLIN = True
FIG_W = 20                   # figure width
ROW_H = 4.2                  # height per row of panels
VIOLIN_H = 4.4               # extra height budget for violin row
HSPACE = 0.62                # vertical spacing between rows
WSPACE = 0.7                # ✅ horizontal spacing between panels (slightly larger)
RIGHT_MARGIN = 0.84          # ✅ shrink plotting area to leave room for RR text outside
BOTTOM_LEGEND_Y = 0.01       # legend anchor y

# E) Forest plot appearance
XLIM = (0.05, 200)
YLAB_FONT = 11
TITLE_FONT = 14
AX_LABEL_FONT = 12
RR_TEXT_FONT = 8.5
RR_TEXT_X = 1.06             # ✅ RR text outside position (axes coord); increase if still tight
POINT_SIZE = 7.5
CI_LW_SIG = 2.2
CI_LW_NS = 1.3
ALPHA_SIG = 0.95
ALPHA_NS = 0.30

# F) Violin appearance
VIOLIN_ALPHA = 0.20          # ✅ more transparent base
MEDIAN_LW = 2.2
SCATTER_MAX_PER_TYPE = 800
JITTER = 0.16
SCATTER_SIZE = 14
SCATTER_ALPHA = 0.45         # ✅ points more visible

# G) Output
SAVE_FIG = True
OUT_PREFIX = "Fig4_forest_4col_violin_noPanelI_srex3letter_widerSpacing"
# -----------------------------------------------------------------------------

# -----------------------------
# Required objects from earlier cells
# -----------------------------
for _name in ["compute_regional_relative_risk", "load_yearly_data", "DATA_YEARLY_DIR", "OUTPUT_DIR"]:
    if _name not in globals():
        raise NameError(f"Missing `{_name}`. Please run earlier cells first.")

# -----------------------------
# Palette unified with Global Hotspot Map v5
# -----------------------------
_default_single_colors = {
    "HOT":   "#7b3294",
    "DRY":   "#fdae61",
    "WINDY": "#66c2a5",
    "FLOOD": "#3288bd",
}
_default_compound_colors = {
    "HOT_DRY": "#984ea3",
    "DRY_WINDY": "#ff7f00",
    "HOT_WINDY": "#4daf4a",
    "HOT_DRY_WINDY": "#a65628",
    "WINDY_FLOOD": "#377eb8",
}

hazard_palette = {}
hazard_palette.update(globals().get("single_colors", _default_single_colors))
hazard_palette.update(globals().get("COMPOUND_COLORS", _default_compound_colors))
DEFAULT_COLOR = "#333333"

# -----------------------------
# Helpers
# -----------------------------
def _normalize_compound_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9_]+", "_", str(s).strip().upper()).strip("_")

def _resolve_compound_type(requested: str, available: list[str]) -> str | None:
    req = _normalize_compound_name(requested)
    if req in available:
        return req
    req_tokens = [t for t in req.split("_") if t]
    req_key = "_".join(sorted(req_tokens))
    avail_map = {}
    for a in available:
        a_norm = _normalize_compound_name(a)
        a_key = "_".join(sorted([t for t in a_norm.split("_") if t]))
        avail_map.setdefault(a_key, a)
    return avail_map.get(req_key)

# 3-letter code (same logic as your hotspot map)
def _srex_code(name):
    """Use canonical SREX abbreviation from global mapping (defined in helpers cell)."""
    if not isinstance(name, str) or not name.strip():
        return "SRE"
    t = name.strip()
    # Use the canonical SREX_NAME_TO_ABBREV mapping
    if 'SREX_NAME_TO_ABBREV' in globals() and t in SREX_NAME_TO_ABBREV:
        return SREX_NAME_TO_ABBREV[t]
    if 'srex_abbrev' in globals():
        return srex_abbrev(t)
    # If already a 3-letter code, return as-is
    if len(t) == 3 and t.isalpha():
        return t.upper()
    # Fallback
    letters = "".join([c for c in t if c.isalpha()])
    return (letters[:3] if len(letters) >= 3 else (letters + "XXX")[:3]).upper()

def _select_top_regions(df_rr: pd.DataFrame, top_high=8, top_low=2):
    df_valid = df_rr[df_rr["RR"].notna() & (df_rr["RR"] > 0)].copy()

    global_rr = None
    global_ci = None
    if (df_valid["region"] == "GLOBAL").any():
        g = df_valid[df_valid["region"] == "GLOBAL"].iloc[0]
        if pd.notna(g["RR"]) and g["RR"] > 0:
            global_rr = float(g["RR"])
        if pd.notna(g.get("CI_lower")) and pd.notna(g.get("CI_upper")):
            global_ci = (float(g["CI_lower"]), float(g["CI_upper"]))

    df_reg = df_valid[df_valid["region"] != "GLOBAL"].copy()
    if df_reg.empty:
        return None, global_rr, global_ci, df_valid

    n_low = min(top_low, len(df_reg))
    n_high = min(top_high, len(df_reg))

    df_low = df_reg.nsmallest(n_low, "RR")
    df_high = df_reg.nlargest(n_high, "RR")

    df_plot = (
        pd.concat([df_low, df_high], ignore_index=True)
          .drop_duplicates(subset=["region"], keep="first")
          .sort_values("RR", ascending=True)
          .reset_index(drop=True)
    )
    return df_plot, global_rr, global_ci, df_valid

# -----------------------------
# Forest panel (RR text outside)
# -----------------------------
def _draw_forest_panel(ax, df_plot: pd.DataFrame, hazard_key: str, global_rr: float | None,
                      xlim=XLIM, show_ylabel=True, panel_letter=None):
    base_color = hazard_palette.get(hazard_key, DEFAULT_COLOR)
    ylabels = [_srex_code(r) for r in df_plot["region"].tolist()]

    for i, (_, row) in enumerate(df_plot.iterrows()):
        rr = float(row["RR"])
        ci_lo = row["CI_lower"] if pd.notna(row["CI_lower"]) else rr * 0.5
        ci_hi = row["CI_upper"] if pd.notna(row["CI_upper"]) else rr * 2
        ci_lo = max(xlim[0], float(ci_lo))
        ci_hi = min(xlim[1], float(ci_hi))

        sig = bool(row.get("significant", False))
        alpha = ALPHA_SIG if sig else ALPHA_NS
        edge = "black" if sig else "gray"
        lw = CI_LW_SIG if sig else CI_LW_NS

        ax.plot([ci_lo, ci_hi], [i, i], color=base_color, linewidth=lw, alpha=alpha, zorder=2)
        ax.plot(rr, i, "o", markersize=POINT_SIZE,
                markerfacecolor=base_color, markeredgecolor=edge,
                markeredgewidth=0.7, alpha=alpha, zorder=3)

        if pd.notna(row.get("CI_lower")) and pd.notna(row.get("CI_upper")):
            txt = f"{row['RR']:.2f} ({row['CI_lower']:.2f}–{row['CI_upper']:.2f})"
        else:
            txt = f"{row['RR']:.2f}"
        if sig:
            txt += " *"

        # ✅ outside axes
        ax.text(
            RR_TEXT_X, i, txt,
            transform=ax.get_yaxis_transform(),  # x=axes coords, y=data coords
            ha="left", va="center",
            fontsize=RR_TEXT_FONT,
            clip_on=False,
            zorder=10
        )

    ax.axvline(1, color="black", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    if global_rr is not None and global_rr > 0:
        ax.axvline(global_rr, color=base_color, linestyle="-.", linewidth=2.3, alpha=0.95, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.invert_yaxis()

    ax.set_yticks(np.arange(len(df_plot)))
    ax.set_yticklabels(ylabels, fontsize=YLAB_FONT)
    ax.tick_params(axis="y", pad=4)

    if show_ylabel:
        ax.set_ylabel("SREX region", fontsize=AX_LABEL_FONT)
    else:
        ax.set_ylabel("")

    ax.set_xlabel("RR (log scale)", fontsize=AX_LABEL_FONT)
    ax.grid(True, axis="x", alpha=0.22)
    ax.set_axisbelow(True)

    ax.set_title(hazard_key.replace("_", " + "), fontsize=TITLE_FONT, fontweight="bold")

    if panel_letter:
        ax.text(-0.14, 1.03, panel_letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="left")

# -----------------------------
# Violin (lighter) + stronger points
# -----------------------------
def _draw_violin(ax, rr_full: dict, global_refs: dict, hazard_list: list[str]):
    rng = np.random.default_rng(42)

    data_log = []
    scatter_x = []
    scatter_y = []
    global_y = []
    labels = []
    colors = []

    for i, hz in enumerate(hazard_list, start=1):
        df_all = rr_full[hz]
        df_reg = df_all[(df_all["region"] != "GLOBAL") & (df_all["RR"].notna()) & (df_all["RR"] > 0)].copy()
        vals = df_reg["RR"].astype(float).values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            vals = np.array([np.nan])

        v_log = np.log10(vals)
        data_log.append(v_log)

        v_s = v_log
        if len(v_s) > SCATTER_MAX_PER_TYPE:
            idx = rng.choice(len(v_s), size=SCATTER_MAX_PER_TYPE, replace=False)
            v_s = v_s[idx]

        x_j = i + rng.uniform(-JITTER, JITTER, size=len(v_s))
        scatter_x.append(x_j)
        scatter_y.append(v_s)

        g_rr, _ = global_refs.get(hz, (None, None))
        global_y.append(np.log10(g_rr) if (g_rr is not None and g_rr > 0) else np.nan)

        labels.append(hz.replace("_", "\n"))
        colors.append(hazard_palette.get(hz, DEFAULT_COLOR))

    parts = ax.violinplot(data_log, showmeans=False, showmedians=True, showextrema=False)

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(VIOLIN_ALPHA)

    if "cmedians" in parts:
        parts["cmedians"].set_color("#1f77b4")
        parts["cmedians"].set_linewidth(MEDIAN_LW)
        parts["cmedians"].set_alpha(0.95)

    for i in range(len(hazard_list)):
        ax.scatter(scatter_x[i], scatter_y[i],
                   s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
                   color=colors[i], edgecolors="none", zorder=3)

    xs = np.arange(1, len(hazard_list)+1)
    ax.scatter(xs, global_y, s=52, color="black", marker="o", zorder=4, label="GLOBAL RR")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    rr_ticks = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.set_yticks(np.log10(rr_ticks))
    ax.set_yticklabels([str(t) for t in rr_ticks], fontsize=10)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_ylabel("RR distribution (log scale)", fontsize=12)
    ax.set_title("RR distribution across all SREX regions (violin + points) + GLOBAL reference",
                 fontsize=13, fontweight="bold", loc="left")

    ax.legend(loc="upper right", frameon=True)

# =============================================================================
# RUN
# =============================================================================
compounds, fires = load_yearly_data(DATA_YEARLY_DIR)
available_types = sorted(list(compounds.keys()))

# FIX: Build land mask from setting data to exclude SREX ocean pixels
# This requires setting_on_fire_grid from the setting analysis (cell 23).
# If not yet available, we build it here.
if 'setting_on_fire_grid' not in dir() or setting_on_fire_grid is None:
    print("Building land mask from WUI data...")
    _wui_tifs = sorted(WUI_DIR.glob('*.tif'))
    _total_wui_path = None
    for p in _wui_tifs:
        if 'total_wui_fraction' in p.name.lower():
            _total_wui_path = p
    if _total_wui_path is None:
        _total_wui_path = _wui_tifs[0]
    with rasterio.open(_total_wui_path) as src:
        _wui_frac = src.read(1).astype(np.float32)
        _wui_transform = src.transform
    _setting_tmp = np.zeros(_wui_frac.shape, dtype=np.uint8)
    _vm = np.isfinite(_wui_frac)
    _setting_tmp[_vm & (_wui_frac >= 0.5)] = 3
    _setting_tmp[_vm & (_wui_frac >= 0.1) & (_wui_frac < 0.5)] = 2
    _setting_tmp[_vm & (_wui_frac >= 0.0) & (_wui_frac < 0.1)] = 1
    # Resample to fire grid
    _any_year = sorted(fires.keys())[0]
    with rasterio.open(fires[_any_year]) as _src:
        _fire_transform = _src.transform
        _fire_shape = (_src.height, _src.width)
    from rasterio.warp import reproject, Resampling as RResampling
    setting_on_fire_grid = np.zeros(_fire_shape, dtype=np.uint8)
    reproject(
        source=_setting_tmp.astype(np.uint8), destination=setting_on_fire_grid,
        src_transform=_wui_transform, src_crs='EPSG:4326',
        dst_transform=_fire_transform, dst_crs='EPSG:4326',
        resampling=RResampling.nearest
    )
    print(f"  Land mask built: {int(np.sum(np.isin(setting_on_fire_grid, [1,2,3]))):,} classified pixels")

_land_mask = np.isin(setting_on_fire_grid, [1, 2, 3])
print(f"Using land mask with {int(_land_mask.sum()):,} pixels (excludes SREX ocean)")

resolved = []
for req in REQUESTED_COMPOUNDS:
    ct = _resolve_compound_type(req, available_types)
    if ct is None:
        print(f"⚠️  Requested '{req}' not found. Example available: {available_types[:12]} ...")
        continue
    if ct not in resolved:
        resolved.append(ct)

# no panel i: keep first N
if len(resolved) > KEEP_N_PANELS:
    removed = resolved[KEEP_N_PANELS:]
    resolved = resolved[:KEEP_N_PANELS]
    print(f"Dropped to keep first {KEEP_N_PANELS} panels:", removed)

print("Final panels:", resolved)

selections = {}
global_refs = {}
rr_full = {}

for ct in resolved:
    df_rr = compute_regional_relative_risk(compounds, fires, ct, compound_pct=COMPOUND_PERCENTILE, land_mask=_land_mask)
    if df_rr is None or df_rr.empty:
        continue
    df_plot, g_rr, g_ci, df_valid = _select_top_regions(df_rr, top_high=TOP_HIGH, top_low=TOP_LOW)
    if df_plot is None or df_plot.empty:
        continue
    selections[ct] = df_plot
    global_refs[ct] = (g_rr, g_ci)
    rr_full[ct] = df_valid

hazard_list = list(selections.keys())
if not hazard_list:
    raise RuntimeError("No hazards produced RR selections. Check SREX mask / input rasters.")

n = len(hazard_list)
nrows = math.ceil(n / NCOLS)

letters_top = list("abcdefghi")[:n]  # up to 9 panels (includes Windy+Flood)
violin_letter = "j"

if INCLUDE_VIOLIN:
    fig = plt.figure(figsize=(FIG_W, ROW_H*nrows + VIOLIN_H))
    gs = GridSpec(nrows + 1, NCOLS, figure=fig,
                  height_ratios=[1]*nrows + [1.2],
                  hspace=HSPACE, wspace=WSPACE)
else:
    fig = plt.figure(figsize=(FIG_W, ROW_H*nrows))
    gs = GridSpec(nrows, NCOLS, figure=fig, hspace=HSPACE, wspace=WSPACE)

# top panels
for idx, ct in enumerate(hazard_list):
    r = idx // NCOLS
    c = idx % NCOLS
    ax = fig.add_subplot(gs[r, c])

    df_plot = selections[ct]
    g_rr, _ = global_refs.get(ct, (None, None))

    _draw_forest_panel(
        ax, df_plot, ct, g_rr,
        xlim=XLIM,
        show_ylabel=(c == 0),
        panel_letter=letters_top[idx] if idx < len(letters_top) else None
    )

# unused slots off
total_slots = nrows * NCOLS
for j in range(n, total_slots):
    r = j // NCOLS
    c = j % NCOLS
    ax_empty = fig.add_subplot(gs[r, c])
    ax_empty.axis("off")

# violin
if INCLUDE_VIOLIN:
    axv = fig.add_subplot(gs[nrows, :])
    _draw_violin(axv, rr_full, global_refs, hazard_list)
    axv.text(-0.02, 1.03, violin_letter, transform=axv.transAxes,
             fontsize=13, fontweight="bold", va="bottom", ha="left")

# shared legend
sig_handle = Line2D([0], [0], marker="o", linestyle="None",
                    markerfacecolor="#777777", markeredgecolor="black",
                    markeredgewidth=0.9, markersize=7, label="Significant (CI excludes 1)  *")
ns_handle = Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor="#777777", markeredgecolor="gray",
                   markeredgewidth=0.8, markersize=7, alpha=0.30, label="Not significant")
rr1_handle = Line2D([0], [0], color="black", linestyle="--", label="RR = 1")
glob_handle = Line2D([0], [0], color="black", linestyle="-.", label="GLOBAL RR (vertical line)")

fig.legend(handles=[sig_handle, ns_handle, rr1_handle, glob_handle],
           loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, BOTTOM_LEGEND_Y))

# ✅ leave room on the right for outside RR text + slightly wider left/right spacing
plt.tight_layout(rect=[0, 0.05, RIGHT_MARGIN, 1])
fig.subplots_adjust(right=RIGHT_MARGIN, wspace=WSPACE, hspace=HSPACE)

if SAVE_FIG:
    out_base = os.path.join(OUTPUT_DIR, OUT_PREFIX)
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".pdf", bbox_inches="tight", facecolor="white")
    print("Saved:", out_base + ".png")
    print("Saved:", out_base + ".pdf")

plt.show()

# %% [markdown]
# ## 9. Figure 4 — Panel a: Global Hotspot Map (bolder colors + distinct markers)
#
# Addresses the reviewer comment that colors on the global RR map are hard to
# differentiate. Singles use hollow circles in saturated hues; 2-way compounds use
# filled squares; the 3-way compound uses a filled triangle; windy+flood uses a
# filled diamond. Each category is visually unambiguous even in grayscale print.

# %%
# =============================================================================
# Shared hotspot styling constants.
# (Originally the 'Version B' Fig 4a hotspot cell. Its rendering was removed
#  because that figure is not used; the SM regional-thresholds figure reuses
#  these colour / colormap / label / SREX-label-position constants.)
# =============================================================================


FIG_W, FIG_H = 22, 10
LAND_COLOR  = "#f5f3ef"
OCEAN_COLOR = "#ffffff"

WF_RGB_MIN_HEX = "#5a0012"
WF_RGB_MAX_HEX = "#d4002a"
WF_CMAP  = LinearSegmentedColormap.from_list("wf_orig", [WF_RGB_MIN_HEX, WF_RGB_MAX_HEX])
WF_ALPHA_MAX = 0.85

SINGLE_COLORS_ORIG = {
    "HOT":   "#7b3294",
    "DRY":   "#fdae61",
    "WINDY": "#66c2a5",
    "FLOOD": "#3288bd",
}

COMPOUND_COLORS_BY_SHAPE = {
    "HOT_DRY":       "#984ea3",
    "DRY_WINDY":     "#ff7f00",
    "HOT_WINDY":     "#4daf4a",
    "HOT_DRY_WINDY": "#a65628",
    "WINDY_FLOOD":   "#377eb8",
}

COMPOUND_MARKERS = {
    "HOT_DRY":       "s",
    "DRY_WINDY":     "s",
    "HOT_WINDY":     "s",
    "WINDY_FLOOD":   "D",
    "HOT_DRY_WINDY": "^",
}

LABELS = {
    "HOT": "Hot (H)", "DRY": "Dry (D)", "WINDY": "Windy (W)", "FLOOD": "Flood (F)",
    "HOT_DRY": "H & D", "DRY_WINDY": "D & W", "HOT_WINDY": "H & W",
    "WINDY_FLOOD": "W & F", "HOT_DRY_WINDY": "H & D & W",
}

# --- Sizing and transparency ---------------------------------------------
SINGLE_SIZE_FACTOR   = 1.00
SINGLE_ALPHA         = 0.50
SINGLE_LW            = 1.0

COMPOUND_SIZE_FACTOR = 0.85       # was 0.85; a touch smaller
COMPOUND_ALPHA       = 0.35

# --- All 26 SREX label positions ----------------------------------------
# Ocean anchors where possible; land-locked regions placed in low-activity
# corners to avoid the heavy fire/compound clusters. Tune any single entry
# by editing its (lon, lat) tuple.
SREX_LABEL_POSITIONS = {
    # --- ocean anchors ---
    "CGI": (-45,   58),   # north of Greenland
    "WNA": (-115,  43),   # Pacific off Oregon
    "CAM": (-100,   8),   # Pacific south of Mexico
    "AMZ": (-45,    5),   # Atlantic off Amazon mouth
    "NEB": (-30,  -10),   # Atlantic east of NE Brazil
    "WSA": (-76,  -30),   # Pacific off Peru
    "SSA": (-47,  -45),   # Atlantic off Argentina
    "WAF": (-10,   -9),   # Gulf of Guinea
    "EAF": (45,     -9),   # Indian Ocean off Somalia
    "SAF": (40,   -30),   # Indian Ocean off Mozambique
    "SAS": (85,    10),   # Bay of Bengal
    "EAS": (132,   40),   # Sea of Japan
    "SEA": (135,   10),   # Banda Sea
    "NAU": (115,   -13),   # Arafura Sea
    "SAU": (120,  -45),   # Tasman Sea
    # --- overlap fixes (moved to clearer water) ---
    "SAH": (2,     20),   # Mediterranean south of Tunisia (away from H&D cluster)
    "MED": (15,    40),   # inland Italy/Tyrrhenian edge (clear of symbols)
    "WAS": (68,    40),   # Persian Gulf (away from H&D cluster in Iran)
    # --- land-locked regions: placed in low-activity corners ---
    "ALA": (-152,  62),   # interior Alaska, south of Brooks Range
    "CNA": (-95,  43),   # N. Great Plains (sparse area)
    "ENA": (-72,   43),   # Great Lakes area (between fire clusters)
    "NEU": (15,    63),   # northern Finland (low fire density)
    "CEU": (15,    50),   # central Poland/Germany
    "NAS": (100,   58),   # central Siberian Lena basin, north corner
    "CAS": (50,    40),   # Aral Sea area (low density corner of CAS)
    "TIB": (88,    40),   # central Tibetan Plateau (interior clear of fire)
}


# %%
# =============================================================================
# FIGURE: Global Hotspots over Wildfire Base Layer (NO halo)
# - Wildfire: ALL fire>0 pixels shown; deep red with mild intensity variation
# - Weakest wildfire NOT too weak (alpha floor + low-end boost)
# - Transparent controls for EVERY layer (wildfire / singles / compounds / SREX / coast / land/ocean)
# REVISION: compound markers now use different SHAPES per hazard
#           (H&D, D&W, H&W = squares; H&D&W = triangle; W&F = diamond).
#           All colors, sizes, transparencies unchanged.
# =============================================================================

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# =============================================================================
# ✅ USER SETTINGS (YOU CAN TUNE THESE)
# =============================================================================

# -----------------------
# A) Global opacity knobs (apply to whole layer)
# -----------------------
OPACITY_WILDFIRE = 0.7
OPACITY_SINGLES  = 0.5
OPACITY_COMPOUND = 0.7
OPACITY_SREX     = 0.60
OPACITY_COAST    = 0.90
OPACITY_LAND     = 1.00
OPACITY_OCEAN    = 1.00

# -----------------------
# B) Wildfire appearance (deep red + mild variation, NO halo)
# -----------------------
WF_NORM_P_LO = 5.0
WF_NORM_P_HI = 99.5
WF_INT_GAMMA = 0.70
WF_ALPHA_MIN = 0.85
WF_ALPHA_MAX = 0.98
WF_RGB_MIN_HEX = "#5a0012"
WF_RGB_MAX_HEX = "#d4002a"
WF_COLOR_FLOOR = 0.12

# -----------------------
# C) Hotspot extraction (circles count & selection)
# -----------------------
SINGLE_HAZARDS_TO_PLOT = ["HOT", "DRY", "WINDY", "FLOOD"]
SINGLE_PIXEL_HOTSPOT_P = 98.5
SINGLE_AGG_DEG         = 1.0
SINGLE_KEEP_TOP_P      = 60.0
SINGLE_MAX_POINTS      = 20000

COMPOUND_PREFERRED_ORDER = ["HOT_DRY", "DRY_WINDY", "HOT_WINDY", "HOT_DRY_WINDY", "WINDY_FLOOD"]
COMP_PIXEL_HOTSPOT_P     = 99.0
COMP_AGG_DEG             = 2.0
COMP_KEEP_TOP_P          = 65.0
COMP_MAX_POINTS          = 9000

R_MIN, R_MAX   = 2.0, 10.0
R2_MIN, R2_MAX = 6.0, 28.0
P_SINGLE       = 0.55
P_COMP         = 0.60

EDGE_ALPHA = 0.70
EDGE_LW    = 1.0
FILL_ALPHA = 0.45

# -----------------------
# D) SREX boundaries + labels
# -----------------------
DRAW_SREX_BOUNDARY = True
SREX_LW = 0.7
SREX_LABEL_FONT = 10
SREX_LABEL_MARGIN_DEG = 14.0

# -----------------------
# E) Plot layout
# -----------------------
LEG_Y = -0.20
LEG_FONT = 11

Z_COMPOUND = 15
Z_SINGLE   = 25
Z_WF_IMAGE = 60
Z_COAST    = 80

OUT_PNG = os.path.join(OUTPUT_DIR, "Fig4a_Hotspot_v11_compoundShapes.png")
OUT_PDF = OUT_PNG.replace(".png", ".pdf")

# =============================================================================
# COLORS (Only wildfire uses red; HOT is purple)
# =============================================================================
single_colors = {
    "HOT":   "#7b3294",
    "DRY":   "#fdae61",
    "WINDY": "#66c2a5",
    "FLOOD": "#3288bd",
}
DEFAULT_COLOR = "#333333"

COMPOUND_COLORS_DISTINCT = {
    "HOT_DRY":       "#984ea3",
    "DRY_WINDY":     "#ff7f00",
    "HOT_WINDY":     "#4daf4a",
    "HOT_DRY_WINDY": "#a65628",
    "WINDY_FLOOD":   "#377eb8",
}
if "COMPOUND_COLORS" in globals():
    COMPOUND_COLORS = dict(COMPOUND_COLORS)
    COMPOUND_COLORS.update(COMPOUND_COLORS_DISTINCT)
else:
    COMPOUND_COLORS = dict(COMPOUND_COLORS_DISTINCT)

# --- NEW: per-compound marker shapes -----------------------------------------
# Squares for 2-way, triangle for 3-way, diamond for windy+flood.
COMPOUND_MARKERS = {
    "HOT_DRY":       "s",
    "DRY_WINDY":     "s",
    "HOT_WINDY":     "s",
    "HOT_DRY_WINDY": "^",
    "WINDY_FLOOD":   "D",
}

# =============================================================================
# HELPERS (unchanged)
# =============================================================================
def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0

def _sum_rasters(file_dict, years):
    with rasterio.open(file_dict[years[0]]) as src:
        transform = src.transform
        bounds = src.bounds
        height, width = src.height, src.width
    s = np.zeros((height, width), dtype=np.float32)
    for y in years:
        with rasterio.open(file_dict[y]) as src:
            a = src.read(1).astype(np.float32)
        a = np.where(np.isfinite(a), a, 0.0)
        s += a
    return s, transform, bounds

def _grid_centers(transform, height, width):
    lon0 = transform.c + transform.a / 2
    lat0 = transform.f + transform.e / 2
    lons = lon0 + np.arange(width) * transform.a
    lats = lat0 + np.arange(height) * transform.e
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lon2d, lat2d, lons, lats

def _build_hotspot_points(sum_arr, lon2d, lat2d, hazard_label,
                          pixel_p=99.0, agg_deg=1.0, keep_top_p=70.0, max_points=10000):
    pos = sum_arr[sum_arr > 0]
    if len(pos) == 0:
        return pd.DataFrame(columns=["hazard", "lon", "lat", "freq"])
    thr = np.percentile(pos, pixel_p)
    m = (sum_arr >= thr) & (sum_arr > 0)
    df = pd.DataFrame({
        "hazard": hazard_label,
        "lon": lon2d[m].ravel(),
        "lat": lat2d[m].ravel(),
        "freq": sum_arr[m].ravel(),
    })
    df["lon_bin"] = np.round(df["lon"] / agg_deg) * agg_deg
    df["lat_bin"] = np.round(df["lat"] / agg_deg) * agg_deg
    df = (
        df.groupby(["hazard", "lon_bin", "lat_bin"], as_index=False)["freq"].sum()
          .rename(columns={"lon_bin": "lon", "lat_bin": "lat"})
    )
    if len(df) > 0:
        thr2 = np.percentile(df["freq"], keep_top_p)
        df = df[df["freq"] >= thr2].copy()
    if len(df) > max_points:
        df = df.nlargest(max_points, "freq").copy()
    return df

def _sizes_from_freq(df, rmin, rmax, p):
    if df.empty:
        return df
    fmax = df["freq"].max()
    fn = (df["freq"] / fmax) if fmax > 0 else 0
    radius = rmin + (np.power(fn, p)) * (rmax - rmin)
    df = df.copy()
    df["s"] = np.square(radius)
    return df

def _resolve_compound_order(available):
    av = set(available)
    picked = [k for k in COMPOUND_PREFERRED_ORDER if k in av]
    for k in available:
        if "_" in k and k not in picked:
            picked.append(k)
    return picked

def _srex_code(name):
    """Use canonical SREX abbreviation from the global mapping."""
    if isinstance(name, str):
        if 'srex_abbrev' in dir() or 'srex_abbrev' in globals():
            return srex_abbrev(name)
        if 'SREX_NAME_TO_ABBREV' in globals() and name in SREX_NAME_TO_ABBREV:
            return SREX_NAME_TO_ABBREV[name]
        t = name.strip()
        if len(t) == 3 and t.isalpha():
            return t.upper()
    return "SRE"

# =============================================================================
# LOAD DATA
# =============================================================================
compounds, fires = load_yearly_data(DATA_YEARLY_DIR)
available_keys = sorted(list(compounds.keys()))
fire_years = sorted(list(fires.keys()))
print("Fire years:", fire_years[:5], "...", fire_years[-5:])

# =============================================================================
# WILDFIRE RGBA (deep red, mild variation, NO halo) - unchanged
# =============================================================================
fire_sum, fire_transform, fire_bounds = _sum_rasters(fires, fire_years)

H, W = fire_sum.shape
lon2d, lat2d, lons_1d, lats_1d = _grid_centers(fire_transform, H, W)
extent = [fire_bounds.left, fire_bounds.right, fire_bounds.bottom, fire_bounds.top]

fire_mask = (fire_sum > 0) & np.isfinite(fire_sum)

fire_log = np.log1p(fire_sum).astype(np.float32)
vals = fire_log[fire_mask]
if vals.size > 0:
    vmin = np.percentile(vals, WF_NORM_P_LO)
    vmax = np.percentile(vals, WF_NORM_P_HI)
else:
    vmin, vmax = 0.0, 1.0

den = (vmax - vmin) if vmax > vmin else 1.0
norm = np.zeros_like(fire_log, dtype=np.float32)
norm[fire_mask] = np.clip((fire_log[fire_mask] - vmin) / den, 0, 1)

inten = np.zeros_like(norm, dtype=np.float32)
inten[fire_mask] = norm[fire_mask] ** WF_INT_GAMMA

if WF_COLOR_FLOOR > 0:
    inten = np.where(fire_mask, np.clip(inten + WF_COLOR_FLOOR * (1 - inten), 0, 1), inten)

rgb_min = _hex_to_rgb01(WF_RGB_MIN_HEX)
rgb_max = _hex_to_rgb01(WF_RGB_MAX_HEX)

rgb = (rgb_min[None, None, :] * (1 - inten[..., None]) +
       rgb_max[None, None, :] * inten[..., None])

alpha_map = np.zeros_like(norm, dtype=np.float32)
alpha_map[fire_mask] = WF_ALPHA_MIN + (WF_ALPHA_MAX - WF_ALPHA_MIN) * inten[fire_mask]
alpha_map *= OPACITY_WILDFIRE
alpha_map = np.clip(alpha_map, 0, 1)

wf_rgba = np.zeros((H, W, 4), dtype=np.float32)
wf_rgba[..., :3] = rgb
wf_rgba[..., 3]  = alpha_map

print(f"Wildfire: deep red w/ mild variation (no halo). "
      f"alpha_min={WF_ALPHA_MIN}, alpha_max={WF_ALPHA_MAX}, gamma={WF_INT_GAMMA}, "
      f"opacity={OPACITY_WILDFIRE}")

# =============================================================================
# BUILD HOTSPOT POINTS - unchanged
# =============================================================================
single_pts = []
for hz in SINGLE_HAZARDS_TO_PLOT:
    hz = hz.upper()
    if hz not in compounds:
        print(f"⚠️ missing single hazard: {hz}")
        continue
    years = sorted(set(compounds[hz].keys()) & set(fires.keys()))
    if not years:
        continue
    hz_sum, _, _ = _sum_rasters(compounds[hz], years)
    df = _build_hotspot_points(
        hz_sum, lon2d, lat2d, hz,
        pixel_p=SINGLE_PIXEL_HOTSPOT_P,
        agg_deg=SINGLE_AGG_DEG,
        keep_top_p=SINGLE_KEEP_TOP_P,
        max_points=SINGLE_MAX_POINTS
    )
    df = _sizes_from_freq(df, R_MIN, R_MAX, P_SINGLE)
    single_pts.append(df)

df_single = pd.concat(single_pts, ignore_index=True) if single_pts else pd.DataFrame(columns=["hazard","lon","lat","freq","s"])

compound_keys = _resolve_compound_order(available_keys)
comp_pts = []
for ct in compound_keys:
    if "_" not in ct:
        continue
    years = sorted(set(compounds[ct].keys()) & set(fires.keys()))
    if not years:
        continue
    ct_sum, _, _ = _sum_rasters(compounds[ct], years)
    df = _build_hotspot_points(
        ct_sum, lon2d, lat2d, ct,
        pixel_p=COMP_PIXEL_HOTSPOT_P,
        agg_deg=COMP_AGG_DEG,
        keep_top_p=COMP_KEEP_TOP_P,
        max_points=COMP_MAX_POINTS
    )
    df = _sizes_from_freq(df, R2_MIN, R2_MAX, P_COMP)
    comp_pts.append(df)

df_comp = pd.concat(comp_pts, ignore_index=True) if comp_pts else pd.DataFrame(columns=["hazard","lon","lat","freq","s"])

# =============================================================================
# PLOT
# =============================================================================
fig = plt.figure(figsize=(16, 8))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()

ax.add_feature(cfeature.LAND,  facecolor="white", edgecolor="none", alpha=OPACITY_LAND,  zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", alpha=OPACITY_OCEAN, zorder=0)

# --- SREX dashed boundaries + margin labels (3 letters) --- unchanged
if DRAW_SREX_BOUNDARY:
    try:
        srex_int, valid_region, region_names = get_srex_mask_and_names(lons_1d, lats_1d)
        if srex_int is not None and ("HAS_REGIONMASK" in globals()) and HAS_REGIONMASK:
            regs = np.unique(srex_int[np.isfinite(srex_int)])
            regs = regs[regs >= 0]
            print(f"✓ Using SREX regions (N={len(regs)})")

            for ridx in regs:
                reg = (srex_int == ridx).astype(np.uint8)
                ax.contour(
                    lons_1d, lats_1d, reg,
                    levels=[0.5],
                    colors="black",
                    linewidths=SREX_LW,
                    linestyles="--",
                    alpha=0.55 * OPACITY_SREX,
                    transform=ccrs.PlateCarree(),
                    zorder=5
                )

            for ridx in regs:
                reg = (srex_int == ridx).astype(np.uint8)
                ys, xs = np.where(reg == 1)
                if len(xs) < 80:
                    continue

                lon_c = float(np.mean(lons_1d[xs]))
                lat_c = float(np.mean(lats_1d[ys]))
                name = region_names.get(ridx, f"SREX_{ridx}")
                code = _srex_code(name)

                lon_lab = lon_c + (SREX_LABEL_MARGIN_DEG if lon_c >= 0 else -SREX_LABEL_MARGIN_DEG)
                lat_lab = lat_c + (SREX_LABEL_MARGIN_DEG * 0.55 if lat_c >= 0 else -SREX_LABEL_MARGIN_DEG * 0.55)
                lon_lab = max(-179, min(179, lon_lab))
                lat_lab = max(-85, min(85, lat_lab))

                ax.plot([lon_c, lon_lab], [lat_c, lat_lab],
                        transform=ccrs.PlateCarree(),
                        color="black", linewidth=0.6, alpha=0.45 * OPACITY_SREX,
                        zorder=6)

                txt = ax.text(lon_lab, lat_lab, code,
                              transform=ccrs.PlateCarree(),
                              fontsize=SREX_LABEL_FONT, color="black",
                              alpha=0.90 * OPACITY_SREX,
                              ha="center", va="center",
                              zorder=7)
                txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.9)])
        else:
            print("⚠️ SREX skipped (regionmask not available).")
    except Exception as e:
        print("⚠️ SREX skipped:", str(e))

# --- Compounds (filled) --- ONLY CHANGE: per-hazard marker shape -----------
for ct in sorted(df_comp["hazard"].unique()) if not df_comp.empty else []:
    sub = df_comp[df_comp["hazard"] == ct]
    ax.scatter(
        sub["lon"].values, sub["lat"].values,
        s=sub["s"].values,
        marker=COMPOUND_MARKERS.get(ct, "o"),      # <-- NEW: per-hazard shape
        facecolors=COMPOUND_COLORS.get(ct, DEFAULT_COLOR),
        edgecolors="none",
        alpha=np.clip(FILL_ALPHA * OPACITY_COMPOUND, 0, 1),
        transform=ccrs.PlateCarree(),
        zorder=Z_COMPOUND
    )

# --- Singles (hollow circles) --- unchanged
for hz in sorted(df_single["hazard"].unique()) if not df_single.empty else []:
    sub = df_single[df_single["hazard"] == hz]
    ax.scatter(
        sub["lon"].values, sub["lat"].values,
        s=sub["s"].values,
        facecolors="none",
        edgecolors=single_colors.get(hz, DEFAULT_COLOR),
        linewidths=EDGE_LW,
        alpha=np.clip(EDGE_ALPHA * OPACITY_SINGLES, 0, 1),
        transform=ccrs.PlateCarree(),
        zorder=Z_SINGLE
    )

# Wildfire raster (deep red, mild variation) - unchanged
ax.imshow(
    wf_rgba,
    origin="upper",
    extent=extent,
    transform=ccrs.PlateCarree(),
    zorder=Z_WF_IMAGE
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.4, alpha=np.clip(OPACITY_COAST, 0, 1), zorder=Z_COAST)

# =============================================================================
# LEGENDS — compound proxies now use their assigned shape
# =============================================================================
wf_handle = Line2D([0], [0], marker="s", linestyle="None",
                   markerfacecolor=WF_RGB_MAX_HEX, markeredgecolor="none",
                   markersize=10, alpha=0.95, label="Wildfire")

single_handles = []
for hz in sorted(df_single["hazard"].unique()) if not df_single.empty else []:
    c = single_colors.get(hz, DEFAULT_COLOR)
    single_handles.append(
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="none", markeredgecolor=c,
               markeredgewidth=1.6, markersize=8, label=hz)
    )

comp_show = [k for k in COMPOUND_PREFERRED_ORDER if (not df_comp.empty and k in df_comp["hazard"].unique())]
comp_handles = []
for ct in comp_show:
    c  = COMPOUND_COLORS.get(ct, DEFAULT_COLOR)
    mk = COMPOUND_MARKERS.get(ct, "o")             # <-- NEW: shape in legend too
    comp_handles.append(
        Line2D([0], [0], marker=mk, linestyle="None",
               markerfacecolor=c, markeredgecolor="none",
               markersize=9, alpha=min(1.0, FILL_ALPHA * 2.5), label=ct)
    )

legA = ax.legend(
    handles=[wf_handle] + single_handles,
    title="Base + Singles",
    loc="lower left",
    bbox_to_anchor=(0.02, LEG_Y),
    frameon=True,
    fontsize=LEG_FONT,
    title_fontsize=LEG_FONT + 1
)
ax.add_artist(legA)

if comp_handles:
    ax.legend(
        handles=comp_handles,
        title="Compounds (shape -> type)",
        loc="lower right",
        bbox_to_anchor=(1.05, LEG_Y),
        frameon=True,
        fontsize=LEG_FONT,
        title_fontsize=LEG_FONT + 1
    )

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
plt.show()

print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)
print("Singles:", len(df_single), "| Compounds:", len(df_comp))
print("Wildfire pixels shown (fire>0):", int(np.sum(fire_sum > 0)))

# %%
# =============================================================================
# SM FIGURE — Global compound + fire hotspot map with PER-SREX-REGION
# percentile thresholds (supplement to main Figure 4a)
#
# - Does NOT modify any variable from the main compute or render cells.
# - New functions/vars use `sm_` prefix or `_regional` suffix.
# - Reuses visual style constants (colors, markers, sizes, legend) from the
#   panel a cell so the two figures are directly visually comparable.
# - Wildfire raster layer is unchanged (absolute fire activity); only the
#   compound/single hotspot thresholds are regionally normalized.
# =============================================================================

# -------- Settings (mirror main figure; regional thresholds) --------------
SM_SINGLE_HAZARDS  = ["HOT", "DRY", "WINDY", "FLOOD"]
SM_COMPOUND_ORDER  = ["HOT_DRY", "DRY_WINDY", "HOT_WINDY",
                      "HOT_DRY_WINDY", "WINDY_FLOOD"]

SM_PIXEL_PCT_SINGLE = 98.5   # per-region percentile (same number, local)
SM_PIXEL_PCT_COMP   = 99.0
SM_KEEP_TOP_PCT_SINGLE = 60.0
SM_KEEP_TOP_PCT_COMP   = 65.0
SM_AGG_DEG_SINGLE = 1.0
SM_AGG_DEG_COMP   = 2.0
SM_MAX_PER_REGION_SINGLE = 2000
SM_MAX_PER_REGION_COMP   = 1200


# -------- 1) Cache the SREX assignment on the raster grid ----------------
# Computed ONCE and reused across all hazard layers.
print("Assigning ERA5-Land raster pixels to SREX regions (one-time)...")
_sm_srex_flat = assign_srex_vectorized(lon2d.ravel(), lat2d.ravel())
sm_srex_grid = np.array(_sm_srex_flat).reshape(lon2d.shape)   # obj dtype
sm_srex_abbrevs = sorted({a for a in _sm_srex_flat if a is not None})
print(f"  Pixels assigned: "
      f"{sum(a is not None for a in _sm_srex_flat):,} / {len(_sm_srex_flat):,}  "
      f"(covering {len(sm_srex_abbrevs)} regions)")


# -------- 2) Per-region hotspot point builder ----------------------------
def _build_hotspot_points_regional(
    raster_sum, lon2d_arr, lat2d_arr, srex_grid, hazard_name,
    pct_per_region, keep_top_p, agg_deg, max_per_region,
    r_min, r_max, p_exp,
    min_pixels_for_region=20,
):
    """Hotspot detection using per-SREX-region percentile thresholds.

    For each SREX region:
      1. Collect nonzero pixel values within the region.
      2. Apply the `pct_per_region`-th percentile threshold LOCALLY.
      3. Aggregate surviving pixels to `agg_deg` lon/lat bins (same as
         the global builder but per-region to avoid cross-region
         aggregation bleed).
      4. Keep the top `keep_top_p`% of aggregated cells by summed freq.
      5. Cap at `max_per_region` points.

    Returns a DataFrame with (hazard, lon, lat, freq, s, srex_region).
    """
    parts = []
    for ab in sm_srex_abbrevs:
        region_mask = (srex_grid == ab) & np.isfinite(raster_sum) & (raster_sum > 0)
        n_pix = int(region_mask.sum())
        if n_pix < min_pixels_for_region:
            continue

        vals_region = raster_sum[region_mask]
        thresh = float(np.percentile(vals_region, pct_per_region))
        if thresh <= 0:
            continue

        hot_mask = region_mask & (raster_sum >= thresh)
        if hot_mask.sum() == 0:
            continue

        df_pix = pd.DataFrame({
            "lon":  lon2d_arr[hot_mask],
            "lat":  lat2d_arr[hot_mask],
            "freq": raster_sum[hot_mask].astype(float),
        })

        # Aggregate to `agg_deg` bins (per-region)
        df_pix["lon"] = np.floor(df_pix["lon"] / agg_deg) * agg_deg + agg_deg / 2.0
        df_pix["lat"] = np.floor(df_pix["lat"] / agg_deg) * agg_deg + agg_deg / 2.0
        df_agg = (df_pix.groupby(["lon", "lat"], as_index=False)["freq"].sum())

        # Keep top `keep_top_p`% within region
        if 0 < keep_top_p < 100 and len(df_agg) > 0:
            cutoff = float(np.percentile(df_agg["freq"], 100 - keep_top_p))
            df_agg = df_agg[df_agg["freq"] >= cutoff]

        if len(df_agg) > max_per_region:
            df_agg = df_agg.nlargest(max_per_region, "freq")

        if len(df_agg) == 0:
            continue

        df_agg["hazard"] = hazard_name
        df_agg["srex_region"] = ab
        parts.append(df_agg)

    if not parts:
        return pd.DataFrame(columns=["hazard", "lon", "lat",
                                     "freq", "s", "srex_region"])

    result = pd.concat(parts, ignore_index=True)
    # Map freq -> radius via the same scaling used by the global builder
    f = result["freq"].to_numpy(float)
    f_norm = (f - f.min()) / (f.max() - f.min()) if f.max() > f.min() else np.zeros_like(f)
    result["s"] = (r_min + (r_max - r_min) * (f_norm ** p_exp)) ** 2
    return result[["hazard", "lon", "lat", "freq", "s", "srex_region"]]


# -------- 3) Build regionally-thresholded hotspot dataframes -------------
print("\nBuilding regionally-thresholded single-hazard points...")
sm_single_parts = []
for hz in SM_SINGLE_HAZARDS:
    if hz not in compounds:
        continue
    yrs = sorted(set(compounds[hz].keys()) & set(fires.keys()))
    if not yrs:
        continue
    hz_sum, _, _ = _sum_rasters(compounds[hz], yrs)
    df_r = _build_hotspot_points_regional(
        hz_sum, lon2d, lat2d, sm_srex_grid, hz,
        pct_per_region=SM_PIXEL_PCT_SINGLE,
        keep_top_p=SM_KEEP_TOP_PCT_SINGLE,
        agg_deg=SM_AGG_DEG_SINGLE,
        max_per_region=SM_MAX_PER_REGION_SINGLE,
        r_min=R_MIN, r_max=R_MAX, p_exp=P_SINGLE,
    )
    sm_single_parts.append(df_r)
    print(f"  {hz:7s}: {len(df_r):5d} points  "
          f"({df_r['srex_region'].nunique()} regions)" if len(df_r)
          else f"  {hz:7s}: 0 points")

sm_df_single = (pd.concat(sm_single_parts, ignore_index=True)
                if sm_single_parts else pd.DataFrame())

print("\nBuilding regionally-thresholded compound-hazard points...")
sm_comp_parts = []
for ct in SM_COMPOUND_ORDER:
    if ct not in compounds:
        continue
    yrs = sorted(set(compounds[ct].keys()) & set(fires.keys()))
    if not yrs:
        continue
    ct_sum, _, _ = _sum_rasters(compounds[ct], yrs)
    df_r = _build_hotspot_points_regional(
        ct_sum, lon2d, lat2d, sm_srex_grid, ct,
        pct_per_region=SM_PIXEL_PCT_COMP,
        keep_top_p=SM_KEEP_TOP_PCT_COMP,
        agg_deg=SM_AGG_DEG_COMP,
        max_per_region=SM_MAX_PER_REGION_COMP,
        r_min=R2_MIN, r_max=R2_MAX, p_exp=P_COMP,
    )
    sm_comp_parts.append(df_r)
    print(f"  {ct:14s}: {len(df_r):5d} points  "
          f"({df_r['srex_region'].nunique()} regions)" if len(df_r)
          else f"  {ct:14s}: 0 points")

sm_df_comp = (pd.concat(sm_comp_parts, ignore_index=True)
              if sm_comp_parts else pd.DataFrame())

print(f"\nTotal regional single points : {len(sm_df_single)}")
print(f"Total regional compound points: {len(sm_df_comp)}")


# -------- 4) Rebuild wildfire raster locally (robust to cell re-run) -----
_sm_fire_mask = (fire_sum > 0) & np.isfinite(fire_sum)
sm_wf_vals = np.where(_sm_fire_mask, np.log1p(fire_sum), np.nan)
sm_wf_vmin = float(np.nanpercentile(sm_wf_vals, 5))
sm_wf_vmax = float(np.nanpercentile(sm_wf_vals, 99))
sm_wf_norm = Normalize(vmin=sm_wf_vmin, vmax=sm_wf_vmax)


# -------- 5) Render SM figure --------------------------------------------
SM_OUT_PNG = OUTPUT_DIR / "SM_FigS_hotspot_regional_thresholds.png"
SM_OUT_PDF = OUTPUT_DIR / "SM_FigS_hotspot_regional_thresholds.pdf"

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
ax.add_feature(cfeature.LAND,  facecolor=LAND_COLOR,  zorder=1)

# Wildfire layer (same absolute scaling as panel a for visual comparison)
ax.pcolormesh(
    lon2d, lat2d, sm_wf_vals,
    cmap=WF_CMAP, norm=sm_wf_norm,
    alpha=WF_ALPHA_MAX, shading="auto",
    transform=ccrs.PlateCarree(), zorder=3,
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#333", zorder=6)
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#777", zorder=6)

draw_srex_polygons(
    ax, highlight_abbrevs=None,
    default_edge="#333333", default_lw=0.5, default_ls="--",
    alpha=0.45, zorder_default=4,
)

# SREX labels (reuse positions from panel a)
for abbrev, (lon_lab, lat_lab) in SREX_LABEL_POSITIONS.items():
    ax.text(lon_lab, lat_lab, abbrev,
            transform=ccrs.PlateCarree(),
            fontsize=10, fontweight="bold", color="#222",
            ha="center", va="center", zorder=95,
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", edgecolor="none", alpha=0.80))

# Single hazards — hollow circles, colors same as panel a
if len(sm_df_single) > 0:
    for ht in SM_SINGLE_HAZARDS:
        d = sm_df_single[sm_df_single["hazard"] == ht]
        if d.empty:
            continue
        col = SINGLE_COLORS_ORIG[ht]
        s = d["s"] * SINGLE_SIZE_FACTOR if "s" in d.columns else 40
        ax.scatter(d["lon"], d["lat"], s=s, marker="o",
                   facecolors="none", edgecolors=col, linewidths=SINGLE_LW,
                   alpha=SINGLE_ALPHA, transform=ccrs.PlateCarree(), zorder=7)

# Compound hazards — shape-differentiated, colors same as panel a
if len(sm_df_comp) > 0:
    for ht in SM_COMPOUND_ORDER:
        d = sm_df_comp[sm_df_comp["hazard"] == ht]
        if d.empty:
            continue
        mk  = COMPOUND_MARKERS[ht]
        col = COMPOUND_COLORS_BY_SHAPE[ht]
        s = d["s"] * COMPOUND_SIZE_FACTOR if "s" in d.columns else 60
        ax.scatter(d["lon"], d["lat"], s=s, marker=mk,
                   facecolors=col, edgecolors="white", linewidths=0.3,
                   alpha=COMPOUND_ALPHA, transform=ccrs.PlateCarree(), zorder=8)

# Legend handles built inline so this cell is self-contained
sm_h_wildfire = [Line2D([0], [0], marker="s", linestyle="none",
                        markerfacecolor=WF_RGB_MAX_HEX, markeredgecolor="none",
                        markersize=12, label="Wildfire (raster)")]
sm_h_singles = [Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor="none",
                       markeredgecolor=SINGLE_COLORS_ORIG[h], markeredgewidth=1.4,
                       markersize=10, label=LABELS[h])
                for h in ["HOT", "DRY", "WINDY", "FLOOD"]]
sm_h_comp_sq = [Line2D([0], [0], marker="s", linestyle="none",
                       markerfacecolor=COMPOUND_COLORS_BY_SHAPE[h],
                       markeredgecolor="white", markeredgewidth=0.5,
                       markersize=10, label=LABELS[h])
                for h in ["HOT_DRY", "DRY_WINDY", "HOT_WINDY"]]
sm_h_comp_tri = [Line2D([0], [0], marker="^", linestyle="none",
                        markerfacecolor=COMPOUND_COLORS_BY_SHAPE["HOT_DRY_WINDY"],
                        markeredgecolor="white", markeredgewidth=0.5,
                        markersize=12, label=LABELS["HOT_DRY_WINDY"])]
sm_h_comp_di  = [Line2D([0], [0], marker="D", linestyle="none",
                        markerfacecolor=COMPOUND_COLORS_BY_SHAPE["WINDY_FLOOD"],
                        markeredgecolor="white", markeredgewidth=0.5,
                        markersize=10, label=LABELS["WINDY_FLOOD"])]

leg = ax.legend(
    handles=sm_h_wildfire + sm_h_singles + sm_h_comp_sq + sm_h_comp_tri + sm_h_comp_di,
    loc="center right",
    bbox_to_anchor=(-0.02, 0.5),
    ncol=1, frameon=True, fontsize=10, framealpha=0.95,
    title="Hazard type\n(singles: hollow circles;\ncompounds: shape -> type)\nthresholds applied PER SREX REGION",
    title_fontsize=10.5, borderaxespad=0.0,
)
leg.get_title().set_fontweight("bold")

ax.set_title(
    "Compound and Fire Hotspot Map — per-SREX-region percentile thresholds",
    fontsize=14, fontweight="bold", pad=12,
)

fig.text(
    0.5, 0.02,
    "Compound-event thresholds are computed within each SREX region's own "
    "non-zero distribution (98.5th percentile for singles, 99th for "
    "compounds), rather than against a global threshold (main Fig. 4a). "
    "Regions such as Alaska (ALA) and Northern Asia (NAS) show locally-"
    "intense wind-containing compounds that are not surfaced by global "
    "thresholding yet drive strong fire sensitivity (see Fig. 4c RR).",
    ha="center", va="bottom", fontsize=9, color="#333", wrap=True,
)

plt.subplots_adjust(left=0.12, bottom=0.10)
plt.savefig(SM_OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(SM_OUT_PDF, bbox_inches="tight", facecolor="white")
plt.show()
print("Saved:", SM_OUT_PNG)
print("Saved:", SM_OUT_PDF)

# %%
# =============================================================================
# SUPPLEMENTARY FIGURE D1, PANEL (a)
# Kelley map with SREX boundaries colour-coded by regime category
# =============================================================================

from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import regionmask

# -------- Regime category assignments (grounded in Kelley et al. 2019, Fig 2a) --------
REGIME_CATEGORY = {
    # Fuel-limited (6)
    'C. Asia': 'fuel',
    'W. Asia': 'fuel',
    'S.E. South America': 'fuel',
    'Tibetan Plateau': 'fuel',
    'S. Australia/New Zealand': 'fuel',
    'N. Australia': 'fuel',

    # Moisture-limited (9)
    'Amazon': 'moisture',
    'S.E. Asia': 'moisture',
    'Central America/Mexico': 'moisture',
    'E. North America': 'moisture',
    'N. Europe': 'moisture',
    'C. Europe': 'moisture',
    'N.E. Brazil': 'moisture',
    'S. Asia': 'moisture',
    'E. Asia': 'moisture',

    # Ignition-limited (3)
    'N. Asia': 'ignition',
    'Alaska/N.W. Canada': 'ignition',
    'Canada/Greenl./Icel.': 'ignition',

    # Intermediate (8)
    'S. Europe/Mediterranean': 'intermediate',
    'W. North America': 'intermediate',
    'Coast South America': 'intermediate',
    'W. Africa': 'intermediate',
    'E. Africa': 'intermediate',
    'Sahara': 'intermediate',
    'C. North America': 'intermediate',
    'S. Africa': 'intermediate',
}

# Category colours (consistent across both panels)
REGIME_COLORS = {
    'fuel':         '#2ca02c',   # green
    'moisture':     '#1f77b4',   # blue
    'ignition':     '#d62728',   # red
    'intermediate': '#ff7f0e',   # orange/yellow
}

REGIME_LABELS = {
    'fuel':         'Fuel-limited',
    'moisture':     'Moisture-limited',
    'ignition':     'Ignition-limited',
    'intermediate': 'Intermediate productivity',
}


# -------- Inputs ---------------------------------------------------------
# Kelley et al. (2019) Fig 2a. NOT included in the repo (copyrighted, Springer Nature);
# download from https://www.nature.com/articles/s41558-019-0540-7 and place it here.
SM_KELLEY_FIG_PATH = Path(os.environ.get("DATA_DIR", "data")) / "external" / "kelley2019_fig2.jpg"
SM_IMG_WEST, SM_IMG_EAST = -178.5, 180.0
SM_IMG_SOUTH, SM_IMG_NORTH = -57.0, 84.5
SM_KELLEY_ALPHA = 0.55
SM_WHITE_THRESH, SM_BLACK_THRESH = 235, 20
SM_OCEAN_MASK_COLOR = "white"
OUTLINE_LW = 2.5


# -------- Load + mask image ---------------------------------------------
img_raw = mpimg.imread(SM_KELLEY_FIG_PATH)
if img_raw.dtype != np.uint8:
    img_u8 = (img_raw * 255).clip(0, 255).astype(np.uint8)
else:
    img_u8 = img_raw.copy()
if img_u8.ndim == 2:
    img_u8 = np.stack([img_u8] * 3, axis=-1)
if img_u8.shape[2] == 3:
    alpha = np.full(img_u8.shape[:2], 255, dtype=np.uint8)
    img_u8 = np.concatenate([img_u8, alpha[..., None]], axis=-1)

r, g, b = img_u8[..., 0], img_u8[..., 1], img_u8[..., 2]
is_white = (r >= SM_WHITE_THRESH) & (g >= SM_WHITE_THRESH) & (b >= SM_WHITE_THRESH)
is_black = (r <= SM_BLACK_THRESH) & (g <= SM_BLACK_THRESH) & (b <= SM_BLACK_THRESH)
img_u8[..., 3] = np.where(is_white | is_black, 0, 255)
kelley_img = img_u8.astype(np.float32) / 255.0


# -------- Render panel (a) ----------------------------------------------
fig_a = plt.figure(figsize=(FIG_W, FIG_H))
ax_a = plt.axes(projection=ccrs.Robinson())
ax_a.set_global()
ax_a.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
ax_a.add_feature(cfeature.LAND,  facecolor=LAND_COLOR,  zorder=1)

ax_a.imshow(
    kelley_img,
    extent=[SM_IMG_WEST, SM_IMG_EAST, SM_IMG_SOUTH, SM_IMG_NORTH],
    transform=ccrs.PlateCarree(), origin="upper",
    alpha=SM_KELLEY_ALPHA, zorder=2, interpolation="bilinear",
)
ax_a.add_feature(cfeature.OCEAN, facecolor=SM_OCEAN_MASK_COLOR,
                 edgecolor="none", alpha=1.0, zorder=3)
ax_a.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#111", zorder=6)
ax_a.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#555", zorder=6)


# -------- Draw SREX polygons coloured by regime --------------------------
srex = regionmask.defined_regions.srex

# Regionmask stores polygons; get name -> polygon mapping
for region_num, region_name in zip(srex.numbers, srex.names):
    category = REGIME_CATEGORY.get(region_name)
    if category is None:
        edge_color = "#888"
        lw = 0.8
        ls = "--"
    else:
        edge_color = REGIME_COLORS[category]
        lw = OUTLINE_LW
        ls = "-"

    # regionmask regions have .polygons accessor
    region_obj = srex[region_num]
    polys = region_obj.polygon
    if hasattr(polys, 'geoms'):
        poly_list = list(polys.geoms)
    else:
        poly_list = [polys]

    for poly in poly_list:
        x, y = poly.exterior.xy
        ax_a.plot(list(x), list(y), color=edge_color, linewidth=lw,
                  linestyle=ls, transform=ccrs.PlateCarree(), zorder=7)

# SREX labels (optional, keep your existing layout)
if 'SREX_LABEL_POSITIONS' in globals():
    for abbrev, (lon_lab, lat_lab) in SREX_LABEL_POSITIONS.items():
        ax_a.text(lon_lab, lat_lab, abbrev,
                  transform=ccrs.PlateCarree(),
                  fontsize=9, fontweight="bold", color="#111",
                  ha="center", va="center", zorder=95,
                  bbox=dict(boxstyle="round,pad=0.2",
                            facecolor="white", edgecolor="none", alpha=0.85))


# -------- Legend ---------------------------------------------------------
legend_handles = [
    Line2D([0], [0], color=REGIME_COLORS[cat], linewidth=OUTLINE_LW,
           label=REGIME_LABELS[cat])
    for cat in ['fuel', 'moisture', 'ignition', 'intermediate']
]
leg = ax_a.legend(
    handles=legend_handles,
    loc="lower left", bbox_to_anchor=(0.02, 0.02),
    ncol=1, frameon=True, fontsize=10, framealpha=0.95,
    title="Fire regime category\n(based on Kelley et al. 2019)",
    title_fontsize=10, borderaxespad=0.0,
)
leg.get_title().set_fontweight("bold")

ax_a.set_title(
    "(a) Kelley et al. (2019) fire regime map with SREX regions "
    "colour-coded by regime category",
    fontsize=13, fontweight="bold", pad=12, loc="left",
)

# Save panel a separately
SM_OUT_PANEL_A = OUTPUT_DIR / "SM_FigD1_panel_a_kelley_coloured.png"
plt.savefig(SM_OUT_PANEL_A, dpi=600, bbox_inches="tight", facecolor="white")
plt.show()
print("Saved panel (a):", SM_OUT_PANEL_A)

# %%
# =============================================================================
# SUPPLEMENTARY FIGURE D1, PANEL (b)
# 26-region forest plot with regime-coloured panel frames
#
# Drop this after your existing 26-panel forest plot rendering.
# Requires: REGIME_CATEGORY and REGIME_COLORS from Cell 1 to be defined.
# =============================================================================

# Name-to-abbrev canonical mapping (should already exist in your notebook)
# If not, reuse from cell 10 of your main notebook
if 'SREX_NAME_TO_ABBREV' not in globals():
    SREX_NAME_TO_ABBREV = {
        "Alaska/N.W. Canada": "ALA", "Canada/Greenl./Icel.": "CGI",
        "W. North America": "WNA", "C. North America": "CNA",
        "E. North America": "ENA", "Central America/Mexico": "CAM",
        "Amazon": "AMZ", "N.E. Brazil": "NEB",
        "Coast South America": "WSA", "S.E. South America": "SSA",
        "N. Europe": "NEU", "C. Europe": "CEU",
        "S. Europe/Mediterranean": "MED", "Sahara": "SAH",
        "W. Africa": "WAF", "E. Africa": "EAF", "S. Africa": "SAF",
        "N. Asia": "NAS", "W. Asia": "WAS", "C. Asia": "CAS",
        "Tibetan Plateau": "TIB", "E. Asia": "EAS", "S. Asia": "SAS",
        "S.E. Asia": "SEA", "N. Australia": "NAU",
        "S. Australia/New Zealand": "SAU",
    }


def add_regime_frames(fig, ax_dict, regime_category, regime_colors,
                     frame_lw=3.0, frame_pad=0.015):
    """
    Draw a coloured frame around each subplot based on its regime category.

    Parameters
    ----------
    fig : matplotlib Figure
        The figure containing the panels.
    ax_dict : dict {region_name: matplotlib Axes}
        Mapping from region name (full SREX name like "C. Asia") to its axes.
    regime_category : dict {region_name: category_str}
    regime_colors : dict {category_str: hex_color}
    frame_lw : float
        Line width of the coloured frame.
    frame_pad : float
        Padding outside the axes bbox (in figure coords) where the frame sits.
    """
    from matplotlib.patches import Rectangle

    for region_name, ax in ax_dict.items():
        category = regime_category.get(region_name)
        if category is None:
            continue
        color = regime_colors[category]

        # Get axes bbox in figure coordinates
        bbox = ax.get_position()
        rect = Rectangle(
            (bbox.x0 - frame_pad, bbox.y0 - frame_pad),
            bbox.width + 2 * frame_pad,
            bbox.height + 2 * frame_pad,
            fill=False, edgecolor=color, linewidth=frame_lw,
            transform=fig.transFigure, clip_on=False, zorder=100,
        )
        fig.add_artist(rect)


def add_regime_corner_badges(fig, ax_dict, regime_category, regime_colors,
                            badge_size=0.012):
    """
    Alternative to full frames: adds a small coloured square in the top-right
    corner of each panel. Less visually heavy than full frames if you have
    many panels.
    """
    from matplotlib.patches import Rectangle
    for region_name, ax in ax_dict.items():
        category = regime_category.get(region_name)
        if category is None:
            continue
        color = regime_colors[category]
        bbox = ax.get_position()
        badge = Rectangle(
            (bbox.x1 - badge_size, bbox.y1 - badge_size),
            badge_size, badge_size,
            facecolor=color, edgecolor="white", linewidth=0.5,
            transform=fig.transFigure, clip_on=False, zorder=100,
        )
        fig.add_artist(badge)


# =============================================================================
# USAGE: after building your 26-panel forest plot, call one of:
#
#   add_regime_frames(fig, panel_axes_by_region, REGIME_CATEGORY, REGIME_COLORS)
#
# or (less visually heavy):
#
#   add_regime_corner_badges(fig, panel_axes_by_region, REGIME_CATEGORY, REGIME_COLORS)
#
# You need to build `panel_axes_by_region` as you create each subplot:
#   panel_axes_by_region[region_name] = ax   # inside your loop that makes panels
#
# Then add a category legend to the figure:
# =============================================================================

def add_regime_legend_to_figure(fig, regime_colors, regime_labels,
                                loc="upper right", bbox_to_anchor=(0.98, 0.99),
                                fontsize=10, title="Fire regime category"):
    """Add a regime category legend at figure level."""
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=regime_colors[cat], edgecolor="black",
              label=regime_labels[cat])
        for cat in ['fuel', 'moisture', 'ignition', 'intermediate']
    ]
    leg = fig.legend(
        handles=handles, loc=loc, bbox_to_anchor=bbox_to_anchor,
        ncol=1, frameon=True, fontsize=fontsize, framealpha=0.95,
        title=title, title_fontsize=fontsize + 0.5,
    )
    leg.get_title().set_fontweight("bold")


# =============================================================================
# EXAMPLE INTEGRATION into your existing 26-panel cell:
#
# Inside your existing loop that creates each subplot, add:
#
#     panel_axes_by_region = {}
#     for i, (region_name, df_region) in enumerate(region_data.items()):
#         ax = fig.add_subplot(nrows, ncols, i + 1)
#         # ... your existing plotting code ...
#         panel_axes_by_region[region_name] = ax
#
# Then after all panels are drawn:
#
#     plt.tight_layout()  # or your existing layout call
#
#     add_regime_frames(
#         fig, panel_axes_by_region,
#         REGIME_CATEGORY, REGIME_COLORS,
#         frame_lw=3.0, frame_pad=0.008,
#     )
#
#     add_regime_legend_to_figure(
#         fig, REGIME_COLORS, REGIME_LABELS,
#         loc="lower center", bbox_to_anchor=(0.5, -0.02),
#         fontsize=10,
#     )
#
# Save:
#     SM_OUT_PANEL_B = OUTPUT_DIR / "SM_FigD1_panel_b_forest_coloured.png"
#     plt.savefig(SM_OUT_PANEL_B, dpi=600, bbox_inches="tight", facecolor="white")
#     plt.show()
# =============================================================================

# %% [markdown]
# ## 10. Figure 4 — Panel c: SREX Region Forest Plot by Fire Regime
#
# 8 regions in three fire-regime groups (Fuel Limited, Climate Limited,
# Intermediate). Each panel shows RR for five compound hazards with confidence
# intervals. Uses the shared `compute_regional_relative_risk` function.

# %%
# =============================================================================
# FIGURE: Relative Risk Forest Plot for ALL SREX Regions
# COMPOUND HAZARDS ONLY — Ranked by significant associations
# Colors aligned with global RR hotspot map
# =============================================================================

import math

# ── Figure dimensions ──
FIG_W = 15
ROW_H_FACTOR = 3
NCOLS = 4

XLIM = (0.05, 200)

HSPACE = 0.3
WSPACE = 0.75

TITLE_FONT = 13
YLAB_FONT = 12
YLAB_FONTWEIGHT = 'bold'
AX_LABEL_FONT = 15
PANEL_LETTER_FONT = 0
LEGEND_FONT = 15

RR_LABEL_FONT = 15
RR_LABEL_COLOR = '#222222'
RR_LABEL_OFFSET_X = 20
RR_LABEL_OFFSET_Y = 0
RR_LABEL_DECIMALS = 2
RR_LABEL_SIG_BOLD = True
RR_LABEL_SIG_MARKER = '*'

POINT_SIZE = 9.0
CI_LW_SIG = 2.8
CI_LW_NS = 1.5
ALPHA_SIG = 0.95
ALPHA_NS = 0.30
MARKER_EDGE_SIG = 'black'
MARKER_EDGE_NS = 'gray'
MARKER_EDGE_WIDTH = 0.8

GLOBAL_BASELINE_LW = 2.0
GLOBAL_BASELINE_ALPHA = 0.55
GLOBAL_BASELINE_LS = ':'

RR1_LW = 1.2
RR1_ALPHA = 0.7

GRID_ALPHA = 0.22

LEGEND_NCOL = 5
LEGEND_Y = -0.04
LEGEND_FRAMEON = True

SAVE_DPI = 600
OUT_FILENAME = "Fig_AllSREX_CompoundOnly_RR_ForestPlots"

# Hazard colors — same palette as the global hotspot map
COMPOUND_HAZ_COLORS = {
    'Hot + Dry':         '#984ea3',
    'Dry + Windy':       '#ff7f00',
    'Hot + Windy':       '#4daf4a',
    'Hot + Dry + Windy': '#a65628',
    'Windy + Flood':     '#377eb8',
}

COMPOUNDS_TO_PLOT_ORDERED = [
    'Hot + Dry', 'Dry + Windy', 'Hot + Windy', 'Hot + Dry + Windy', 'Windy + Flood'
]
COMPOUND_RAW_KEYS = ['HOT_DRY', 'DRY_WINDY', 'HOT_WINDY', 'HOT_DRY_WINDY', 'WINDY_FLOOD']

HAZARD_DISPLAY_MAPPING = {
    'HOT': 'Hot', 'DRY': 'Dry', 'WINDY': 'Windy', 'FLOOD': 'Flood',
    'HOT_DRY': 'Hot + Dry', 'DRY_WINDY': 'Dry + Windy',
    'HOT_WINDY': 'Hot + Windy', 'HOT_DRY_WINDY': 'Hot + Dry + Windy',
    'WINDY_FLOOD': 'Windy + Flood'
}

# ---------- Prerequisites ----------
# If srex_int / valid_region / region_names don't exist yet in globals(),
# build them from the unified regionmask source (Section 4).
if ('srex_int' not in globals()) or ('valid_region' not in globals()) or ('region_names' not in globals()):
    # Build mask on the fire grid (same grid used for _land_mask)
    _any_year = sorted(fires.keys())[0]
    with rasterio.open(fires[_any_year]) as _src:
        _ft = _src.transform
        _H, _W = _src.height, _src.width
    _, _, _lons_1d, _lats_1d = _grid_centers(_ft, _H, _W)
    srex_int, valid_region, region_names = get_srex_mask_and_names(_lons_1d, _lats_1d)
    print(f"Built srex_int from unified regionmask source: shape={srex_int.shape}")
# ---------- Prerequisites (build missing vars from unified SREX source) ----------
# Build srex_int / valid_region / region_names on the fire grid if absent
if ('srex_int' not in globals()) or ('valid_region' not in globals()) or ('region_names' not in globals()):
    _any_year = sorted(fires.keys())[0]
    with rasterio.open(fires[_any_year]) as _src:
        _ft = _src.transform
        _H, _W = _src.height, _src.width
    _, _, _lons_1d, _lats_1d = _grid_centers(_ft, _H, _W)
    srex_int, valid_region, region_names = get_srex_mask_and_names(_lons_1d, _lats_1d)
    print(f"Built srex_int from unified regionmask: shape={srex_int.shape}")

# COMPOUND_PERCENTILE should already exist from the Section 8 forest-plot cell;
# fall back to 75 (the Section 8 default) if not.
if 'COMPOUND_PERCENTILE' not in globals():
    COMPOUND_PERCENTILE = 75
    print("COMPOUND_PERCENTILE not set — defaulting to 75")

# Build global_rr_baselines: GLOBAL row of compute_regional_relative_risk
# for each compound hazard. This matches what the original prep cell produced.
if 'global_rr_baselines' not in globals():
    print("Building global_rr_baselines from compute_regional_relative_risk...")
    global_rr_baselines = {}
    for _hz_key in ['HOT_DRY', 'DRY_WINDY', 'HOT_WINDY', 'HOT_DRY_WINDY', 'WINDY_FLOOD']:
        if _hz_key not in compounds:
            continue
        _df = compute_regional_relative_risk(
            compounds, fires, _hz_key,
            compound_pct=COMPOUND_PERCENTILE, land_mask=_land_mask
        )
        if _df is None or _df.empty:
            continue
        _glob = _df[_df['region'] == 'GLOBAL']
        if len(_glob) > 0 and pd.notna(_glob.iloc[0]['RR']):
            global_rr_baselines[_hz_key] = float(_glob.iloc[0]['RR'])
    print(f"  Built {len(global_rr_baselines)} global baselines: {global_rr_baselines}")

for _n in ['srex_int', 'valid_region', 'region_names', 'COMPOUND_PERCENTILE', '_land_mask', 'global_rr_baselines']:
    if _n not in globals():
        raise NameError(f"{_n} not found. Please run earlier cells.")

# ---------- Identify SREX regions ----------
all_srex_regions_idx = np.unique(srex_int[valid_region])
all_srex_regions_idx = all_srex_regions_idx[all_srex_regions_idx >= 0]

regions_with_data = []
for ridx in all_srex_regions_idx:
    region_name = region_names.get(ridx, f'Region_{ridx}') if isinstance(region_names, dict) else region_names.get(ridx, f'Region_{ridx}')
    regions_with_data.append({'region_idx': ridx, 'region_name': region_name})
df_regions = pd.DataFrame(regions_with_data)
print(f"Total SREX regions with data: {len(df_regions)}")

# ---------- Collect RR data (COMPOUND only) ----------
rr_data_for_plotting = []

for hazard_key_raw in COMPOUND_RAW_KEYS:
    hazard_display_name = HAZARD_DISPLAY_MAPPING.get(hazard_key_raw, hazard_key_raw.replace('_', ' + '))
    if hazard_key_raw not in compounds:
        print(f"  skip {hazard_key_raw} (not in compounds)")
        continue
    df_rr_all = compute_regional_relative_risk(
        compounds, fires, hazard_key_raw,
        compound_pct=COMPOUND_PERCENTILE, land_mask=_land_mask
    )
    if df_rr_all is None or df_rr_all.empty:
        continue
    df_rr_filtered = df_rr_all[df_rr_all['region_idx'].isin(all_srex_regions_idx)].copy()
    df_rr_filtered['Hazard'] = hazard_key_raw
    df_rr_filtered['Hazard_Display'] = hazard_display_name
    rr_data_for_plotting.append(df_rr_filtered)

if not rr_data_for_plotting:
    raise RuntimeError("No RR data collected for plotting.")

df_plot_all_regions = pd.concat(rr_data_for_plotting, ignore_index=True)

present_hazards_display = df_plot_all_regions['Hazard_Display'].unique()
HAZARDS_FILTERED = [h for h in COMPOUNDS_TO_PLOT_ORDERED if h in present_hazards_display]
N_HAZARDS = len(HAZARDS_FILTERED)
hazard_y_map = {hz: i for i, hz in enumerate(HAZARDS_FILTERED)}
print(f"Plotting {N_HAZARDS} compound hazards: {HAZARDS_FILTERED}")

# ---------- Rank regions by number of significant compound RR associations ----
sig_stats = []
for _, region_info in df_regions.iterrows():
    region_name = region_info['region_name']
    df_reg = df_plot_all_regions[df_plot_all_regions['region'] == region_name]
    n_sig = int(df_reg['significant'].sum()) if 'significant' in df_reg.columns else 0
    sig_rows = df_reg[df_reg['significant'] == True] if 'significant' in df_reg.columns else df_reg.iloc[0:0]
    mean_sig_rr = float(sig_rows['RR'].mean()) if len(sig_rows) > 0 else 0.0
    sig_stats.append({
        'region_idx': region_info['region_idx'],
        'region_name': region_name,
        'n_sig': n_sig,
        'mean_sig_rr': mean_sig_rr
    })

df_sig_stats = pd.DataFrame(sig_stats)
df_sig_stats = df_sig_stats.sort_values(by=['n_sig', 'mean_sig_rr'], ascending=[False, False]).reset_index(drop=True)
df_sig_stats['sig_rank'] = range(1, len(df_sig_stats) + 1)

df_regions = df_regions.merge(df_sig_stats[['region_idx', 'n_sig', 'mean_sig_rr', 'sig_rank']],
                               on='region_idx', how='left')
df_regions = df_regions.sort_values('sig_rank').reset_index(drop=True)

print(f"\nRegion ranking (compound RR significance):")
for _, r in df_sig_stats.head(10).iterrows():
    print(f"  {srex_abbrev(r['region_name']):>3s} ({r['region_name']}): {r['n_sig']}/{N_HAZARDS} sig")

# ---------- Plotting ----------
N_REGIONS = len(df_regions)
NROWS = math.ceil(N_REGIONS / NCOLS)
FIG_H = NROWS * ROW_H_FACTOR

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(FIG_W, FIG_H), sharex=True, sharey=True)
axes = axes.flatten()
panel_letters = 'abcdefghijklmnopqrstuvwxyz'

for i, (_, region_info) in enumerate(df_regions.iterrows()):
    ax = axes[i]
    region_name = region_info['region_name']
    n_sig = int(region_info['n_sig'])

    region_abbrev = srex_abbrev(region_name)

    ax.text(0.02, 0.98, panel_letters[i % 26], transform=ax.transAxes,
            fontsize=PANEL_LETTER_FONT, fontweight='bold', va='top', ha='left')

    ax.set_title(f"{region_abbrev} ({n_sig}/{N_HAZARDS} sig.)",
                 fontsize=TITLE_FONT, fontweight="bold", pad=8)

    df_region = df_plot_all_regions[df_plot_all_regions['region'] == region_name].copy()

    ax.axvline(1, color="black", linestyle="--", linewidth=RR1_LW, alpha=RR1_ALPHA, zorder=1)

    # Global RR baselines
    for hazard_display in HAZARDS_FILTERED:
        raw_key = None
        for k, v in HAZARD_DISPLAY_MAPPING.items():
            if v == hazard_display:
                raw_key = k
                break
        if raw_key is None:
            continue
        global_rr_val = global_rr_baselines.get(raw_key)
        if global_rr_val is not None and global_rr_val > 0:
            base_color = COMPOUND_HAZ_COLORS.get(hazard_display, '#333333')
            ax.axvline(global_rr_val, color=base_color, linestyle=GLOBAL_BASELINE_LS,
                       linewidth=GLOBAL_BASELINE_LW, alpha=GLOBAL_BASELINE_ALPHA, zorder=1)

    # Plot RR points + CI + number label
    for _, row in df_region.iterrows():
        hazard_display = row['Hazard_Display']
        rr = row['RR']
        ci_lo = row['CI_lower']
        ci_hi = row['CI_upper']
        sig = row['significant']

        if pd.isna(rr) or rr <= 0:
            continue
        if hazard_display not in hazard_y_map:
            continue

        y_pos = hazard_y_map[hazard_display]
        base_color = COMPOUND_HAZ_COLORS.get(hazard_display, '#333333')

        alpha = ALPHA_SIG if sig else ALPHA_NS
        lw = CI_LW_SIG if sig else CI_LW_NS
        edge_color = MARKER_EDGE_SIG if sig else MARKER_EDGE_NS

        ci_lo_c = max(XLIM[0], float(ci_lo)) if pd.notna(ci_lo) else rr * 0.5
        ci_hi_c = min(XLIM[1], float(ci_hi)) if pd.notna(ci_hi) else rr * 2
        ax.plot([ci_lo_c, ci_hi_c], [y_pos, y_pos], color=base_color,
                linewidth=lw, alpha=alpha, zorder=2)

        ax.plot(rr, y_pos, "o", markersize=POINT_SIZE,
                markerfacecolor=base_color, markeredgecolor=edge_color,
                markeredgewidth=MARKER_EDGE_WIDTH, alpha=alpha, zorder=3)

        rr_txt = f"{rr:.{RR_LABEL_DECIMALS}f}"
        if sig:
            rr_txt += RR_LABEL_SIG_MARKER

        ax.annotate(rr_txt, (rr, y_pos),
                    xytext=(RR_LABEL_OFFSET_X, RR_LABEL_OFFSET_Y),
                    textcoords='offset points',
                    fontsize=RR_LABEL_FONT,
                    color=RR_LABEL_COLOR,
                    fontweight='bold' if (RR_LABEL_SIG_BOLD and sig) else 'normal',
                    ha='left', va='center',
                    clip_on=False, zorder=10)

    ax.set_xscale("log")
    ax.set_xlim(XLIM[0], XLIM[1])
    ax.set_yticks(np.arange(N_HAZARDS))

    if i % NCOLS == 0:
        ax.set_yticklabels(HAZARDS_FILTERED, fontsize=YLAB_FONT, fontweight=YLAB_FONTWEIGHT)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="y", pad=6)
    ax.grid(True, axis="x", alpha=GRID_ALPHA)
    ax.set_axisbelow(True)

    if i >= N_REGIONS - NCOLS:
        ax.set_xlabel("Relative Risk (log scale)", fontsize=AX_LABEL_FONT)
    else:
        ax.set_xlabel('')

for j in range(N_REGIONS, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(h_pad=HSPACE, w_pad=WSPACE)

# Legend
sig_handle = Line2D([0], [0], marker="o", linestyle="None",
                    markerfacecolor="#777777", markeredgecolor="black",
                    markeredgewidth=1.0, markersize=8, label="Significant *")
ns_handle = Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor="#777777", markeredgecolor="gray",
                   markeredgewidth=0.8, markersize=8, alpha=0.30, label="Not significant")
rr1_handle = Line2D([0], [0], color="black", linestyle="--", linewidth=RR1_LW, label="RR = 1")
global_rr_handle = Line2D([0], [0], color="black", linestyle=GLOBAL_BASELINE_LS,
                          linewidth=GLOBAL_BASELINE_LW, label="Global RR baseline")

legend_hazard_handles = []
for hz_display in HAZARDS_FILTERED:
    color = COMPOUND_HAZ_COLORS.get(hz_display, '#333333')
    legend_hazard_handles.append(Line2D([0], [0], marker="o", linestyle="None",
                                        markerfacecolor=color, markeredgecolor=color,
                                        markersize=POINT_SIZE, label=hz_display))

fig.legend(handles=[rr1_handle, global_rr_handle, sig_handle, ns_handle] + legend_hazard_handles,
           loc="lower center", ncol=LEGEND_NCOL, frameon=LEGEND_FRAMEON,
           framealpha=0.95, edgecolor='#cccccc', fontsize=LEGEND_FONT,
           bbox_to_anchor=(0.5, LEGEND_Y))

plt.subplots_adjust(bottom=0.08)

out_png = OUTPUT_DIR / f"{OUT_FILENAME}.png"
out_pdf = OUTPUT_DIR / f"{OUT_FILENAME}.pdf"
fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
plt.show()
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")

# %%
# =============================================================================
# Print regional RR table in a format easy to paste back for review
# =============================================================================

import pandas as pd

COMPOUND_ORDER = ["HOT_DRY", "DRY_WINDY", "HOT_WINDY", "HOT_DRY_WINDY", "WINDY_FLOOD"]

# Collect per-region RR across compound types into a single dataframe
rr_rows = []
for ct in COMPOUND_ORDER:
    if ct not in rr_full:
        continue
    df = rr_full[ct].copy()
    df["compound"] = ct
    rr_rows.append(df[["region", "compound", "RR", "CI_lower", "CI_upper", "significant", "n_cells"]])

if rr_rows:
    rr_table = pd.concat(rr_rows, ignore_index=True)

    # Pivot to wide format: regions as rows, compounds as columns
    rr_wide = rr_table.pivot(index="region", columns="compound", values="RR")
    sig_wide = rr_table.pivot(index="region", columns="compound", values="significant")

    # Reorder columns to match the figure
    rr_wide = rr_wide[COMPOUND_ORDER]
    sig_wide = sig_wide[COMPOUND_ORDER]

    # Format each cell as "RR" or "RR*" if significant
    def _fmt(rr, sig):
        if pd.isna(rr):
            return "  nan"
        star = "*" if sig else " "
        return f"{rr:5.2f}{star}"

    formatted = rr_wide.copy().astype(str)
    for region in rr_wide.index:
        for ct in COMPOUND_ORDER:
            formatted.loc[region, ct] = _fmt(rr_wide.loc[region, ct], sig_wide.loc[region, ct])

    # Print in a paste-friendly format
    print("=" * 70)
    print("REGIONAL RR TABLE (asterisk = significant, CI excludes 1)")
    print("=" * 70)
    # Use to_string for a clean fixed-width table
    print(formatted.to_string())
    print()

    # Also print as CSV for easier paste
    print("=" * 70)
    print("CSV FORMAT (copy everything below this line):")
    print("=" * 70)
    csv_str = formatted.to_csv()
    print(csv_str)

    # Save CSV to disk too
    csv_path = OUTPUT_DIR / "regional_rr_table.csv"
    rr_table.to_csv(csv_path, index=False)
    print(f"Full long-format table saved to: {csv_path}")
else:
    print("No RR data available. Run the RR computation cell first.")
