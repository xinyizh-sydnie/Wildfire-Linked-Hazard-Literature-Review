# %% [markdown]
# # Figure 5 + Resolution Sensitivity (ERL Revision)
#
# Focused deliverable notebook. Drops the Figure 4 pipeline from v4.
#
# **What this notebook produces**
#
# 1. **Figure 5 (a–d).** Four pairwise / triple mismatch maps at the 2 deg baseline
#    using the areal-weighted research-attention surface.
# 2. **Supplementary Figure SENS-1.** Triple-mismatch panels at 5, 2, 1 and
#    0.5 deg grids, side by side.
# 3. **Supplementary Table SENS-A.** Cross-resolution Spearman rho at the cell
#    level (finer grid aggregated up to the coarser grid by cos-lat area mean)
#    and at the SREX region level.
# 4. **Supplementary Table SENS-B.** Top-10 SREX regions by triple mismatch at
#    each resolution, with stability measured by Jaccard overlap against the 2 deg
#    baseline.
#
# **Why these resolutions?**
#
# * The MODIS hotspot product used for fire burden is at a native 50 km grid,
#   so 0.5 deg is the honest refinement floor. Going finer interpolates rather
#   than resolves.
# * Literature footprint spans analyzed from `geo_regions_json`: median finest
#   MOB span per paper is roughly 7 deg, p10 is 0.5 deg, p25 is 2 deg, p75 is 14 deg.
#   Only about 11 percent of papers (precision 3 or 4) resolve below 1 deg; the
#   remaining 89 percent are effectively unchanged by finer grids.
# * 5 deg is included as a coarsening stress test: if the Global-South pattern
#   survives at 5 deg, it is robust to the analyst's grid choice.
#
# **Runtime note.** The 0.5 deg run processes roughly 209 k cells and takes
# approximately 15 to 30 minutes in Colab. All finer runs use vectorized
# areal weighting and windowed zonal statistics; peak memory stays under 4 GB.
#
# **Running order.** Top to bottom. Paths in section 2 assume the same Google
# Drive layout as v4.

# %% [markdown]
# ## 1. Install dependencies (Colab)

# %%
# Colab: install geospatial deps. For local runs use the repo's requirements.txt instead.
# !pip install -q cartopy regionmask rasterio rioxarray geopandas   # notebook shell/magic - install via requirements.txt

# %% [markdown]
# ## 2. Imports

# %%
import warnings
warnings.filterwarnings('ignore')

import os, re, json, math, logging, pickle, time
from pathlib import Path
from collections import Counter

import glob as _glob_module
from glob import glob
glob.glob = _glob_module.glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D

import rasterio
from rasterio.transform import rowcol
from rasterio.windows import from_bounds

from scipy.stats import rankdata, spearmanr
from scipy.ndimage import gaussian_filter, zoom
from scipy.interpolate import griddata

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from shapely.ops import unary_union
from shapely.geometry import Point, box

try:
    from shapely import contains_xy
    def _shape_contains_xy(geom, xs, ys):
        return contains_xy(geom, xs, ys)
except ImportError:
    from shapely.vectorized import contains as _vcontains
    def _shape_contains_xy(geom, xs, ys):
        return _vcontains(geom, xs, ys)

import regionmask

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('erl_fig5_sens')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
})

print('Imports OK.')

# %% [markdown]
# ## 3. File paths
#
# Edit the paths below to point at your local or Drive copies. Defaults match
# the v4 notebook's Drive layout.

# %%
# Mount Google Drive (Colab only). Skips automatically when not on Colab.
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
except ModuleNotFoundError:
    print("Not on Colab; skipping Drive mount. Using local DATA_DIR / OUTPUT_DIR instead.")

# %%
# ---------------------------------------------------------------------------
# File paths. By default these resolve to a local ./data and ./outputs tree
# (see the canonical repo layout in the README). To point at Google Drive
# instead, set the DATA_DIR / OUTPUT_DIR environment variables, e.g.
#   os.environ["DATA_DIR"] = "/content/drive/MyDrive/ERL/data"
# ---------------------------------------------------------------------------
from pathlib import Path
import os

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
OUT_BASE = Path(os.environ.get("OUTPUT_DIR", "outputs")) / "fig5_sensitivity"

DATA_YEARLY_DIR       = DATA_DIR / "hotspot_50km"          # output of 00_preprocess_hotspots
PHYSICAL_REALITY_PATH = DATA_YEARLY_DIR
LITERATURE_PATH       = DATA_DIR / "study_table.csv"       # trimmed LLM classification output
GRDI_PATH             = DATA_DIR / "grdi" / "povmap-grdi-v1.tif"

OUTPUT_DIR = OUT_BASE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR  = OUT_BASE / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

for lbl, p in [('DATA_YEARLY_DIR', DATA_YEARLY_DIR),
               ('LITERATURE_PATH', LITERATURE_PATH),
               ('GRDI_PATH',       GRDI_PATH),
               ('OUTPUT_DIR',      OUTPUT_DIR)]:
    print(f"  {'OK ' if os.path.exists(p) else 'MISS'} {lbl}: {p}")

# %% [markdown]
# ## 4. SREX polygons (single source of truth)

# %%
# SREX reference regions are now defined once in code/srex_regions.py
# (keep that file alongside this notebook). This replaces the inline copy
# that was duplicated across the figure notebooks.
from srex_regions import (
    SREX_NAME_TO_ABBREV, SREX_ABBREV_TO_NAME, SREX_POLYGONS,
    assign_srex_vectorized, draw_srex_polygons,
)
print(f'{len(SREX_POLYGONS)} SREX polygons indexed (from srex_regions.py).')

# %% [markdown]
# ## 5. Literature loading
#
# Parse `geo_regions_json` to recover **all** region MOBs per paper (not just
# the primary centroid). Detect equity dimensions from title + definition +
# keywords. Sample GRDI at paper centroids for diagnostic columns only; the
# gridded analysis uses its own cell-level GRDI aggregation later.

# %%
def extract_all_centroids(geo_json_str):
    """Return a list of dicts (one per region) with lat, lon, lat_span, lon_span."""
    if pd.isna(geo_json_str) or not isinstance(geo_json_str, str):
        return []
    try:
        regs = json.loads(geo_json_str)
    except (json.JSONDecodeError, TypeError):
        return []
    out = []
    if not isinstance(regs, list):
        return out
    for r in regs:
        try:
            mob = r.get('MOB')
            if mob is None:
                continue
            tl = mob['top_left']; br = mob['bottom_right']
            lat = (tl[0] + br[0]) / 2
            lon = (tl[1] + br[1]) / 2
            lat_span = abs(tl[0] - br[0])
            lon_span = abs(tl[1] - br[1])
            out.append({'lat': lat, 'lon': lon,
                        'lat_span': lat_span, 'lon_span': lon_span,
                        'region_name': r.get('region_name', '')})
        except (KeyError, TypeError, IndexError):
            continue
    return out


def detect_equity_dimensions(row):
    text = ' '.join([str(row.get('title','')), str(row.get('definition_summary','')),
                     str(row.get('normalized_keywords',''))])
    d = {}
    d['vulnerability'] = bool(re.search(
        r'vulnerab|\bSVI\b|disadvant|marginal(?:iz)|low.?income|poverty|depriv',
        text, re.IGNORECASE))
    d['social_vulnerability'] = bool(re.search(
        r'social\s*vulnerab|socio.?(?:economic|demographic)\s+(?:vulnerab|inequal|dispar)',
        text, re.IGNORECASE))
    d['health'] = bool(re.search(
        r'(?:human|public|population|child|infant|respiratory|cardiovascular)\s*health|'
        r'mortalit|respirat|cardio|hospital|\bPM2\.?5\b|smoke.*expos|'
        r'\basthma\b|\bCOPD\b|\blung\b(?!.*forest)|mental\s*health',
        text, re.IGNORECASE))
    d['recovery'] = bool(re.search(
        r'recover|restor|rebuild|resilien|\badapt(?:ation|ive)\b|aftermath',
        text, re.IGNORECASE))
    d['env_justice'] = bool(re.search(
        r'justice|\bequit|\bEJ\b|\bracial\b|ethnic|indigenous|tribal|native\s+(?:american|communit)|'
        r'disparit|\binequal',
        text, re.IGNORECASE))
    d['community'] = bool(re.search(
        r'communit(?:y|ies)\s+(?:impact|risk|expos|resilien|recover)|'
        r'household|housing|evacu|displace|livelihood|insur(?:ance)',
        text, re.IGNORECASE))
    return d


df = pd.read_csv(LITERATURE_PATH)
print(f'Total rows: {len(df)}')
linked = df[df['linked_label'] == 'linked'].copy()
print(f'Linked-hazard rows: {len(linked)}')

# Primary centroid for sensitivity reference
linked['centroid_data'] = linked['geo_regions_json'].apply(
    lambda s: (extract_all_centroids(s) or [None])[0])
linked['lat'] = linked['centroid_data'].apply(lambda x: x['lat'] if x else None)
linked['lon'] = linked['centroid_data'].apply(lambda x: x['lon'] if x else None)

# Equity dimensions
eq = linked.apply(detect_equity_dimensions, axis=1, result_type='expand')
EQUITY_DIMS = ['vulnerability','social_vulnerability','health','recovery','env_justice','community']
for c in EQUITY_DIMS:
    linked[c] = eq[c]
linked['has_equity'] = linked[EQUITY_DIMS].any(axis=1)

print(f'Papers with valid primary centroid: {linked["lat"].notna().sum()}')
print(f'Papers flagged equity: {linked["has_equity"].sum()} '
      f'({100*linked["has_equity"].mean():.1f}%)')

# %% [markdown]
# ## 6. Load resolution-independent rasters
#
# These are loaded once and reused at every resolution:
# - Accumulated 24-year MODIS fire frequency raster
# - GRDI v1 raster
# - Natural Earth 50 m land polygon union

# %%
# --- Fire: accumulate across 24 yearly rasters --------------------------------
print('Accumulating 24-year fire raster...')
fire_tifs = sorted(glob(os.path.join(PHYSICAL_REALITY_PATH, 'hotspot_*_fire_frequency.tif')))
print(f'  {len(fire_tifs)} yearly fire TIFFs found')

with rasterio.open(fire_tifs[0]) as src:
    fire_shape = (src.height, src.width)
    fire_transform = src.transform
    fire_crs = src.crs

fire_total = np.zeros(fire_shape, dtype=np.float64)
for p in fire_tifs:
    with rasterio.open(p) as src:
        d = src.read(1).astype(np.float64)
        m = np.isfinite(d) & (d > 0)
        fire_total[m] += d[m]
print(f'  Fire pixels > 0: {(fire_total > 0).sum():,} / {fire_total.size:,}')

# --- GRDI ---------------------------------------------------------------------
print('\nLoading GRDI raster...')
with rasterio.open(GRDI_PATH) as src:
    grdi_data_full = src.read(1).astype(np.float64)
    grdi_tf = src.transform
    if src.nodata is not None:
        grdi_data_full[grdi_data_full == src.nodata] = np.nan
print(f'  GRDI shape {grdi_data_full.shape}, '
      f'valid fraction {np.isfinite(grdi_data_full).mean()*100:.1f}%')

# --- Land union (50m Natural Earth) -------------------------------------------
print('\nBuilding 50m Natural Earth land union...')
_lshp = shpreader.natural_earth('50m', 'physical', 'land')
land_union = unary_union(list(shpreader.Reader(_lshp).geometries()))
print('  Land union built.')

# %% [markdown]
# ## 7. Core: `build_mismatch_grid(res)`
#
# Every step (zonal stats, GRDI gap-filling, areal-weighted attention,
# percentile ranking, geometric-mean aggregation, SREX assignment) from the v4
# notebook is reproduced here, but parameterized on grid resolution `res` in
# degrees. Numerical behavior at `res = 2.0` matches the v4 Figure 5 exactly.

# %%
def build_mismatch_grid(
    res,
    linked_df=None,
    fire_total=None, fire_transform=None,
    grdi_data_full=None, grdi_tf=None,
    land_union=None,
    lat_range=(-60, 85),
    lon_range=(-180, 180),
    verbose=True,
):
    """Build research-risk mismatch grid at `res` deg. Returns dict."""
    t0 = time.time()
    if verbose:
        print(f'\n{"="*60}')
        print(f'BUILDING MISMATCH GRID @ {res} deg')
        print(f'{"="*60}')

    # --- 1) Bins and cell centers
    lat_bins = np.arange(lat_range[0], lat_range[1], res)
    lon_bins = np.arange(lon_range[0], lon_range[1], res)
    n_lat = len(lat_bins) - 1
    n_lon = len(lon_bins) - 1
    lat_c = lat_bins[:-1] + res / 2
    lon_c = lon_bins[:-1] + res / 2
    cos_lat = np.cos(np.deg2rad(lat_c))
    if verbose:
        print(f'  Grid: {n_lat} x {n_lon} = {n_lat*n_lon:,} cells')

    # --- 2) Land mask (point-in-polygon at each cell center)
    lon_mesh, lat_mesh = np.meshgrid(lon_c, lat_c)
    land_flat = _shape_contains_xy(land_union, lon_mesh.ravel(), lat_mesh.ravel())
    land_mask_2d = land_flat.reshape(n_lat, n_lon)
    if verbose:
        print(f'  Land cells: {land_mask_2d.sum():,} ({100*land_mask_2d.mean():.1f}%)')

    # --- 3) Zonal statistics (fire: sum, GRDI: mean)
    def _zonal(raster, tf, lat0, lon0, r, agg):
        half = r / 2
        try:
            w = from_bounds(lon0-half, lat0-half, lon0+half, lat0+half, tf)
            r0 = max(0, int(w.row_off)); r1 = min(raster.shape[0], int(w.row_off + w.height))
            c0 = max(0, int(w.col_off)); c1 = min(raster.shape[1], int(w.col_off + w.width))
            if r0 >= r1 or c0 >= c1:
                return 0.0 if agg == 'sum' else np.nan
            chunk = raster[r0:r1, c0:c1]
            fin = np.isfinite(chunk)
            if agg == 'sum':
                m = fin & (chunk > 0)
                return float(chunk[m].sum()) if chunk.size else 0.0
            else:
                if fin.sum() == 0 or fin.sum()/chunk.size < 0.01:
                    return np.nan
                return float(chunk[fin].mean())
        except Exception:
            return np.nan if agg == 'mean' else 0.0

    if verbose: print('  Zonal stats for fire + GRDI...')
    fire_arr = np.full((n_lat, n_lon), np.nan)
    grdi_arr = np.full((n_lat, n_lon), np.nan)
    for i in range(n_lat):
        for j in range(n_lon):
            fire_arr[i, j] = _zonal(fire_total, fire_transform,
                                    lat_c[i], lon_c[j], res, 'sum')
            grdi_arr[i, j] = _zonal(grdi_data_full, grdi_tf,
                                    lat_c[i], lon_c[j], res, 'mean')
        if verbose and (i + 1) % max(1, n_lat // 10) == 0:
            pct = 100 * (i + 1) / n_lat
            print(f'    zonal progress: {i+1}/{n_lat} ({pct:.0f}%)')

    # --- Build base DataFrame
    LAT, LON = np.meshgrid(lat_c, lon_c, indexing='ij')
    grid_df = pd.DataFrame({
        'lat': LAT.ravel(), 'lon': LON.ravel(),
        'fire_count': fire_arr.ravel(),
        'grdi': grdi_arr.ravel(),
        'is_land': land_mask_2d.ravel(),
    })

    # --- 4) GRDI gap-fill (nearest neighbor)
    gv = grid_df['grdi'].notna()
    gn = grid_df['grdi'].isna()
    grid_df['grdi_interpolated'] = False
    if gv.sum() > 10 and gn.sum() > 0:
        sc = grid_df.loc[gv, ['lat','lon']].to_numpy()
        sv = grid_df.loc[gv, 'grdi'].to_numpy()
        tc = grid_df.loc[gn, ['lat','lon']].to_numpy()
        fill = griddata(sc, sv, tc, method='nearest')
        grid_df.loc[gn, 'grdi'] = fill
        grid_df.loc[gn, 'grdi_interpolated'] = True

    # --- 5) Areal-weighted research attention (Goodchild & Lam 1980 + dasymetric land)
    def _mob_weights(tl, br):
        """Return (n_lat, n_lon) weights summing to 1 over land cells."""
        lat_lo, lat_hi = sorted([tl[0], br[0]])
        lon_lo, lon_hi = sorted([tl[1], br[1]])
        if (lat_hi - lat_lo) < res:
            c = (lat_lo + lat_hi) / 2
            lat_lo, lat_hi = c - res/2, c + res/2
        if (lon_hi - lon_lo) < res:
            c = (lon_lo + lon_hi) / 2
            lon_lo, lon_hi = c - res/2, c + res/2
        lat_ov = np.clip(np.minimum(lat_bins[1:], lat_hi) - np.maximum(lat_bins[:-1], lat_lo), 0, None)
        lon_ov = np.clip(np.minimum(lon_bins[1:], lon_hi) - np.maximum(lon_bins[:-1], lon_lo), 0, None)
        if lat_ov.sum() == 0 or lon_ov.sum() == 0:
            return None
        area = (lat_ov * cos_lat)[:, None] * lon_ov[None, :]
        area = area * land_mask_2d
        tot = area.sum()
        return area / tot if tot > 0 else None

    n_papers_wt = np.zeros((n_lat, n_lon))
    n_equity_wt = np.zeros((n_lat, n_lon))
    n_papers_ct = np.zeros((n_lat, n_lon), dtype=int)
    n_equity_ct = np.zeros((n_lat, n_lon), dtype=int)

    n_used = n_nofoot = n_noland = 0
    for _, row in linked_df.iterrows():
        regs = extract_all_centroids(row.get('geo_regions_json', ''))
        if not regs:
            n_nofoot += 1; continue
        grids_ = []
        for r in regs:
            tl = (r['lat'] + r['lat_span']/2, r['lon'] - r['lon_span']/2)
            br = (r['lat'] - r['lat_span']/2, r['lon'] + r['lon_span']/2)
            g = _mob_weights(tl, br)
            if g is not None:
                grids_.append(g)
        if not grids_:
            n_noland += 1; continue
        paper_grid = sum(grids_) / len(grids_)
        n_papers_wt += paper_grid
        is_eq = bool(row.get('has_equity', False))
        if is_eq:
            n_equity_wt += paper_grid
        # Centroid counterpart (for sensitivity reference)
        pl = row.get('lat'); po = row.get('lon')
        if pd.notna(pl) and pd.notna(po):
            ri = int(np.floor((pl - lat_bins[0]) / res))
            ci = int(np.floor((po - lon_bins[0]) / res))
            if 0 <= ri < n_lat and 0 <= ci < n_lon:
                n_papers_ct[ri, ci] += 1
                if is_eq:
                    n_equity_ct[ri, ci] += 1
        n_used += 1

    grid_df['n_papers']          = n_papers_wt.ravel()
    grid_df['n_equity']          = n_equity_wt.ravel()
    grid_df['n_papers_centroid'] = n_papers_ct.ravel()
    grid_df['n_equity_centroid'] = n_equity_ct.ravel()
    if verbose:
        print(f'  Papers used in areal surface: {n_used}  '
              f'(no footprint: {n_nofoot}, no land overlap: {n_noland})')
        print(f'  Total areal weight on grid: {n_papers_wt.sum():.1f} '
              f'(expected: {n_used})')

    # --- 6) Normalize
    fire_log = np.log1p(grid_df['fire_count'].fillna(0))
    q95f = fire_log.quantile(0.95)
    grid_df['fire_norm'] = (fire_log / max(q95f, 1e-12)).clip(0, 1)
    grid_df['grdi_norm'] = (grid_df['grdi'] / 100.0).clip(0, 1).fillna(0)
    q95p = max(grid_df['n_papers'].quantile(0.95), 1)
    grid_df['research_intensity'] = (np.log1p(grid_df['n_papers']) /
                                     np.log1p(q95p)).clip(0, 1)
    grid_df['research_gap'] = 1 - grid_df['research_intensity']

    # --- 7) Percentile rank
    def _rnk_nz(s):
        out = np.zeros(len(s))
        v = s.to_numpy()
        m = v > 0
        if m.sum() > 0:
            r = rankdata(v[m], method='average'); out[m] = r / r.max()
        return out
    def _rnk_all(s):
        out = np.zeros(len(s))
        v = s.to_numpy()
        m = np.isfinite(v) & (v >= 0)
        if m.sum() > 0:
            r = rankdata(v[m], method='average'); out[m] = r / r.max()
        return out
    grid_df['fire_pct'] = _rnk_nz(grid_df['fire_norm'])
    grid_df['grdi_pct'] = _rnk_all(grid_df['grdi_norm'])
    grid_df['gap_pct']  = _rnk_all(grid_df['research_gap'])

    # --- 8) Mismatch indices (geometric mean + convergence penalty for triple)
    grid_df['fire_research_mismatch']   = np.sqrt(grid_df['fire_pct'] * grid_df['gap_pct'])
    grid_df['fire_grdi_index']          = np.sqrt(grid_df['fire_pct'] * grid_df['grdi_pct'])
    grid_df['grdi_research_mismatch']   = np.sqrt(grid_df['grdi_pct'] * grid_df['gap_pct'])
    prod3 = grid_df['fire_pct'] * grid_df['grdi_pct'] * grid_df['gap_pct']
    base3 = np.cbrt(prod3)
    dmin = np.minimum(np.minimum(grid_df['fire_pct'], grid_df['grdi_pct']),
                       grid_df['gap_pct'])
    conv = np.where(dmin < 0.25, dmin / 0.25, 1.0)
    grid_df['triple_mismatch'] = base3 * conv

    nf = grid_df['fire_pct'] == 0
    for c in ['fire_research_mismatch', 'fire_grdi_index', 'triple_mismatch']:
        grid_df.loc[nf, c] = 0

    # --- 9) SREX assignment on land
    ab = assign_srex_vectorized(grid_df['lon'].values, grid_df['lat'].values)
    grid_df['srex_region'] = [a if (a is not None and l) else None
                              for a, l in zip(ab, grid_df['is_land'])]

    elapsed = time.time() - t0
    if verbose:
        nz = (grid_df['triple_mismatch'] > 0).sum()
        print(f'  Nonzero triple mismatch: {nz:,} cells')
        print(f'  Total runtime: {elapsed:.1f} s')

    return dict(grid_df=grid_df, lat_bins=lat_bins, lon_bins=lon_bins,
                land_mask_2d=land_mask_2d, GRID_RESOLUTION=res,
                n_lat=n_lat, n_lon=n_lon, elapsed_seconds=elapsed)

print('build_mismatch_grid(res) defined.')

# %% [markdown]
# ## 8. Baseline run at 2 deg
#
# This is the run that produces the headline Figure 5. Exposed as module-level
# globals so the plotting code in section 9 can consume them without touching
# the sensitivity dictionary.

# %%
BASELINE_RES = 2.0
baseline = build_mismatch_grid(
    res=BASELINE_RES,
    linked_df=linked,
    fire_total=fire_total, fire_transform=fire_transform,
    grdi_data_full=grdi_data_full, grdi_tf=grdi_tf,
    land_union=land_union,
)

# Expose globals expected by the plotting code
full_grid_df    = baseline['grid_df']
lat_bins        = baseline['lat_bins']
lon_bins        = baseline['lon_bins']
land_mask_2d    = baseline['land_mask_2d']
GRID_RESOLUTION = baseline['GRID_RESOLUTION']

# Cache to disk
with open(CACHE_DIR / f'mismatch_res_{BASELINE_RES}.pkl', 'wb') as f:
    pickle.dump(baseline, f)
print(f'\nCached baseline to {CACHE_DIR / f"mismatch_res_{BASELINE_RES}.pkl"}')

# %% [markdown]
# ## 9. Figure 5 plotting (a–d)
#
# Four separate files preserving the original color schemes:
# - **a.** Triple mismatch with geolocated study points (equity vs non-equity).
# - **b.** Fire x research gap (pink).
# - **c.** Fire x GRDI (orange).
# - **d.** GRDI x research gap (blue).

# %%
# =============================================================================
# Figure 5 — four separate mismatch maps with gradient colormap stretched
# so that panel p25/p50/p75 land at fixed visual positions on the colorbar.
#
# Each panel stays a smooth gradient, but the color at "p50" is the same
# mid-tone in every panel, the color at "p75" is the same darker tone, etc.
# Tier labels (Minimal / Elevated / Moderate / Severe / Extreme) sit on top
# of the colorbar at those fixed visual positions; the numeric p25/p50/p75
# values for that panel sit underneath.
# =============================================================================

from matplotlib.colors import FuncNorm

SMOOTH_SIGMA     = 0.7
PLOT_UPSAMPLE    = 2
MIN_VALID_WEIGHT = 0.25
OCEAN_COLOR      = '#E8F4F8'
LAND_COLOR       = '#F5F5F5'
TOP_N            = 10
MAX_POINTS       = 2500
RANDOM_SEED      = 7

# Tier boundaries as percentiles of each panel's nonzero distribution,
# and the visual position on the colorbar each should map to.
TIER_QUANTILES  = [0.25, 0.50, 0.75]        # panel-specific numeric breaks
TIER_VISUAL_POS = [0.25, 0.50, 0.75]        # fixed visual positions (same for all panels)
TIER_LABELS     = ['Minimal', 'Elevated', 'Moderate', 'Severe']
# 4 tiers between the 3 breaks: [0, p25] [p25, p50] [p50, p75] [p75, max]
# (Add p95 back if you want 5 tiers — see note at bottom.)

CMAPS = {
    'triple': LinearSegmentedColormap.from_list('triple_charcoal',
        ['#FFFFFF', '#F2F3F5', '#D5D9E0', '#A8AFBB',
         '#7B8494', '#667085', '#3D4556', '#1D2027']),
    'fire_research': LinearSegmentedColormap.from_list('pairs_pink',
        ['#FFFFFF', '#FDE5F2', '#F9B8DC', '#E973B1', '#C63384', '#7A1155']),
    'fire_grdi': LinearSegmentedColormap.from_list('pairs_orange',
        ['#FFFFFF', '#FFEAD0', '#FFC277', '#F2883A', '#B95A1C', '#5E2A0B']),
    'grdi_research': LinearSegmentedColormap.from_list('pairs_blue',
        ['#FFFFFF', '#DDEAF7', '#9ABEDC', '#3E7CB1', '#1C4E8A', '#0A254F']),
}
EQUITY_STYLE     = dict(s=18, c='#0D47A1', edgecolors='white',
                        linewidths=0.35, alpha=0.6)
NON_EQUITY_STYLE = dict(s=10, facecolors='none', edgecolors='#0D47A1',
                        linewidths=0.6, alpha=0.4)


def normalize_lon_to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def nan_gaussian(arr, sigma, min_w=MIN_VALID_WEIGHT):
    m = np.isfinite(arr)
    v = np.where(m, arr, 0.0)
    vs = gaussian_filter(v, sigma=sigma, mode='nearest')
    ws = gaussian_filter(m.astype(float), sigma=sigma, mode='nearest')
    out = vs / np.maximum(ws, 1e-12)
    out[ws < min_w] = np.nan
    return out


def nan_upsample(arr, factor):
    if factor == 1:
        return arr
    m = np.isfinite(arr).astype(float)
    v = np.where(np.isfinite(arr), arr, 0.0)
    vz = zoom(v, (factor, factor), order=1)
    mz = zoom(m, (factor, factor), order=1)
    out = vz / np.maximum(mz, 1e-12)
    out[mz < 0.5] = np.nan
    return out


def build_land_path(geom):
    try:
        from cartopy.mpl.path import shapely_to_path
        p = shapely_to_path(geom)
        if isinstance(p, list):
            p = mpath.Path.make_compound_path(*p)
        return p
    except Exception:
        from cartopy.mpl.patch import geos_to_path
        return mpath.Path.make_compound_path(*geos_to_path(geom))


n_lat_m   = len(lat_bins) - 1
n_lon_m   = len(lon_bins) - 1
_lat_c_m  = lat_bins[:-1] + GRID_RESOLUTION / 2
_lon_raw  = lon_bins[:-1] + GRID_RESOLUTION / 2
_lon_c_m  = normalize_lon_to_180(_lon_raw)
_lon_ord  = np.argsort(_lon_c_m)
_lon_c_m  = _lon_c_m[_lon_ord]
_lon_msh, _lat_msh = np.meshgrid(_lon_c_m, _lat_c_m)
_land_plot = _shape_contains_xy(land_union, _lon_msh, _lat_msh)


def gridify(colname):
    arr = np.full((n_lat_m, n_lon_m), np.nan, dtype=float)
    lat_left = full_grid_df['lat'].to_numpy(float) - GRID_RESOLUTION / 2
    lon_left = full_grid_df['lon'].to_numpy(float) - GRID_RESOLUTION / 2
    li = np.searchsorted(lat_bins, lat_left, side='right') - 1
    ci = np.searchsorted(lon_bins, lon_left, side='right') - 1
    ok = (li >= 0) & (li < n_lat_m) & (ci >= 0) & (ci < n_lon_m)
    arr[li[ok], ci[ok]] = full_grid_df.loc[ok, colname].to_numpy(float)
    arr = arr[:, _lon_ord]
    arr[~_land_plot] = np.nan
    return arr


def prep_plotgrid(base):
    sm = nan_gaussian(base, SMOOTH_SIGMA)
    sm[~_land_plot] = np.nan
    if PLOT_UPSAMPLE > 1:
        up = nan_upsample(sm, PLOT_UPSAMPLE)
        lon_f = np.linspace(_lon_c_m.min(), _lon_c_m.max(),
                            n_lon_m * PLOT_UPSAMPLE)
        lat_f = np.linspace(_lat_c_m.min(), _lat_c_m.max(),
                            n_lat_m * PLOT_UPSAMPLE)
        lm, ltm = np.meshgrid(lon_f, lat_f)
        up[~_shape_contains_xy(land_union, lm, ltm)] = np.nan
        return np.ma.masked_invalid(up), lm, ltm
    return np.ma.masked_invalid(sm), _lon_msh, _lat_msh


def top_srex(colname, top_n=TOP_N):
    raw = gridify(colname)
    smn = nan_gaussian(raw, SMOOTH_SIGMA); smn[~_land_plot] = np.nan
    ab = assign_srex_vectorized(_lon_msh.ravel(), _lat_msh.ravel())
    vals = smn.ravel()
    byr = {}
    for a, v in zip(ab, vals):
        if a is not None and np.isfinite(v):
            byr.setdefault(a, []).append(v)
    ranked = sorted(
        {a: float(np.mean(vs)) for a, vs in byr.items() if vs}.items(),
        key=lambda x: x[1], reverse=True)[:top_n]
    return [a for a, _ in ranked]


papers = linked[linked['lat'].notna() & linked['lon'].notna()].copy()
papers['lon'] = normalize_lon_to_180(papers['lon'].to_numpy(float))
_rng = np.random.default_rng(RANDOM_SEED)
def _maybe_dn(d):
    if len(d) <= MAX_POINTS: return d
    idx = _rng.choice(d.index.to_numpy(), size=MAX_POINTS, replace=False)
    return d.loc[idx]
_eq  = _maybe_dn(papers[ papers['has_equity']])
_neq = _maybe_dn(papers[~papers['has_equity']])


def _percentile_norm(data_breaks, visual_positions, vmin=0.0, vmax=None):
    """Piecewise-linear norm that maps data_breaks to visual_positions in [0,1]."""
    data_breaks = np.asarray(data_breaks, dtype=float)
    vis = np.asarray(visual_positions, dtype=float)
    if vmax is None:
        vmax = max(1.0, float(data_breaks[-1]) * 1.05)
    x_pts = np.concatenate([[vmin], data_breaks, [vmax]])
    y_pts = np.concatenate([[0.0],  vis,         [1.0]])

    def _forward(v):
        return np.interp(v, x_pts, y_pts)
    def _inverse(y):
        return np.interp(y, y_pts, x_pts)

    return FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax), x_pts, y_pts


def _panel_percentiles(colname):
    lv = full_grid_df[colname].to_numpy(dtype=float)
    if 'srex_region' in full_grid_df.columns:
        lv = lv[full_grid_df['srex_region'].notna().to_numpy()]
    nz = lv[np.isfinite(lv) & (lv > 0)]
    return np.quantile(nz, TIER_QUANTILES), nz.max()


def make_mismatch_map(colname, cmap, title, cbar_label,
                      out_stem, draw_points=True):
    raw = gridify(colname)
    plot_arr, lm, ltm = prep_plotgrid(raw)
    tabs = top_srex(colname)

    # Panel-specific percentile breaks → piecewise norm
    q, panel_max = _panel_percentiles(colname)
    vmax = max(1.0, float(panel_max) * 1.02)
    norm, _, _ = _percentile_norm(q, TIER_VISUAL_POS, vmin=0.0, vmax=vmax)

    # Fine levels for smooth gradient rendering
    n_fine = 12
    fine_levels = np.unique(np.concatenate([
        np.linspace(0.0,  q[0],  n_fine // 3),
        np.linspace(q[0], q[1],  n_fine // 3),
        np.linspace(q[1], q[2],  n_fine // 3),
        np.linspace(q[2], vmax,  n_fine // 3),
    ]))

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor=LAND_COLOR,  zorder=1)

    cf = ax.contourf(lm, ltm, plot_arr, levels=fine_levels,
                     cmap=cmap, norm=norm,
                     transform=ccrs.PlateCarree(), extend='max', zorder=3)

    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=4)
    ax.add_feature(cfeature.LAKES, facecolor=OCEAN_COLOR,
                   edgecolor='#999999', linewidth=0.2, zorder=4)
    clip = PathPatch(build_land_path(land_union),
                     transform=ccrs.PlateCarree(), facecolor='none')
    for art in (getattr(cf, 'collections', None)
                or [c for c in cf.get_children() if hasattr(c, 'set_clip_path')]):
        art.set_clip_path(clip)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor='#666', zorder=6)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25, edgecolor='#999', zorder=6)

    draw_srex_polygons(ax, highlight_abbrevs=tabs,
                       default_edge='#999999', default_lw=0.5, default_ls='--',
                       highlight_edge='#D32F2F', highlight_lw=1.8, highlight_ls='-',
                       zorder_default=7, zorder_highlight=8)
    if draw_points:
        ax.scatter(_neq['lon'], _neq['lat'], transform=ccrs.PlateCarree(),
                   zorder=9, rasterized=True, **NON_EQUITY_STYLE)
        ax.scatter(_eq['lon'],  _eq['lat'],  transform=ccrs.PlateCarree(),
                   zorder=10, rasterized=True, **EQUITY_STYLE)

    # Gradient colorbar. Ticks sit at p25/p50/p75 (numerically specific to
    # this panel) but because we used the piecewise norm, they ALWAYS fall
    # at the same visual positions (25% / 50% / 75% along the bar).
    cb = fig.colorbar(cf, ax=ax, orientation='horizontal',
                      pad=0.07, fraction=0.04, aspect=40, extend='max')
    cb.set_label(cbar_label, fontsize=9, labelpad=26)
    cb.set_ticks(q)
    cb.ax.set_xticklabels([f'{v:.2f}' for v in q])
    cb.ax.tick_params(axis='x', labelsize=9, length=6, width=1.2,
                      colors='#111', pad=3)

    # Tier labels sit on top of the colorbar, centered between breakpoints.
    tier_centers = [
        0.5 * TIER_VISUAL_POS[0],                             # 0 to p25
        0.5 * (TIER_VISUAL_POS[0] + TIER_VISUAL_POS[1]),      # p25 to p50
        0.5 * (TIER_VISUAL_POS[1] + TIER_VISUAL_POS[2]),      # p50 to p75
        0.5 * (TIER_VISUAL_POS[2] + 1.0),                     # p75 to max
    ]
    for c, lab in zip(tier_centers, TIER_LABELS):
        cb.ax.text(c, 1.55, lab,
                   transform=cb.ax.transAxes,
                   ha='center', va='bottom',
                   fontsize=9.5, fontweight='bold', color='#111')
    # Small "p25 / p50 / p75" markers above the numeric ticks
    for pos, lab in zip(TIER_VISUAL_POS, ['p25', 'p50', 'p75']):
        cb.ax.text(pos, 1.05, lab,
                   transform=cb.ax.transAxes,
                   ha='center', va='bottom',
                   fontsize=8, color='#666')

    print(f'  [{colname}] p25={q[0]:.3f}  p50={q[1]:.3f}  p75={q[2]:.3f}  '
          f'max={panel_max:.3f}')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    out_png = OUTPUT_DIR / f'{out_stem}.png'
    out_pdf = OUTPUT_DIR / f'{out_stem}.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf,            bbox_inches='tight', facecolor='white')
    plt.show()
    print('Saved:', out_png)


make_mismatch_map('triple_mismatch', CMAPS['triple'],
    'a. Social Vulnerability x Fire x Research Attention Mismatch',
    'Triple Mismatch Index (percentile-stretched gradient, panel-specific)',
    'Fig5a_triple_mismatch', draw_points=True)

make_mismatch_map('fire_research_mismatch', CMAPS['fire_research'],
    'b. Fire Occurrence x Research Attention Mismatch',
    'Mismatch Index (percentile-stretched gradient, panel-specific)',
    'Fig5b_fire_research_mismatch', draw_points=False)

make_mismatch_map('fire_grdi_index', CMAPS['fire_grdi'],
    'c. Fire Occurrence x Social Vulnerability Mismatch',
    'Mismatch Index (percentile-stretched gradient, panel-specific)',
    'Fig5c_fire_grdi_mismatch', draw_points=False)

make_mismatch_map('grdi_research_mismatch', CMAPS['grdi_research'],
    'd. Social Vulnerability x Research Attention Mismatch',
    'Mismatch Index (percentile-stretched gradient, panel-specific)',
    'Fig5d_grdi_research_mismatch', draw_points=False)

# %% [markdown]
# ## 10. Sensitivity loop: 5, 2, 1, 0.5 deg
#
# Each resolution is cached to disk so re-running the notebook only redoes the
# missing runs. The 2 deg run is read from cache since we already computed it
# in section 8.
#
# **Runtime expectations (Colab free tier, one run each):**
# - 5.0 deg: < 1 min
# - 2.0 deg: cached (already done)
# - 1.0 deg: 4 to 8 min
# - 0.5 deg: 20 to 40 min
#
# You can skip the 0.5 deg run if time is tight; the 5 deg and 1 deg coverage
# alone already brackets the relevant grid-choice range for a sensitivity
# argument.

# %%
SENS_RESOLUTIONS = [5.0, 2.0, 1.0, 0.5]
# SENS_RESOLUTIONS = [5.0, 2.0, 1.0]  # <<< uncomment to skip 0.5 deg

sens = {}
for r in SENS_RESOLUTIONS:
    cache_path = CACHE_DIR / f'mismatch_res_{r}.pkl'
    if cache_path.exists():
        print(f'  [cache hit]  res={r}: loading from {cache_path.name}')
        with open(cache_path, 'rb') as f:
            sens[r] = pickle.load(f)
        continue
    print(f'  [cache miss] res={r}: running build_mismatch_grid')
    sens[r] = build_mismatch_grid(
        res=r, linked_df=linked,
        fire_total=fire_total, fire_transform=fire_transform,
        grdi_data_full=grdi_data_full, grdi_tf=grdi_tf,
        land_union=land_union,
    )
    with open(cache_path, 'wb') as f:
        pickle.dump(sens[r], f)
    print(f'    cached to {cache_path.name}')

print('\n=== Summary ===')
for r in SENS_RESOLUTIONS:
    g = sens[r]['grid_df']
    nz = (g['triple_mismatch'] > 0).sum()
    el = sens[r].get('elapsed_seconds', np.nan)
    print(f'  {r} deg: {len(g):>8,} cells, {nz:>7,} nonzero triple mismatch, '
          f'runtime {el:6.1f}s')

# %% [markdown]
# ## 11. Cross-resolution comparison: SREX-region level
#
# For each resolution, compute the cos-latitude-weighted mean of each mismatch
# index within each of the 26 SREX regions. Then compute Spearman rho of those
# 26 means between every pair of resolutions. This is the comparison that
# matters for the paper's SREX-level headline finding.

# %%
MISMATCH_COLS = ['triple_mismatch', 'fire_research_mismatch',
                 'fire_grdi_index', 'grdi_research_mismatch']


def srex_area_weighted_means(grid_df, cols=MISMATCH_COLS):
    g = grid_df[grid_df['srex_region'].notna()].copy()
    g['_w'] = np.cos(np.deg2rad(g['lat']))
    def _wm(x, col):
        w = x['_w'].to_numpy(); v = x[col].to_numpy()
        ok = np.isfinite(v) & np.isfinite(w)
        return np.average(v[ok], weights=w[ok]) if ok.sum() and w[ok].sum() > 0 else np.nan
    rows = []
    for reg, x in g.groupby('srex_region'):
        rows.append({'srex_region': reg,
                     **{c: _wm(x, c) for c in cols}})
    return pd.DataFrame(rows).set_index('srex_region').sort_index()


srex_means = {r: srex_area_weighted_means(sens[r]['grid_df']) for r in SENS_RESOLUTIONS}

# Spearman rho matrix between resolutions, one per index
rho_tables = {}
for col in MISMATCH_COLS:
    mat = np.full((len(SENS_RESOLUTIONS), len(SENS_RESOLUTIONS)), np.nan)
    for i, ri in enumerate(SENS_RESOLUTIONS):
        for j, rj in enumerate(SENS_RESOLUTIONS):
            a = srex_means[ri][col]; b = srex_means[rj][col]
            idx = a.index.intersection(b.index)
            if len(idx) >= 3:
                mat[i, j] = spearmanr(a.loc[idx], b.loc[idx]).correlation
    rho_tables[col] = pd.DataFrame(
        mat, index=[f'{r}°' for r in SENS_RESOLUTIONS],
        columns=[f'{r}°' for r in SENS_RESOLUTIONS])

print('\n=== SREX-region Spearman rho matrix (26 regions) ===')
for col, tbl in rho_tables.items():
    print(f'\n  {col}:')
    print(tbl.round(3).to_string())

# Save the region-mean tables for the SM
for r in SENS_RESOLUTIONS:
    srex_means[r].to_csv(OUTPUT_DIR / f'SREX_region_means_{r}deg.csv')
pd.concat({r: srex_means[r] for r in SENS_RESOLUTIONS}, axis=1).to_csv(
    OUTPUT_DIR / 'SREX_region_means_all_resolutions.csv')
print(f'\nSaved SREX region-mean tables to {OUTPUT_DIR}/')

# %% [markdown]
# ## 12. Cross-resolution comparison: cell level
#
# For each pair `(fine, coarse)` with `coarse / fine` integer, aggregate the
# finer grid to the coarser grid by cos-latitude-weighted mean of the mismatch
# indices. Compute Spearman rho across all land cells that exist at both
# resolutions.

# %%
def aggregate_finer_to_coarser(fine_grid_df, fine_res, coarse_res,
                               cols=MISMATCH_COLS,
                               lat_range=(-60, 85), lon_range=(-180, 180)):
    """Cos-lat-weighted mean of fine cells inside each coarse cell."""
    if (coarse_res / fine_res) != round(coarse_res / fine_res):
        raise ValueError(f'coarse_res/fine_res must be integer; got {coarse_res/fine_res}')
    g = fine_grid_df.copy()
    g = g[g['is_land']].copy()
    g['_w'] = np.cos(np.deg2rad(g['lat']))

    # Snap lat/lon to coarse cell centers
    cl_lat = np.arange(lat_range[0], lat_range[1], coarse_res) + coarse_res / 2
    cl_lon = np.arange(lon_range[0], lon_range[1], coarse_res) + coarse_res / 2
    g['coarse_lat'] = cl_lat[np.clip(
        np.floor((g['lat'] - lat_range[0]) / coarse_res).astype(int),
        0, len(cl_lat) - 1)]
    g['coarse_lon'] = cl_lon[np.clip(
        np.floor((g['lon'] - lon_range[0]) / coarse_res).astype(int),
        0, len(cl_lon) - 1)]

    def _wmean_block(x, c):
        w = x['_w'].to_numpy(); v = x[c].to_numpy()
        ok = np.isfinite(v) & np.isfinite(w)
        return np.average(v[ok], weights=w[ok]) if ok.sum() and w[ok].sum() > 0 else np.nan

    rows = []
    for (cla, clo), x in g.groupby(['coarse_lat', 'coarse_lon']):
        rows.append({'coarse_lat': cla, 'coarse_lon': clo,
                     **{c: _wmean_block(x, c) for c in cols}})
    return pd.DataFrame(rows)


cell_rho_rows = []
for fine in SENS_RESOLUTIONS:
    for coarse in SENS_RESOLUTIONS:
        if coarse < fine:
            continue
        if (coarse / fine) != round(coarse / fine):
            continue
        if fine == coarse:
            r = 1.0; n = (sens[coarse]['grid_df']['is_land']).sum()
        else:
            agg = aggregate_finer_to_coarser(sens[fine]['grid_df'], fine, coarse)
            ref = sens[coarse]['grid_df'][['lat','lon'] + MISMATCH_COLS].copy()
            ref = ref.rename(columns={'lat':'coarse_lat','lon':'coarse_lon'})
            merged = ref.merge(agg, on=['coarse_lat','coarse_lon'],
                               suffixes=('_coarse','_fine'))
            r_dict = {}
            for c in MISMATCH_COLS:
                a = merged[f'{c}_coarse']; b = merged[f'{c}_fine']
                ok = np.isfinite(a) & np.isfinite(b)
                r_dict[c] = (spearmanr(a[ok], b[ok]).correlation
                             if ok.sum() >= 10 else np.nan)
            cell_rho_rows.append({'fine_res': fine, 'coarse_res': coarse,
                                  'n_cells': len(merged), **r_dict})
            continue
        cell_rho_rows.append({'fine_res': fine, 'coarse_res': coarse,
                              'n_cells': int(n),
                              **{c: r for c in MISMATCH_COLS}})

cell_rho_df = pd.DataFrame(cell_rho_rows)
print('\n=== Cell-level Spearman rho (finer -> coarser aggregation) ===')
print(cell_rho_df.round(3).to_string(index=False))
cell_rho_df.to_csv(OUTPUT_DIR / 'Cell_level_Spearman_rho.csv', index=False)
print(f'\nSaved: {OUTPUT_DIR / "Cell_level_Spearman_rho.csv"}')

# %% [markdown]
# ## 13. SM figure: triple mismatch at each resolution
#
# Four Robinson-projection panels, one per resolution. No study-point overlay
# to keep the figure readable.

# %%
from matplotlib.colors import FuncNorm

# Tier definitions: same scheme as Figure 5
TIER_QUANTILES  = [0.25, 0.50, 0.75]
TIER_VISUAL_POS = [0.25, 0.50, 0.75]
TIER_LABELS     = ['Low', 'Elevated', 'Moderate', 'Severe']


def _percentile_norm(data_breaks, visual_positions, vmin=0.0, vmax=None):
    """Piecewise-linear norm: data_breaks → visual_positions on [0,1]."""
    data_breaks = np.asarray(data_breaks, dtype=float)
    vis = np.asarray(visual_positions, dtype=float)
    if vmax is None:
        vmax = max(1.0, float(data_breaks[-1]) * 1.05)
    x_pts = np.concatenate([[vmin], data_breaks, [vmax]])
    y_pts = np.concatenate([[0.0],  vis,         [1.0]])

    def _forward(v): return np.interp(v, x_pts, y_pts)
    def _inverse(y): return np.interp(y, y_pts, x_pts)

    return FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)


def plot_triple_mismatch_panel(ax, result_dict, cmap,
                                smooth_sigma=0.7, upsample=2, title=None):
    res = result_dict['GRID_RESOLUTION']
    lat_b = result_dict['lat_bins']; lon_b = result_dict['lon_bins']
    gdf = result_dict['grid_df']
    n_lat = len(lat_b) - 1; n_lon = len(lon_b) - 1
    lat_c = lat_b[:-1] + res / 2
    lon_c_raw = lon_b[:-1] + res / 2
    lon_c = normalize_lon_to_180(lon_c_raw)
    lon_ord = np.argsort(lon_c)
    lon_c = lon_c[lon_ord]
    lon_m, lat_m = np.meshgrid(lon_c, lat_c)
    lmask = _shape_contains_xy(land_union, lon_m, lat_m)

    arr = np.full((n_lat, n_lon), np.nan)
    li = np.searchsorted(lat_b, gdf['lat'].to_numpy(float) - res/2, side='right') - 1
    ci = np.searchsorted(lon_b, gdf['lon'].to_numpy(float) - res/2, side='right') - 1
    ok = (li >= 0) & (li < n_lat) & (ci >= 0) & (ci < n_lon)
    arr[li[ok], ci[ok]] = gdf.loc[ok, 'triple_mismatch'].to_numpy(float)
    arr = arr[:, lon_ord]
    arr[~lmask] = np.nan

    sm = nan_gaussian(arr, smooth_sigma); sm[~lmask] = np.nan
    if upsample > 1:
        sm = nan_upsample(sm, upsample)
        lon_f = np.linspace(lon_c.min(), lon_c.max(), n_lon * upsample)
        lat_f = np.linspace(lat_c.min(), lat_c.max(), n_lat * upsample)
        lm, ltm = np.meshgrid(lon_f, lat_f)
        sm[~_shape_contains_xy(land_union, lm, ltm)] = np.nan
    else:
        lm, ltm = lon_m, lat_m

    plot_arr = np.ma.masked_invalid(sm)

    # Compute panel-specific percentile breaks
    nz = gdf.loc[gdf['srex_region'].notna() & (gdf['triple_mismatch'] > 0),
                 'triple_mismatch'].to_numpy()
    if len(nz) >= 4:
        q = np.quantile(nz, TIER_QUANTILES)
        panel_max = float(nz.max())
    else:
        q = np.array([0.05, 0.15, 0.30])
        panel_max = 0.5
    vmax = max(1.0, panel_max * 1.02)
    norm = _percentile_norm(q, TIER_VISUAL_POS, vmin=0.0, vmax=vmax)

    # Fine levels for smooth gradient
    n_fine = 24
    fine_levels = np.unique(np.concatenate([
        np.linspace(0.0,  q[0],  n_fine // 3),
        np.linspace(q[0], q[1],  n_fine // 3),
        np.linspace(q[1], q[2],  n_fine // 3),
        np.linspace(q[2], vmax,  n_fine // 3),
    ]))

    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor=LAND_COLOR,  zorder=1)
    cf = ax.contourf(lm, ltm, plot_arr, levels=fine_levels,
                     cmap=cmap, norm=norm,
                     transform=ccrs.PlateCarree(), extend='max', zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#666', zorder=6)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.2, edgecolor='#999', zorder=6)
    draw_srex_polygons(ax, default_edge='#888', default_lw=0.4, default_ls='--',
                       zorder_default=7)
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    return cf, q, vmax


# ---------- Figure assembly --------------------------------------------------
n_panels = len(SENS_RESOLUTIONS)
ncols = 2
nrows = int(np.ceil(n_panels / ncols))
fig = plt.figure(figsize=(16, 5 * nrows))
panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']

cf_last = None
q_per_panel = {}

for k, r in enumerate(SENS_RESOLUTIONS):
    ax = fig.add_subplot(nrows, ncols, k + 1, projection=ccrs.Robinson())
    cf_last, q, vmax = plot_triple_mismatch_panel(
        ax, sens[r], cmap=CMAPS['triple'],
        title=f'{panel_labels[k]}. Triple mismatch at {r}° grid '
              f'({sens[r]["n_lat"]} x {sens[r]["n_lon"]} cells)')
    q_per_panel[r] = q
    n_total = sens[r]['n_lat'] * sens[r]['n_lon']
    n_land  = int(sens[r]['land_mask_2d'].sum())
    ax.text(0.02, -0.06,
            f'Land cells: {n_land:,} / {n_total:,}  |  '
            f'p25={q[0]:.2f}  p50={q[1]:.2f}  p75={q[2]:.2f}',
            transform=ax.transAxes, fontsize=9, color='#444', ha='left')

# Shared bottom colorbar with tier labels at fixed visual positions
cbar_ax = fig.add_axes([0.20, 0.05, 0.6, 0.022])
cb = plt.colorbar(cf_last, cax=cbar_ax, orientation='horizontal',
                  extend='max', ticks=[])
cb.set_label('Triple Mismatch Index (panel-specific percentile-stretched gradient)',
             fontsize=10, labelpad=22)

# Tier labels above the bar
tier_centers = [
    0.5 * TIER_VISUAL_POS[0],
    0.5 * (TIER_VISUAL_POS[0] + TIER_VISUAL_POS[1]),
    0.5 * (TIER_VISUAL_POS[1] + TIER_VISUAL_POS[2]),
    0.5 * (TIER_VISUAL_POS[2] + 1.0),
]
for c, lab in zip(tier_centers, TIER_LABELS):
    cbar_ax.text(c, 1.55, lab, transform=cbar_ax.transAxes,
                 ha='center', va='bottom',
                 fontsize=9.5, fontweight='bold', color='#111')

# p25 / p50 / p75 markers
for pos, lab in zip(TIER_VISUAL_POS, ['p25', 'p50', 'p75']):
    cbar_ax.text(pos, 1.05, lab, transform=cbar_ax.transAxes,
                 ha='center', va='bottom',
                 fontsize=8, color='#666')

fig.suptitle('Triple mismatch index at four analysis-grid resolutions',
             fontsize=14, fontweight='bold', y=0.98)
plt.subplots_adjust(top=0.93, bottom=0.10, hspace=0.10, wspace=0.05)

for ext in ['png', 'pdf']:
    out = OUTPUT_DIR / f'FigSENS1_triple_mismatch_all_res.{ext}'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print('Saved:', out)
plt.show()

# Print panel breakpoints for the SM caption
print('\nPanel-specific p25/p50/p75 (for caption):')
for r, q in q_per_panel.items():
    print(f'  {r}°: p25={q[0]:.3f}, p50={q[1]:.3f}, p75={q[2]:.3f}')

# %% [markdown]
# ## 14. Top-10 SREX stability table
#
# For each resolution and each mismatch index, list the 10 SREX regions with
# the highest area-weighted mean. Jaccard index is computed against the 2 deg
# baseline as a single-number stability summary.

# %%
def top_n_regions(srex_mean_df, col, n=10):
    return list(srex_mean_df[col].sort_values(ascending=False).head(n).index)


def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / max(len(A | B), 1)


rows_top = []
for col in MISMATCH_COLS:
    baseline_top = top_n_regions(srex_means[BASELINE_RES], col, 10)
    for r in SENS_RESOLUTIONS:
        toplist = top_n_regions(srex_means[r], col, 10)
        rows_top.append({
            'index': col, 'res': r,
            'top_10_SREX': ', '.join(toplist),
            'jaccard_vs_2deg': round(jaccard(toplist, baseline_top), 3),
        })

top_df = pd.DataFrame(rows_top)
print('\n=== Top-10 SREX regions per index per resolution ===')
with pd.option_context('display.width', 220, 'display.max_colwidth', 120):
    print(top_df.to_string(index=False))

top_df.to_csv(OUTPUT_DIR / 'Top10_SREX_by_resolution.csv', index=False)
print(f'\nSaved: {OUTPUT_DIR / "Top10_SREX_by_resolution.csv"}')

# %% [markdown]
# ## 15. Bundle outputs for the supplement
#
# Writes a single folder containing every table and figure the sensitivity
# section of the SM will reference.

# %%
written = []
for p in OUTPUT_DIR.iterdir():
    if p.is_file():
        written.append(p.name)
print(f'Output folder: {OUTPUT_DIR}')
print(f'Files ({len(written)}):')
for f in sorted(written):
    print(f'  {f}')

# %%
# =============================================================================
# Diagnostics for revision text (Section 5.2 + SM C5.1a + SM C5.7)
# =============================================================================

# ---------- (i) Footprint precision distribution of papers in the surface ----
# Recover the per-paper footprint statistics directly from `linked` so we can
# report what fraction of papers have region-specific (small) vs. large MOBs.
import json

def _parse_regs(s):
    if not isinstance(s, str) or s in ('', '[]'):
        return []
    try:
        return json.loads(s) or []
    except Exception:
        return []

def _finest_span(s):
    regs = _parse_regs(s)
    spans = []
    for r in regs:
        try:
            mob = r.get('MOB')
            if not mob: continue
            tl, br = mob['top_left'], mob['bottom_right']
            spans.append(max(abs(tl[0]-br[0]), abs(tl[1]-br[1])))
        except Exception:
            continue
    return min(spans) if spans else None

linked['_finest_span'] = linked['geo_regions_json'].apply(_finest_span)
in_surface = linked[linked['_finest_span'].notna()].copy()
n_used = len(in_surface)

bins = [
    ('region-specific (< 1°)',     in_surface['_finest_span'] <  1.0),
    ('subnational (1–5°)',          (in_surface['_finest_span'] >= 1.0) & (in_surface['_finest_span'] < 5.0)),
    ('country / subcontinental (5–20°)', (in_surface['_finest_span'] >= 5.0) & (in_surface['_finest_span'] < 20.0)),
    ('continental (20–60°)',        (in_surface['_finest_span'] >= 20.0) & (in_surface['_finest_span'] < 60.0)),
    ('global (≥ 60°)',              in_surface['_finest_span'] >= 60.0),
]
print('=== (i) Footprint precision of papers in the gridded surface ===')
print(f'Total papers contributing to surface: n = {n_used}')
for label, mask in bins:
    k = int(mask.sum())
    print(f'  {label:42s}: {k:5d} ({100*k/n_used:5.1f}%)')

# ---------- (ii) High-mismatch land share at p75 threshold ------------------
# Re-create the % land share that gets called "high-mismatch" in Section 5.2.
print('\n=== (ii) High-mismatch land share (>= panel p75) at 2° baseline ===')
land_df = full_grid_df[full_grid_df['srex_region'].notna()].copy()
n_land = len(land_df)
for col, pretty in [
    ('fire_research_mismatch', 'Fire × Research'),
    ('fire_grdi_index',        'Fire × GRDI'),
    ('grdi_research_mismatch', 'GRDI × Research'),
    ('triple_mismatch',        'Triple'),
]:
    nz = land_df[col][land_df[col] > 0]
    p75 = float(nz.quantile(0.75)) if len(nz) else float('nan')
    high = (land_df[col] >= p75).sum()
    print(f'  {pretty:20s}  p75={p75:5.3f}   high-mismatch land share = '
          f'{100*high/n_land:5.1f}%   ({high}/{n_land} cells)')

# ---------- (iii) Centroid vs. areal-allocation Spearman (provenance) -------
# We already stored both surfaces in `full_grid_df`; this gives the headline
# robustness number for the SM provenance subsection.
from scipy.stats import spearmanr

print('\n=== (iii) Centroid vs. areal-weighted research attention (2° baseline) ===')

# Build a centroid-only triple_mismatch alongside the areal-weighted one
def build_triple_from_paper_count(df, n_col):
    f = df['fire_pct'].copy()
    g = df['grdi_pct'].copy()
    log_n = pd.Series(__import__('numpy').log1p(df[n_col]), index=df.index)
    q95 = max(log_n.quantile(0.95), 1.0)
    research_intensity = (log_n / __import__('numpy').log1p(q95)).clip(0, 1)
    research_gap = 1 - research_intensity
    from scipy.stats import rankdata
    import numpy as _np
    v = research_gap.to_numpy()
    rk = _np.zeros_like(v, dtype=float)
    m = _np.isfinite(v) & (v >= 0)
    if m.sum() > 0:
        r = rankdata(v[m], method='average')
        rk[m] = r / r.max()
    gap_pct = pd.Series(rk, index=df.index)
    prod = f * g * gap_pct
    base = _np.cbrt(prod)
    dmin = _np.minimum(_np.minimum(f, g), gap_pct)
    conv = _np.where(dmin < 0.25, dmin / 0.25, 1.0)
    return base * conv

land = full_grid_df[full_grid_df['srex_region'].notna()].copy()
land['triple_centroid'] = build_triple_from_paper_count(land, 'n_papers_centroid')
ok = (land['triple_mismatch'] > 0) | (land['triple_centroid'] > 0)
rho_cell, _ = spearmanr(land.loc[ok, 'triple_mismatch'],
                        land.loc[ok, 'triple_centroid'])
print(f'  cell-level Spearman ρ (centroid vs. areal): {rho_cell:.3f}  (n={int(ok.sum())})')

# SREX-region level
def _srex_mean(df, col):
    df = df.copy()
    import numpy as _np
    df['_w'] = _np.cos(_np.deg2rad(df['lat']))
    out = {}
    for reg, x in df.groupby('srex_region'):
        ok2 = x[col].notna() & x['_w'].notna()
        if ok2.sum() and x.loc[ok2, '_w'].sum() > 0:
            out[reg] = float(_np.average(x.loc[ok2, col], weights=x.loc[ok2, '_w']))
    return pd.Series(out)

a = _srex_mean(land, 'triple_mismatch')
b = _srex_mean(land, 'triple_centroid')
both = a.index.intersection(b.index)
rho_srex, _ = spearmanr(a.loc[both], b.loc[both])
print(f'  SREX-region Spearman ρ (centroid vs. areal): {rho_srex:.3f}  (k={len(both)} regions)')

# Top-10 overlap
top_a = set(a.sort_values(ascending=False).head(10).index)
top_b = set(b.sort_values(ascending=False).head(10).index)
jacc = len(top_a & top_b) / len(top_a | top_b)
print(f'  Top-10 SREX Jaccard (centroid vs. areal): {jacc:.3f}')
print(f'    Areal:    {sorted(top_a)}')
print(f'    Centroid: {sorted(top_b)}')

# %%
# =============================================================================
# SREX-region tier classification: how many of the 26 regions have area-weighted
# mean mismatch in Low / Elevated / Moderate / Severe?
#
# This is empirically informative even though cell-level binning is tautological,
# because regions span many cells with different cos-lat weights — a region's mean
# is not forced to fall in any particular tier by the binning.
# =============================================================================

# Step 1: For each index, get cell-level p25/p50/p75 of the nonzero distribution
# (same breakpoints the figure colorbar uses).
# Step 2: Compute area-weighted mean per SREX region.
# Step 3: Classify each region's mean into one of four tiers.
# Step 4: Count regions per tier.

land_df = full_grid_df[full_grid_df['srex_region'].notna()].copy()
land_df['_w'] = np.cos(np.deg2rad(land_df['lat']))

INDICES = {
    'Triple mismatch':                'triple_mismatch',
    'Fire × Research gap':            'fire_research_mismatch',
    'Fire × Social vulnerability':    'fire_grdi_index',
    'Social vulnerability × Research': 'grdi_research_mismatch',
}
TIERS = ['Low', 'Elevated', 'Moderate', 'Severe']


def srex_means(df, col):
    """Cos-latitude weighted mean of `col` for each SREX region."""
    out = {}
    for reg, x in df.groupby('srex_region'):
        v = x[col].to_numpy(); w = x['_w'].to_numpy()
        ok = np.isfinite(v) & np.isfinite(w)
        if ok.sum() and w[ok].sum() > 0:
            out[reg] = float(np.average(v[ok], weights=w[ok]))
    return pd.Series(out)


def classify(value, p25, p50, p75):
    if value < p25:  return 'Low'
    if value < p50:  return 'Elevated'
    if value < p75:  return 'Moderate'
    return 'Severe'


region_summary = []
region_per_tier = {}  # for printing the actual region names per tier

for label, col in INDICES.items():
    # Cell-level breakpoints (panel-specific, nonzero only — matches figures)
    nz = land_df[col][land_df[col] > 0].to_numpy()
    p25, p50, p75 = np.quantile(nz, [0.25, 0.50, 0.75])

    # Region means
    means = srex_means(land_df, col)
    tiers = means.apply(lambda v: classify(v, p25, p50, p75))

    # Counts per tier
    counts = {t: int((tiers == t).sum()) for t in TIERS}
    region_summary.append({
        'Index': label,
        **counts,
        'p25': round(float(p25), 3),
        'p50': round(float(p50), 3),
        'p75': round(float(p75), 3),
    })

    # Region lists per tier (for the SM caption / rebuttal letter)
    region_per_tier[label] = {t: sorted(tiers[tiers == t].index.tolist()) for t in TIERS}

# Display
region_tier_table = pd.DataFrame(region_summary)
print('=== SREX-region tier classification (n = 26 regions per row) ===\n')
print(region_tier_table[['Index'] + TIERS + ['p25', 'p50', 'p75']].to_string(index=False))
print()
print(f"Sanity check: row sums should equal 26 (the number of SREX regions)")
print(f"  {region_tier_table[TIERS].sum(axis=1).tolist()}")

# Save for the SM
out_csv = OUTPUT_DIR / 'SREX_tier_counts.csv'
region_tier_table.to_csv(out_csv, index=False)
print(f'\nSaved counts: {out_csv}')

# Print region names per tier (useful for the rebuttal and the SM caption)
print('\n=== Regions per tier (Severe + Moderate, the bins worth naming) ===')
for label in INDICES:
    print(f'\n{label}:')
    for tier in ['Severe', 'Moderate']:
        regs = region_per_tier[label][tier]
        print(f'  {tier:8s} ({len(regs):2d}): {", ".join(regs) if regs else "—"}')

# LaTeX-ready table
print('\n--- LaTeX (copy into SM) ---')
print(r'\begin{tabular}{lcccc}')
print(r'\toprule')
print(r'\textbf{Index} & \textbf{Low} & \textbf{Elevated} & \textbf{Moderate} & \textbf{Severe} \\')
print(r'\midrule')
for _, r in region_tier_table.iterrows():
    print(f"{r['Index']} & {r['Low']} & {r['Elevated']} & {r['Moderate']} & {r['Severe']} \\\\")
print(r'\bottomrule')
print(r'\end{tabular}')

# %%
# =============================================================================
# Regenerate SM Tables E1, E2, E3 from the footprint-aware areal surface
# Format matches Supplementary_Material.tex exactly so output can be pasted in.
# =============================================================================

# Mapping from SREX abbreviation to the full name used in the SM tables
SREX_FULL_NAME = {
    'ALA': 'Alaska / N.W. Canada',
    'CGI': 'Canada / Greenland / Iceland',
    'WNA': 'West North America',
    'CNA': 'Central North America',
    'ENA': 'East North America',
    'CAM': 'Central America / Mexico',
    'AMZ': 'South America (Amazon)',
    'NEB': 'Northeast Brazil',
    'WSA': 'West Coast South America',
    'SSA': 'Southeastern South America',
    'NEU': 'North Europe',
    'CEU': 'Central Europe',
    'MED': 'South Europe / Mediterranean',
    'SAH': 'Sahara',
    'WAF': 'West Africa',
    'EAF': 'East Africa',
    'SAF': 'Southern Africa',
    'NAS': 'North Asia',
    'WAS': 'West Asia',
    'CAS': 'Central Asia',
    'TIB': 'Tibetan Plateau',
    'EAS': 'East Asia',
    'SAS': 'South Asia',
    'SEA': 'Southeast Asia',
    'NAU': 'Northern Australia',
    'SAU': 'Southern Australia / New Zealand',
}

# --- Compute area-weighted SREX region means for each pairwise index ---
land_df = full_grid_df[full_grid_df['srex_region'].notna()].copy()
land_df['_w'] = np.cos(np.deg2rad(land_df['lat']))


def srex_means(df, col):
    out = {}
    for reg, x in df.groupby('srex_region'):
        v = x[col].to_numpy(); w = x['_w'].to_numpy()
        ok = np.isfinite(v) & np.isfinite(w)
        if ok.sum() and w[ok].sum() > 0:
            out[reg] = float(np.average(v[ok], weights=w[ok]))
    return pd.Series(out).sort_values(ascending=False)


def render_top10_table(means_series, table_label, caption, math_symbol, top_n=10):
    """Build LaTeX table matching the SM E1/E2/E3 format."""
    out = []
    out.append(r'\begin{table}[H]')
    out.append(r'\centering')
    out.append(rf'\caption{{{caption}}}')
    out.append(rf'\label{{{table_label}}}')
    out.append(r'\small')
    out.append(r'\begin{tabular}{clc}')
    out.append(r'\toprule')
    out.append(rf'\textbf{{Rank}} & \textbf{{SREX Region}} & \textbf{{Mean ${math_symbol}$}} \\')
    out.append(r'\midrule')
    for i, (abbrev, val) in enumerate(means_series.head(top_n).items(), 1):
        full = SREX_FULL_NAME.get(abbrev, abbrev)
        out.append(f'{i} & {full} & {val:.3f} \\\\')
    out.append(r'\bottomrule')
    out.append(r'\end{tabular}')
    out.append(r'\end{table}')
    return '\n'.join(out)


# --- Tables E1, E2, E3 ---
specs = [
    ('fire_research_mismatch', 'tab:E1_rankings',
     r'SREX region rankings by mean fire-research mismatch index ($M_{\mathrm{FR}}$). Top ten regions shown; full rankings are available in the data repository.',
     r'M_{\mathrm{FR}}'),
    ('fire_grdi_index', 'tab:E2_rankings',
     r'SREX region rankings by mean fire-vulnerability mismatch index ($M_{\mathrm{FV}}$). Top ten regions shown.',
     r'M_{\mathrm{FV}}'),
    ('grdi_research_mismatch', 'tab:E3_rankings',
     r'SREX region rankings by mean vulnerability-research mismatch index ($M_{\mathrm{VG}}$). Top ten regions shown.',
     r'M_{\mathrm{VG}}'),
]

print('=' * 70)
print('SREX-region top-10 rankings (footprint-aware areal allocation, 2 deg)')
print('=' * 70)
for col, label, caption, sym in specs:
    means = srex_means(land_df, col)
    print(f'\n--- {label} ({col}) ---')
    for i, (abbrev, val) in enumerate(means.head(10).items(), 1):
        print(f'  {i:2d}. {abbrev}  {SREX_FULL_NAME.get(abbrev, abbrev):35s}  {val:.3f}')

# --- LaTeX output for direct paste into SM ---
print('\n\n' + '=' * 70)
print('LATEX OUTPUT — paste these blocks into Supplementary_Material.tex')
print('replacing the existing E1, E2, E3 tables')
print('=' * 70)
for col, label, caption, sym in specs:
    means = srex_means(land_df, col)
    print('\n')
    print(render_top10_table(means, label, caption, sym))

# --- Save full rankings to CSV (all 26 regions, not just top 10) ---
all_means = pd.DataFrame({
    label.split('_')[0]: srex_means(land_df, col)
    for col, label, _, _ in specs
})
all_means.index = [SREX_FULL_NAME.get(a, a) for a in all_means.index]
all_means = all_means.sort_index()
out_csv = OUTPUT_DIR / 'SREX_full_rankings_E1_E2_E3.csv'
all_means.to_csv(out_csv)
print(f'\n\nSaved full 26-region rankings to: {out_csv}')
