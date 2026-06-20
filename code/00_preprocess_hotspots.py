# %% [markdown]
# # Global Wildfire-Linked Compound Hazard Hotspot Analysis
#
# **Purpose:** Export annual compound climate hazard layers (2000-2024) for global wildfire research.
#
# **Outputs per year:**
# - HOT, DRY, WINDY, FLOOD (individual hazards)
# - HOT_DRY, HOT_WINDY, DRY_WINDY, HOT_DRY_WINDY, WINDY_FLOOD (compounds)
# - Fire frequency from MODIS MCD64A1
#
# **Data Sources:**
# - ECMWF/ERA5_LAND/DAILY_AGGR (climate variables)
# - MODIS/061/MCD64A1 (burned area)

# %% [markdown]
# ## Cell 1: Setup & Configuration


# %%
import ee
import time
import os
import glob

# =============================================
# CONFIGURATION - EDIT THESE AS NEEDED
# =============================================

# Your Google Earth Engine / Cloud project ID.
EE_PROJECT = "your-ee-project-id"  # <-- REQUIRED: change to your own project ID

# Time configuration
YEAR_START = 2001
YEAR_END = 2024  # Inclusive

# Baseline period for percentile thresholds
BASELINE_START = '2001-01-01'
BASELINE_END = '2024-12-31'

# Spatial configuration
SCALE = 50000          # 50 km resolution (memory-safe for global exports)
SUMMARY_SCALE = 250000  # For reduceRegion stats

GLOBAL_REGION = ee.Geometry.Rectangle(
    [-180, -60, 180, 85],
    proj='EPSG:4326',
    geodesic=False
)

# Export configuration.
# GEE exports land in Google Drive. After they finish, move this folder into the
# repository's data/ directory (e.g. data/hotspot_50km/) so the downstream
# notebooks (02_rr_figure4, 03_mismatch_sensitivity) can read it.
FOLDER_NAME = "Global_Hotspot_50km_2001_2024"
DRIVE_BASE = os.environ.get("DRIVE_BASE", "/content/drive/MyDrive")
EXPORT_FOLDER = os.path.join(DRIVE_BASE, FOLDER_NAME)

# Percentile thresholds (Supplementary Material, Table C1)
P_HOT = 90    # Temperature percentile (hot = above p90)
P_DRY = 10    # Precipitation percentile (dry = below p10)
P_WIND = 99   # Wind percentile (windy = above p99)
P_FLOOD = 99  # Runoff percentile (flood = above p99)

# Compound event pairs to compute (Supplementary Material, Table C2)
PAIRS = ['HOT_DRY', 'HOT_WINDY', 'DRY_WINDY', 'HOT_DRY_WINDY', 'WINDY_FLOOD']

# Expected files per year (4 hazards + 5 compounds + 1 fire = 10)
EXPECTED_PER_YEAR = 10

# ---------------------------------------------
# Authenticate and initialize Earth Engine.
# Placed AFTER EE_PROJECT is defined (the original ran this before the
# variable existed, which raised NameError on a clean kernel).
# ---------------------------------------------
try:
    ee.Initialize(project=EE_PROJECT)
    print("Earth Engine already authenticated")
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)
    print("Earth Engine authenticated")

print("Configuration loaded")
print(f"  Analysis period: {YEAR_START}-{YEAR_END}")
print(f"  Baseline: {BASELINE_START} to {BASELINE_END}")
print(f"  Scale: {SCALE/1000:.0f} km")
print(f"  Export folder: {FOLDER_NAME}")

# %% [markdown]
# ## Cell 3: Compute Baseline Percentile Thresholds
#
# Using ERA5-Land daily aggregates to compute global percentiles for each hazard variable.

# %%
# ERA5-Land band names
B_TMAX_K = "temperature_2m_max"        # Kelvin
B_TP_M = "total_precipitation_sum"     # meters (daily sum)
B_U10 = "u_component_of_wind_10m"      # m/s
B_V10 = "v_component_of_wind_10m"      # m/s
B_SRO_M = "surface_runoff_sum"         # meters (daily sum) - flood proxy


def preprocess_era5land_image(image):
    """
    Convert ERA5-Land image to analysis bands:
      - temp_celsius (°C)
      - precip_mm (mm/day)
      - wind_speed (m/s)
      - runoff_mm (mm/day) - flood proxy
    """
    image = ee.Image(image)

    temp_celsius = image.select(B_TMAX_K).subtract(273.15).rename('temp_celsius')
    precip_mm = image.select(B_TP_M).multiply(1000).rename('precip_mm')
    u = image.select(B_U10)
    v = image.select(B_V10)
    wind_speed = u.hypot(v).rename('wind_speed')
    runoff_mm = image.select(B_SRO_M).multiply(1000).rename('runoff_mm')

    out = image.addBands([temp_celsius, precip_mm, wind_speed, runoff_mm]).clip(GLOBAL_REGION)
    return ee.Image(out.copyProperties(image, ['system:time_start']))


# Build baseline collection
print(f"Building baseline from ERA5-Land ({BASELINE_START} to {BASELINE_END})...")

era5land_baseline = (
    ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    .filterDate(BASELINE_START, BASELINE_END)
    .filterBounds(GLOBAL_REGION)
    .map(preprocess_era5land_image)
)

print(f"  Baseline collection size: {era5land_baseline.size().getInfo()} images")

# Compute percentile thresholds
print("Computing percentile thresholds...")

temp_threshold = (
    era5land_baseline.select('temp_celsius')
    .reduce(ee.Reducer.percentile([P_HOT]))
    .rename('temp_p90')
)

precip_threshold = (
    era5land_baseline.select('precip_mm')
    .reduce(ee.Reducer.percentile([P_DRY]))
    .rename('precip_p10')
)

wind_threshold = (
    era5land_baseline.select('wind_speed')
    .reduce(ee.Reducer.percentile([P_WIND]))
    .rename('wind_p99')
)

runoff_threshold = (
    era5land_baseline.select('runoff_mm')
    .reduce(ee.Reducer.percentile([P_FLOOD]))
    .rename('runoff_p99')
)

# Combine all thresholds into single image
percentiles = temp_threshold.addBands([precip_threshold, wind_threshold, runoff_threshold])

print("✓ Baseline percentiles computed")

# Print threshold summaries
print("\n--- Threshold Summaries (global mean/min/max) ---")

reducer = ee.Reducer.mean().combine(ee.Reducer.minMax(), '', True)

for band in ['temp_p90', 'precip_p10', 'wind_p99', 'runoff_p99']:
    band_img = percentiles.select(band)
    stats = band_img.reduceRegion(
        reducer=reducer,
        geometry=GLOBAL_REGION,
        scale=SUMMARY_SCALE,
        bestEffort=True,
        maxPixels=1e8
    ).getInfo()

    print(f"\n{band}:")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if v else f"  {k}: None")

# %% [markdown]
# ## Cell 4: Define Hazard Detection Functions

# %%
def detect_hazards(img):
    """
    Convert one ERA5-Land daily image into HOT/DRY/WINDY/FLOOD boolean bands
    using precomputed percentile thresholds.
    """
    img = ee.Image(img)

    hot = img.select('temp_celsius').gt(percentiles.select('temp_p90')).rename('HOT')
    dry = img.select('precip_mm').lt(percentiles.select('precip_p10')).rename('DRY')
    windy = img.select('wind_speed').gt(percentiles.select('wind_p99')).rename('WINDY')
    flood = img.select('runoff_mm').gt(percentiles.select('runoff_p99')).rename('FLOOD')

    out = hot.addBands([dry, windy, flood]).clip(GLOBAL_REGION)
    return ee.Image(out.copyProperties(img, ['system:time_start']))


def detect_compound(hazards_ic, pair_name):
    """
    Given a hazards ImageCollection (with HOT, DRY, WINDY, FLOOD bands),
    return an ImageCollection with a single 'compound' boolean band.
    """
    def mapper(img):
        img = ee.Image(img)
        hot = img.select('HOT')
        dry = img.select('DRY')
        windy = img.select('WINDY')
        flood = img.select('FLOOD')

        if pair_name == "HOT_DRY":
            compound = hot.And(dry)
        elif pair_name == "HOT_WINDY":
            compound = hot.And(windy)
        elif pair_name == "DRY_WINDY":
            compound = dry.And(windy)
        elif pair_name == "HOT_DRY_WINDY":
            compound = hot.And(dry).And(windy)
        elif pair_name == "WINDY_FLOOD":
            compound = windy.And(flood)
        else:
            compound = hot.And(dry)  # Fallback

        out = compound.rename('compound')
        return ee.Image(out.copyProperties(img, ['system:time_start']))

    return hazards_ic.map(mapper)


def get_yearly_hazards(year):
    """
    Get HOT/DRY/WINDY/FLOOD daily maps for a single calendar year.
    """
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, 'year')

    era5_year = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .filterBounds(GLOBAL_REGION)
        .map(preprocess_era5land_image)
    )

    return era5_year.map(detect_hazards)


def get_yearly_fire_frequency(year):
    """
    MODIS burned-area based fire frequency (count of burn events) for a year.
    """
    start = f"{year}-01-01"
    end = f"{year+1}-01-01"

    modis_ba = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(start, end)
        .filterBounds(GLOBAL_REGION)
    )

    fire_ic = modis_ba.map(
        lambda img: img.select('BurnDate')
                       .gt(0)
                       .rename('fire')
                       .copyProperties(img, ['system:time_start'])
    )

    fire_freq_image = (
        fire_ic.select('fire')
        .sum()
        .rename('fire_frequency')
        .toFloat()
        .clip(GLOBAL_REGION)
    )

    return fire_freq_image


print("✓ Hazard detection functions defined")

# %% [markdown]
# ## Cell 5: Define Export Functions

# %%
def export_to_drive(image, description, region, scale, folder=FOLDER_NAME, crs='EPSG:4326'):
    """
    Export one image to Google Drive and wait for completion.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e10,
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True}
    )
    task.start()
    print(f"    Started export: {description}")

    # Poll task status
    while task.active():
        status = task.status()
        state = status.get('state', 'UNKNOWN')
        print(f"      Status: {state} ...", end='\r')
        time.sleep(10)

    status = task.status()
    state = status.get('state', 'UNKNOWN')
    print(f"      Final status: {state}                      ")

    if state == 'COMPLETED':
        print(f"    ✓ Exported: {description}")
        return description
    else:
        print(f"    ✗ Export FAILED for {description}: {status}")
        return None


def export_if_missing(image, description, region, scale, folder=FOLDER_NAME, crs='EPSG:4326'):
    """
    Export image to Drive only if the .tif file doesn't already exist.
    """
    tif_name = f"{description}.tif"
    full_path = os.path.join(EXPORT_FOLDER, tif_name)

    if os.path.exists(full_path):
        print(f"    ⏭️  {tif_name} already exists, skipping.")
        return description

    print(f"    ⬆️  Exporting {tif_name} ...")
    return export_to_drive(
        image=image,
        description=description,
        region=region,
        scale=scale,
        folder=folder,
        crs=crs,
    )


def summarize_frequency(image, label):
    """
    Print basic occurrence stats for a frequency image (mean/min/max days per year).
    """
    reducer = ee.Reducer.mean().combine(ee.Reducer.minMax(), '', True)
    stats = image.reduceRegion(
        reducer=reducer,
        geometry=GLOBAL_REGION,
        scale=SUMMARY_SCALE,
        bestEffort=True,
        maxPixels=1e8
    ).getInfo()

    print(f"  >> Occurrence summary for {label}:")
    for k, v in stats.items():
        if v is not None:
            print(f"     {k}: {v:.2f}")
        else:
            print(f"     {k}: None")


print("✓ Export functions defined")

# %% [markdown]
# ## Cell 6: Process Single Year Function

# %%
def process_one_year(year):
    """
    Compute and export all hazard, compound, and fire layers for a single year.
    """
    print("\n" + "="*50)
    print(f"Processing year {year}")
    print("="*50)

    hazards_year = get_yearly_hazards(year)
    n_days = hazards_year.size().getInfo()
    print(f"  • ERA5-Land daily images: {n_days}")

    # --- Individual hazard frequencies ---
    hazard_names = ['HOT', 'DRY', 'WINDY', 'FLOOD']

    for hazard in hazard_names:
        print(f"  • Computing {hazard} frequency...")

        freq_img = (
            hazards_year.select(hazard)
            .sum()
            .rename('frequency')
            .toFloat()
            .clip(GLOBAL_REGION)
        )

        summarize_frequency(freq_img, f"{hazard} ({year})")

        desc = f"hotspot_{year}_{hazard}_days"
        export_if_missing(freq_img, desc, region=GLOBAL_REGION, scale=SCALE)

    # --- Compound event frequencies ---
    for pair in PAIRS:
        print(f"  • Computing compound {pair} frequency...")

        compound_ic = detect_compound(hazards_year, pair)

        freq_img = (
            compound_ic.select('compound')
            .sum()
            .rename('frequency')
            .toFloat()
            .clip(GLOBAL_REGION)
        )

        summarize_frequency(freq_img, f"{pair} ({year})")

        desc = f"hotspot_{year}_{pair}_days"
        export_if_missing(freq_img, desc, region=GLOBAL_REGION, scale=SCALE)

    # --- Fire frequency ---
    print("  • Computing fire frequency (MODIS MCD64A1)...")
    fire_freq_img = get_yearly_fire_frequency(year)
    summarize_frequency(fire_freq_img, f"FIRE ({year})")

    desc = f"hotspot_{year}_fire_frequency"
    export_if_missing(fire_freq_img, desc, region=GLOBAL_REGION, scale=SCALE)

    print(f"  ✓ Finished year {year}")
    return year


print("✓ Year processing function defined")

# %% [markdown]
# ## Cell 7: Check Existing Exports

# %%
# Check what files already exist
print(f"Checking for existing exports in: {EXPORT_FOLDER}")

existing_files_by_year = {}

if os.path.isdir(EXPORT_FOLDER):
    all_existing_files = glob.glob(os.path.join(EXPORT_FOLDER, "*.tif"))
    print(f"  Found {len(all_existing_files)} existing file(s)")

    for path in all_existing_files:
        name = os.path.basename(path)
        for year in range(YEAR_START, YEAR_END + 1):
            if str(year) in name:
                existing_files_by_year.setdefault(year, []).append(name)

    # Summary
    print("\nExisting files by year:")
    for y in range(YEAR_START, YEAR_END + 1):
        count = len(existing_files_by_year.get(y, []))
        status = "✓ complete" if count >= EXPECTED_PER_YEAR else f"({count}/{EXPECTED_PER_YEAR})"
        print(f"  {y}: {status}")
else:
    print("  Export folder does not exist yet (no previous exports)")
    all_existing_files = []

# %% [markdown]
# ## Cell 8: Run Export Loop
#
# This cell processes all years that don't have complete exports yet.

# %%
# Process all years with missing/incomplete exports
processed_years = []
skipped_years = []

for year in range(YEAR_START, YEAR_END + 1):
    current_files = existing_files_by_year.get(year, [])
    count = len(current_files)

    if count >= EXPECTED_PER_YEAR:
        print(f"\n⏭️  Year {year}: {count}/{EXPECTED_PER_YEAR} files → skipping")
        skipped_years.append(year)
    else:
        print(f"\n▶️  Year {year}: {count}/{EXPECTED_PER_YEAR} files → processing")
        process_one_year(year)
        processed_years.append(year)

print("\n" + "="*50)
print("Export Summary")
print("="*50)
print(f"Processed: {len(processed_years)} years")
print(f"Skipped (already complete): {len(skipped_years)} years")

if processed_years:
    print(f"\nProcessed years: {processed_years}")
if skipped_years:
    print(f"Skipped years: {skipped_years}")

# %% [markdown]
# ## Cell 9: Process Single Year (Optional)
#
# Use this to process a specific year if needed.

# %%
# Uncomment to process a specific year:
process_one_year(2001)

# %% [markdown]
# ## Cell 10: Async Export (Faster Alternative)
#
# Launch all exports without waiting for each to complete. Useful for submitting many tasks quickly.

# %%
def export_async(image, description, region, scale, folder=FOLDER_NAME, crs='EPSG:4326'):
    """
    Start export task without waiting for completion.
    Returns the task object.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e10,
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True}
    )
    task.start()
    return task


def process_year_async(year):
    """
    Launch all export tasks for a year without waiting.
    Returns list of task objects.
    """
    tasks = []
    hazards_year = get_yearly_hazards(year)

    # Individual hazards
    for hazard in ['HOT', 'DRY', 'WINDY', 'FLOOD']:
        freq_img = (
            hazards_year.select(hazard)
            .sum()
            .rename('frequency')
            .toFloat()
            .clip(GLOBAL_REGION)
        )
        desc = f"hotspot_{year}_{hazard}_days"

        tif_path = os.path.join(EXPORT_FOLDER, f"{desc}.tif")
        if not os.path.exists(tif_path):
            task = export_async(freq_img, desc, GLOBAL_REGION, SCALE)
            tasks.append((desc, task))
            print(f"  Started: {desc}")

    # Compounds
    for pair in PAIRS:
        compound_ic = detect_compound(hazards_year, pair)
        freq_img = (
            compound_ic.select('compound')
            .sum()
            .rename('frequency')
            .toFloat()
            .clip(GLOBAL_REGION)
        )
        desc = f"hotspot_{year}_{pair}_days"

        tif_path = os.path.join(EXPORT_FOLDER, f"{desc}.tif")
        if not os.path.exists(tif_path):
            task = export_async(freq_img, desc, GLOBAL_REGION, SCALE)
            tasks.append((desc, task))
            print(f"  Started: {desc}")

    # Fire
    fire_freq_img = get_yearly_fire_frequency(year)
    desc = f"hotspot_{year}_fire_frequency"

    tif_path = os.path.join(EXPORT_FOLDER, f"{desc}.tif")
    if not os.path.exists(tif_path):
        task = export_async(fire_freq_img, desc, GLOBAL_REGION, SCALE)
        tasks.append((desc, task))
        print(f"  Started: {desc}")

    return tasks


print("✓ Async export functions defined")

# %%
# Launch async exports for all incomplete years
all_tasks = []

for year in range(YEAR_START, YEAR_END + 1):
    current_count = len(existing_files_by_year.get(year, []))

    if current_count < EXPECTED_PER_YEAR:
        print(f"\n▶️  Year {year}:")
        year_tasks = process_year_async(year)
        all_tasks.extend(year_tasks)

print(f"\n{'='*50}")
print(f"Launched {len(all_tasks)} export tasks")
