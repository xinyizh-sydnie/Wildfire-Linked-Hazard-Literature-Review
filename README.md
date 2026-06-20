# Wildfire as Urban Risk: Global Synthesis of Compound Hazards, Cascade Pathways, and Research–Exposure–Vulnerability Mismatch

Code and processed data for the manuscript by Xinyi Zhang and Lu Liang (Landscape Architecture and Environmental Planning, University of California, Berkeley). Corresponding author: lianglu@berkeley.edu.

This repository contains the analysis code and the processed data needed to reproduce the figures and supplementary tables. The large input rasters and the subscription-restricted literature corpus are not redistributed here; the sources and the steps to obtain or regenerate them are documented below.

## Repository structure

```
.
├── README.md
├── LICENSE                       MIT (code). Data and processed outputs: CC BY 4.0 (see License).
├── requirements.txt
├── .gitignore
├── code/
│   ├── 00_preprocess_hotspots.py   Earth Engine: build 50 km fire + compound-hazard rasters (SI C1, C2)
│   ├── 01_llm_classification.py    LLM screening + classification of the literature corpus (SI B)
│   ├── 02_rr.py            Relative-risk analysis and Figure 4 / Figure D1 (SI C3, D)
│   ├── 03_mismatch_sensitivity.py  Mismatch indices, resolution sensitivity, SM Tables E1–E4, C7–C9 (SI C5)
│   └── srex_regions.py             Helper: loads the IPCC SREX regions via regionmask (see SREX reference regions)
└── data/
    ├── study_table.csv             Classified literature table (output of 01, trimmed). Tracked.
    ├── validation_set.csv          350-record validation subset. Tracked.
    ├── hotspot_50km/               Output of 00 (yearly fire + compound rasters). Not tracked; large.
    ├── grdi/                       GRDI raster (download from SEDAC). Not tracked.
    ├── wui/                        10 km WUI fraction rasters (derived from Schug et al. 2023). Not tracked.
    ├── land_10km/                  10 km land/setting rasters (derived from Schug et al. 2023). Not tracked.
    └── external/                   Kelley et al. (2019) figure (copyright; obtain separately). Not tracked.
```

Scripts read from `data/` and write to `outputs/` by default. Both paths can be redirected with the `DATA_DIR` and `OUTPUT_DIR` environment variables (useful when running on Google Colab with a Drive copy of the data).

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Notes:
- `cartopy`, `regionmask`, and `rasterio` are sensitive to version and to system geospatial libraries (GEOS, PROJ, GDAL). If `pip` install fails, installing these through `conda-forge` is the most reliable route.
- `code/00_preprocess_hotspots.py` runs on Google Earth Engine and requires a registered Earth Engine project. Set your own project ID at the top of the file (the placeholder is `your-ee-project-id`). It is most easily run on Google Colab.
- `code/01_llm_classification.py` calls the OpenAI API. Set your key first: `export OPENAI_API_KEY="sk-..."`.

## Data sources

The global analysis uses four publicly available datasets plus a subscription database. None of the raw inputs are redistributed in this repository.

| Dataset | Source | Used by |
| --- | --- | --- |
| MODIS MCD64A1 Collection 6.1 burned area | NASA LP DAAC, https://lpdaac.usgs.gov | 00 |
| ERA5-Land daily aggregates | Copernicus CDS via Google Earth Engine, https://developers.google.com/earth-engine/datasets | 00 |
| Global 10 m wildland–urban interface classification | Schug et al. (2023), Nature, https://www.nature.com/articles/s41586-023-06320-0 | 02 (after aggregation to 10 km) |
| Global Gridded Relative Deprivation Index (GRDI) | NASA SEDAC, https://sedac.ciesin.columbia.edu | 02, 03 |
| Web of Science Core Collection (literature corpus) | Subscription required; search strings in SI Section A | 01 |

Not included, and why:
- Raw Web of Science records and abstract text are not shared (Web of Science terms and publisher copyright). `study_table.csv` contains only the derived classification labels and geoparsed bounding boxes, not abstract text. The article identifiers in `study_table.csv` allow re-retrieval of the corpus by anyone with Web of Science access.
- The four raw rasters above are large and externally hosted, so they are cited rather than mirrored.
- The Kelley et al. (2019) fire-regime figure used in the Figure D1 overlay is copyrighted (Springer Nature). It is not committed; download it from https://www.nature.com/articles/s41558-019-0540-7 and place it at `data/external/kelley2019_fig2.jpg`.
- The 10 km WUI and land/setting rasters (`data/wui/`, `data/land_10km/`) are aggregated from the Schug et al. (2023) 10 m product following SI Section C1.3. The aggregation step itself is described in the SI but is not included as a separate script in this repository.

### SREX reference regions

The 26 IPCC SREX reference regions are not stored in this repository. `code/srex_regions.py` loads them at runtime from the `regionmask` package and adds helper functions (region assignment and plotting). The region definitions originate from the IPCC Special Report on Managing the Risks of Extreme Events and Disasters to Advance Climate Change Adaptation (SREX): Seneviratne et al. (2012), in *Managing the Risks of Extreme Events and Disasters to Advance Climate Change Adaptation*, Cambridge University Press, pp. 109–230. The `regionmask` implementation is documented at https://regionmask.readthedocs.io. Please use the SREX citation as it appears in the manuscript for consistency.

## Reproducing the analysis

The shipped `study_table.csv` is the already-classified output, so steps 1 and 2 below are only needed to reproduce the classification itself.

1. **Preprocessing (optional; Earth Engine).** Run `code/00_preprocess_hotspots.py` to export the yearly fire-frequency and compound-hazard rasters to `data/hotspot_50km/`.
2. **Classification (optional; OpenAI API).** Run `code/01_llm_classification.py` on a raw Web of Science export (with abstracts; user-supplied) to regenerate the classification table. The published `study_table.csv` is the trimmed result of this step.
3. **Relative risk and figures.** Run `code/02_rr_figure4.py` to produce the relative-risk estimates, Figure 4, and Supplementary Figure D1.
4. **Mismatch indices and sensitivity.** Run `code/03_mismatch_sensitivity.py` to build the mismatch indices, run the resolution sensitivity analysis, and regenerate Supplementary Tables E1–E4 and C7–C9.

Scripts 02 and 03 both import `code/srex_regions.py`; run them from the `code/` directory (or add it to `PYTHONPATH`) so the import resolves. The figure scripts are saved in the percent-cell format (`# %%`), so they run top to bottom as plain scripts and can also be opened cell by cell in VS Code or Jupyter.

## Data availability statement

The global analysis draws on four publicly available datasets (MODIS MCD64A1, ERA5-Land, the global 10 m WUI classification of Schug et al. 2023, and the GRDI), each accessible at the URLs above. The literature corpus was retrieved from the Web of Science Core Collection, which requires an institutional subscription; full search strings are in Supplementary Material A. The processed data and analysis code are available in this repository.

## License

Code is released under the MIT License (see `LICENSE`). The processed data files in `data/` (`study_table.csv`, `validation_set.csv`) are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Third-party datasets remain under their original licenses.
