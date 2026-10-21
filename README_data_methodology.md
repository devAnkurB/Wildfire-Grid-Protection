# Data Methodology

This document lists the data methodology techniques used in this project pipeline.

## Data Sources Used
- CPUC utility ignition/incident Excel reports (PG&E, SCE, SDG&E; 2014-2024)
- GridMET daily climate NetCDF data (`bi`, `erc`, `rmin`, `vs`, `tmmx`, `pr`)
- LANDFIRE FBFM40 fuel raster (`LF2024_FBFM40_CONUS.tif`)
- USGS 3DEP DEM tiles (for centroid-level topography)

## Data Methodology Techniques
- Multi-file ingestion from heterogeneous yearly utility reports
- Dynamic header-row detection for inconsistent Excel schemas
- Column alias mapping to a canonical schema
- Duplicate-column handling after schema harmonization
- Utility normalization from text values with filename fallback
- Type coercion and standardization (`date`, coordinates, voltage)
- California spatial bounding-box filtering
- Longitude sign correction for malformed positive longitudes
- Fire-size bucketization into standardized categories
- Cause/event bucketization into normalized categories
- Line-type bucketization (`overhead`, `underground`, etc.)
- Inflation-period tagging for PG&E 2020-2021 rows
- Confirmed ignition label construction from fire-size buckets (>= 0.25 acres)
- Use of pre-2021 rows as denominator non-ignition days due to reporting-format mismatch
- Unique-location caching for faster geospatial weather joins
- Nearest-grid-cell spatial join from incident location to GridMET lat/lon grid
- Day-of-year temporal indexing into daily NetCDF weather variables
- Year-partitioned NetCDF dataset caching to reduce file-open overhead
- GridMET fill-value and invalid-range handling
- Temperature unit conversion (Kelvin to Celsius)
- LANDFIRE CRS-aware sampling (lat/lon to EPSG:5070 Albers)
- Strip/window-based raster sampling to avoid loading full CONUS fuel raster in memory
- Fuel code-to-name mapping (Scott & Burgan FBFM40)
- Fuel code-to-category mapping (`grass`, `shrub`, `timber_litter`, etc.)
- Cluster-centroid topography extraction from DEM tiles only for required tiles
- 3x3 neighborhood terrain derivative for slope estimation
- Assembly of final feature dataset for modeling (`incidents_features.csv`)

## Data Products Generated
- `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/fire_incidents/incidents_clean.csv`
- `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/fire_incidents/incidents_gridmet.csv`
- `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/fire_incidents/incidents_features.csv`
