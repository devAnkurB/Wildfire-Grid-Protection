"""
add_fuels_topo.py  v4
=====================
Step 3: Add LANDFIRE fuel model features to incidents_gridmet.csv.

Reads:   fire_incidents/incidents_gridmet.csv
Reads:   environmental_data/CONUS/Tif/*.tif       (LANDFIRE FBFM40)
Writes:  fire_incidents/incidents_features.csv

New columns
───────────
  fuel_model_code  FBFM40 integer code
  fuel_model_name  Scott & Burgan (2005) description
  fuel_category    grass / grass_shrub / shrub / timber_understory /
                   timber_litter / slash / nonburnable / unknown

Key fixes vs v2
───────────────
  1. LANDFIRE raster is in EPSG:5070 Albers Equal Area Conic (metres),
     NOT geographic lat/lon. The bounding box calculation was converting
     lat/lon directly to pixel coords which produced a negative-height window.
     Fixed: pure-numpy Albers forward projection converts each incident's
     lat/lon to Albers (x, y) metres before computing pixel indices.

  2. LZW compression cannot be read by tifffile without imagecodecs.
     Fixed: use Pillow (PIL) which links against libtiff and handles LZW
     natively. PIL also reads ModelPixelScaleTag / ModelTiepointTag for
     the geotransform. The CONUS raster (~2 GB) is sampled using
     PIL.Image.crop() strips sorted by row, avoiding full load into RAM.

  3. Nodata value 32767 (int16 max) handled correctly.

No additional packages required — uses Pillow (PIL) for all raster I/O.
both already installed.

Run from project root:
    python add_fuels_topo.py
"""

from __future__ import annotations
import logging, math, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HERE       = Path(__file__).resolve().parent
INPUT_CSV  = HERE / "fire_incidents" / "incidents_gridmet.csv"
OUTPUT_CSV = HERE / "fire_incidents" / "incidents_features.csv"
TOPO_DIR   = HERE / "environmental_data" / "topo"
FUELS_DIR  = HERE / "environmental_data" / "CONUS" / "Tif"

FBFM40_NODATA = 32767

# ── EPSG:5070 Albers Equal Area Conic forward projection ─────────────────────
# Parameters from EPSG registry (NAD83 / Conus Albers)
_A   = 6378137.0
_E2  = 2*(1/298.257222101) - (1/298.257222101)**2
_E   = math.sqrt(_E2)
_PHI1 = math.radians(29.5)
_PHI2 = math.radians(45.5)
_PHI0 = math.radians(23.0)
_LAM0 = math.radians(-96.0)

def _m(phi: float) -> float:
    s = math.sin(phi)
    return math.cos(phi) / math.sqrt(1 - _E2 * s * s)

def _q(phi: float) -> float:
    s = math.sin(phi)
    return (1 - _E2) * (s / (1 - _E2*s*s) - (1/(2*_E)) * math.log((1-_E*s)/(1+_E*s)))

_m1 = _m(_PHI1); _m2 = _m(_PHI2)
_q0 = _q(_PHI0); _q1 = _q(_PHI1); _q2 = _q(_PHI2)
_N   = (_m1**2 - _m2**2) / (_q2 - _q1)
_C   = _m1**2 + _N * _q1
_RHO0 = _A * math.sqrt(_C - _N * _q0) / _N

def latlon_to_albers(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84/NAD83 lat/lon (degrees) to EPSG:5070 Albers (x, y) metres."""
    phi = math.radians(lat)
    lam = math.radians(lon)
    qq  = _q(phi)
    rho = _A * math.sqrt(_C - _N * qq) / _N
    theta = _N * (lam - _LAM0)
    x = rho * math.sin(theta)
    y = _RHO0 - rho * math.cos(theta)
    return x, y


# ── GeoTIFF geotransform utilities ───────────────────────────────────────────

def parse_gt_tifffile(page) -> dict:
    """Extract geotransform from tifffile page tags."""
    tags = {tag.name: tag.value for tag in page.tags.values()}
    scale    = tags.get("ModelPixelScaleTag") or tags.get(33550)
    tiepoint = tags.get("ModelTiepointTag")   or tags.get(33922)
    if scale is None or tiepoint is None:
        raise ValueError("Missing GeoTIFF geotransform tags")
    s = np.array(scale,    dtype=float).flatten()
    t = np.array(tiepoint, dtype=float).flatten()
    return dict(origin_x = float(t[3]) - float(t[0]) * float(s[0]),
                origin_y = float(t[4]) - float(t[1]) * (-float(s[1])),
                pixel_w  =  float(s[0]),
                pixel_h  = -float(s[1]))


def parse_gt_pil(img: Image.Image) -> dict:
    """Extract geotransform from PIL TiffImagePlugin tag_v2."""
    tag_v2 = getattr(img, "tag_v2", {})
    # Also check tag (PIL's older interface)
    tag = getattr(img, "tag", {})

    scale    = tag_v2.get(33550) or tag.get(33550)
    tiepoint = tag_v2.get(33922) or tag.get(33922)

    if scale is None or tiepoint is None:
        raise ValueError("PIL: missing ModelPixelScaleTag/ModelTiepointTag")

    s = np.array(scale,    dtype=float).flatten()
    t = np.array(tiepoint, dtype=float).flatten()
    return dict(origin_x = float(t[3]) - float(t[0]) * float(s[0]),
                origin_y = float(t[4]) - float(t[1]) * (-float(s[1])),
                pixel_w  =  float(s[0]),
                pixel_h  = -float(s[1]))


def xy_to_pixel(x: float, y: float, gt: dict) -> tuple[int, int]:
    """Convert projected (x, y) to (row, col) pixel indices."""
    col = int((x - gt["origin_x"]) / gt["pixel_w"])
    row = int((y - gt["origin_y"]) / gt["pixel_h"])
    return row, col


def latlon_to_pixel_geographic(lat, lon, gt) -> tuple[int, int]:
    """For geographic CRS rasters (DEM tiles): lat/lon directly to pixel."""
    col = int((lon - gt["origin_x"]) / gt["pixel_w"])
    row = int((lat - gt["origin_y"]) / gt["pixel_h"])
    return row, col


# ── Scott & Burgan FBFM40 lookup ─────────────────────────────────────────────

FBFM40_NAMES = {
    91:"Urban/Developed", 92:"Snow/Ice", 93:"Agriculture",
    98:"Water", 99:"Bare Ground",
    101:"GR1 Short sparse dry climate grass",
    102:"GR2 Low load dry climate grass",
    103:"GR3 Low load very coarse humid climate grass",
    104:"GR4 Moderate load dry climate grass",
    105:"GR5 Low load humid climate grass",
    106:"GR6 Moderate load humid climate grass",
    107:"GR7 High load dry climate grass",
    108:"GR8 High load very coarse humid climate grass",
    109:"GR9 Very high load humid climate grass",
    121:"GS1 Low load dry climate grass-shrub",
    122:"GS2 Moderate load dry climate grass-shrub",
    123:"GS3 Moderate load humid climate grass-shrub",
    124:"GS4 High load humid climate grass-shrub",
    141:"SH1 Low load dry climate shrub",
    142:"SH2 Moderate load dry climate shrub",
    143:"SH3 Moderate load humid climate shrub",
    144:"SH4 Low load humid climate shrub",
    145:"SH5 High load dry climate shrub",
    146:"SH6 Low load humid climate shrub",
    147:"SH7 Very high load dry climate shrub",
    148:"SH8 High load humid climate shrub",
    149:"SH9 Very high load humid climate shrub",
    161:"TU1 Low load dry climate timber-grass-shrub",
    162:"TU2 Moderate load humid climate timber-shrub",
    163:"TU3 Moderate load humid climate timber-grass-shrub",
    164:"TU4 Dwarf conifer understory",
    165:"TU5 Very high load dry climate timber-shrub",
    181:"TL1 Low load compact conifer litter",
    182:"TL2 Low load broadleaf litter",
    183:"TL3 Moderate load conifer litter",
    184:"TL4 Small downed logs",
    185:"TL5 High load conifer litter",
    186:"TL6 High load broadleaf litter",
    187:"TL7 Large downed logs",
    188:"TL8 Long-needle litter",
    189:"TL9 Very high load broadleaf litter",
    201:"SB1 Low load activity fuel",
    202:"SB2 Moderate load activity fuel or low load blowdown",
    203:"SB3 High load activity fuel or moderate load blowdown",
    204:"SB4 High load blowdown",
}

def fuel_category(code: int) -> str:
    if code in (91,92,93,98,99): return "nonburnable"
    if 101 <= code <= 109:       return "grass"
    if 121 <= code <= 124:       return "grass_shrub"
    if 141 <= code <= 149:       return "shrub"
    if 161 <= code <= 165:       return "timber_understory"
    if 181 <= code <= 189:       return "timber_litter"
    if 201 <= code <= 204:       return "slash"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# LANDFIRE FUELS — PIL-based strip reading (handles LZW, Albers CRS)
# ─────────────────────────────────────────────────────────────────────────────

class FuelsRaster:
    """
    Samples LANDFIRE FBFM40 raster using PIL for LZW decompression.

    Strategy: sort all query points by raster row, then read horizontal
    strips (100-row bands) covering clusters of nearby points. Each strip
    is ~100 × 20000 × 2 bytes ≈ 4 MB — completely manageable.
    The full CONUS raster is never loaded into memory.

    CRS: EPSG:5070 Albers Equal Area Conic.
    Incident lat/lon is converted to Albers (x,y) before pixel lookup.
    """

    def __init__(self, fuels_dir: Path):
        tifs = list(fuels_dir.glob("*.tif"))
        if not tifs:
            raise FileNotFoundError(f"No .tif found in {fuels_dir}")
        self.path = tifs[0]
        log.info(f"LANDFIRE raster: {self.path.name}")

        # Disable PIL decompression bomb check — CONUS raster is legitimately
        # large (~15B pixels). We never load it fully; only small strips are read.
        Image.MAX_IMAGE_PIXELS = None

        # Open once to read metadata only (no pixel data yet)
        with Image.open(self.path) as img:
            self.gt    = parse_gt_pil(img)
            self.nrows = img.height
            self.ncols = img.width
            self.mode  = img.mode

        log.info(f"  Size: {self.nrows:,} × {self.ncols:,}  mode={self.mode}")
        log.info(f"  Origin: ({self.gt['origin_x']:.0f}, {self.gt['origin_y']:.0f})")
        log.info(f"  Pixel: {self.gt['pixel_w']:.1f} m")

        # Verify projection by checking Sacramento
        x_sac, y_sac = latlon_to_albers(38.5, -121.5)
        r_sac, c_sac = xy_to_pixel(x_sac, y_sac, self.gt)
        log.info(f"  Sanity (Sacramento 38.5N,121.5W): "
                 f"row={r_sac}, col={c_sac}  "
                 f"in-bounds={0<=r_sac<self.nrows and 0<=c_sac<self.ncols}")

    def _albers_pixel(self, lat: float, lon: float) -> tuple[int,int]:
        x, y = latlon_to_albers(lat, lon)
        return xy_to_pixel(x, y, self.gt)

    def sample_batch(self, coords: list[tuple[float,float]]) -> list[int|None]:
        """
        Sample FBFM40 at a list of (lat, lon) tuples.
        Reads the raster in sorted row-order strips to minimise I/O.
        Returns a list of integer codes (or None for nodata/OOB).
        """
        n = len(coords)
        # Convert all coords to pixel (row, col)
        pixels = []
        for lat, lon in coords:
            r, c = self._albers_pixel(lat, lon)
            pixels.append((r, c))

        # Sort by row for sequential strip reading
        order = sorted(range(n), key=lambda i: pixels[i][0])
        results = [None] * n

        STRIP_HEIGHT = 150   # rows per strip
        i = 0
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(self.path) as img:
            while i < len(order):
                idx     = order[i]
                row_ctr = pixels[idx][0]

                # Collect all points within this strip
                strip_top = max(0, row_ctr - STRIP_HEIGHT // 2)
                strip_bot = min(self.nrows, strip_top + STRIP_HEIGHT)
                strip_top = max(0, strip_bot - STRIP_HEIGHT)  # re-anchor

                # Find all query points in this strip
                batch = []
                j = i
                while j < len(order):
                    r, c = pixels[order[j]]
                    if strip_top <= r < strip_bot:
                        batch.append(order[j])
                        j += 1
                    else:
                        break

                # Compute column range for this batch
                cols_in_batch = [pixels[k][1] for k in batch]
                c_min = max(0, min(cols_in_batch) - 2)
                c_max = min(self.ncols, max(cols_in_batch) + 3)

                # Read the strip (PIL crop: left, upper, right, lower)
                strip = np.array(
                    img.crop((c_min, strip_top, c_max, strip_bot))
                ).astype(np.int32)

                # Sample each point
                for k in batch:
                    r, c = pixels[k]
                    local_r = r - strip_top
                    local_c = c - c_min
                    if (0 <= local_r < strip.shape[0] and
                        0 <= local_c < strip.shape[1]):
                        val = int(strip[local_r, local_c])
                        if val != FBFM40_NODATA and val > 0:
                            results[k] = val

                i = j

        return results


# ─────────────────────────────────────────────────────────────────────────────
# DEM TILE MANAGER (unchanged — DEM tiles are geographic CRS, uncompressed)
# ─────────────────────────────────────────────────────────────────────────────

class DEMTileManager:
    """
    USGS 3DEP 1/3 arc-second tiles. Geographic CRS (lat/lon).
    Filename: USGS_13_nLATwLON_DATE.tif

    Memory strategy: never load full tiles into RAM.
    Use PIL.Image.crop() to read only a small window around each query
    point. Each crop reads ~(2*half+1)^2 pixels — a few KB vs ~940 MB
    for the full tile. The geotransform is cached (negligible size).
    """

    def __init__(self, topo_dir: Path):
        self.topo_dir = topo_dir
        self._gt_cache: dict = {}   # key -> geotransform dict only (no array)
        self._index    = self._build_index()
        log.info(f"DEM tiles: {len(self._index)} found in {topo_dir}")

    def _build_index(self):
        idx = {}
        pat = re.compile(r"USGS_13_n(\d+)w(\d+)", re.IGNORECASE)
        for f in sorted(self.topo_dir.glob("*.tif")):
            m = pat.search(f.name)
            if m:
                idx[(int(m.group(1)), int(m.group(2)))] = f
        return idx

    def _key(self, lat, lon):
        return (int(math.ceil(lat)), int(math.ceil(-lon)))

    def _get_gt(self, key) -> dict | None:
        """Return geotransform for tile, reading metadata only (no pixel data)."""
        if key in self._gt_cache:
            return self._gt_cache[key]
        if key not in self._index:
            return None
        path = self._index[key]
        log.info(f"  Loading DEM metadata: {path.name}")
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as img:
            gt = parse_gt_pil(img)
            nrows, ncols = img.height, img.width
        gt["nrows"] = nrows
        gt["ncols"] = ncols
        gt["path"]  = path
        self._gt_cache[key] = gt
        return gt

    def _crop_window(self, gt: dict, row: int, col: int, half: int) -> np.ndarray | None:
        """Read a (2*half+1)x(2*half+1) window via PIL crop — no full tile load."""
        c0 = max(0, col - half);  c1 = min(gt["ncols"], col + half + 1)
        r0 = max(0, row - half);  r1 = min(gt["nrows"], row + half + 1)
        if c1 <= c0 or r1 <= r0:
            return None
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(gt["path"]) as img:
            arr = np.array(img.crop((c0, r0, c1, r1))).astype(float)
        arr[(arr < -500) | (arr > 9000)] = np.nan
        # Pad back to full (2*half+1)x(2*half+1) if near edge
        full = (2*half+1, 2*half+1)
        if arr.shape != full:
            padded = np.full(full, np.nan)
            pr = row - half - r0 + (row - half < 0 and half - row or 0)
            pc = col - half - c0 + (col - half < 0 and half - col or 0)
            pr = max(0, r0 - (row - half))
            pc = max(0, c0 - (col - half))
            padded[pr:pr+arr.shape[0], pc:pc+arr.shape[1]] = arr
            return padded
        return arr

    def sample(self, lat, lon) -> float | None:
        gt = self._get_gt(self._key(lat, lon))
        if gt is None: return None
        row, col = latlon_to_pixel_geographic(lat, lon, gt)
        if not (0 <= row < gt["nrows"] and 0 <= col < gt["ncols"]):
            return None
        w = self._crop_window(gt, row, col, half=0)
        if w is None or np.isnan(w[0,0]): return None
        return float(w[0, 0])

    def window(self, lat, lon, half=5) -> np.ndarray | None:
        gt = self._get_gt(self._key(lat, lon))
        if gt is None: return None
        row, col = latlon_to_pixel_geographic(lat, lon, gt)
        if not (0 <= row < gt["nrows"] and 0 <= col < gt["ncols"]):
            return None
        return self._crop_window(gt, row, col, half=half)


# ─────────────────────────────────────────────────────────────────────────────
# TERRAIN DERIVATIVES
# ─────────────────────────────────────────────────────────────────────────────

def compute_terrain(window: np.ndarray | None, cell_m=10.0) -> dict:
    """Compute slope only (Horn 1981)."""
    empty = {"slope_deg": np.nan}
    if window is None or window.shape[0] < 3 or window.shape[1] < 3:
        return empty
    h = window.shape[0] // 2
    w = window.shape[1] // 2
    z = window[h-1:h+2, w-1:w+2]
    if np.isnan(z).any():
        return empty

    dz_dx = ((z[0,2]+2*z[1,2]+z[2,2]) - (z[0,0]+2*z[1,0]+z[2,0])) / (8*cell_m)
    dz_dy = ((z[2,0]+2*z[2,1]+z[2,2]) - (z[0,0]+2*z[0,1]+z[0,2])) / (8*cell_m)
    slope  = float(np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))))
    return {"slope_deg": slope}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN JOIN
# ─────────────────────────────────────────────────────────────────────────────

def join_topo_fuels(df: pd.DataFrame, fuels: FuelsRaster, dem: DEMTileManager | None = None) -> pd.DataFrame:
    unique = (df[["latitude", "longitude"]].dropna()
              .drop_duplicates().reset_index(drop=True))
    n_uniq = len(unique)
    log.info(f"Processing {n_uniq:,} unique locations...")

    # ── Fuels (batch strip-read for efficiency) ───────────────────────────────
    log.info("Sampling LANDFIRE FBFM40 fuels (strip-based I/O)...")
    coord_list = [(float(r["latitude"]), float(r["longitude"]))
                  for _, r in unique.iterrows()]
    fuel_codes = fuels.sample_batch(coord_list)

    fuel_cache: dict = {}
    for (lat, lon), code in zip(coord_list, fuel_codes):
        fname = FBFM40_NAMES.get(code, f"Unknown({code})") if code else "unknown"
        fcat = fuel_category(code) if code else "unknown"
        fuel_cache[(lat, lon)] = {
            "fuel_model_code": float(code) if code else np.nan,
            "fuel_model_name": fname,
            "fuel_category": fcat,
        }

    # ── Map results back to all incident rows ─────────────────────────────────
    log.info("Attaching features to incident rows...")
    records = {"fuel_model_code": [np.nan] * len(df),
               "fuel_model_name": ["unknown"] * len(df),
               "fuel_category":   ["unknown"] * len(df)}

    for idx, row in df.iterrows():
        lat = row.get("latitude"); lon = row.get("longitude")
        if pd.isna(lat) or pd.isna(lon): continue
        key = (float(lat), float(lon))
        fc = fuel_cache.get(key, {})
        for c in ["fuel_model_code", "fuel_model_name", "fuel_category"]:
            records[c][idx] = fc.get(c, records[c][idx])

    for col, vals in records.items():
        df[col] = vals

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  FUEL MODEL DISTRIBUTION")
    print("="*60)
    if "fuel_category" in df.columns:
        print(df["fuel_category"].value_counts(dropna=False).to_string())
    if "fuel_model_code" in df.columns:
        print(f"\n  Coverage: {df['fuel_model_code'].notna().mean()*100:.1f}%")

    if "confirmed_ignition" in df.columns and "fuel_category" in df.columns:
        print("\n" + "="*60)
        print("  FUELS vs CONFIRMED IGNITION")
        print("="*60)
        conf = df[df["confirmed_ignition"] == 1]
        print("  Fuel category — confirmed ignitions:")
        print(conf["fuel_category"].value_counts(dropna=False).to_string())


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("STEP 1: Load incidents")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    log.info(f"  {len(df):,} records")

    log.info("STEP 2: LANDFIRE fuels")
    fuels = FuelsRaster(FUELS_DIR)
    log.info("STEP 3: Join")
    df = join_topo_fuels(df, fuels, dem=None)

    summarize(df)

    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Saved → {OUTPUT_CSV}")
    log.info(f"Shape : {df.shape}")
    log.info("New cols: fuel_model_code fuel_model_name fuel_category")
