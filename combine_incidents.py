import os
import re
import pandas as pd
import numpy as np

FOLDER = "fire_incidents"
OUTPUT  = "master_ignitions.csv"

# ── 1. CANONICAL COLUMN MAP ────────────────────────────────────────────────
# Maps every raw column name variant seen across 2014-2024 files
# to a single clean name used in the master dataframe.
COLUMN_MAP = {
    # ── Utility / source
    "utility name"                          : "utility",

    # ── Date / time
    "date"                                  : "date",
    "fire start_date"                       : "date",
    "time"                                  : "time",
    "fire start_time"                       : "time",

    # ── Location
    "latitude"                              : "latitude",
    "longitude"                             : "longitude",
    "material at origin"                    : "material_at_origin",
    "land use at origin"                    : "land_use",
    "land use"                              : "land_use",
    "county"                                : "county",
    "city"                                  : "city",

    # ── Fire outcome  ← THIS IS OUR TARGET VARIABLE
    "size"                                  : "fire_size",
    "fire size"                             : "fire_size",
    "suppressed by"                         : "suppressed_by",
    "suppressing agency"                    : "suppressing_agency",

    # ── Utility facility / asset features
    "facility identification"               : "facility_id",
    "voltage\n(volts)"                      : "voltage",
    "voltage (volts)"                       : "voltage",
    "voltage (kv)"                          : "voltage",
    "nominal voltage"                       : "voltage",
    "voltage"                               : "voltage",
    "equipment involved with ignition"      : "equipment_type",
    "equipment involved"                    : "equipment_type",
    "type"                                  : "facility_type",       # overhead/underground
    "facility type"                         : "facility_type",
    "other companies"                       : "other_companies",

    # ── Outage
    "was there an outage"                   : "outage_occurred",
    "outage_date"                           : "outage_date",
    "outage_time"                           : "outage_time",

    # ── Cause features
    "suspected ignition cause"              : "ignition_cause",
    "suspected cause"                       : "ignition_cause",
    "equipment /facility failure"           : "equipment_failure_type",
    "equipment/facility failure"            : "equipment_failure_type",
    "contact from object"                   : "contact_object",
    "facility contacted"                    : "facility_contacted",
    "contributing factor"                   : "contributing_factor",

    # ── HFTD zone (present in some newer files)
    "hftd tier"                             : "hftd_tier",
    "fire threat zone"                      : "hftd_tier",
}

# Columns we actually want in the final master dataframe (in order)
KEEP_COLS = [
    "utility", "year", "date", "time",
    "latitude", "longitude", "county", "city",
    "material_at_origin", "land_use",
    "fire_size",                          # target variable (raw string)
    "fire_spread",                        # target variable (binary, derived)
    "suppressed_by", "suppressing_agency",
    "voltage", "equipment_type", "facility_type",
    "outage_occurred",
    "ignition_cause", "equipment_failure_type",
    "contact_object", "contributing_factor",
    "hftd_tier",
]

# ── 2. FIRE SIZE → BINARY SPREAD LABEL ────────────────────────────────────
# Anything larger than "Less Than .25 Acres" that is not NaN counts as spread.
NO_SPREAD_PATTERNS = [
    r"less than",
    r"contained at point",
    r"< ?0?\.25",
    r"no fire",
    r"no spread",
]

def encode_spread(size_val):
    """Return 1 if fire spread beyond equipment, 0 if contained, NaN if unknown."""
    if pd.isna(size_val):
        return np.nan
    s = str(size_val).strip().lower()
    if s in ("", "nan", "n/a", "unknown"):
        return np.nan
    for pat in NO_SPREAD_PATTERNS:
        if re.search(pat, s):
            return 0
    return 1

# ── 3. UTILITY DETECTION FROM FILENAME ────────────────────────────────────
def detect_utility(fname):
    f = fname.upper()
    if "PGE" in f or "PG&E" in f or "PG_E" in f:
        return "PG&E"
    if "SCE" in f:
        return "SCE"
    if "SDGE" in f or "SDG&E" in f:
        return "SDG&E"
    return "UNKNOWN"

# ── 4. YEAR DETECTION FROM FILENAME ───────────────────────────────────────
def detect_year(fname):
    # Prefer a 4-digit year that appears early in filename
    matches = re.findall(r"(20\d{2})", fname)
    if matches:
        return int(matches[0])
    return np.nan

# ── 5. HEADER RECONSTRUCTION ──────────────────────────────────────────────
def build_column_names(row0, row1):
    """
    Combine two header rows into clean column name strings.
    Row 0 has section labels (with NaN gaps from merged cells).
    Row 1 has field names.
    Strategy: forward-fill row0 section labels, then concatenate
    section_label + '_' + field_name only when section label adds info.
    """
    section = None
    cols = []
    for s0, s1 in zip(row0, row1):
        # Forward-fill section label
        if pd.notna(s0) and str(s0).strip() not in ("nan", ""):
            section = str(s0).strip()

        field = str(s1).strip() if pd.notna(s1) else ""

        # Build combined name
        if field == "" or field == "nan":
            cols.append(f"_unnamed_{len(cols)}")
        elif section and section.lower() not in field.lower() \
                and section.lower() not in (
                    "utility name", "fire", "location",
                    "utility facility", "outage", "suspected cause"
                ):
            cols.append(f"{section}_{field}")
        else:
            cols.append(field)

    return cols

# ── 6. NORMALISE ONE DATAFRAME ─────────────────────────────────────────────
def normalise(df, utility, year):
    # Lowercase + strip all column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Apply canonical map
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    # Drop fully unnamed / all-NaN columns
    df = df.loc[:, ~df.columns.str.startswith("_unnamed")]
    df = df.dropna(how="all")

    # ── DEDUPLICATE COLUMN NAMES ──────────────────────────────────────────
    # If two columns map to the same name, keep the first and drop the rest.
    # Print a warning so we know which columns were affected.
    seen = {}
    new_cols = []
    for i, col in enumerate(df.columns):
        if col in seen:
            print(f"    WARN [{utility} {year}] duplicate column '{col}' "
                  f"at position {i} — dropping duplicate")
            new_cols.append(f"_dup_{i}_{col}")
        else:
            seen[col] = i
            new_cols.append(col)
    df.columns = new_cols
    # Now drop the renamed duplicate columns
    df = df.loc[:, ~df.columns.str.startswith("_dup_")]

    # Inject metadata
    df["utility"] = utility
    df["year"]    = year

    # Derive binary spread label
    if "fire_size" in df.columns:
        df["fire_spread"] = df["fire_size"].apply(encode_spread)
    else:
        df["fire_spread"] = np.nan

    # Keep only canonical columns that exist
    present = [c for c in KEEP_COLS if c in df.columns]
    df = df[present]

    return df

# ── 7. READ ONE XLSX FILE ──────────────────────────────────────────────────
def read_fire_xlsx(fpath, utility, year):
    # Read raw rows (no header parsing yet)
    raw = pd.read_excel(fpath, header=None, nrows=4)

    row0 = list(raw.iloc[0])
    row1 = list(raw.iloc[1])

    # Decide if file uses the 2-row merged header format
    # Heuristic: row0 has many NaN values (merged cell artifact)
    nan_count_row0 = sum(1 for v in row0 if pd.isna(v) or str(v).strip() in ("nan",""))
    uses_two_row_header = nan_count_row0 > len(row0) * 0.3

    if uses_two_row_header:
        col_names = build_column_names(row0, row1)
        df = pd.read_excel(fpath, header=None, skiprows=2)
        df.columns = col_names[:len(df.columns)]
    else:
        # Single header row (some newer files)
        df = pd.read_excel(fpath, header=0)

    return normalise(df, utility, year)

# ── 8. MAIN LOOP ───────────────────────────────────────────────────────────
all_frames = []
errors     = []

for fname in sorted(os.listdir(FOLDER)):
    if not fname.endswith(".xlsx"):
        continue

    fpath   = os.path.join(FOLDER, fname)
    utility = detect_utility(fname)
    year    = detect_year(fname)

    try:
        df = read_fire_xlsx(fpath, utility, year)
        all_frames.append(df)
        print(f"  OK  [{utility} {year}]  {len(df):>4} rows  |  {fname}")
    except Exception as e:
        errors.append((fname, str(e)))
        print(f"  ERR [{utility} {year}]  {fname}  →  {e}")

# ── 9. COMBINE & BASIC QC ──────────────────────────────────────────────────
# Validate each frame before concat to catch any remaining duplicate columns
for i, df in enumerate(all_frames):
    if not df.columns.is_unique:
        dupes = df.columns[df.columns.duplicated()].tolist()
        print(f"  WARN frame {i} still has duplicate columns: {dupes} — forcing unique")
        # Manually rename duplicates by appending suffix
        seen = {}
        new_cols = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
        all_frames[i] = df

master = pd.concat(all_frames, ignore_index=True)

print(f"\n{'='*60}")
print(f"MASTER DATAFRAME: {len(master):,} rows × {len(master.columns)} columns")
print(f"Columns present: {list(master.columns)}")
print(f"\nUtility breakdown:\n{master['utility'].value_counts()}")
print(f"\nYear breakdown:\n{master['year'].value_counts().sort_index()}")
print(f"\nTarget variable (fire_spread):\n{master['fire_spread'].value_counts(dropna=False)}")
print(f"\nMissing values per column:\n{master.isnull().sum()}")

if errors:
    print(f"\nFILES WITH ERRORS ({len(errors)}):")
    for fname, err in errors:
        print(f"  {fname}: {err}")

# ── 10. SAVE ───────────────────────────────────────────────────────────────
master.to_csv(OUTPUT, index=False)
print(f"\nSaved to: {OUTPUT}")