"""Convert TableResolution.csv -> a Parquet cache + parameter metadata JSON.

The raw CSV has a multi-line preamble:
  lines 1-10: free-text metadata (operator, registration, project title, ...)
  line 11   : the literal token "DATA"
  line 12   : the column header ("Time, ...")
  line 13   : units row "(sec),(),(deg),..."
  line 14   : decoder row, e.g. '%N(0.0:0.0="OFF",1.0:1.0="ON")' for booleans, NUMBER otherwise
  line 15+  : data

Some rows in the dense CSV are sentinel/saturation frames where many sensors all
read their "max" value at once (Altitude Press = -1, Airspeed Comp = 511.75,
Ground Spd = 1023.5, Heading = 359.65, N1 = 127.88, Roll Angle = -0.18, ...).
These rows do not represent real flight state and would otherwise produce huge
spurious spikes in any plot, so we filter them out at preprocessing time.

Run:
    uv run python preprocess.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC_CSV = ROOT / "TableResolution.csv"
EXACT_CSV = ROOT / "ExactSample.csv"
OUT_DIR = ROOT / "cache"
OUT_PARQUET = OUT_DIR / "table_resolution.parquet"
OUT_EXACT_PARQUET = OUT_DIR / "exact_sample.parquet"
OUT_META = OUT_DIR / "parameter_meta.json"


def _parse_decoder(decoder: str) -> dict[float, str] | None:
    """Parse a discrete-state decoder string like:
        %N(0.0:0.0="OFF",1.0:1.0="ON")
    into a dict {0.0: "OFF", 1.0: "ON"}. Return None for plain numerics.
    """
    if not isinstance(decoder, str) or not decoder.startswith("%N"):
        return None
    pairs = re.findall(r'([-+]?\d*\.?\d+):[-+]?\d*\.?\d+="([^"]+)"', decoder)
    if not pairs:
        return None
    out: dict[float, str] = {}
    for val, label in pairs:
        try:
            out[float(val)] = label
        except ValueError:
            pass
    return out or None


def _read_metadata_header(path: Path) -> dict[str, str]:
    """Pull the free-text header (lines 1-10) into a dict."""
    meta: dict[str, str] = {}
    with path.open("r") as fh:
        for i, line in enumerate(fh):
            if i >= 10:
                break
            line = line.rstrip("\n")
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
            else:
                meta[f"_line_{i}"] = line.strip()
    return meta


# Sentinel saturation values that mark a "frame artifact" row, not real flight
# data. If a row has several of these simultaneously, we drop it. Values were
# verified by inspecting the data: they all read at the same instant, even
# while the legit cruise channel reads ~265 KIAS / 29,100 ft.
_SENTINELS = {
    "Altitude Press": -1.0,
    "Airspeed Comp": 511.75,
    "Ground Spd": 1023.5,
    "Heading": 359.65,
}


def _filter_sentinel_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows where ALL of the sentinel columns read their saturation value.

    A row is considered a sentinel/garbage frame if every sentinel column that
    has a value matches its expected sentinel (with a small tolerance). Rows
    where the sentinel columns are NaN are kept (they're just sparsely sampled).
    """
    cols = [c for c in _SENTINELS if c in df.columns]
    if not cols:
        return df, 0

    # For each row, count (a) sentinel cols that are NOT NaN and (b) of those,
    # how many match the sentinel value. If (a) >= 3 and (b) == (a), drop it.
    masks_present: list[pd.Series] = []
    masks_match: list[pd.Series] = []
    for c in cols:
        series = pd.to_numeric(df[c], errors="coerce")
        present = series.notna()
        match = present & np.isclose(series, _SENTINELS[c], atol=0.05)
        masks_present.append(present)
        masks_match.append(match)
    n_present = sum(m.astype(int) for m in masks_present)
    n_match = sum(m.astype(int) for m in masks_match)
    is_sentinel = (n_present >= 3) & (n_match == n_present)
    n_dropped = int(is_sentinel.sum())
    return df.loc[~is_sentinel].reset_index(drop=True), n_dropped


def load_fdr(path: Path) -> tuple[pd.DataFrame, dict, dict[str, dict]]:
    """Load an NTSB FDR CSV (TableResolution / ExactSample format).

    Returns: (data_df, file_metadata, per_parameter_metadata)
    """
    file_meta = _read_metadata_header(path)

    # header is line 12 (0-indexed 11) -> use header=11 with skiprows of [0..10]
    # Actually pandas: skiprows=11 skips lines 0..10, then line 11 becomes the
    # header. The next two rows are units + decoders, then real data.
    df = pd.read_csv(path, skiprows=11, low_memory=False)

    # Pull units and decoders out of the first two data rows
    units_row = df.iloc[0].to_dict()
    decoder_row = df.iloc[1].to_dict()
    df = df.iloc[2:].reset_index(drop=True)

    # Coerce all columns to numeric where possible (textual states stay as text)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows that are entirely empty (no Time)
    df = df.dropna(subset=["Time"]).reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)

    # Build per-parameter metadata
    param_meta: dict[str, dict] = {}
    for col in df.columns:
        unit = str(units_row.get(col, "")).strip()
        # units look like "(sec)" / "(deg)" / "()" -> strip parens
        unit = unit.strip("()") if unit.startswith("(") and unit.endswith(")") else unit
        decoder = _parse_decoder(str(decoder_row.get(col, "")))
        s = pd.to_numeric(df[col], errors="coerce")
        n_valid = int(s.notna().sum())
        info: dict = {
            "unit": unit,
            "is_discrete": decoder is not None,
            "decoder": decoder,
            "n_valid": n_valid,
        }
        if n_valid:
            info["min"] = float(s.min())
            info["max"] = float(s.max())
        param_meta[col] = info

    return df, file_meta, param_meta


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Loading {SRC_CSV.name} ...")
    df, file_meta, param_meta = load_fdr(SRC_CSV)
    print(f"  raw rows={len(df):,}, columns={len(df.columns)}")

    df, n_dropped = _filter_sentinel_rows(df)
    print(f"  dropped {n_dropped:,} sentinel/saturation rows")
    print(f"  kept {len(df):,} rows, time {df['Time'].min():.2f} -> {df['Time'].max():.2f}")

    # Time relative to start, in seconds, for friendlier plotting.
    df["t_rel"] = df["Time"] - df["Time"].min()

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  wrote {OUT_PARQUET}")

    # Metadata bundle
    bundle = {
        "file_metadata": file_meta,
        "time_range": [float(df["Time"].min()), float(df["Time"].max())],
        "duration_s": float(df["Time"].max() - df["Time"].min()),
        "sample_hz": round(1.0 / df["Time"].diff().median(), 3),
        "parameters": param_meta,
        "sentinels_dropped": n_dropped,
    }
    OUT_META.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"  wrote {OUT_META}")

    # Optional: also process ExactSample.csv if present (slow, sparse, larger)
    if EXACT_CSV.exists():
        print(f"\nLoading {EXACT_CSV.name} (sparse / exact-sample format) ...")
        df2, _, _ = load_fdr(EXACT_CSV)
        print(f"  raw rows={len(df2):,}, columns={len(df2.columns)}")
        df2, n2 = _filter_sentinel_rows(df2)
        print(f"  dropped {n2:,} sentinel rows; kept {len(df2):,}")
        df2["t_rel"] = df2["Time"] - df2["Time"].min()
        df2.to_parquet(OUT_EXACT_PARQUET, index=False)
        print(f"  wrote {OUT_EXACT_PARQUET}")


if __name__ == "__main__":
    main()
