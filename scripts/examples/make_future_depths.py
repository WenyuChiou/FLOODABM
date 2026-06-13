# -*- coding: utf-8 -*-
"""
Example: build a FUTURE flood-depth file for a climate scenario.

This is an ILLUSTRATIVE starter, not a calibrated projection. It takes the shipped
historical depths file and writes a new file with future `YY_mean` columns, so you
can exercise the future-simulation path end to end. Replace the projection logic
(`future_depth_for`) with your own CAT-model output under the chosen RCP/SSP pathway.

Output schema matches the model's input exactly (see docs/DATA_FORMATS.md):
a JSON array of {"CensusTract": "<11-digit GEOID>", "11_mean": <m>, ..., "NN_mean": <m>}.
Years are decoded by the model as 2000 + YY.

Usage:
    python scripts/examples/make_future_depths.py \
        --input config/overall_md_mean_by_tract_2011_2023.json \
        --output config/depths_future_2024_2050.json \
        --end-year 2050 --uplift 1.25

Then point config/abm_params.yaml -> files.depths_overall at the output file and set
years.mode: explicit with years.list covering the future years (see docs/FUTURE_SIMULATION.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

YEAR_COL = "{yy:02d}_mean"   # e.g. 24_mean -> 2024


def historical_years(record: dict) -> list[int]:
    """Return the 4-digit years present as YY_mean columns in one tract record."""
    yrs = []
    for k in record:
        if k.endswith("_mean") and k[:2].isdigit():
            yrs.append(2000 + int(k[:2]))
    return sorted(yrs)


def future_depth_for(record: dict, year: int, uplift: float) -> float:
    """ILLUSTRATIVE projection: representative historical depth x uplift.

    Uses the mean of the tract's non-zero historical depths (a tract with no
    historical flooding stays at 0.0). REPLACE this with your CAT-model depths.
    """
    vals = [float(record[f"{y - 2000:02d}_mean"])
            for y in historical_years(record)
            if float(record.get(f"{y - 2000:02d}_mean", 0.0)) > 0.0]
    base = (sum(vals) / len(vals)) if vals else 0.0
    return round(base * uplift, 4)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build an illustrative future flood-depth file.")
    ap.add_argument("--input", default="config/overall_md_mean_by_tract_2011_2023.json",
                    help="Historical depths JSON (model input schema).")
    ap.add_argument("--output", required=True, help="Where to write the future depths JSON.")
    ap.add_argument("--end-year", type=int, default=2050, help="Last future year to generate.")
    ap.add_argument("--start-year", type=int, default=None,
                    help="First future year (default: one year after the last historical year).")
    ap.add_argument("--uplift", type=float, default=1.25,
                    help="Multiplicative depth uplift applied to the historical base.")
    ap.add_argument("--keep-historical", action="store_true",
                    help="Also keep the historical YY_mean columns in the output.")
    args = ap.parse_args()

    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        print(f"[error] {args.input} is not a non-empty JSON array.", file=sys.stderr)
        return 1

    last_hist = max(historical_years(records[0]))
    start = args.start_year if args.start_year is not None else last_hist + 1
    future_years = list(range(start, args.end_year + 1))
    if not future_years:
        print(f"[error] no future years to generate (start {start} > end {args.end_year}).", file=sys.stderr)
        return 1

    out = []
    for rec in records:
        new_rec = {"CensusTract": str(rec["CensusTract"])}
        if args.keep_historical:
            for y in historical_years(rec):
                new_rec[f"{y - 2000:02d}_mean"] = float(rec[f"{y - 2000:02d}_mean"])
        for y in future_years:
            new_rec[f"{y - 2000:02d}_mean"] = future_depth_for(rec, y, args.uplift)
        out.append(new_rec)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ok] wrote {len(out)} tracts x {len(future_years)} future years "
          f"({future_years[0]}-{future_years[-1]}, uplift={args.uplift}) -> {args.output}")
    print("Next: set files.depths_overall to this file and years.mode: explicit "
          "with years.list covering these years (see docs/FUTURE_SIMULATION.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
