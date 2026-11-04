#!/usr/bin/env python3
"""
well_lithology_export.py - export every lithologic-log interval from the
Treasure Valley OCR database into one Excel workbook, with QA flags.

Input : welllogs_TV.sqlite (output of OCR_main8.py). Reads normalized.json_final
        (falls back to json_text) joined to ocr_pages.well_id. Multi-page
        reports are merged per well: continuation-page log rows are appended
        in page order, and a page whose log exactly repeats an earlier page of
        the same well is skipped.

Output: one .xlsx with three sheets
  Lithology_Intervals  one row per interval: well id, location (TRS, QQ,
                       lat/long, county), well site address, total depth,
                       from/to depths, raw remarks, standardized lithology
                       type, water Y/N, interval flags and well flags
  Wells                one row per well (including wells with no log or
                       unreadable JSON) with log bottom vs total depth and
                       counts of each flag
  Flag_Legend          what every flag means

Interval flags (sequence checks run in the order the driller wrote the rows):
  GAP                  from_ft > previous to_ft  (a missing entry)
  OVERLAP              from_ft < previous to_ft  (overlap or out of order)
  NOT_FROM_SURFACE     first interval does not start at 0 ft
  BAD_INTERVAL         to_ft <= from_ft
  DEPTH_UNPARSEABLE    from or to depth missing / not a number
  WATER_UNRESOLVED     water column has a mark that is not Y or N (e.g. "X")
  POSSIBLE_NON_LITHOLOGY  remark looks like a seal/casing/construction note

Well flags:
  NO_TOTAL_DEPTH       no completed depth on the report
  LOG_SHORT_OF_TD      deepest log entry < total depth - tolerance
                       (missing entries at the bottom of the log)
  LOG_DEEPER_THAN_TD   deepest log entry > total depth + tolerance
                       (drilled deeper than completed, or a misread depth)
  plus HAS_GAPS / HAS_OVERLAPS / ... when any interval carries that flag.

Usage:
  python well_lithology_export.py outputs/welllogs_TV.sqlite
  python well_lithology_export.py outputs/welllogs_TV.sqlite --out outputs/TV_lithology.xlsx --tol 2

Requires openpyxl. Uses lithology_extract.py (same folder) to classify remarks.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lithology_extract import (DITTO_RE, NON_LITHO_PATTERNS, clean_text,  # noqa: E402
                               parse_materials, primary_type)

try:
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError:
    sys.exit("openpyxl is required:  pip install openpyxl")

EPS = 0.01  # ft; depth comparisons inside the log

# =========================================================
# value parsing
# =========================================================

_FT_IN = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:'|ft\.?)?\s*[- ]?\s*(\d+(?:\.\d+)?)\s*(?:\"|in\.?)$")
_MIXED = re.compile(r"^(\d+)\s+(\d+)\s*/\s*(\d+)")
_FRAC = re.compile(r"^(\d+)\s*/\s*(\d+)")


def parse_depth(v):
    """'12', "12'", '12.5', "12'6\"", '12-6"', '3 1/2', '1,200', 'Surface' -> float ft."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace("�", "").replace("’", "'")
    if not s or s in {"?", "-"}:
        return None
    if s.startswith(("surface", "grade", "gl")):
        return 0.0
    s = s.lstrip("+-~ ").rstrip("+.? ")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)      # 1,200 -> 1200
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)           # 12,5  -> 12.5
    if s.startswith("."):
        s = "0" + s
    m = _FT_IN.match(s)
    if m and ("'" in s or "-" in s or " " in s or "ft" in s):
        inches = float(m.group(2))
        if inches < 12:
            return float(m.group(1)) + inches / 12
    m = _MIXED.match(s)
    if m and float(m.group(3)):
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    m = _FRAC.match(s)
    if m and float(m.group(2)) and not re.match(r"^\d+\s*/\s*\d+\s*\d", s):
        return int(m.group(1)) / int(m.group(2))
    m = re.match(r"^(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def parse_coord(v, is_long):
    """Decimal degrees, 'DD MM.mmm', 'DD:MM:SS.s', 'DD o MM.mmm'N' -> signed float.
    Longitudes are forced negative (western hemisphere)."""
    if v in (None, ""):
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", str(v))
    if not nums:
        return None
    vals = [float(n) for n in nums[:3]]
    deg = vals[0]
    if len(vals) >= 2:
        deg += vals[1] / 60
    if len(vals) >= 3:
        deg += vals[2] / 3600
    if is_long:
        deg = -abs(deg)
    lo, hi = ((-118.0, -110.0) if is_long else (41.0, 49.5))
    return round(deg, 6) if lo <= deg <= hi else None


def norm_water(v):
    """-> ('Y'|'N'|None, unresolved_mark_or_None)"""
    if v in (None, ""):
        return None, None
    s = str(v).strip().lower()
    if s in {"y", "yes", "water"}:
        return "Y", None
    if s in {"n", "no", "no water"}:
        return "N", None
    return None, str(v)


def norm_qq(q):
    if not q:
        return None
    s = re.sub(r"\s+", " ", str(q).upper().replace("¼", "1/4")).strip()
    return s or None


def trs_string(L):
    t, r, s = L.get("twp"), L.get("rge"), L.get("sec")
    if not (t or r or s):
        return None
    return (f"T{t or '?'}{(L.get('twp_dir') or '').upper()} "
            f"R{r or '?'}{(L.get('rge_dir') or '').upper()} S{s or '?'}")


def completed_depth(d):
    """completed_depth_ft, else a numeric 'Completed Depth...' unmapped field."""
    raw = d.get("completed_depth_ft")
    val = parse_depth(raw)
    if val is not None:
        return raw, val
    for u in d.get("unmapped_fields") or []:
        if not isinstance(u, dict):
            continue
        if re.search(r"completed depth|total depth", str(u.get("label")), re.I):
            v = parse_depth(u.get("value"))
            if v is not None:
                return u.get("value"), v
    return raw, None


def page_no(page_name):
    m = re.search(r"_page_(\d+)", str(page_name))
    return int(m.group(1)) if m else 0


def first_nonempty(dicts, key):
    for x in dicts:
        if x.get(key) not in (None, ""):
            return x.get(key)
    return None

# =========================================================
# main
# =========================================================


WELL_COLS = ["well_id", "well_id_no", "county", "township_range_section",
             "quarter_quarter", "latitude", "longitude", "lat_raw", "long_raw",
             "well_site_address", "well_site_city", "subdivision_lot_blk",
             "total_depth_ft", "total_depth_raw", "first_water_ft"]
INTERVAL_COLS = ["page", "row", "from_ft", "to_ft", "thickness_ft",
                 "from_raw", "to_raw", "bore_dia_in", "lithology_remarks",
                 "lithology_type", "materials", "water_present",
                 "water_raw", "interval_flags"]

LEGEND = [
    ("GAP", "interval", "from_ft is deeper than the previous row's to_ft - an entry is missing between them (gap size in ft)."),
    ("OVERLAP", "interval", "from_ft is shallower than the previous row's to_ft - rows overlap or are out of order."),
    ("NOT_FROM_SURFACE", "interval", "First interval of the log does not start at 0 ft."),
    ("BAD_INTERVAL", "interval", "to_ft is less than or equal to from_ft."),
    ("DEPTH_UNPARSEABLE", "interval", "from_ft or to_ft is blank or not a number; raw values kept in from_raw/to_raw."),
    ("WATER_UNRESOLVED", "interval", "Water column has a mark that is not Y or N (e.g. 'X'); see water_raw."),
    ("POSSIBLE_NON_LITHOLOGY", "interval", "Remark looks like a seal/casing/construction note rather than a material."),
    ("DITTO_EXPANDED", "interval", "Remark was a ditto mark; lithology_type is taken from the row above."),
    ("NO_TOTAL_DEPTH", "well", "No completed/total depth could be read from the report."),
    ("LOG_SHORT_OF_TD", "well", "Deepest lithology entry is shallower than total depth by more than the tolerance - bottom of log missing."),
    ("LOG_DEEPER_THAN_TD", "well", "Deepest lithology entry is deeper than total depth by more than the tolerance - hole drilled deeper than completed (backfilled) or a misread depth."),
    ("HAS_GAPS / HAS_OVERLAPS / ...", "well", "At least one interval in the well carries that interval flag."),
    ("MULTI_PAGE", "well", "Log was assembled from more than one scanned page."),
    ("NO_LITHOLOGY_LOG", "well", "Report has no lithologic log rows (Wells sheet only)."),
    ("INVALID_JSON", "well", "OCR normalizer output could not be read (Wells sheet only)."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", help="path to welllogs_TV.sqlite")
    ap.add_argument("--out", default=None,
                    help="output .xlsx (default: TV_lithology_logs.xlsx next to the db)")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="ft tolerance for log-bottom vs total-depth check (default 1)")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.db)),
                                   "TV_lithology_logs.xlsx")

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    cols = {r[1] for r in con.execute("PRAGMA table_info(normalized)")}
    jexpr = ("COALESCE(NULLIF(n.json_final,''), n.json_text)"
             if "json_final" in cols else "n.json_text")
    rows = con.execute(f"SELECT o.well_id, n.page_name, {jexpr} FROM normalized n "
                       f"LEFT JOIN ocr_pages o ON o.page_name = n.page_name").fetchall()
    con.close()

    by_well = defaultdict(list)
    for wid, page, j in rows:
        wid = str(wid) if wid else os.path.basename(str(page)).split("_")[0]
        by_well[wid].append((page_no(page), page, j))

    interval_rows, well_rows = [], []
    flag_totals = Counter()

    for wid in sorted(by_well, key=lambda w: (len(w), w)):
        pages = sorted(by_well[wid])
        docs, n_bad = [], 0
        for pn, page, j in pages:
            try:
                d = json.loads(j) if j else None
            except (json.JSONDecodeError, TypeError):
                d = None
            if isinstance(d, dict):
                docs.append((pn, d))
            else:
                n_bad += 1

        well_flags = []
        if not docs:
            well_rows.append({"well_id": wid, "pages": len(pages), "well_flags": "INVALID_JSON"})
            flag_totals["INVALID_JSON"] += 1
            continue

        ds = [d for _, d in docs]
        locs = [d.get("location") or {} for d in ds]
        L = next((x for x in locs if x.get("twp") or x.get("quarter_quarter")), locs[0])
        lat_raw = first_nonempty(locs, "lat")
        long_raw = first_nonempty(locs, "long")
        td_raw, td = None, None
        for d in ds:
            td_raw, td = completed_depth(d)
            if td is not None:
                break
        lot_bits = [f"{k.upper()} {first_nonempty(locs, k)}" for k in ("subdivision", "lot", "blk")
                    if first_nonempty(locs, k)]
        well = {
            "well_id": wid,
            "well_id_no": first_nonempty(ds, "well_id_no"),
            "county": first_nonempty(locs, "county"),
            "township_range_section": trs_string(L),
            "quarter_quarter": norm_qq(L.get("quarter_quarter") or first_nonempty(locs, "quarter_quarter")),
            "latitude": parse_coord(lat_raw, False),
            "longitude": parse_coord(long_raw, True),
            "lat_raw": lat_raw, "long_raw": long_raw,
            "well_site_address": first_nonempty(locs, "well_site_address"),
            "well_site_city": first_nonempty(locs, "city"),
            "subdivision_lot_blk": ", ".join(lot_bits) or None,
            "total_depth_ft": td, "total_depth_raw": td_raw,
            "first_water_ft": parse_depth(first_nonempty(ds, "depth_first_water_encountered_ft")),
        }

        # ---- assemble the log across pages
        log, seen, used_pages = [], set(), set()
        for pn, d in docs:
            entries = [e for e in (d.get("lithologic_log") or []) if isinstance(e, dict)]
            entries = [e for e in entries if any(str(e.get(k) or "").strip()
                                                 for k in ("from_ft", "to_ft", "remarks"))]
            if not entries:
                continue
            sig = tuple((str(e.get("from_ft")), str(e.get("to_ft")), str(e.get("remarks")))
                        for e in entries)
            if sig in seen:
                continue
            seen.add(sig)
            used_pages.add(pn)
            log.extend((pn, e) for e in entries)

        if len(used_pages) > 1:
            well_flags.append("MULTI_PAGE")
        if td is None:
            well_flags.append("NO_TOTAL_DEPTH")

        if not log:
            well_flags.insert(0, "NO_LITHOLOGY_LOG")
            flag_totals.update(well_flags)
            well_rows.append({**well, "pages": len(pages), "intervals": 0,
                              "well_flags": "; ".join(well_flags)})
            continue

        # ---- per-interval parsing + sequence checks
        prev_to, prev_remark, log_bottom = None, None, None
        iflag_counts = Counter()
        n_gap_ft = 0.0
        recs = []
        for idx, (pn, e) in enumerate(log, 1):
            fr_raw, to_raw = e.get("from_ft"), e.get("to_ft")
            fr, to = parse_depth(fr_raw), parse_depth(to_raw)
            raw = str(e.get("remarks") or "").strip()
            flags = []

            if fr is None or to is None:
                flags.append("DEPTH_UNPARSEABLE")
            if fr is not None and to is not None and to <= fr + EPS:
                flags.append("BAD_INTERVAL")
            if fr is not None:
                if prev_to is None and idx == 1 and fr > EPS:
                    flags.append(f"NOT_FROM_SURFACE ({fr:g} ft)")
                elif prev_to is not None and fr > prev_to + EPS:
                    gap = fr - prev_to
                    n_gap_ft += gap
                    flags.append(f"GAP ({prev_to:g}-{fr:g} ft missing)")
                elif prev_to is not None and fr < prev_to - EPS:
                    flags.append(f"OVERLAP (starts {prev_to - fr:g} ft above previous bottom)")
            if to is not None:
                prev_to = to
                log_bottom = to if log_bottom is None else max(log_bottom, to)

            water, water_odd = norm_water(e.get("water"))
            if water_odd:
                flags.append("WATER_UNRESOLVED")

            remark = raw
            if DITTO_RE.match(raw) and prev_remark:
                remark = prev_remark
                flags.append("DITTO_EXPANDED")
            clean = clean_text(remark)
            terms, _, _ = parse_materials(clean)
            if any(re.search(p, clean) for p in NON_LITHO_PATTERNS):
                flags.append("POSSIBLE_NON_LITHOLOGY")
            if raw and not DITTO_RE.match(raw):
                prev_remark = remark

            for f in flags:
                iflag_counts[f.split(" (")[0]] += 1
            recs.append({
                "page": pn, "row": idx, "from_ft": fr, "to_ft": to,
                "thickness_ft": round(to - fr, 2) if fr is not None and to is not None and to > fr else None,
                "from_raw": fr_raw, "to_raw": to_raw, "bore_dia_in": e.get("bore_dia_in"),
                "lithology_remarks": raw or None,
                "lithology_type": primary_type(clean, terms),
                "materials": "; ".join(dict.fromkeys(t[0] for t in terms)) or None,
                "water_present": water, "water_raw": e.get("water"),
                "interval_flags": "; ".join(flags) or None,
            })

        # ---- well-level checks
        name_map = {"GAP": "HAS_GAPS", "OVERLAP": "HAS_OVERLAPS",
                    "NOT_FROM_SURFACE": "NOT_FROM_SURFACE", "BAD_INTERVAL": "HAS_BAD_INTERVALS",
                    "DEPTH_UNPARSEABLE": "HAS_UNPARSEABLE_DEPTHS",
                    "WATER_UNRESOLVED": "HAS_UNRESOLVED_WATER"}
        for k, name in name_map.items():
            if iflag_counts[k]:
                well_flags.append(name)
        td_diff = None
        if td is not None and log_bottom is not None:
            td_diff = round(log_bottom - td, 2)
            if td_diff < -args.tol:
                well_flags.append(f"LOG_SHORT_OF_TD ({-td_diff:g} ft)")
            elif td_diff > args.tol:
                well_flags.append(f"LOG_DEEPER_THAN_TD ({td_diff:g} ft)")

        waters = [r["water_present"] for r in recs]
        wf = "; ".join(well_flags) or None
        for r in recs:
            interval_rows.append({**well, **r, "log_bottom_ft": log_bottom,
                                  "log_bottom_minus_td_ft": td_diff, "well_flags": wf})
        flag_totals.update(f.split(" (")[0] for f in well_flags)
        well_rows.append({
            **well, "pages": len(pages), "intervals": len(recs),
            "log_bottom_ft": log_bottom, "log_bottom_minus_td_ft": td_diff,
            "any_water_Y": "Y" if "Y" in waters else ("N" if "N" in waters else None),
            "water_bearing_intervals": waters.count("Y"),
            "gaps": iflag_counts["GAP"], "gap_total_ft": round(n_gap_ft, 2) or None,
            "overlaps": iflag_counts["OVERLAP"],
            "unparseable_depths": iflag_counts["DEPTH_UNPARSEABLE"],
            "well_flags": wf,
        })

    # ---- write workbook
    iv_cols = WELL_COLS + INTERVAL_COLS + ["log_bottom_ft", "log_bottom_minus_td_ft", "well_flags"]
    w_cols = WELL_COLS + ["pages", "intervals", "log_bottom_ft", "log_bottom_minus_td_ft",
                          "any_water_Y", "water_bearing_intervals", "gaps", "gap_total_ft",
                          "overlaps", "unparseable_depths", "well_flags"]
    wb = Workbook(write_only=True)
    hdr_font = Font(bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", fgColor="1F3864")
    widths = {"lithology_remarks": 40, "interval_flags": 40, "well_flags": 40,
              "well_site_address": 28, "quarter_quarter": 18, "township_range_section": 16,
              "materials": 22, "lithology_type": 16, "subdivision_lot_blk": 26}

    def sheet(name, cols, data):
        ws = wb.create_sheet(name)
        for i, c in enumerate(cols):
            ws.column_dimensions[get_column_letter(i + 1)].width = widths.get(c, max(10, len(c) + 2))
        ws.freeze_panes = "B2"
        hdr = []
        for c in cols:
            cell = WriteOnlyCell(ws, value=c)
            cell.font, cell.fill = hdr_font, hdr_fill
            hdr.append(cell)
        ws.append(hdr)
        for r in data:
            ws.append([r.get(c) for c in cols])
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(data) + 1}"

    sheet("Lithology_Intervals", iv_cols, interval_rows)
    sheet("Wells", w_cols, well_rows)
    sheet("Flag_Legend", ["flag", "level", "meaning"],
          [{"flag": f, "level": l, "meaning": m} for f, l, m in LEGEND]
          + [{"flag": "", "level": "", "meaning": ""},
             {"flag": "tolerance_ft", "level": "setting", "meaning": args.tol},
             {"flag": "source_db", "level": "setting", "meaning": os.path.abspath(args.db)}])
    wb.save(out)

    n_logged = sum(1 for w in well_rows if w.get("intervals"))
    print(f"wells: {len(well_rows)}  with lithology: {n_logged}  intervals: {len(interval_rows)}")
    print("well flag counts:")
    for k, v in flag_totals.most_common():
        print(f"  {k:<24} {v}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
