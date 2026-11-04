#!/usr/bin/env python3
"""
well_column.py - draw a lithologic column for one well from the OCR database.

The column runs from the surface to the deepest logged interval. Each interval
is colored (and patterned) by its standardized lithology type, the driller's
remarks are written beside it with a leader line, water-bearing intervals are
marked, and the completed depth is drawn as a dashed line. Header: well ID,
county, date drilled, location.

Input : welllogs_TV.sqlite (output of OCR_main8.py). Reads normalized.json_final
        (falls back to json_text) joined to ocr_pages.well_id; multi-page
        reports are merged in page order, exactly like well_lithology_export.py.

Usage:
  python well_column.py 344163
  python well_column.py 344163 --db outputs/welllogs_TV.sqlite --out col.pdf
  python well_column.py 344163 --show

The well ID is ocr_pages.well_id (the scan's file name number). If nothing
matches, the report's own well_id_no field (e.g. "D 19439") is searched too.
Output format follows the --out extension (.png, .pdf, .svg).

Requires matplotlib. Uses lithology_extract.py and well_lithology_export.py
from the same folder.
"""

import argparse
import json
import os
import sqlite3
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lithology_extract import (DITTO_RE, clean_text, parse_materials,  # noqa: E402
                               primary_type)
from well_lithology_export import (completed_depth, norm_qq, page_no,  # noqa: E402
                                   parse_depth, trs_string)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# =========================================================
# lithology colors + patterns (loosely after USGS/FGDC lithology conventions)
# (fill, hatch)
# =========================================================

STYLE = {
    "topsoil":         ("#A0703C", ""),
    "soil":            ("#C49A6C", ""),
    "fill":            ("#BDB5A6", "x"),
    "pavement":        ("#8C8C8C", ""),
    "peat / muck":     ("#5C4632", ""),
    "clay":            ("#93B193", "--"),
    "silt":            ("#CBD9A4", "-."),
    "sand":            ("#F2D86B", ".."),
    "gravel":          ("#F0A04B", "oo"),
    "cemented gravel": ("#D9822B", "oo"),
    "cobbles":         ("#E07B39", "OO"),
    "boulders":        ("#C0612B", "OO"),
    "hardpan":         ("#B3A38B", "xx"),
    "caliche":         ("#EDE6D2", "xx"),
    "cinders":         ("#B5523B", ".."),
    "ash":             ("#DADADA", ".."),
    "lava":            ("#6A6A80", "//"),
    "basalt":          ("#4A4A5C", "//"),
    "rhyolite":        ("#D9A3B8", "\\\\"),
    "tuff":            ("#E6C3D0", ".."),
    "obsidian":        ("#262626", ""),
    "pumice":          ("#F0F0F0", ".."),
    "sandstone":       ("#E3C26B", "..."),
    "siltstone":       ("#BFC98A", "-."),
    "claystone":       ("#9DB59D", "--"),
    "mudstone":        ("#8C9C8C", "--"),
    "shale":           ("#7D8C7D", "---"),
    "shell":           ("#E8E2C8", ""),
    "soapstone":       ("#A9B7A9", ""),
    "conglomerate":    ("#D49A5A", "oo"),
    "limestone":       ("#9BC4E2", "+"),
    "granite":         ("#E7A6A6", "++"),
    "quartzite":       ("#F3E3E3", "+"),
    "schist":          ("#9B8FB0", "~~"),
    "quartz":          ("#F7F7F7", "+"),
    "breccia":         ("#B39088", "xx"),
    "rock":            ("#A0A0A0", ""),
    "red rock":        ("#B5524A", ""),
    "black rock":      ("#2E2E2E", ""),
    "crevice / fracture / void": ("#FFFFFF", "\\\\"),
}
UNCLASSIFIED = ("#FFFFFF", "")
DARK_FILLS = {"lava", "basalt", "obsidian", "black rock", "peat / muck", "shale"}

WRAP_CHARS = 48       # characters per remark line
MAX_LINES = 3         # longer remarks are truncated with "..."
FONT = 9              # remark font size (pt)
LINE_PT = FONT * 1.3  # height of one text line (pt)
LABEL_PAD_PT = 4      # vertical gap between labels (pt)

# =========================================================
# data
# =========================================================


def load_well(db_path, well_id):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = {r[1] for r in con.execute("PRAGMA table_info(normalized)")}
    jexpr = ("COALESCE(NULLIF(n.json_final,''), n.json_text)"
             if "json_final" in cols else "n.json_text")
    rows = con.execute(f"SELECT o.well_id, n.page_name, {jexpr} FROM normalized n "
                       f"JOIN ocr_pages o ON o.page_name = n.page_name "
                       f"WHERE o.well_id = ?", (str(well_id),)).fetchall()
    if not rows:  # fall back to the well number written on the report
        rows = con.execute(f"SELECT o.well_id, n.page_name, {jexpr} FROM normalized n "
                           f"JOIN ocr_pages o ON o.page_name = n.page_name "
                           f"WHERE {jexpr} LIKE ?", (f'%"well_id_no": "{well_id}"%',)).fetchall()
        if rows:
            wid = rows[0][0]
            rows = [r for r in rows if r[0] == wid]
    con.close()
    docs = []
    for wid, page, j in sorted(rows, key=lambda r: page_no(r[1])):
        try:
            d = json.loads(j)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(d, dict):
            docs.append((page_no(page), wid, d))
    return docs


def first(dicts, key):
    for x in dicts:
        if x.get(key) not in (None, ""):
            return x[key]
    return None


def build_log(docs):
    """Merge pages (skipping exact repeats), parse depths, classify remarks."""
    out, seen, prev = [], set(), None
    for _, _, d in docs:
        entries = [e for e in (d.get("lithologic_log") or []) if isinstance(e, dict)]
        sig = tuple((str(e.get("from_ft")), str(e.get("to_ft")), str(e.get("remarks")))
                    for e in entries)
        if not entries or sig in seen:
            continue
        seen.add(sig)
        for e in entries:
            raw = str(e.get("remarks") or "").strip()
            remark = prev if (DITTO_RE.match(raw) and prev) else raw
            if raw and not DITTO_RE.match(raw):
                prev = raw
            fr, to = parse_depth(e.get("from_ft")), parse_depth(e.get("to_ft"))
            if fr is None or to is None or to <= fr:
                continue  # cannot be drawn
            clean = clean_text(remark)
            terms, _, _ = parse_materials(clean)
            water = str(e.get("water") or "").strip().lower()
            out.append({"from": fr, "to": to, "remark": remark or "(no description)",
                        "type": primary_type(clean, terms) or "unclassified",
                        "water": water in ("y", "yes")})
    return out

# =========================================================
# label placement: keep labels from overlapping, stay near their interval
# =========================================================


def wrap(s):
    lines = textwrap.wrap(" ".join(s.split()), WRAP_CHARS) or [""]
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]
        lines[-1] = lines[-1][:WRAP_CHARS - 3].rstrip() + "..."
    return lines


def place_labels(wanted, heights, top, bottom):
    """1-D repel. wanted = desired label centers, heights = label heights (both
    in depth units). Returns centers kept within [top, bottom], no overlaps."""
    order = sorted(range(len(wanted)), key=lambda i: wanted[i])
    pos = [0.0] * len(wanted)
    cursor = top
    for i in order:                       # push down
        c = max(wanted[i], cursor + heights[i] / 2)
        pos[i] = c
        cursor = c + heights[i] / 2
    cursor = bottom
    for i in reversed(order):             # pull back up if we ran off the bottom
        c = min(pos[i], cursor - heights[i] / 2)
        pos[i] = c
        cursor = c - heights[i] / 2
    return pos

# =========================================================
# plot
# =========================================================


def plot(docs, log, out_path, show=False):
    ds = [d for _, _, d in docs]
    locs = [d.get("location") or {} for d in ds]
    well_id = docs[0][1]
    county = first(locs, "county")
    county = f"{str(county).title()} County" if county else "County unknown"
    started, completed = first(ds, "date_started"), first(ds, "date_completed")
    if started and completed and str(started) != str(completed):
        date = f"{started} to {completed}"
    else:
        date = completed or started or "unknown"
    L = next((x for x in locs if x.get("twp") or x.get("quarter_quarter")), locs[0])
    loc_bits = [b for b in (trs_string(L), norm_qq(L.get("quarter_quarter"))) if b]
    td_raw, td = None, None
    for d in ds:
        td_raw, td = completed_depth(d)
        if td is not None:
            break
    tag = first(ds, "well_id_no")

    bottom = max(r["to"] for r in log)
    depth_max = max(bottom, td or 0)

    # figure height: enough room for every label, at least 7 in, at most 60 in
    labels = [wrap(f"{r['from']:g}–{r['to']:g} ft   {r['remark']}") for r in log]
    need_pt = sum(len(l) * LINE_PT + LABEL_PAD_PT for l in labels)
    axis_h_in = min(60, max(5.5, need_pt / 72 * 1.15))
    legend_rows = (len({r["type"] for r in log}) + 2) // 3
    fig_h = axis_h_in + 1.6 + 0.25 * legend_rows

    fig = plt.figure(figsize=(8.5, fig_h))
    top_frac = 1 - 1.25 / fig_h
    bot_frac = (0.35 + 0.25 * legend_rows) / fig_h
    ax = fig.add_axes([0.11, bot_frac, 0.86, top_frac - bot_frac])
    ax.set_xlim(0, 10)
    ax.set_ylim(depth_max * 1.02, 0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_ylabel("Depth below surface (ft)")
    ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    ax.set_axisbelow(True)

    BAR_L, BAR_R, TXT_X = 0.9, 2.6, 3.4
    ft_per_pt = (depth_max * 1.02) / (axis_h_in * 72)

    # gaps in the record (shown hatched gray so they are not mistaken for rock)
    covered = sorted((r["from"], r["to"]) for r in log)
    cur = 0.0
    for f, t in covered + [(bottom, bottom)]:
        if f > cur + 0.01:
            ax.add_patch(Rectangle((BAR_L, cur), BAR_R - BAR_L, f - cur, facecolor="#F4F4F4",
                                   edgecolor="#B0B0B0", hatch="////", lw=0.6))
            ax.text(BAR_R + 0.08, (cur + f) / 2, "no record", fontsize=7, color="#8A8A8A",
                    va="center", style="italic")
        cur = max(cur, t)

    # intervals
    for r in log:
        fill, hatch = STYLE.get(r["type"], UNCLASSIFIED)
        hatch_col = "#FFFFFF80" if r["type"] in DARK_FILLS else "#00000055"
        ax.add_patch(Rectangle((BAR_L, r["from"]), BAR_R - BAR_L, r["to"] - r["from"],
                               facecolor=fill, edgecolor=hatch_col, hatch=hatch, lw=0))
        ax.add_patch(Rectangle((BAR_L, r["from"]), BAR_R - BAR_L, r["to"] - r["from"],
                               facecolor="none", edgecolor="#333333", lw=0.7))
        if r["water"]:
            ax.plot(BAR_L - 0.32, (r["from"] + r["to"]) / 2, marker="v", ms=8,
                    color="#1F77D0", markeredgecolor="#0B4C8C")

    # labels with leader lines
    heights = [(len(l) * LINE_PT + LABEL_PAD_PT) * ft_per_pt for l in labels]
    wanted = [(r["from"] + r["to"]) / 2 for r in log]
    placed = place_labels(wanted, heights, 0, depth_max * 1.02)
    for r, lines, y_mid, y_lab in zip(log, labels, wanted, placed):
        ax.plot([BAR_R, BAR_R + 0.3, TXT_X - 0.12, TXT_X - 0.05],
                [y_mid, y_mid, y_lab, y_lab], color="#9A9A9A", lw=0.6)
        ax.text(TXT_X, y_lab, "\n".join(lines), fontsize=FONT, va="center",
                ha="left", color="#222222", linespacing=1.15)

    # completed depth: dashed line across the column, short "TD" tag on the left
    if td is not None:
        ax.plot([BAR_L - 0.55, BAR_R + 0.2], [td, td], color="#C0392B", ls="--", lw=1.4,
                clip_on=False)
        ax.text(BAR_L - 0.6, td, "TD", color="#C0392B", fontsize=8, fontweight="bold",
                va="center", ha="right", clip_on=False)

    # header
    fig.text(0.04, 1 - 0.42 / fig_h, f"Well {well_id}", fontsize=16, fontweight="bold",
             color="#1F3864")
    sub = [county, f"Drilled: {date}"] + loc_bits
    fig.text(0.04, 1 - 0.78 / fig_h, "   |   ".join(sub), fontsize=10, color="#44546A")
    note = [f"Logged to {bottom:g} ft",
            f"Completed depth (TD) {td:g} ft" if td is not None else "No completed depth reported"]
    if tag and str(tag) != str(well_id):
        note.append(f"Well no. on form: {tag}")
    fig.text(0.04, 1 - 1.02 / fig_h, "   ·   ".join(note), fontsize=8.5, color="#6B7A90")

    # legend: lithology types present, in order of first appearance
    present = list(dict.fromkeys(r["type"] for r in log))
    handles = []
    for t in present:
        fill, hatch = STYLE.get(t, UNCLASSIFIED)
        handles.append(Patch(facecolor=fill, edgecolor="#333333", hatch=hatch, lw=0.6,
                             label=t))
    if any(r["water"] for r in log):
        handles.append(plt.Line2D([], [], marker="v", ls="", ms=8, color="#1F77D0",
                                  markeredgecolor="#0B4C8C", label="water reported"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.05 / fig_h), handlelength=2.2, handleheight=1.2)

    fig.savefig(out_path, dpi=200)
    if show:
        matplotlib.use("TkAgg", force=True)
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("well_id", help="well ID (ocr_pages.well_id, or the report's well number)")
    ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "outputs", "welllogs_TV.sqlite"),
                    help="SQLite database (default: outputs/welllogs_TV.sqlite)")
    ap.add_argument("--out", default=None, help="output file (default: well_<id>_column.png)")
    ap.add_argument("--show", action="store_true", help="also open the figure in a window")
    args = ap.parse_args()

    docs = load_well(args.db, args.well_id)
    if not docs:
        sys.exit(f"well {args.well_id} not found in {args.db}")
    log = build_log(docs)
    if not log:
        sys.exit(f"well {args.well_id} has no lithology intervals with readable depths")
    out = args.out or f"well_{docs[0][1]}_column.png"
    plot(docs, log, out, args.show)
    print(f"{len(log)} intervals, 0-{max(r['to'] for r in log):g} ft -> {out}")


if __name__ == "__main__":
    main()
