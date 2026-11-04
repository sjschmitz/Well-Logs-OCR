#!/usr/bin/env python3
"""
lithology_extract.py - pull every lithologic-log row out of the OCR database
and count each unique lithology type across all reports.

Input : welllogs_TV.sqlite (output of OCR_main8.py). Reads normalized.json_final
        (falls back to json_text when json_final is empty) and joins
        ocr_pages.well_id. Each record's "lithologic_log" array holds
        {bore_dia_in, from_ft, to_ft, remarks, water}.

Output (written to --out folder, default ./lithology_output):
  lithology_rows.csv         every interval: well, page, depths, water, raw
                             remarks, cleaned remarks, materials found,
                             primary material, combination, flags
  lithology_type_counts.csv  MAIN RESULT - one row per unique primary
                             lithology type (lava, clay, sand, ...) with
                             intervals, wells, total thickness, % water=Y
  lithology_term_counts.csv  every material term counted wherever it appears
                             (a "brown clay w/ sand" row counts clay AND sand)
  lithology_combo_counts.csv unique material combinations ("clay + sand")
  unique_remarks_counts.csv  each distinct cleaned remark string, as written
  unrecognized_words.csv     words not in the vocabulary, most common first -
                             use this to extend MATERIALS / MODIFIERS below
  lithology_summary.xlsx     all of the above as sheets (if openpyxl present)

Usage:
  python lithology_extract.py welllogs_TV.sqlite
  python lithology_extract.py welllogs_TV.sqlite --out results --no-fuzzy

Pure standard library (openpyxl optional, only for the .xlsx summary).
"""

import argparse
import csv
import difflib
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict

# =========================================================
# Vocabulary. Keys are the canonical lithology type; values are the spellings
# that map to it (lower case, matched as whole words / phrases). Extend these
# lists using unrecognized_words.csv after a first run.
# =========================================================

MATERIALS = {
    # unconsolidated
    "topsoil":      ["topsoil", "top soil", "top-soil", "top soi", "top sol", "top so",
                     "overburden", "over burden", "sod", "grass", "sod grass"],
    "soil":         ["soil", "dirt", "loam"],
    "clay":         ["clay", "clays", "clayey", "clayy", "caly", "cley", "clav", "gumbo", "mud", "adobe"],
    "silt":         ["silt", "silty", "silts"],
    "sand":         ["sand", "sands", "sandy", "snd", "quicksand", "quick sand"],
    "gravel":       ["gravel", "gravels", "gravelly", "gravely", "grvl", "gravle", "gravles",
                     "pea gravel", "pebbles", "pebble", "pit run", "pitrun", "river run"],
    "cemented gravel": ["cemented gravel", "cemented gravels", "cement gravel"],
    "cobbles":      ["cobble", "cobbles", "cobbly", "cobblestone", "cobblestones",
                     "river rock", "rocks and gravel"],
    "boulders":     ["boulder", "boulders", "bolders", "bouldery"],
    "hardpan":      ["hardpan", "hard pan", "hard-pan", "hrd pan", "hid pan", "hardpack"],
    "caliche":      ["caliche", "cleache", "cleachy", "caleche", "cliche", "calichie"],
    "cinders":      ["cinder", "cinders", "cender", "cenders", "sinder", "sinders",
                     "cynders", "cin", "scoria"],
    "ash":          ["ash", "volcanic ash"],
    "peat / muck":  ["peat", "muck", "mucky", "organic"],
    "fill":         ["fill", "backfill", "road base", "road mix", "landfill"],
    "pavement":     ["asphalt", "pavement", "concrete", "blacktop"],
    "crevice / fracture / void": ["crevice", "crevices", "crevas", "creviced", "void",
                     "voids", "cavern", "cave", "cavity", "crack", "cracks", "fracture",
                     "fractures", "lost circulation", "lost circ"],
    "breccia":      ["breccia"],
    # volcanic
    "basalt":       ["basalt", "basalts", "basaltic", "bassalt", "basault", "basolt"],
    "lava":         ["lava", "lava rock", "laua"],
    "rhyolite":     ["rhyolite", "rhyolitic"],
    "tuff":         ["tuff", "tuft", "welded tuff", "welded tuft"],
    "obsidian":     ["obsidian"],
    "pumice":       ["pumice"],
    # sedimentary rock
    "sandstone":    ["sandstone", "sand stone", "sand rock"],
    "siltstone":    ["siltstone", "silt stone"],
    "claystone":    ["claystone", "clay stone"],
    "mudstone":     ["mudstone", "mud stone"],
    "shale":        ["shale", "shales", "shaley"],
    "shell":        ["shell", "shells", "shell rock", "shil"],
    "soapstone":    ["soapstone", "soap stone"],
    "conglomerate": ["conglomerate"],
    "limestone":    ["limestone", "lime stone", "lime"],
    # plutonic / metamorphic
    "granite":      ["granite", "granitic", "decomposed granite", "dg"],
    "quartzite":    ["quartzite"],
    "schist":       ["schist"],
    "quartz":       ["quartz"],
    # generic rock words (kept separate on purpose - "red rock", "rock")
    "rock":         ["rock", "rocks", "stone", "stones", "bedrock", "ledge"],
    "red rock":     ["red rock"],
    "black rock":   ["black rock"],
}

# Words that describe a material but are not a material themselves. They are
# stripped when deciding the type but kept in the cleaned remark text.
COLORS = {"red", "brown", "brn", "tan", "gray", "grey", "gry", "black", "blk",
          "blue", "white", "wht", "green", "grn", "yellow", "orange", "pink",
          "rose", "purple", "dark", "dk", "light", "lt", "reddish", "brownish",
          "greenish", "bluish", "grayish", "greyish", "rust", "buff", "cream",
          "pale", "olive", "yellowish", "rusty", "redish", "whitish", "lite",
          "blu", "bl", "multi", "salt", "pepper", "speckled", "mottled", "beige",
          "silver", "gold", "golden", "maroon", "lavender", "violet", "charcoal"}
MODIFIERS = {"hard", "soft", "firm", "loose", "broken", "fractured", "fractures",
             "fracture", "cracked", "cracks", "crack", "decomposed", "weathered",
             "fine", "medium", "coarse", "course", "very", "porous", "dense",
             "tight", "sticky", "caving", "cave", "large", "small", "big",
             "layers", "layer", "layered", "seams", "seam", "strips", "strip",
             "streaks", "streak", "stringers", "lenses", "spots", "pockets",
             "some", "trace", "little", "lots", "mixed", "mix", "w", "with",
             "and", "of", "in", "&", "to", "a", "the", "water", "wet", "dry",
             "damp", "moist", "bearing", "good", "poor", "formation", "cemented",
             "vesicular", "welded", "grained", "gr", "sorted", "packed",
             "crevice", "creviced", "fissure", "lost", "circulation", "flow",
             "gpm", "seep", "seeps", "seepage", "hole", "zone", "type", "ft",
             "multi", "colored", "solid", "chunks", "pieces", "rotten",
             "interbedded", "occasional", "stratas", "strata", "cobbley",
             "thin", "heaving", "heave", "heavy", "tacky", "baked", "burnt", "grain",
             "grains", "chips", "muddy", "stiff", "poorly", "well", "color", "colors",
             "lean", "softer", "harder", "multicolor", "multicolored", "mostly",
             "slightly", "highly", "moderate", "moderately", "rounded", "interbeds",
             "beds", "bed", "lens", "fines", "embedded", "saturated", "few", "mixture",
             "thick", "streaking", "stripes", "striped", "graded", "clean", "dirty",
             "brittle", "pea", "mica", "micaceous", "glassy", "lots", "lot", "up",
             "from", "at", "top", "bottom", "no", "was", "is", "on", "then", "by",
             "it", "or", "as", "like", "only", "very", "real", "but", "also", "for",
             "rocky", "loamy", "soily", "occ", "occasionally", "frac", "fract",
             "fractures", "cracks", "pan", "base", "material", "run", "pit"}
# Adjective forms: they count as a material term but never become the primary
# type when a noun-form material is also present ("sandy clay" -> clay).
ADJECTIVE_FORMS = {"sandy", "clayey", "clayy", "silty", "gravelly", "gravely",
                   "cobbly", "bouldery", "basaltic", "granitic", "rhyolitic",
                   "shaley", "mucky", "muck", "organic",
                   # nouns that usually qualify another material ("quartz sand")
                   "quartz", "pebbles", "pebble",
                   # fracture/void words: only the type when nothing else is named
                   "crevice", "crevices", "crevas", "creviced", "void", "voids",
                   "cavern", "cave", "cavity", "crack", "cracks", "fracture",
                   "fractures"}

# Driller abbreviations and common OCR/spelling slips, applied word by word
# BEFORE matching. (Material misspellings go straight into MATERIALS.)
ABBREV = {"hrd": "hard", "hd": "hard", "brn": "brown", "bm": "brown",
          "brm": "brown", "br": "brown", "brwn": "brown", "med": "medium",
          "md": "medium", "crse": "coarse", "corse": "coarse", "crs": "coarse",
          "fin": "fine", "fn": "fine", "strks": "streaks", "stks": "streaks",
          "stk": "streaks", "fract": "fractured", "fx": "fractured",
          "sm": "small", "lg": "large", "gry": "gray", "wht": "white",
          "blk": "black", "grn": "green", "dk": "dark", "lt": "light",
          "sl": "slightly", "v": "very", "heevy": "heavy", "blu": "blue"}

# Remarks that are not lithology at all (seal rows, test notes, stamps ...).
NON_LITHO_PATTERNS = [
    r"\bseal\b", r"\bbentonite\b", r"\bgrout\b", r"\bcement\b(?! gravel)",
    r"\bpumped\b", r"\bgpm\b.*\bat\b", r"\breceived\b", r"\bmicrofilm",
    r"\bidwr\b", r"\bwater resources\b", r"\bsee attached\b",
    r"\bcasing\b", r"\bscreen\b", r"\bperforat", r"\babandon", r"\bplug",
    r"\btremie\b", r"\bpvc\b", r"\bslot", r"\bliner\b", r"\bwelded\b(?! tuf)",
    r"\btag\b", r"\bsch\b", r"\bstainless\b", r"\bwire wrap", r"\bjohnson\b",
    r"\bhole ?plug\b", r"\bbaroid\b", r"\bpacker\b", r"\bpiezo", r"\binclinometer",
    r"\boriginal log\b", r"\bexisting (?:well|casing)\b", r"\btotal depth\b",
    r"\bcompleted depth\b", r"\bpsi\b", r"\bcut off\b", r"\bpipe\b", r"\bsealing procedures\b",
    r"\binstalled\b", r"\bbackfilled\b", r"\bfilled (?:in|with)\b",
]
DITTO_RE = re.compile(r'^\s*(?:"+|\'\'+|〃+|ditto|same|same as above|do\.?)\s*$', re.I)
DITTO_TOKEN = re.compile(r'(?:^|\s)(?:"|\'\'|〃|ditto)(?=\s|$)', re.I)

# =========================================================
# vocabulary index
# =========================================================

PHRASES = []          # (phrase_tokens, canonical, surface) longest first
WORD2CANON = {}
for canon, spellings in MATERIALS.items():
    for s in spellings:
        toks = tuple(s.split())
        if len(toks) == 1:
            WORD2CANON[toks[0]] = canon
        else:
            PHRASES.append((toks, canon, s))
PHRASES.sort(key=lambda p: -len(p[0]))
KNOWN_WORDS = set(WORD2CANON) | COLORS | MODIFIERS
FUZZY_TARGETS = sorted(w for w in WORD2CANON if len(w) >= 4)


def clean_text(s):
    """Lower case, unify separators, drop the OCR '?' uncertainty marker."""
    s = str(s or "").lower()
    s = s.replace("w/", " with ").replace("w/o", " without ")
    s = s.replace("(?)", " ").replace("?", " ")
    s = re.sub(r"[/\\+;:,.()\[\]{}*_=|-]", " ", s)
    s = re.sub(r"&", " & ", s)
    s = re.sub(r"\d+(\.\d+)?\s*(ft|')?", " ", s)      # stray depths / numbers
    return re.sub(r"\s+", " ", s).strip()


_SPLIT_PARTS = None


def _split_concat(w):
    """'bmclay' -> ['brown','clay'], 'sandstrips' -> ['sand','strips'] when
    both halves are known words; otherwise None."""
    global _SPLIT_PARTS
    if _SPLIT_PARTS is None:
        _SPLIT_PARTS = set(WORD2CANON) | COLORS | MODIFIERS | set(ABBREV)
    if len(w) < 6:
        return None
    for i in range(2, len(w) - 2):
        a, b = w[:i], w[i:]
        if a in _SPLIT_PARTS and b in _SPLIT_PARTS and (
                a in WORD2CANON or b in WORD2CANON):
            return [ABBREV.get(a, a), ABBREV.get(b, b)]
    return None


def tokenize(clean):
    out = []
    for t in clean.split(" "):
        if not t:
            continue
        t = t.strip('"\'!')
        if not t:
            continue
        t = ABBREV.get(t, t)
        if t not in KNOWN_WORDS:
            parts = _split_concat(t)
            if parts:
                out.extend(parts)
                continue
        out.append(t)
    return out


def parse_materials(clean, use_fuzzy=True):
    """Return (terms_in_order, unknown_words, fuzzy_hits).
    terms_in_order: list of (canonical, surface_word, is_adjective)."""
    toks = tokenize(clean)
    terms, unknown, fuzzy = [], [], []
    i = 0
    while i < len(toks):
        hit = None
        for ptoks, canon, surface in PHRASES:
            n = len(ptoks)
            if tuple(toks[i:i + n]) == ptoks:
                hit = (canon, surface, False, n)
                break
        if hit:
            terms.append(hit[:3]); i += hit[3]; continue
        w = toks[i]
        if w in WORD2CANON:
            terms.append((WORD2CANON[w], w, w in ADJECTIVE_FORMS))
        elif w in COLORS or w in MODIFIERS or len(w) <= 1:
            pass
        else:
            m = (difflib.get_close_matches(w, FUZZY_TARGETS, n=1, cutoff=0.84)
                 if use_fuzzy and len(w) >= 5 else [])
            if m:
                terms.append((WORD2CANON[m[0]], w, m[0] in ADJECTIVE_FORMS))
                fuzzy.append(f"{w}->{m[0]}")
            else:
                unknown.append(w)
        i += 1
    return terms, unknown, fuzzy


def primary_type(clean, terms):
    """Main material of the interval: first noun-form material in the part of
    the remark BEFORE 'with' / 'some' / 'trace' (the qualifier clause).
    Falls back to the first material of any kind."""
    if not terms:
        return None
    head = re.split(r"\b(?:with|some|trace|little|occasional|seams of|"
                    r"strips of|streaks of|lenses of|stringers of)\b", clean)[0]
    head_terms, _, _ = parse_materials(head, use_fuzzy=True)
    for pool in (head_terms, terms):
        nouns = [t for t in pool if not t[2]]
        if nouns:
            # "red rock" etc. phrases already resolved; prefer a specific
            # material over the generic word "rock" when both appear
            specific = [t for t in nouns if t[0] != "rock"]
            return (specific or nouns)[0][0]
    return terms[0][0]


def num(v):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(v or ""))
    return float(m.group()) if m else None


# =========================================================
# database reading
# =========================================================

def load_records(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "normalized" not in tables:
        sys.exit(f"{db_path}: no 'normalized' table (tables: {sorted(tables)})")
    cols = {r[1] for r in con.execute("PRAGMA table_info(normalized)")}
    json_expr = ("COALESCE(NULLIF(n.json_final,''), n.json_text)"
                 if "json_final" in cols else "n.json_text")
    if "ocr_pages" in tables:
        sql = (f"SELECT n.page_name, o.well_id, {json_expr} FROM normalized n "
               f"LEFT JOIN ocr_pages o ON o.page_name = n.page_name "
               f"ORDER BY n.page_name")
    else:
        sql = f"SELECT n.page_name, NULL, {json_expr} FROM normalized n ORDER BY n.page_name"
    rows = con.execute(sql).fetchall()
    con.close()
    return rows


def well_id_for(page, well_id, d):
    if well_id:
        return str(well_id)
    if d.get("well_id_no"):
        return str(d["well_id_no"])
    return os.path.basename(str(page)).split("_")[0].split(".")[0]


# =========================================================
# main
# =========================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", help="path to welllogs_TV.sqlite")
    ap.add_argument("--out", default="lithology_output", help="output folder")
    ap.add_argument("--no-fuzzy", action="store_true",
                    help="disable fuzzy spelling matches (e.g. 'basolt'->basalt)")
    ap.add_argument("--include-notes", action="store_true",
                    help="also parse lithology_notes[] free text (off by default)")
    args = ap.parse_args()
    use_fuzzy = not args.no_fuzzy
    os.makedirs(args.out, exist_ok=True)

    records = load_records(args.db)
    out_rows = []
    pages_total = pages_bad_json = pages_no_log = 0
    pages_duplicate = 0
    seen_logs = defaultdict(set)

    for page, well_id, jtxt in records:
        pages_total += 1
        try:
            d = json.loads(jtxt) if jtxt else None
        except (json.JSONDecodeError, TypeError):
            d = None
        if not isinstance(d, dict):
            pages_bad_json += 1
            continue
        log = d.get("lithologic_log") or []
        if args.include_notes:
            log = list(log) + [{"remarks": n, "_note": True}
                               for n in (d.get("lithology_notes") or []) if n]
        log = [e for e in log if isinstance(e, dict)]
        if not any((e.get("remarks") or "").strip() for e in log):
            pages_no_log += 1
            continue
        wid = well_id_for(page, well_id, d)
        # skip an exact duplicate log already read from another page of the
        # same well (duplicate scans / pdf+jpg copies)
        sig = tuple((str(e.get("from_ft")), str(e.get("to_ft")),
                     str(e.get("remarks"))) for e in log)
        if sig in seen_logs[wid]:
            pages_duplicate += 1
            continue
        seen_logs[wid].add(sig)
        prev_remark, prev_bore = None, None

        for idx, e in enumerate(log, 1):
            raw = str(e.get("remarks") or "").strip()
            flags = []
            bore = e.get("bore_dia_in")
            if bore in (None, "") and prev_bore not in (None, ""):
                bore, _ = prev_bore, flags.append("bore_dia_filled_down")
            prev_bore = bore if bore not in (None, "") else prev_bore
            if not raw:
                continue
            remark = raw
            if DITTO_RE.match(raw) and prev_remark:
                remark = prev_remark; flags.append("ditto_expanded")
            elif DITTO_TOKEN.search(raw) and prev_remark:
                # partial ditto: '" " gray' -> previous remark's materials + rest
                remark = DITTO_TOKEN.sub(" ", raw) + " " + prev_remark
                flags.append("partial_ditto_expanded")
            if raw.endswith("?") or "(?)" in raw:
                flags.append("ocr_uncertain")
            if e.get("_note"):
                flags.append("from_lithology_notes")

            clean = clean_text(remark)
            terms, unknown, fuzzy = parse_materials(clean, use_fuzzy)
            non_litho = any(re.search(p, clean) for p in NON_LITHO_PATTERNS)
            if fuzzy:
                flags.append("fuzzy:" + ",".join(fuzzy))
            if non_litho:
                flags.append("possible_non_lithology")
            if not terms:
                flags.append("no_material_recognized")

            prim = primary_type(clean, terms)
            combo = " + ".join(sorted({t[0] for t in terms})) if terms else None
            f_, t_ = num(e.get("from_ft")), num(e.get("to_ft"))
            thick = (t_ - f_) if (f_ is not None and t_ is not None and t_ > f_) else None
            if f_ is not None and t_ is not None and t_ <= f_:
                flags.append("bad_interval")
            water = e.get("water")
            water = water if water in ("Y", "N") else None

            n_words = len(clean.split())
            excluded = non_litho and (not terms or n_words > 8)
            if excluded:
                flags.append("excluded_from_counts")
            elif n_words > 15:
                flags.append("long_narrative")
            out_rows.append({
                "well_id": wid, "page_name": page, "row": idx,
                "bore_dia_in": bore, "from_ft": e.get("from_ft"),
                "to_ft": e.get("to_ft"), "thickness_ft": thick,
                "water": water, "remarks_raw": raw, "remarks_clean": clean,
                "materials": "; ".join(dict.fromkeys(t[0] for t in terms)),
                "primary_type": prim, "combination": combo,
                "unrecognized": " ".join(unknown), "flags": "; ".join(flags),
                "excluded": int(excluded),
            })
            if not DITTO_RE.match(raw):
                prev_remark = remark

    # ---------------- aggregation
    def tally(key_fn):
        agg = defaultdict(lambda: {"n": 0, "wells": set(), "thick": 0.0,
                                   "wy": 0, "wn": 0, "examples": Counter()})
        for r in out_rows:
            if r["excluded"]:
                continue
            for k in key_fn(r):
                a = agg[k]
                a["n"] += 1
                a["wells"].add(r["well_id"])
                a["thick"] += r["thickness_ft"] or 0
                a["wy"] += r["water"] == "Y"
                a["wn"] += r["water"] == "N"
                a["examples"][r["remarks_clean"]] += 1
        out = []
        for k, a in agg.items():
            marked = a["wy"] + a["wn"]
            out.append({
                "lithology": k, "intervals": a["n"], "wells": len(a["wells"]),
                "total_thickness_ft": round(a["thick"], 1),
                "water_Y": a["wy"], "water_N": a["wn"],
                "pct_water_Y_of_marked": round(100 * a["wy"] / marked, 1) if marked else None,
                "top_examples": " | ".join(x for x, _ in a["examples"].most_common(5)),
            })
        return sorted(out, key=lambda x: (-x["intervals"], str(x["lithology"])))

    type_counts = tally(lambda r: [r["primary_type"] or "(unclassified)"])
    term_counts = tally(lambda r: r["materials"].split("; ") if r["materials"] else ["(unclassified)"])
    combo_counts = tally(lambda r: [r["combination"] or "(unclassified)"])
    remark_counts = Counter(r["remarks_clean"] for r in out_rows)
    unk = Counter(w for r in out_rows if not r["excluded"]
                  for w in r["unrecognized"].split())

    # ---------------- write
    def write_csv(name, rows, fields=None):
        path = os.path.join(args.out, name)
        rows = list(rows)
        fields = fields or (list(rows[0].keys()) if rows else ["empty"])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        return path

    remark_rows = [{"remark": k, "count": v} for k, v in remark_counts.most_common()]
    unk_rows = [{"word": k, "count": v} for k, v in unk.most_common()]
    sheets = [("lithology_type_counts", type_counts),
              ("lithology_term_counts", term_counts),
              ("lithology_combo_counts", combo_counts),
              ("unique_remarks_counts", remark_rows),
              ("unrecognized_words", unk_rows),
              ("lithology_rows", out_rows)]
    for name, rows in sheets:
        write_csv(name + ".csv", rows)

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        wb = Workbook(); wb.remove(wb.active)
        for name, rows in sheets:
            ws = wb.create_sheet(name[:31])
            if not rows:
                ws.append(["(no rows)"]); continue
            hdr = list(rows[0].keys()); ws.append(hdr)
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1F3864")
            for r in rows:
                ws.append([r[h] for h in hdr])
            ws.freeze_panes = "A2"
            for i, h in enumerate(hdr, 1):
                ws.column_dimensions[ws.cell(1, i).column_letter].width = \
                    min(60, max(10, len(h) + 2))
        wb.save(os.path.join(args.out, "lithology_summary.xlsx"))
        xlsx_note = "lithology_summary.xlsx"
    except ImportError:
        xlsx_note = "(openpyxl not installed - CSVs only)"

    # ---------------- report
    n_flag_nolith = sum("no_material_recognized" in r["flags"] and not r["excluded"]
                        for r in out_rows)
    n_excl = sum(r["excluded"] for r in out_rows)
    print(f"database           : {args.db}")
    print(f"pages read         : {pages_total}  (invalid JSON {pages_bad_json}, "
          f"no lithologic log {pages_no_log}, "
          f"duplicate log skipped {pages_duplicate})")
    print(f"intervals extracted: {len(out_rows)} from "
          f"{len({r['well_id'] for r in out_rows})} wells")
    print(f"excluded (seal/grout/construction notes, not lithology): {n_excl}")
    print(f"unique remarks     : {len(remark_counts)}")
    print(f"unique primary types: {len(type_counts)}   "
          f"(rows with no recognized material: {n_flag_nolith})")
    print(f"\nTop lithology types (primary):")
    for t in type_counts[:25]:
        print(f"  {t['lithology']:<16} {t['intervals']:>7} intervals  "
              f"{t['wells']:>6} wells")
    if unk_rows:
        print(f"\nMost common unrecognized words (add to MATERIALS/MODIFIERS):")
        print("  " + ", ".join(f"{u['word']}({u['count']})" for u in unk_rows[:20]))
    print(f"\nwrote CSVs + {xlsx_note} to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
