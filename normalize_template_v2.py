#!/usr/bin/env python3
"""
normalize_template_v2.py - improved normalization template for the well
driller's report OCR pipeline (stage 2 of OCR_main8.py), designed to shrink
unmapped_fields.

Built from
  * the 13 historic sample forms (main/historic samples: 1950s .. Rev 08/13/25)
    and the crosswalk in drillingreport_crosswalk_flags_v0.1.xlsx
  * the labels the v1 template actually pushed into unmapped_fields across
    136,669 normalized records (welllogs_main + welllogs_TV): 2.55 M entries,
    40 % of them with no value, the rest dominated by ~60 recurring form
    fields that v1 has no slot for.

What changed from v1 (all v1 keys keep their name and meaning, so existing
code - export, flagger, well_column - keeps working):
  new identifiers   form.form_number / form.form_revision, other_idwr_no,
                    owners_permit_no, idwr_document_no
  sealing           surface seal provided/depth, surface casing used/cemented,
                    additional_seals[] (2nd/3rd seal rows)
  casing_details    joining method, weight per ft, wall thickness, material,
                    headpipe / tailpipe length, packer used / type
  casing rows       + length_ft (1950s casing record)
  perforations      perforated Y/N, perforation size and count, screen
                    installed Y/N, screen manufacturer / model
  filter_pack       gravel packed Y/N, gravel size
  water             flowing Y/N, flow_gpm, artesian_conditions, controlled_by,
                    leaks_around_casing, unusable_water
  well tests        + set_depth_ft, recovery_time_hr, recovered_level_ft;
                    well_test_status ('not_tested'), test_pump_size
  borehole          drilled_hole_diameter
  lithology rows    + drilling_time, casing_perforated (1950s well log)
  driller           + address, license_no, signatures[] {role, name, date}
  annotations       stamps[], page_number, office_use{}, location_correction,
                    clerk_trs_note
  prompt rules      no empty / label-only values, no printed boilerplate,
                    no repeats of captured values, unmapped_fields capped at 20

Two ways to use it
  1. New normalizations: in OCR_main8.py set
         from normalize_template_v2 import NORMALIZE_PROMPT_V2 as NORMALIZE_PROMPT
     (affects pages normalized afterwards).
  2. Existing records, no model calls: migrate_record(d) moves recognised
     unmapped_fields entries into the v2 fields and drops empty/boilerplate
     ones. Run this file to measure it on a database (read-only):
         python normalize_template_v2.py ../OCR/outputs/welllogs_TV.sqlite [more.sqlite ...]
"""

import copy
import json
import re
import sqlite3
import sys
from collections import Counter

# =========================================================================
# v2 schema (v1 keys unchanged; additions marked in the module docstring)
# =========================================================================

SCHEMA_V2 = {
    "form": {"form_number": None, "form_revision": None},
    "well_id_no": None,
    "well_tag_no": None,
    "drilling_permit_no": None,
    "water_right_no": None,
    "other_idwr_no": None,
    "owners_permit_no": None,
    "idwr_document_no": None,
    "owner": {"name": None, "address": None, "city": None, "state": None, "zip": None},
    "location": {
        "twp": None, "twp_dir": None, "rge": None, "rge_dir": None, "sec": None,
        "quarter_quarter": None, "gov_lot": None, "county": None,
        "lat": None, "long": None, "well_site_address": None, "city": None,
        "lot": None, "blk": None, "subdivision": None, "offset_from_corner": None,
    },
    "use": [],
    "type_of_work": [],
    "drill_method": [],
    "sealing": {
        "material": None, "from_ft": None, "to_ft": None, "weight_volume": None,
        "placement_method": None, "drive_shoe_used": None, "shoe_depths": None,
        "drive_shoe_seal_tested": None, "how_tested": None,
        "surface_seal_provided": None, "surface_seal_depth_ft": None,
        "surface_casing_used": None, "surface_casing_cemented": None,
        "additional_seals": [],
    },
    "casing_liner": [],
    "casing_details": {
        "method_of_joining": None, "weight_per_ft": None, "thickness": None, "material": None,
        "headpipe_length_ft": None, "tailpipe_length_ft": None,
        "packer_used": None, "packer_type": None,
    },
    "perforations_screens": {
        "perforated": None, "perforation_method": None, "perforation_size": None,
        "perforation_count": None, "screen_installed": None, "screen_type": None,
        "screen_manufacturer": None, "screen_model": None, "installation_method": None,
        "entries": [],
    },
    "filter_pack": {
        "gravel_packed": None, "gravel_size": None, "material": None, "from_ft": None,
        "to_ft": None, "weight_volume": None, "placement_method": None,
    },
    "drilled_hole_diameter": None,
    "static_water_level_ft": None,
    "depth_first_water_encountered_ft": None,
    "flowing": None,
    "flow_gpm": None,
    "artesian_conditions": None,
    "artesian_pressure": None,
    "depth_flow_encountered_ft": None,
    "controlled_by": None,
    "access_port_or_control": None,
    "leaks_around_casing": None,
    "unusable_water": None,
    "well_test_status": None,
    "test_pump_size": None,
    "well_tests": [],
    "water_temp": None,
    "bottom_hole_temp": None,
    "water_quality_comments": None,
    "lithologic_log": [],
    "lithology_notes": [],
    "completed_depth_ft": None,
    "date_started": None,
    "date_completed": None,
    "driller": {
        "company_name": None, "firm_no": None, "license_no": None, "address": None,
        "principal_driller": None, "date": None, "driller_or_operator_2": None, "operator_1": None,
        "signatures": [],
    },
    "received_stamp": None,
    "annotations": {
        "stamps": [], "page_number": None, "location_correction": None, "clerk_trs_note": None,
        "office_use": {"inspected_by": None, "twp": None, "rge": None, "sec": None,
                       "quarter_quarter": None, "lat": None, "long": None, "well_id_no": None},
    },
    "unmapped_fields": [],
}

ROW_SHAPES = {
    "casing_liner": {"diameter": None, "from_ft": None, "to_ft": None, "length_ft": None, "gauge": None,
                     "material": None, "casing_or_liner": None, "welded_or_threaded": None},
    "perforations_screens.entries": {"from_ft": None, "to_ft": None, "slot_size": None, "number_per_ft": None,
                                     "diameter": None, "material": None, "casing_or_liner": None},
    "sealing.additional_seals": {"material": None, "from_ft": None, "to_ft": None, "weight_volume": None,
                                 "placement_method": None},
    "well_tests": {"type": None, "yield_gpm": None, "drawdown_ft": None, "pumping_level_ft": None,
                   "duration": None, "set_depth_ft": None, "recovery_time_hr": None, "recovered_level_ft": None},
    "lithologic_log": {"bore_dia_in": None, "from_ft": None, "to_ft": None, "remarks": None, "water": None,
                       "drilling_time": None, "casing_perforated": None},
    "driller.signatures": {"role": None, "name": None, "date": None},
}


def _schema_text():
    s = copy.deepcopy(SCHEMA_V2)
    for path, shape in ROW_SHAPES.items():
        cur = s
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = [shape]
    s["unmapped_fields"] = [{"label": None, "value": None}]
    return json.dumps(s, indent=2)


NORMALIZE_PROMPT_V2 = """You are given an OCR markdown transcription of an Idaho Department of \
Water Resources "Well Driller's Report" (Form 238-7, its 1950s-1980s predecessors, or a \
continuation sheet). Convert it to JSON.

Rules:
- Return VALID JSON ONLY. No prose, no markdown fences.
- Use EXACTLY the field names in the schema below. Use null for anything absent or blank.
- Copy digits exactly as transcribed. NEVER infer, round, or "correct" a number.
- If a value is illegible or uncertain in the transcription (marked with (?)), keep it and append "?".
- Checkboxes appear as "Label [x]" (checked) or "Label [ ]" (unchecked). Record ONLY checked options.
  Yes/No questions (flowing, perforated, screen installed, gravel packed, packer, drive shoe,
  surface seal, surface casing, cemented, leaks around casing, artesian conditions) are "Y", "N",
  or null when neither box is checked.
- CHECKBOX TABLE COLUMNS (Casing/Liner, Welded/Threaded, Water Y/N, test method Pump/Bailer/Air):
  column ORDER differs between form revisions. Always map each [x] to the column HEADER directly
  above it - never assume a position.
- A VALUE is something the driller or IDWR wrote, typed, stamped or checked. Printed form text is
  NEVER a value. Leave a field null when its box is empty - do not copy the label, unit words
  ("ft.", "lb.", "gpm or cfs", "p.s.i.", "inches by inches", "feet to feet"), underscores, or
  "(Measurable)" into it.
- Never output acreage notes ("10 acres", "40 acres", "160 acres") or bare "1/4" strings
  (gov_lot is almost always blank).
- Put the printed form number and revision in form ("Form 238-7", "11/97 JGE", "Rev 08/13/25").
- Depths: completed_depth_ft is the number only. On older forms "Total depth of well" goes in
  completed_depth_ft. "Size of drilled hole" / "Diameter of hole" goes in drilled_hole_diameter.
  "Work started" / "Date of commencement" -> date_started; "Work finished" / "Date of completion"
  -> date_completed.
- Seals: the first seal row goes in sealing; every further seal or filter row goes in
  sealing.additional_seals (or filter_pack). Older forms: "Surface seal depth" / "To what depth?"
  -> sealing.surface_seal_depth_ft; "Material used in seal" -> sealing.material; "Sealing procedure
  used" -> sealing.placement_method.
- Casing: older forms with one "Thickness"/"Casing schedule"/"Method of joining casing" for all
  rows use casing_details. Headpipe / tailpipe lengths and packer answers go in casing_details.
- Well tests: one entry per test row/block. "Hours pumped", "Time", "Length of time of test" ->
  duration. "NOT TESTED" -> well_test_status "not_tested". "Size of pump and motor used to make
  test" -> test_pump_size.
- Copy EVERY data row of the casing, perforation/screen, and lithologic-log tables into the
  corresponding array. Do not drop rows. Lithologic log: one entry per depth interval; record the
  Water column as "Y" or "N" PER ROW exactly as marked - null only if that row is truly blank.
- Section 3 location: capture township number AND direction (N or S), range number AND direction
  (E or W), section, and quarter-quarter calls (e.g. "SE 1/4 SE 1/4").
- Signatures: one driller.signatures entry per signed line {role: "principal_driller", "driller",
  "operator", "firm_official", "supervisor_or_operator", ..., name, date}; write "[signature]"
  for an unreadable signature. Also keep principal_driller / date as before.
- Stamps and office marks: RECEIVED date -> received_stamp; every stamp text (RECEIVED, POSTED,
  MICROFILMED, region) -> annotations.stamps; a stamped/handwritten document serial number ->
  idwr_document_no; "Location Corrected by IDWR ..." -> annotations.location_correction; a
  handwritten township/range/section note at the page bottom -> annotations.clerk_trs_note;
  filled-in "Office Use Only" box -> annotations.office_use. Stamp text never goes in lithology
  remarks or any other field.
- NEVER put these in unmapped_fields: certification statements, instructions ("Sign once if...",
  "Give at least name of road...", "Sketch map location must agree..."), footers ("FORWARD WHITE
  COPY TO WATER RESOURCES", "Use additional sheets..."), table headers, lists of checkbox options,
  empty boxes, or any value already placed in another field.
- unmapped_fields is ONLY for filled-in values that fit no field above: {"label": ..., "value": ...},
  at most 20 entries, no duplicates.

Schema:
""" + _schema_text()

# =========================================================================
# migration of existing (v1) records: unmapped_fields -> v2 fields
# =========================================================================

EMPTY_VALUES = {"", "none", "null", "n/a", "na", "n.a.", "[ ]", "-", "--", "---", "—", "blank", "unchecked",
                "no entries", "not filled", "not provided", "empty", "illegible", "false"}
UNIT_ONLY_RE = re.compile(
    r"^[\s_\-:/.,()\[\]x°]*(?:(?:ft|feet|foot|lb|lbs|p\.?s\.?i\.?|psig|g\.?p\.?m\.?|c\.?f\.?s\.?|inches|inch|in"
    r"|hrs?|hours|min(?:utes)?|°?\s*f(?:ahr(?:enheit)?)?|degrees?|acres|1/4|measurable|gal\.?/min\.?"
    r"|below ground|below land surface|above land surface|or|to|by|and)[\s_\-:/.,()\[\]x°]*)*$", re.I)
BOILERPLATE_RE = re.compile(
    r"forward (?:white )?(?:copy|note|instruction)|^footer|sketch map|give at least name of road|"
    r"location address note|sign once|signature note|certif|i/we certify|sheet note|"
    r"if more space|use additional sheets|log header|lithology log header|sealing procedure options|"
    r"slot size set from|from to slot size|placed from feet to feet|set from feet to feet|"
    r"(?:artesian pressure|yield|well test yield) unit|state law requires|do not fill in|"
    r"describe repairs or abandonment$|\bheader\b|instructions?$|note$", re.I)

YES_RE = re.compile(r"^\s*(?:y|yes|true|checked|\[x\]\s*y(?:es)?|y(?:es)?\s*\[x\])\b", re.I)
NO_RE = re.compile(r"^\s*(?:n|no|false|\[x\]\s*no?|no?\s*\[x\])\b", re.I)


def yn(v):
    s = str(v).strip()
    if re.search(r"\[x\]\s*y", s, re.I) or (YES_RE.match(s) and not re.search(r"\[\s*\]\s*y", s, re.I)):
        return "Y"
    if re.search(r"\[x\]\s*n", s, re.I) or NO_RE.match(s):
        return "N"
    return None


def depth_text(v):
    s = re.sub(r"\(?measurable\)?", "", str(v), flags=re.I).strip(" :")
    return s if re.search(r"\d", s) else None


def rev_mark(v):
    m = re.search(r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?(?:\s*JGE)?|Rev\.?\s*\d{1,2}/\d{1,2}/\d{2})\b", str(v), re.I)
    return m.group(1) if m else None


def text(v):
    return str(v).strip()


# (label regex on the normalised label, target path, value transform)
LABEL_RULES = [
    (r"^(?:form number|form no|form)$", "form.form_number", text),
    (r"^(?:form date|date code|date form|date on form|date printed|form revision|revision|rev)$", "form.form_revision", rev_mark),
    (r"^date$", "form.form_revision", rev_mark),
    (r"^other idwr no", "other_idwr_no", text),
    (r"^owner s permit no|^owners permit no", "owners_permit_no", text),
    (r"^(?:lithologic log id|document (?:no|number)|idwr document)", "idwr_document_no", text),
    (r"^page(?: number)?$", "annotations.page_number", text),
    # sealing
    (r"^(?:surface seal depth|to what depth|surface seal depth feet)", "sealing.surface_seal_depth_ft", depth_text),
    (r"^(?:surface seal provided|surface seal)$", "sealing.surface_seal_provided", yn),
    (r"^material used in seal", "sealing.material", text),
    (r"^sealing procedure(?: used)?$", "sealing.placement_method", text),
    (r"^surface casing used", "sealing.surface_casing_used", yn),
    (r"^cemented in place", "sealing.surface_casing_cemented", yn),
    (r"^(?:was )?(?:casing )?drive shoe used|^was casing drive shoe used", "sealing.drive_shoe_used", yn),
    # casing
    (r"^(?:casing )?length of headpipe|^top packer or headpipe|^headpipe", "casing_details.headpipe_length_ft", text),
    (r"^(?:casing )?length of tailpipe|^tailpipe", "casing_details.tailpipe_length_ft", text),
    (r"^(?:was a )?packer or seal used|^packer(?: y/n)?$", "casing_details.packer_used", yn),
    (r"^packer type", "casing_details.packer_type", text),
    (r"^method of joining casing", "casing_details.method_of_joining", text),
    (r"^weight of casing per lin", "casing_details.weight_per_ft", text),
    (r"^(?:thickness of casing|thickness|casing thickness)$", "casing_details.thickness", text),
    (r"^(?:casing material|casing schedule)$", "casing_details.material", text),
    # perforations / screens
    (r"^(?:perforated|perforations y/n|perforations)$", "perforations_screens.perforated", yn),
    (r"^(?:how perforated|perforations? method|type of perforator used)$", "perforations_screens.perforation_method", text),
    (r"^(?:size of perforations?|perforation size)", "perforations_screens.perforation_size", text),
    (r"^(?:number of perforations|perforation count)$", "perforations_screens.perforation_count", text),
    (r"^(?:well screen installed|was screen installed|manufactured screen)", "perforations_screens.screen_installed", yn),
    (r"^manufacturer s name", "perforations_screens.screen_manufacturer", text),
    (r"^(?:model no|type model no)$", "perforations_screens.screen_model", text),
    (r"^(?:screens? screen type|screen type)$", "perforations_screens.screen_type", text),
    # filter pack
    (r"^(?:well )?gravel packed", "filter_pack.gravel_packed", yn),
    (r"^size of gravel", "filter_pack.gravel_size", text),
    # water
    (r"^(?:flowing|flowing artesian y/n|flowing artesian)$", "flowing", yn),
    (r"^(?:flowing )?g p m flow$|^flow gpm|^give flow", "flow_gpm", text),
    (r"^artesian conditions", "artesian_conditions", yn),
    (r"^(?:static water level )?artesian (?:closed in )?pressure|^shutoff pressure", "artesian_pressure", text),
    (r"^controlled by", "controlled_by", text),
    (r"^does well leak around casing", "leaks_around_casing", yn),
    (r"^did any strata contain unusable water", "unusable_water", text),
    (r"^depth first water encounter", "depth_first_water_encountered_ft", depth_text),
    (r"^(?:static water level|standing water level below ground|depth of standing water)$", "static_water_level_ft", depth_text),
    (r"^(?:quality|water quality test or comments|temperature of quality)$", "water_quality_comments", text),
    (r"^(?:temperature|water temp|temp fahr|temperature f)$", "water_temp", text),
    (r"^bottom hole temp", "bottom_hole_temp", text),
    (r"^describe (?:access port|control)|^describe controlling devices", "access_port_or_control", text),
    (r"^size of pump and motor", "test_pump_size", text),
    (r"^(?:test method|well test (?:type|method)|well test data)$", "well_tests[0].type", text),
    (r"^(?:hours pumped|(?:well tests? )?time|length of time of test)$", "well_tests[0].duration", text),
    (r"^(?:(?:well tests? )?draw ?down)$", "well_tests[0].drawdown_ft", depth_text),
    (r"^(?:(?:well tests? )?pumping level)$", "well_tests[0].pumping_level_ft", depth_text),
    (r"^(?:discharge(?: g p m)?|(?:well tests? )?yield(?: gal /min)?|test delivery)$", "well_tests[0].yield_gpm", text),
    # borehole / depth / dates
    (r"^(?:diameter of hole|size of drilled hole|hole diam)$", "drilled_hole_diameter", text),
    (r"^(?:completed depth(?: measurable| note)?|total depth(?: of well)?)$", "completed_depth_ft", depth_text),
    (r"^(?:work started|date started|date of commencement)", "date_started", text),
    (r"^(?:finished|work finished|completed|date completed|date of completion)$", "date_completed", text),
    # driller
    (r"^(?:driller address|address driller)$", "driller.address", text),
    (r"^(?:license no|lic no)$", "driller.license_no", text),
    (r"^firm no$", "driller.firm_no", text),
    (r"^(?:firm name|driller s or firm s name)$", "driller.company_name", text),
    (r"^(?:signed by firm official|firm official(?: signature)?|signed by)$", "driller.signatures+firm_official", text),
    (r"^(?:and operator|signed by operator|supervisor or operator|driller or operator)$", "driller.signatures+operator", text),
    # annotations / location
    (r"^(?:stamps?|stamp text|microfilm(?:ed)?(?: stamp)?|stamp microfilmed|posted)$", "annotations.stamps+", text),
    (r"^(?:office use only|inspected by)", "annotations.office_use.raw", text),
    (r"^county$", "location.county", text),
    (r"^address of well site", "location.well_site_address", text),
    (r"^(?:lot no|lot)$", "location.lot", text),
    (r"^(?:block no|block)$", "location.blk", text),
    (r"^subdivision name", "location.subdivision", text),
]
LABEL_RULES = [(re.compile(rx), tgt, fn) for rx, tgt, fn in LABEL_RULES]


def norm_label(label):
    s = str(label or "").lower()
    s = re.sub(r"^\s*(?:section\s*)?\d+[a-z]?[.)]\s*", "", s)
    s = re.sub(r"\b(?:section|sec)\s*\d+\s*[-:]\s*", "", s)
    s = re.sub(r"[^a-z0-9/# ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_empty_value(label_n, v):
    if v is None:
        return True
    s = str(v).strip()
    if s.lower() in EMPTY_VALUES or UNIT_ONLY_RE.match(s):
        return True
    vn = norm_label(s)
    return bool(vn) and (vn == label_n or (len(vn) > 6 and vn in label_n) or (len(label_n) > 6 and label_n in vn
                                                                               and not re.search(r"\d", s)))


def _get_parent(d, path):
    """Walk/create dicts along path; supports 'well_tests[0]'. -> (parent, key)."""
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        m = re.fullmatch(r"(\w+)\[(\d+)\]", p)
        if m:
            lst = cur.setdefault(m.group(1), [])
            if not isinstance(lst, list):
                return None, None
            while len(lst) <= int(m.group(2)):
                lst.append({})
            cur = lst[int(m.group(2))]
        else:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
    return cur, parts[-1]


def migrate_record(d, drop=False):
    """-> (new record, Counter of outcomes).

    drop=False (default): NOTHING is discarded. An entry whose label matches a
    v2 field is moved into that field when the field is empty (outcome
    'moved'); every other entry stays in unmapped_fields unchanged, counted as
    kept_empty / kept_boilerplate / kept_duplicate / kept so you can see what
    a cleanup WOULD remove.
    drop=True: empty, boilerplate and duplicate entries are removed as well
    (outcomes dropped_empty / dropped_boilerplate / duplicate)."""
    d = copy.deepcopy(d)
    outcome = Counter()
    kept, seen = [], set()

    def keep(u, why):
        kept.append(u)
        outcome[why] += 1

    for u in d.get("unmapped_fields") or []:
        if not isinstance(u, dict):
            if drop:
                outcome["dropped_empty"] += 1
            else:
                keep(u, "kept_empty")
            continue
        ln, v = norm_label(u.get("label")), u.get("value")
        if BOILERPLATE_RE.search(ln) or (isinstance(v, str) and BOILERPLATE_RE.search(norm_label(v))):
            if drop:
                outcome["dropped_boilerplate"] += 1
            else:
                keep(u, "kept_boilerplate")
            continue
        if is_empty_value(ln, v):
            if drop:
                outcome["dropped_empty"] += 1
            else:
                keep(u, "kept_empty")
            continue
        rule = next(((t, fn) for rx, t, fn in LABEL_RULES if rx.search(ln)), None)
        if rule:
            target, fn = rule
            val = fn(v)
            if val is None:                       # e.g. a Y/N field holding '[ ] Y [ ] N'
                if drop:
                    outcome["dropped_empty"] += 1
                else:
                    keep(u, "kept_empty")
                continue
            if "+" in target:                     # list targets
                path, role = target.split("+", 1)
                parent, key = _get_parent(d, path)
                lst = parent.setdefault(key, []) if parent is not None else None
                if isinstance(lst, list):
                    lst.append({"role": role, "name": val, "date": None} if role else val)
                    outcome["moved"] += 1
                    continue
            else:
                parent, key = _get_parent(d, target)
                if parent is not None:
                    cur = parent.get(key)
                    if cur in (None, "", []):
                        parent[key] = val
                        outcome["moved"] += 1
                    elif drop:
                        outcome["duplicate"] += 1
                    else:
                        keep(u, "kept_duplicate")  # field already filled: keep this value too
                    continue
        sig = (ln, str(v).strip().lower())
        if sig in seen and drop:
            outcome["duplicate"] += 1
            continue
        seen.add(sig)
        keep(u, "kept")
    d["unmapped_fields"] = kept
    return d, outcome


def ensure_schema(d, template=None):
    """Add every v2 key the record lacks (as null / empty list), recursively.
    Existing keys and values are never changed or removed."""
    template = SCHEMA_V2 if template is None else template
    for k, tv in template.items():
        if k not in d:
            d[k] = copy.deepcopy(tv)
        elif isinstance(tv, dict) and isinstance(d[k], dict):
            ensure_schema(d[k], tv)
    return d


def measure(db_paths):
    total, per_rec_before, per_rec_after = Counter(), [], []
    residual = Counter()
    n = 0
    for db in db_paths:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for (j,) in con.execute("SELECT COALESCE(NULLIF(json_final,''), json_text) FROM normalized WHERE valid_json=1"):
            try:
                d = json.loads(j)
            except (json.JSONDecodeError, TypeError):
                continue
            n += 1
            before = len(d.get("unmapped_fields") or [])
            new, oc = migrate_record(d)
            total += oc
            per_rec_before.append(before)
            per_rec_after.append(len(new["unmapped_fields"]))
            for u in new["unmapped_fields"]:
                residual[norm_label(u.get("label"))[:50]] += 1
        con.close()
    entries = sum(per_rec_before)
    print(f"records: {n:,}   unmapped entries before: {entries:,}  (mean {entries / max(n, 1):.1f} per record)")
    for k in ("moved", "kept_empty", "kept_boilerplate", "kept_duplicate", "kept",
              "dropped_empty", "dropped_boilerplate", "duplicate"):
        print(f"  {k:<20} {total[k]:>10,}  ({100 * total[k] / max(entries, 1):.1f}%)")
    after = sum(per_rec_after)
    srt = sorted(per_rec_after)
    print(f"unmapped entries after: {after:,} (mean {after / max(n, 1):.1f}, median {srt[len(srt) // 2]}, "
          f"records with none {sum(1 for x in per_rec_after if x == 0):,})  -> "
          f"{100 * (1 - after / max(entries, 1)):.0f}% fewer")
    print("most common labels still unmapped:")
    for k, v in residual.most_common(25):
        print(f"  {v:>8,}  {k}")


def apply(db_path):
    """Write migrated records into a database (use a COPY): adds columns to
    'normalized' and fills them for every valid record. The original
    json_text / json_final are left untouched.
      json_v2        the v2 record (v1 fields + new fields, cleaned unmapped_fields)
      unmapped_v1    unmapped_fields entries before
      unmapped_v2    unmapped_fields entries after
      v2_outcome     JSON counts: moved / kept_empty / kept_boilerplate / kept_duplicate / kept
    Nothing is discarded: every v1 unmapped entry is either moved into a v2
    field or still in unmapped_fields.
    Re-running overwrites these columns."""
    import time
    con = sqlite3.connect(db_path)
    have = {r[1] for r in con.execute("PRAGMA table_info(normalized)")}
    for col, typ in (("json_v2", "TEXT"), ("unmapped_v1", "INTEGER"), ("unmapped_v2", "INTEGER"),
                     ("v2_outcome", "TEXT")):
        if col not in have:
            con.execute(f"ALTER TABLE normalized ADD COLUMN {col} {typ}")
    con.commit()
    rows = con.execute("SELECT id, COALESCE(NULLIF(json_final,''), json_text) FROM normalized "
                       "WHERE valid_json=1").fetchall()
    t0, batch, total, n_before, n_after = time.time(), [], Counter(), 0, 0
    for i, (rid, j) in enumerate(rows, 1):
        try:
            d = json.loads(j)
        except (json.JSONDecodeError, TypeError):
            continue
        new, oc = migrate_record(d)              # drop=False: nothing is discarded
        ensure_schema(new)
        b, a = len(d.get("unmapped_fields") or []), len(new["unmapped_fields"])
        n_before, n_after, total = n_before + b, n_after + a, total + oc
        batch.append((json.dumps(new, ensure_ascii=False, indent=2), b, a, json.dumps(dict(oc)), rid))
        if len(batch) >= 5000 or i == len(rows):
            con.executemany("UPDATE normalized SET json_v2=?, unmapped_v1=?, unmapped_v2=?, v2_outcome=? "
                            "WHERE id=?", batch)
            con.commit()
            batch = []
            print(f"  {i:,}/{len(rows):,} records  ({time.time() - t0:.0f}s)", flush=True)
    con.close()
    print(f"done: {len(rows):,} records migrated in {time.time() - t0:.0f}s")
    print(f"unmapped_fields entries {n_before:,} -> {n_after:,} "
          f"({100 * (1 - n_after / max(n_before, 1)):.0f}% fewer)")
    for k in ("moved", "kept_empty", "kept_boilerplate", "kept_duplicate", "kept",
              "dropped_empty", "dropped_boilerplate", "duplicate"):
        print(f"  {k:<20} {total[k]:>10,}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "apply":
        apply(sys.argv[2])
    elif len(sys.argv) >= 2:
        measure(sys.argv[1:])
    else:
        print(__doc__)
        sys.exit(1)
