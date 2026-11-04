#!/usr/bin/env python3
"""
Well log OCR pipeline - v8 (production / HPC batch version).

Built for unattended runs over thousands of scans. Differences from v7:
- No evaluation machinery (score/gold removed) - this version just RUNS
  the process and produces data + a review queue.
- Per-page error isolation: a failing page is recorded with status='error'
  and the run continues; it is retried on the next invocation. Only
  authentication failures (401/403) abort the whole run.
- Recursive image discovery: WELLLOG_IMG_DIR is walked, subdirectories
  included; pages are keyed by their relative path.
- Sharding for HPC array jobs: WELLLOG_SHARD="k/n" processes only the
  k-th of n stable hash-based slices and writes to an auto-suffixed
  shard database. "merge" combines shard DBs afterwards.
- The zoom proofread pass can be disabled (WELLLOG_ZOOM=off) to roughly
  halve model time per page, at some accuracy cost on handwriting.
- Everything printed is also appended to a run log file (WELLLOG_LOG).
- "export" writes all validator flags + repair notes to a CSV review
  queue for humans.

Stages (same core process as v5-v7, ~95% field accuracy on the 10-page
audited benchmark):
  1  "ocr"        full-page transcription with the vision model
  1b "zoom"       magnified-crop proofread of the draft (optional)
  2  "normalize"  markdown -> fixed-schema JSON via the text model
  2b "repair"     deterministic re-derivation of checkbox fields from the
                  markdown + OpenCV water-column second witness (offline)
  3  "validate"   consistency checks -> normalized.flags (offline)

Usage:
    python3 OCR_main8.py both               # full pipeline (default)
    python3 OCR_main8.py ocr|zoom|normalize|repair|validate   # one stage
    python3 OCR_main8.py audit              # print flags per page
    python3 OCR_main8.py export             # write review_queue.csv
    python3 OCR_main8.py merge OUT.sqlite SHARD1.sqlite SHARD2.sqlite ...

Typical SLURM array over 8 shards:
    #SBATCH --array=1-8
    export WELLLOG_SHARD="${SLURM_ARRAY_TASK_ID}/8"
    python3 OCR_main8.py both
then:
    python3 OCR_main8.py merge well_logs_all.sqlite well_logs.shard*.sqlite

Environment variables (or .env; see .env.template):
    MINDROUTER_API_KEY   (required for ocr/zoom/normalize stages)
    MINDROUTER_BASE_URL  WELLLOG_IMG_DIR  WELLLOG_DB
    WELLLOG_OCR_MODEL    WELLLOG_NORMALIZE_MODEL
    WELLLOG_WORKERS      (parallel requests, default 20)
    WELLLOG_ZOOM         (on | off; default on)
    WELLLOG_SHARD        (e.g. "3/8"; default: no sharding)
    WELLLOG_CV_POLICY    (off | flag | override; default flag - switch to
                          override once the CV detector is spot-checked on
                          a sample of the real corpus)
    WELLLOG_LOG          (run log path, default well_logs_run.log)
    WELLLOG_REVIEW_CSV   (export path, default review_queue.csv)
"""

import io
import os
import re
import csv
import sys
import json
import time
import zlib
import base64
import sqlite3
import threading
import traceback
import random
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False   # auto-rotation disabled, everything else works

# =========================================================
# config - everything comes from environment variables (or a .env file in
# the working directory / next to this script). No secrets in this file.
# See .env.template for the full list.
# =========================================================

def _load_dotenv():
    """Tiny .env loader (no dependency): KEY=VALUE lines, # comments.
    Looks in the current directory, then next to this script. Real
    environment variables always win over .env values."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(os.getcwd(), ".env"),
                 os.path.join(here, ".env"),
                 os.path.join(here, "..", ".env")):
        if not os.path.isfile(cand):
            continue
        with open(cand) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        break


_load_dotenv()

API_KEY   = os.environ.get("MINDROUTER_API_KEY", "")
BASE_URL  = os.environ.get("MINDROUTER_BASE_URL",
                           "https://mindrouter.uidaho.edu")

IMG_DIR   = os.environ.get("WELLLOG_IMG_DIR", "jpg")
SQLITE_DB = os.environ.get("WELLLOG_DB", "well_logs.sqlite")
LOG_PATH  = os.environ.get("WELLLOG_LOG", "well_logs_run.log")
REVIEW_CSV = os.environ.get("WELLLOG_REVIEW_CSV", "review_queue.csv")

# Zoom proofread pass: "on" (default, best accuracy) or "off" (~2x faster)
ZOOM_ENABLED = os.environ.get("WELLLOG_ZOOM", "on").lower() != "off"

# Sharding for HPC array jobs: WELLLOG_SHARD="k/n" (1-based). Each shard
# takes a stable hash-based slice of the image list and writes to an
# auto-suffixed copy of WELLLOG_DB; combine with the "merge" mode.
SHARD_K = SHARD_N = 0
_shard = os.environ.get("WELLLOG_SHARD", "").strip()
if _shard:
    try:
        SHARD_K, SHARD_N = (int(x) for x in _shard.split("/"))
        assert 1 <= SHARD_K <= SHARD_N
    except (ValueError, AssertionError):
        print(f"Bad WELLLOG_SHARD {_shard!r} - expected e.g. 3/8")
        sys.exit(1)
    _root, _ext = os.path.splitext(SQLITE_DB)
    SQLITE_DB = f"{_root}.shard{SHARD_K}of{SHARD_N}{_ext or '.sqlite'}"

# CV second witness for the lithology water column:
#   "off"      - don't run the detector
#   "flag"     - note disagreements in repair_notes, keep LLM value (default
#                for corpus runs; switch to "override" after spot-checking
#                the detector on a sample of the real data)
#   "override" - CV wins Y<->N disagreements
CV_POLICY = os.environ.get("WELLLOG_CV_POLICY", "flag")

OCR_MODEL       = os.environ.get("WELLLOG_OCR_MODEL", "qwen/qwen3.6-27b")
NORMALIZE_MODEL = os.environ.get("WELLLOG_NORMALIZE_MODEL", "qwen/qwen3.6-35b")

MAX_WORKERS      = int(os.environ.get("WELLLOG_WORKERS", "20"))
REQUEST_TIMEOUT  = 300       # seconds per request
MAX_RETRIES      = 5
MIN_CONTENT_CHARS = 200      # OCR output shorter than this gets status='short'

# Upload size handling. The nginx proxy rejects bodies over its limit with
# HTTP 413 (default 1 MB). Images travel base64-encoded inside the JSON
# body, which inflates size ~33%: 650 KB of JPEG -> ~870 KB of JSON body.
# v4: if the original file already fits the budget (and needs no rotation)
# it is sent byte-for-byte - no recompression generation loss. Otherwise:
# grayscale + downscale + JPEG, NO binarization (hard thresholding destroys
# faint digits and checkmarks). On a 413 the budget halves automatically.
MAX_UPLOAD_BYTES = 650_000
MAX_IMAGE_DIM    = 2000      # px on the long edge

# Generation settings for BOTH stages. Qwen models on MindRouter toggle
# thinking via chat_template_kwargs (per MindRouter docs). Thinking is OFF:
# transcription and restructuring don't need extended reasoning, and
# thinking tokens count against the output budget.
GEN_MAX_TOKENS      = 16384
GEN_ENABLE_THINKING = False

# Stage 1b zoom pass. The page is cut into a ZOOM_COLS x ZOOM_ROWS grid of
# crops, each expanded by ZOOM_OVERLAP of the page dimension on every side so
# nothing is lost at tile borders, then upscaled. Sending a CROP gives the
# vision encoder far more effective resolution on that region than the
# full-page image - the same reason a human zooms in. Each crop is also
# upscaled (up to ZOOM_UPSCALE x, capped at ZOOM_MAX_DIM px).
ZOOM_COLS    = 2
ZOOM_ROWS    = 3
ZOOM_OVERLAP = 0.06          # fraction of page width/height added per side
ZOOM_UPSCALE = 2.0
ZOOM_MAX_DIM = 1600

CHAT_URL = f"{BASE_URL}/v1/chat/completions"

# HTTP statuses worth retrying (rate limits / transient server trouble).
# Anything else (400/401/403/404...) fails immediately - retrying won't help.
RETRYABLE_STATUS = {408, 419, 429, 500, 502, 503, 504}


# =========================================================
# stage 1 prompt. v4 additions target errors found in the 10-page TV4 audit:
# strikethrough corrections kept as the old value, rubber stamps transcribed
# as table-cell contents, and checkboxes marked [x] on guesswork.
# =========================================================

TRANSCRIBE_PROMPT = """Transcribe this scanned well drilling report form COMPLETELY and \
FAITHFULLY as markdown.

Rules:
- Transcribe ALL text: printed form labels, handwritten entries, stamps, and \
margin notes. Do not summarize, interpret, or omit anything.
- CHECKBOXES ARE CRITICAL. For every checkbox on the form, write its label \
followed by [x] if it is marked or [ ] if it is not. Look carefully for check \
marks, X marks, slashes, and filled boxes - marks are often faint or sloppy. \
Never mark a box [x] because that answer would be typical; if you cannot see \
a mark, write [ ]. If you are unsure, write [x](?) or [ ](?).
- Tables: render as markdown tables (pipe syntax), one row per form row, \
keeping every column. In the lithologic log, include the Water Y/N column for \
EVERY row exactly as marked ([x] under Y, [x] under N, or blank).
- If a handwritten remark continues across multiple lines of the same table \
row, keep it together in that row's cell.
- If a value is struck through and replaced, transcribe the REPLACEMENT as \
the value and note the original in parentheses, e.g. 21' (was 22').
- Rubber stamps (RECEIVED, date stamps, PERFORATED - SEE DIAGRAM, MICROFILMED, \
etc.) are overlays, not form data. Transcribe them in a separate "Stamps" \
section at the end - NEVER as the contents of a table cell, even when the \
stamp overlaps a table.
- Copy handwriting EXACTLY as written, letter by letter - especially names \
and place names. Never substitute a more familiar word for an unusual one.
- Transcribe digits exactly. If a character is ambiguous or illegible, give \
your best reading followed by (?).
- Preserve the form's section structure (numbered sections) with headings."""


# =========================================================
# stage 1b prompt - draft + magnified crops -> corrected transcription.
# The crops give the model the effective resolution it lacked on the full
# page; the draft keeps the overall structure so it only has to PROOFREAD.
# =========================================================

ZOOM_PROMPT = """You previously transcribed a scanned well drilling report from a single \
full-page image. Below is your DRAFT transcription. After it you are given \
{ntiles} MAGNIFIED, OVERLAPPING crops of the same page ({cols} columns x \
{rows} rows, reading order left-to-right then top-to-bottom, with overlap at \
every border).

Your job is to PROOFREAD the draft against the magnified crops and return \
the FULL corrected transcription.

Rules:
- The crops show the same page at much higher magnification. Wherever a crop \
clearly contradicts the draft, the crop wins.
- Pay special attention to the things magnification reveals:
  * handwritten digits (4 vs 9, 7 vs 1, 2 vs 0, 6 vs 0)
  * handwritten names and place names, letter by letter
  * which checkbox actually holds the mark
  * which column (Y or N) each lithology water mark sits in
  * struck-through values and their replacements (use the replacement, note \
the original in parentheses)
- Do NOT rewrite, rephrase, or reformat anything the crops do not contradict: \
reproduce the draft verbatim except for the specific corrections.
- Keep the exact same markdown structure (headings, pipe tables, checkbox \
[x]/[ ] notation, Stamps section).
- If a character is still ambiguous even at this magnification, give your \
best reading followed by (?).
- Return ONLY the corrected markdown transcription. No commentary, no list \
of changes.

DRAFT TRANSCRIPTION:

{draft}"""


# =========================================================
# stage 2 prompt - markdown -> fixed-schema JSON. v4 additions: map checkbox
# columns by HEADER (column order differs between form revisions), never use
# form label fragments as values, copy every table data row.
# =========================================================

NORMALIZE_PROMPT = """You are given an OCR markdown transcription of an Idaho Department of \
Water Resources "Well Driller's Report" (Form 238-7 or similar). Convert it to JSON.

Rules:
- Return VALID JSON ONLY. No prose, no markdown fences.
- Use EXACTLY the field names in the schema below. Use null for anything absent.
- Copy digits exactly as transcribed. NEVER infer, round, or "correct" a number.
- If a value is illegible or uncertain in the transcription (marked with (?)), \
keep it and append "?".
- Checkboxes appear in the transcription as "Label [x]" (checked) or \
"Label [ ]" (unchecked). Record ONLY checked options.
- CHECKBOX TABLE COLUMNS (Casing/Liner, Welded/Threaded, Water Y/N, test \
method Pump/Bailer/Air): column ORDER differs between form revisions. Always \
map each [x] to the column HEADER directly above it - never assume a position.
- Form label fragments are NOT data. Never output acreage notes ("10 acres", \
"40 acres", "160 acres"), bare "1/4" strings, or other printed label text as \
field values (gov_lot in particular is almost always blank).
- Stamp text (RECEIVED, dates of receipt, SEE DIAGRAM, etc.) belongs ONLY in \
received_stamp or unmapped_fields - never in lithology remarks or other fields.
- Copy EVERY data row of the casing, perforation/screen, and lithologic-log \
tables into the corresponding array. Do not drop rows.
- Section 3 location: capture township number AND direction (N or S), range number \
AND direction (E or W), section, and quarter-quarter calls (e.g. "SE 1/4 SE 1/4").
- Lithologic log: one entry per depth interval. Record the Water column as "Y" or "N" \
PER ROW exactly as marked - use null only if that row's column is truly blank.
- Anything that does not fit the schema goes in unmapped_fields as \
{"label": ..., "value": ...}. Discard nothing.

Schema:
{
  "well_id_no": null,
  "well_tag_no": null,
  "drilling_permit_no": null,
  "water_right_no": null,
  "owner": {"name": null, "address": null, "city": null, "state": null, "zip": null},
  "location": {
    "twp": null, "twp_dir": null, "rge": null, "rge_dir": null, "sec": null,
    "quarter_quarter": null, "gov_lot": null, "county": null,
    "lat": null, "long": null, "well_site_address": null, "city": null,
    "lot": null, "blk": null, "subdivision": null
  },
  "use": [],
  "type_of_work": [],
  "drill_method": [],
  "sealing": {"material": null, "from_ft": null, "to_ft": null,
              "weight_volume": null, "placement_method": null,
              "drive_shoe_used": null, "shoe_depths": null,
              "drive_shoe_seal_tested": null, "how_tested": null},
  "casing_liner": [{"diameter": null, "from_ft": null, "to_ft": null,
                    "gauge": null, "material": null, "casing_or_liner": null,
                    "welded_or_threaded": null}],
  "perforations_screens": {"perforation_method": null, "screen_type": null,
                           "installation_method": null, "entries": []},
  "filter_pack": {"material": null, "from_ft": null, "to_ft": null,
                  "weight_volume": null, "placement_method": null},
  "static_water_level_ft": null,
  "artesian_pressure": null,
  "depth_flow_encountered_ft": null,
  "access_port_or_control": null,
  "well_tests": [{"type": null, "yield_gpm": null, "drawdown_ft": null,
                  "pumping_level_ft": null, "duration": null}],
  "water_temp": null,
  "bottom_hole_temp": null,
  "water_quality_comments": null,
  "depth_first_water_encountered_ft": null,
  "lithologic_log": [{"bore_dia_in": null, "from_ft": null, "to_ft": null,
                      "remarks": null, "water": null}],
  "lithology_notes": [],
  "completed_depth_ft": null,
  "date_started": null,
  "date_completed": null,
  "driller": {"company_name": null, "firm_no": null, "principal_driller": null,
              "date": null, "driller_or_operator_2": null, "operator_1": null},
  "received_stamp": null,
  "unmapped_fields": []
}"""


# =========================================================
# database - one shared connection, writes serialized by a lock
# =========================================================

db_lock = threading.Lock()
db_conn = None


def init_db():
    global db_conn
    db_conn = sqlite3.connect(SQLITE_DB, check_same_thread=False)
    db_conn.execute("PRAGMA journal_mode=WAL;")
    db_conn.execute("PRAGMA synchronous=NORMAL;")
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS ocr_pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            well_id TEXT,
            page_name TEXT UNIQUE,
            markdown TEXT,
            model TEXT,
            pages INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            runtime_sec REAL,
            status TEXT,
            timestamp_utc TEXT,
            markdown_zoom TEXT,
            zoom_status TEXT,
            zoom_runtime_sec REAL
        )
    """)
    # migrate a pre-v5 ocr_pages table (e.g. a copied TV5 db)
    for col, typ in (("markdown_zoom", "TEXT"), ("zoom_status", "TEXT"),
                     ("zoom_runtime_sec", "REAL")):
        try:
            db_conn.execute(f"ALTER TABLE ocr_pages ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS normalized (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_name TEXT UNIQUE,
            json_text TEXT,
            model TEXT,
            valid_json INTEGER,
            runtime_sec REAL,
            timestamp_utc TEXT,
            flags TEXT,
            json_final TEXT,
            repair_notes TEXT
        )
    """)
    # migrate a pre-v6 normalized table
    for col in ("flags", "json_final", "repair_notes"):
        try:
            db_conn.execute(f"ALTER TABLE normalized ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    db_conn.commit()


def db_write(sql, params):
    with db_lock:
        db_conn.execute(sql, params)
        db_conn.commit()


def db_select(sql, params=()):
    with db_lock:
        return db_conn.execute(sql, params).fetchall()


def now_utc():
    return datetime.now(timezone.utc).isoformat()


# =========================================================
# one requests.Session per thread (connection reuse = faster)
# =========================================================

thread_local = threading.local()


def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        s.headers.update({"Authorization": f"Bearer {API_KEY}"})
        thread_local.session = s
    return thread_local.session


class _Tee:
    """Mirror stdout to the run log so multi-day jobs leave a record."""

    def __init__(self, path):
        self.file = open(path, "a")
        self.stdout = sys.stdout
        self.file.write(f"\n===== run started {now_utc()} "
                        f"(db={SQLITE_DB}) =====\n")

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class FatalAPIError(Exception):
    """Non-retryable failure (bad key, bad model name, bad request)."""


def _is_auth_error(exc):
    """401/403 mean every request will fail - abort the run. Anything else
    fatal is recorded per page and the batch continues."""
    return "HTTP 401" in str(exc) or "HTTP 403" in str(exc)


def post_with_retry(url, *, json_body=None, label=""):
    """POST with retry on transient errors, fail-fast on permanent ones."""
    session = get_session()
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(url, json=json_body, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in RETRYABLE_STATUS:
                last_err = f"HTTP {resp.status_code}"
            else:
                raise FatalAPIError(
                    f"HTTP {resp.status_code}: {resp.text[:300]}"
                )
        except FatalAPIError:
            raise
        except requests.exceptions.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"

        wait = min(2 ** attempt * 5, 120) + random.random() * 3
        print(f"[RETRY {attempt}/{MAX_RETRIES}] {label}: {last_err} "
              f"- waiting {wait:.0f}s")
        time.sleep(wait)

    raise Exception(f"{label}: failed after {MAX_RETRIES} retries ({last_err})")


# =========================================================
# shared progress printer
# =========================================================

stats_lock = threading.Lock()


class Progress:
    def __init__(self, total, stage):
        self.total = total
        self.stage = stage
        self.done = 0
        self.runtime_sum = 0.0
        self.start = time.perf_counter()

    def tick(self, name, runtime, extra=""):
        with stats_lock:
            self.done += 1
            self.runtime_sum += runtime
            elapsed = time.perf_counter() - self.start
            rate = self.done / elapsed if elapsed > 0 else 0
            eta = (self.total - self.done) / rate if rate > 0 else 0
            print(f"[{self.stage} {self.done}/{self.total}] {name}  "
                  f"{runtime:.1f}s  "
                  f"throughput={rate * 3600:.0f}/hr  "
                  f"ETA={int(eta // 3600):02d}:{int(eta % 3600 // 60):02d}:{int(eta % 60):02d}"
                  f"{('  ' + extra) if extra else ''}")


# =========================================================
# STAGE 1 - transcription (chat completions + our prompt)
# =========================================================

def _flip_score(img):
    """Up/down discriminator for THESE IDWR forms, validated on the 10-page
    test batch (9/9 informative pages, including the one rotated scan).

    For each detected text line, compare ink in the top vs bottom half.
    Form text is printed ON ruled underlines, so upright pages are
    bottom-heavy => LOWER score means upright. (Note this is the opposite
    of the usual ascender heuristic for plain text - the ruled lines
    dominate.) Returns mean (top-bottom)/(top+bottom) over text lines."""
    g = img.convert("L").copy()
    g.thumbnail((900, 900))
    a = np.asarray(g) < 128
    # mask solid bars (redactions, heavy rules): rows that are >50% ink are
    # not text and would swamp the statistics
    bar = a.sum(axis=1) > 0.5 * a.shape[1]
    a = a.copy()
    a[bar] = False
    rows = a.sum(axis=1)
    thr = max(3, int(rows.mean()))
    inline = rows > thr
    score, n, i = 0.0, 0, 0
    while i < len(inline):
        if inline[i]:
            j = i
            while j < len(inline) and inline[j]:
                j += 1
            h = j - i
            if 5 <= h <= 40:           # plausible text-line height
                mid = i + h // 2
                top, bot = a[i:mid].sum(), a[mid:j].sum()
                if top + bot > 0:
                    score += (top - bot) / (top + bot)
                    n += 1
            i = j
        else:
            i += 1
    return score / n if n else 0.0


def auto_orient(img):
    """These IDWR forms are portrait. If a scan is landscape, rotate it to
    portrait, choosing 90 vs 270 by the flip score (found one 90-degree
    -rotated page in the first 10-image test batch). Upside-down portrait
    scans are corrected too, with a safety margin against false flips.
    Returns (image, degrees_rotated_ccw)."""
    if not HAVE_NUMPY:
        return img, 0
    if img.width > img.height:
        cand = {90: img.rotate(90, expand=True),
                270: img.rotate(270, expand=True)}
        best = min(cand, key=lambda a: _flip_score(cand[a]))
        return cand[best], best
    flipped = img.rotate(180)
    if _flip_score(flipped) < _flip_score(img) - 0.03:
        return flipped, 180
    return img, 0


def prepare_image(path, max_bytes):
    """Return JPEG bytes for upload.

    Fast path: if the file is already a JPEG under the budget and needs no
    rotation, send the original bytes untouched (no recompression loss).

    Slow path: grayscale + downscale + JPEG quality ladder. Deliberately NO
    hard thresholding/binarization - that destroys faint digits and
    checkmarks. If lowering JPEG quality isn't enough, shrink dimensions
    until it fits."""
    with open(path, "rb") as fh:
        raw = fh.read()
    img = Image.open(io.BytesIO(raw))
    img, rotated = auto_orient(img)

    if (rotated == 0 and len(raw) <= max_bytes
            and (img.format or "").upper() == "JPEG"
            and max(img.size) <= MAX_IMAGE_DIM):
        return raw, rotated

    img = img.convert("L")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)),
                         Image.Resampling.LANCZOS)

    while True:
        for quality in (95, 88, 80, 70, 60, 50, 40):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= max_bytes:
                return buf.getvalue(), rotated

        # still too big at lowest quality: shrink 20% and try again,
        # down to a floor where further shrinking would hurt OCR too much
        w, h = img.size
        if max(w, h) <= 800:
            return buf.getvalue(), rotated  # best effort; caller handles a 413
        img = img.resize((int(w * 0.8), int(h * 0.8)),
                         Image.Resampling.LANCZOS)


def ocr_one(rel, progress):
    """rel is the page's path RELATIVE to IMG_DIR (also its DB key)."""
    path = os.path.join(IMG_DIR, rel)
    page = rel
    well_id = os.path.basename(rel).split("_")[0]
    start = time.perf_counter()

    try:
        budget = MAX_UPLOAD_BYTES
        while True:
            img_bytes, rotated = prepare_image(path, budget)
            img_b64 = base64.b64encode(img_bytes).decode()
            body = {
                "model": OCR_MODEL,
                "temperature": 0,
                "max_completion_tokens": GEN_MAX_TOKENS,
                "chat_template_kwargs": {
                    "enable_thinking": GEN_ENABLE_THINKING
                },
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }},
                    ],
                }],
            }
            try:
                resp = post_with_retry(CHAT_URL, json_body=body, label=page)
                break
            except FatalAPIError as e:
                # 413 = this payload too big for the proxy. Halve the image
                # budget and recompress; only give up below 100 KB.
                if "HTTP 413" in str(e) and budget > 100_000:
                    budget //= 2
                    print(f"[413] {page}: rejected at this size - "
                          f"recompressing to <= {budget // 1000} KB and retrying")
                    continue
                raise

        result = resp.json()
        content = result["choices"][0]["message"]["content"] or ""
        usage = result.get("usage", {}) or {}
        status = "ok" if len(content) >= MIN_CONTENT_CHARS else "short"

        runtime = time.perf_counter() - start
        # OR REPLACE so a re-run can overwrite a previous 'short' attempt
        db_write("""
            INSERT OR REPLACE INTO ocr_pages
            (well_id, page_name, markdown, model, pages,
             prompt_tokens, completion_tokens, runtime_sec, status, timestamp_utc)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (well_id, page, content, OCR_MODEL, 1,
              usage.get("prompt_tokens"),
              usage.get("completion_tokens"), runtime, status, now_utc()))

        notes = []
        if rotated:
            notes.append(f"rotated {rotated} deg")
        if status == "short":
            notes.append("** SHORT OUTPUT **")
        progress.tick(page, runtime, extra="  ".join(notes))

    except FatalAPIError as e:
        if _is_auth_error(e):
            print(f"AUTH FAILURE on {page}: {e} - aborting run")
            raise
        # record the failure and keep the batch moving; this page is
        # retried automatically on the next invocation
        print(f"ERROR (recorded, run continues): {page}: {e}")
        db_write("""
            INSERT OR REPLACE INTO ocr_pages
            (well_id, page_name, markdown, model, status, timestamp_utc)
            VALUES (?,?,NULL,?, 'error', ?)
        """, (well_id, page, OCR_MODEL, now_utc()))
    except Exception:
        print(f"FAILED: {page}")
        traceback.print_exc()


def list_images():
    """All images under IMG_DIR (recursive), as sorted relative paths,
    filtered to this process's shard when WELLLOG_SHARD is set."""
    rels = []
    for root, _, files in os.walk(IMG_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                rels.append(os.path.relpath(os.path.join(root, f), IMG_DIR))
    rels.sort()
    if SHARD_N:
        rels = [r for r in rels
                if zlib.crc32(r.encode()) % SHARD_N == SHARD_K - 1]
    return rels


def run_ocr():
    images = list_images()
    # only pages with a GOOD previous transcription are skipped;
    # 'short'/'error' pages are retried on every run
    done = {r[0] for r in db_select(
        "SELECT page_name FROM ocr_pages WHERE status='ok'")}
    todo = [r for r in images if r not in done]

    print(f"\n=== STAGE 1: TRANSCRIBE ===\n"
          f"total images : {len(images)}"
          f"{f'  (shard {SHARD_K}/{SHARD_N})' if SHARD_N else ''}\n"
          f"already done  : {len(done)}\n"
          f"to process    : {len(todo)}\n"
          f"workers       : {MAX_WORKERS}\n")

    if not todo:
        return

    progress = Progress(len(todo), "OCR")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(ocr_one, p, progress) for p in todo]
        for f in as_completed(futures):
            try:
                f.result()
            except FatalAPIError:
                ex.shutdown(cancel_futures=True)
                sys.exit(1)


# =========================================================
# STAGE 1b - zoom refinement: draft + magnified crops -> corrected draft
# =========================================================

def make_tiles(img):
    """Cut the (already correctly oriented) page into ZOOM_COLS x ZOOM_ROWS
    overlapping crops and upscale each. Returns a list of PIL images in
    reading order (left-to-right, top-to-bottom)."""
    w, h = img.size
    ox, oy = int(w * ZOOM_OVERLAP), int(h * ZOOM_OVERLAP)
    tiles = []
    for r in range(ZOOM_ROWS):
        for c in range(ZOOM_COLS):
            x0 = max(0, w * c // ZOOM_COLS - ox)
            x1 = min(w, w * (c + 1) // ZOOM_COLS + ox)
            y0 = max(0, h * r // ZOOM_ROWS - oy)
            y1 = min(h, h * (r + 1) // ZOOM_ROWS + oy)
            t = img.crop((x0, y0, x1, y1))
            scale = min(ZOOM_UPSCALE, ZOOM_MAX_DIM / max(t.size))
            if scale > 1.0:
                t = t.resize((int(t.width * scale), int(t.height * scale)),
                             Image.Resampling.LANCZOS)
            tiles.append(t)
    return tiles


def encode_tile(tile, max_bytes):
    """Grayscale JPEG under max_bytes, walking down the quality ladder and
    shrinking 15% per lap if quality alone is not enough."""
    t = tile.convert("L")
    while True:
        for quality in (85, 75, 65, 55, 45):
            buf = io.BytesIO()
            t.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= max_bytes:
                return buf.getvalue()
        if max(t.size) <= 500:
            return buf.getvalue()   # best effort
        t = t.resize((int(t.width * 0.85), int(t.height * 0.85)),
                     Image.Resampling.LANCZOS)


def zoom_one(rel, draft, progress):
    path = os.path.join(IMG_DIR, rel)
    page = rel
    start = time.perf_counter()
    try:
        img = Image.open(path)
        img, _ = auto_orient(img)
        ntiles = ZOOM_COLS * ZOOM_ROWS

        budget = MAX_UPLOAD_BYTES
        while True:
            per_tile = max(40_000, (budget - len(draft)) // ntiles)
            content = [{"type": "text",
                        "text": ZOOM_PROMPT.format(ntiles=ntiles,
                                                   cols=ZOOM_COLS,
                                                   rows=ZOOM_ROWS,
                                                   draft=draft)}]
            for tile in make_tiles(img):
                b64 = base64.b64encode(encode_tile(tile, per_tile)).decode()
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"}})
            body = {
                "model": OCR_MODEL,
                "temperature": 0,
                "max_completion_tokens": GEN_MAX_TOKENS,
                "chat_template_kwargs": {
                    "enable_thinking": GEN_ENABLE_THINKING
                },
                "messages": [{"role": "user", "content": content}],
            }
            try:
                resp = post_with_retry(CHAT_URL, json_body=body, label=page)
                break
            except FatalAPIError as e:
                if "HTTP 413" in str(e) and budget > 150_000:
                    budget //= 2
                    print(f"[413] {page}: zoom payload too big - "
                          f"recompressing tiles to <= {per_tile // 2000} KB each")
                    continue
                raise

        refined = resp.json()["choices"][0]["message"]["content"] or ""
        # a refinement that lost big chunks of the draft is worse than no
        # refinement: keep it only if it is at least 70% of the draft length
        status = "ok" if len(refined) >= 0.7 * len(draft) else "suspect_short"
        runtime = time.perf_counter() - start

        db_write("""
            UPDATE ocr_pages
            SET markdown_zoom=?, zoom_status=?, zoom_runtime_sec=?
            WHERE page_name=?
        """, (refined if status == "ok" else None, status, runtime, page))
        # the old normalized row was built from the draft - force a re-run
        if status == "ok":
            db_write("DELETE FROM normalized WHERE page_name=?", (page,))

        progress.tick(page, runtime,
                      extra="" if status == "ok"
                      else "** SUSPECT (kept draft) **")

    except FatalAPIError as e:
        if _is_auth_error(e):
            print(f"AUTH FAILURE on {page}: {e} - aborting run")
            raise
        print(f"ERROR (recorded, run continues): {page}: {e}")
        db_write("UPDATE ocr_pages SET zoom_status='error' WHERE page_name=?",
                 (page,))
    except Exception:
        print(f"FAILED: {page}")
        traceback.print_exc()


def run_zoom():
    if not ZOOM_ENABLED:
        print("\n=== STAGE 1b: ZOOM REFINE === skipped (WELLLOG_ZOOM=off)")
        return
    rows = db_select("""
        SELECT page_name, markdown FROM ocr_pages
        WHERE status='ok'
          AND markdown IS NOT NULL AND length(markdown) > 0
          AND (markdown_zoom IS NULL OR markdown_zoom = '')
          AND (zoom_status IS NULL OR zoom_status NOT IN ('ok', 'suspect_short'))
    """)
    print(f"\n=== STAGE 1b: ZOOM REFINE ===\n"
          f"to process : {len(rows)}\n"
          f"tiles/page : {ZOOM_COLS * ZOOM_ROWS}\n"
          f"workers    : {MAX_WORKERS}\n")

    if not rows:
        return

    progress = Progress(len(rows), "ZOOM")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(zoom_one, page, md, progress)
                   for page, md in rows
                   if os.path.exists(os.path.join(IMG_DIR, page))]
        for f in as_completed(futures):
            try:
                f.result()
            except FatalAPIError:
                ex.shutdown(cancel_futures=True)
                sys.exit(1)


# =========================================================
# STAGE 2 - normalize markdown -> structured JSON
# =========================================================

def extract_json_text(text):
    """Salvage JSON from model output that may include thinking text or
    markdown fences. Returns (json_string, valid_flag)."""
    text = (text or "").strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        candidate = text[s:e + 1]
        try:
            json.loads(candidate)
            return candidate, 1
        except json.JSONDecodeError:
            pass
    try:
        json.loads(text)
        return text, 1
    except json.JSONDecodeError:
        return text, 0


def normalize_one(page, markdown, progress):
    start = time.perf_counter()
    try:
        body = {
            "model": NORMALIZE_MODEL,
            "temperature": 0,
            "max_completion_tokens": GEN_MAX_TOKENS,
            "chat_template_kwargs": {
                "enable_thinking": GEN_ENABLE_THINKING
            },
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "user",
                 "content": f"{NORMALIZE_PROMPT}\n\nTranscription:\n\n{markdown}"}
            ],
        }
        resp = post_with_retry(CHAT_URL, json_body=body, label=page)
        raw = resp.json()["choices"][0]["message"]["content"]
        text, valid = extract_json_text(raw)

        runtime = time.perf_counter() - start
        db_write("""
            INSERT OR REPLACE INTO normalized
            (page_name, json_text, model, valid_json, runtime_sec,
             timestamp_utc, flags)
            VALUES (?,?,?,?,?,?,NULL)
        """, (page, text, NORMALIZE_MODEL, valid, runtime, now_utc()))

        progress.tick(page, runtime,
                      extra="** INVALID JSON **" if not valid else "")

    except FatalAPIError as e:
        if _is_auth_error(e):
            print(f"AUTH FAILURE on {page}: {e} - aborting run")
            raise
        if "HTTP 422" in str(e):
            print(f"SKIPPED (422 token budget): {page} - "
                  f"raise GEN_MAX_TOKENS if this persists")
            return
        # no row is written, so this page is retried on the next run
        print(f"ERROR (recorded, run continues): {page}: {e}")
    except Exception:
        print(f"FAILED: {page}")
        traceback.print_exc()


def run_normalize():
    # process pages with no normalized row yet OR an invalid previous attempt;
    # prefer the zoom-refined markdown over the draft
    rows = db_select("""
        SELECT o.page_name,
               COALESCE(NULLIF(o.markdown_zoom, ''), o.markdown)
        FROM ocr_pages o
        LEFT JOIN normalized n ON n.page_name = o.page_name
        WHERE (n.page_name IS NULL OR n.valid_json = 0)
              AND o.markdown IS NOT NULL
              AND length(o.markdown) > 0
    """)

    print(f"\n=== STAGE 2: NORMALIZE ===\n"
          f"to process : {len(rows)}\n"
          f"workers    : {MAX_WORKERS}\n")

    if not rows:
        return

    progress = Progress(len(rows), "NORM")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(normalize_one, page, md, progress)
                   for page, md in rows]
        for f in as_completed(futures):
            try:
                f.result()
            except FatalAPIError:
                ex.shutdown(cancel_futures=True)
                sys.exit(1)


# =========================================================
# STAGE 2b - deterministic repair (pure Python, no API).
# The 27b OCR reads checkboxes into the markdown reliably; the 35b
# normalizer mapping those marks into JSON is where flips, drops, and swaps
# happen (every audited stage-2 regression was of this kind). So: re-derive
# those fields mechanically from the markdown tables and OVERWRITE the
# normalizer's answer. Optionally cross-check the lithology water column
# with the OpenCV detector as an independent witness.
# =========================================================

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cv_watercol import detect_water_column
    HAVE_CV = True
except Exception:
    HAVE_CV = False


def _repair_casing(d, tables, notes):
    table = next((t for t in tables
                  if _find_col(t["headers"], "welded") is not None
                  and _find_col(t["headers"], "threaded") is not None), None)
    entries = d.get("casing_liner") or []
    if table is None or not entries:
        return
    h = table["headers"]
    iw, it = _find_col(h, "welded"), _find_col(h, "threaded")
    ic, il = _find_col(h, "casing"), _find_col(h, "liner")
    idia = _find_col(h, "diameter", "diameternominal")
    ifrom = _find_col(h, "from", "fromft")
    ito = _find_col(h, "to", "toft")
    for e in entries:
        row = _match_row(table["rows"],
                         [i for i in (idia, ifrom, ito) if i is not None],
                         [e.get("diameter"), e.get("from_ft"), e.get("to_ft")])
        if row is None or len(row) != len(h):
            continue                      # unverifiable; validate will flag
        exp_wt = ("Welded" if _checked(row[iw]) else
                  "Threaded" if _checked(row[it]) else None)
        if _squash(e.get("welded_or_threaded") or "") != _squash(exp_wt or ""):
            notes.append(f"casing {e.get('diameter')} {e.get('from_ft')}-"
                         f"{e.get('to_ft')}: welded_or_threaded "
                         f"{e.get('welded_or_threaded')!r} -> {exp_wt!r} (md)")
            e["welded_or_threaded"] = exp_wt
        if ic is not None and il is not None:
            exp_cl = ("Casing" if _checked(row[ic]) else
                      "Liner" if _checked(row[il]) else None)
            if _squash(e.get("casing_or_liner") or "") != _squash(exp_cl or ""):
                notes.append(f"casing {e.get('diameter')} {e.get('from_ft')}-"
                             f"{e.get('to_ft')}: casing_or_liner "
                             f"{e.get('casing_or_liner')!r} -> {exp_cl!r} (md)")
                e["casing_or_liner"] = exp_cl


def _repair_litho_water_md(d, tables, notes):
    entries = d.get("lithologic_log") or []
    table = None
    for t in tables:
        iy = _find_col(t["headers"], "y", "watery")
        in_ = _find_col(t["headers"], "n", "watern")
        if iy is not None and in_ is not None:
            table = t
            break
    if table is None or not entries:
        return
    h = table["headers"]
    iy, in_ = _find_col(h, "y", "watery"), _find_col(h, "n", "watern")
    ifrom, ito = _find_col(h, "from", "fromft"), _find_col(h, "to", "toft")
    for e in entries:
        row = _match_row(table["rows"],
                         [i for i in (ifrom, ito) if i is not None],
                         [e.get("from_ft"), e.get("to_ft")])
        if row is None or len(row) != len(h):
            continue
        exp = ("Y" if _checked(row[iy]) else
               "N" if _checked(row[in_]) else None)
        if (e.get("water") or None) != exp:
            notes.append(f"litho {e.get('from_ft')}-{e.get('to_ft')}: "
                         f"water {e.get('water')!r} -> {exp!r} (md)")
            e["water"] = exp


def _repair_litho_water_cv(d, img_path, notes):
    entries = d.get("lithologic_log") or []
    if not entries or not img_path or not os.path.exists(img_path):
        return
    try:
        cv = detect_water_column(img_path)
    except Exception as exc:
        notes.append(f"cv: detector failed ({exc})")
        return
    if not cv:
        notes.append("cv: water table not found in image")
        return
    llm = [e.get("water") or None for e in entries]
    # align by the offset that best agrees with the LLM's column
    best_off, best_agree = 0, -1
    for off in (-2, -1, 0, 1, 2):
        cand = (cv[off:] if off >= 0 else [None] * (-off) + cv)
        cand = (cand + [None] * len(llm))[:len(llm)]
        agree = sum(1 for a, b in zip(cand, llm)
                    if a is not None and b is not None and a == b)
        if agree > best_agree:
            best_agree, best_off = agree, off
    cand = (cv[best_off:] if best_off >= 0 else [None] * (-best_off) + cv)
    cand = (cand + [None] * len(llm))[:len(llm)]
    for k, (c, e) in enumerate(zip(cand, entries)):
        got = e.get("water") or None
        if c is None or c == got:
            continue
        tag = (f"litho {e.get('from_ft')}-{e.get('to_ft')}: water "
               f"{got!r} vs CV {c!r}")
        # CV arbitrates Y vs N only - mark POSITION is its home turf, and on
        # the gold set unconditional Y<->N override never lost a row. But a
        # blank in the transcription stays blank: CV "marks" on blank rows
        # are usually row-alignment noise, so those are flagged instead.
        if (CV_POLICY == "override"
                and got in ("Y", "N") and c in ("Y", "N")):
            e["water"] = c
            notes.append(tag + " -> CV wins")
        else:
            notes.append(tag + " (flagged)")


def _repair_test_type(d, md_norm, tables, notes):
    tests = d.get("well_tests") or []
    if not tests:
        return
    checked = set()
    for m in re.finditer(r"\[x\]\s*([a-z][a-z ]{2,20})", md_norm):
        checked.add(_squash(m.group(1)))
    for t in tables:
        all_rows = [t["headers"]] + t["rows"]
        for i, r in enumerate(all_rows):
            sq = [_squash(c) for c in r]
            if "pump" in sq and "air" in sq:
                for rr in all_rows[i + 1:]:
                    for j, cell in enumerate(rr):
                        if j < len(sq) and _checked(cell) and sq[j]:
                            checked.add(sq[j])
    methods = {"pump": "Pump", "bailer": "Bailer", "bailor": "Bailer",
               "bailier": "Bailer", "air": "Air", "airlift": "Air Lift",
               "flowingartesian": "Flowing Artesian"}
    hits = [methods[c] for c in checked if c in methods]
    if len(set(hits)) == 1:
        want = hits[0]
        for te in tests:
            got = te.get("type")
            if _squash(got or "") not in (_squash(want), ):
                notes.append(f"well_test type {got!r} -> {want!r} (md)")
                te["type"] = want


def _repair_directions(d, md_norm, notes):
    loc = d.get("location") or {}
    full = {"n": "north", "s": "south", "e": "east", "w": "west"}

    def is_checked(word):
        return bool(re.search(word + r"\s*\[x\]", md_norm)
                    or re.search(r"\[x\]\s*" + word, md_norm))

    for field, (a, b) in (("twp_dir", ("north", "south")),
                          ("rge_dir", ("east", "west"))):
        v = loc.get(field)
        if not v:
            continue
        word = full.get(_norm(v)[0], _norm(v))
        if word not in (a, b):
            continue
        opposite = b if word == a else a
        if not is_checked(word) and is_checked(opposite):
            fixed = opposite.capitalize()[0]
            notes.append(f"location.{field} {v!r} -> {fixed!r} (md checkbox)")
            loc[field] = fixed


def _repair_gov_lot(d, notes):
    loc = d.get("location") or {}
    gv = loc.get("gov_lot")
    if gv is not None and ("acre" in _norm(gv)
                          or _squash(gv) in ("10", "40", "160", "14", "1414",
                                             "141414", "104160")):
        notes.append(f"location.gov_lot {gv!r} -> None (label artifact)")
        loc["gov_lot"] = None


def _repair_screens(d, tables, notes):
    ps = d.setdefault("perforations_screens", {})
    entries = ps.setdefault("entries", [])
    md_rows = []
    for t in tables:
        isl = _find_col(t["headers"], "slotsize", "slot")
        if isl is None:
            continue
        h = t["headers"]
        cols = {"from_ft": _find_col(h, "from", "fromft"),
                "to_ft": _find_col(h, "to", "toft"),
                "slot_size": isl,
                "diameter": _find_col(h, "diameter", "diameternominal"),
                "material": _find_col(h, "material")}
        for r in t["rows"]:
            real = [c for c in r if c and not set(_norm(c)) <= set("[ x](?)")]
            if len(real) >= 2:
                md_rows.append({k: (r[i] if i is not None and i < len(r)
                                    and r[i] else None)
                                for k, i in cols.items()})
    if len(md_rows) > len(entries):
        have = {(_squash(e.get("from_ft")), _squash(e.get("to_ft")))
                for e in entries}
        for row in md_rows:
            key = (_squash(row.get("from_ft")), _squash(row.get("to_ft")))
            if key not in have:
                entries.append(row)
                notes.append(f"screens: restored dropped row "
                             f"{row.get('from_ft')}-{row.get('to_ft')} (md)")


def _flag_interval_sanity(d, notes):
    """Physics checks on the lithologic log: intervals should be contiguous,
    non-zero-length, and end near the completed depth. Flag only (a wrong
    digit needs a human or a re-zoom, not a guess)."""
    entries = d.get("lithologic_log") or []

    def num(v):
        m = re.search(r"-?\d+\.?\d*", str(v or ""))
        return float(m.group()) if m else None

    prev_to = None
    for e in entries:
        f_, t_ = num(e.get("from_ft")), num(e.get("to_ft"))
        if f_ is None or t_ is None:
            continue
        if t_ <= f_:
            notes.append(f"litho sanity: interval {e.get('from_ft')}-"
                         f"{e.get('to_ft')} is zero/negative length (flagged)")
        if prev_to is not None and f_ != prev_to:
            notes.append(f"litho sanity: gap/overlap at {prev_to:g} -> "
                         f"{e.get('from_ft')} (flagged)")
        prev_to = t_
    comp = num(d.get("completed_depth_ft"))
    if comp is not None and prev_to is not None and abs(comp - prev_to) > 0.1 \
            and comp > prev_to:
        notes.append(f"litho sanity: last interval ends at {prev_to:g} but "
                     f"completed depth is {comp:g} (flagged)")


def repair_page(json_text, markdown, img_path):
    """Return (final_json_text, notes). Never raises on bad input."""
    try:
        d = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return json_text, ["invalid JSON - repair skipped"]
    md = markdown or ""
    md_norm = _norm(md)
    tables = parse_md_tables(md)
    notes = []
    _repair_casing(d, tables, notes)
    _repair_litho_water_md(d, tables, notes)
    _repair_test_type(d, md_norm, tables, notes)
    _repair_directions(d, md_norm, notes)
    _repair_gov_lot(d, notes)
    _repair_screens(d, tables, notes)
    _flag_interval_sanity(d, notes)
    if CV_POLICY != "off":
        if HAVE_CV:
            _repair_litho_water_cv(d, img_path, notes)
        else:
            notes.append("cv: cv_watercol/cv2 not available - skipped")
    return json.dumps(d), notes


def run_repair():
    rows = db_select("""
        SELECT n.page_name, n.json_text,
               COALESCE(NULLIF(o.markdown_zoom, ''), o.markdown)
        FROM normalized n JOIN ocr_pages o ON o.page_name = n.page_name
        WHERE n.valid_json = 1
    """)
    print(f"\n=== STAGE 2b: REPAIR ===\n"
          f"pages     : {len(rows)}\n"
          f"cv policy : {CV_POLICY}"
          f"{'' if HAVE_CV or CV_POLICY == 'off' else '  (cv2 NOT available)'}\n")
    changed = 0
    for page, json_text, md in rows:
        img_path = os.path.join(IMG_DIR, page)
        final, notes = repair_page(json_text, md, img_path)
        db_write("""
            UPDATE normalized SET json_final=?, repair_notes=?
            WHERE page_name=?
        """, (final, json.dumps(notes), page))
        if notes:
            changed += 1
            print(f"[{len(notes)} notes] {page}")
            for nt in notes:
                print(f"    - {nt}")
    print(f"\nrepair: {changed}/{len(rows)} pages touched")
# Catches normalizer regressions only: an error present in BOTH the markdown
# and the JSON (i.e. an OCR-stage mistake) is invisible here by design.
# Every check below corresponds to a real error class found when auditing
# the TV4 output against the source images.
# =========================================================

def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _squash(s):
    """Aggressive normalization for cell matching: keep [a-z0-9] only."""
    return re.sub(r"[^a-z0-9]", "", _norm(s))


def _checked(cell):
    c = _norm(cell)
    return "[x]" in c or c in ("x", "xx", "xxx")


def parse_md_tables(md):
    """Parse pipe-syntax markdown tables -> [{'headers': [...], 'rows': [[...]]}]."""
    tables, block = [], []
    for line in md.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            block.append(line)
            continue
        if len(block) >= 2:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")]
                    for r in block]
            data = [r for r in rows[1:]
                    if not all(set(c) <= set(":- ") for c in r)]
            tables.append({"headers": rows[0], "rows": data})
        block = []
    return tables


def _find_col(headers, *names):
    sq = [_squash(h) for h in headers]
    for n in names:
        if n in sq:
            return sq.index(n)
    return None


def _match_row(rows, key_cols, key_vals):
    """Find the table row whose cells at key_cols squash-match key_vals."""
    targets = [_squash(v) for v in key_vals]
    for r in rows:
        if all(i < len(r) and _squash(r[i]) == t
               for i, t in zip(key_cols, targets) if t):
            return r
    return None


def check_casing(d, tables):
    flags = []
    table = next((t for t in tables
                  if _find_col(t["headers"], "welded") is not None
                  and _find_col(t["headers"], "threaded") is not None), None)
    entries = d.get("casing_liner") or []
    if not entries:
        return flags
    if table is None:
        return ["casing: no welded/threaded table in markdown to verify against"]
    h = table["headers"]
    iw, it = _find_col(h, "welded"), _find_col(h, "threaded")
    ic, il = _find_col(h, "casing"), _find_col(h, "liner")
    idia, ifrom, ito = (_find_col(h, "diameter", "diameternominal"),
                        _find_col(h, "from", "fromft"), _find_col(h, "to", "toft"))
    ncols = len(h)
    for e in entries:
        row = _match_row(table["rows"],
                         [i for i in (idia, ifrom, ito) if i is not None],
                         [e.get("diameter"), e.get("from_ft"), e.get("to_ft")])
        if row is None:
            flags.append(f"casing: row {e.get('diameter')}/{e.get('from_ft')}"
                         f"-{e.get('to_ft')} not found in markdown table "
                         f"(possible column shift)")
            continue
        if len(row) != ncols:
            # merged or split cell in the md row: positions are unreliable,
            # so don't compare - just ask a human to look
            flags.append(f"casing {e.get('diameter')} {e.get('from_ft')}-"
                         f"{e.get('to_ft')}: markdown row has {len(row)} "
                         f"cells vs {ncols} headers (merged cell?) - "
                         f"checkbox columns not verifiable")
            continue
        exp_wt = ("Welded" if iw < len(row) and _checked(row[iw]) else
                  "Threaded" if it < len(row) and _checked(row[it]) else None)
        got_wt = e.get("welded_or_threaded")
        if _squash(got_wt or "") != _squash(exp_wt or ""):
            flags.append(f"casing {e.get('diameter')} {e.get('from_ft')}-"
                         f"{e.get('to_ft')}: welded_or_threaded='{got_wt}' "
                         f"but markdown marks '{exp_wt}'")
        if ic is not None and il is not None:
            exp_cl = ("Casing" if ic < len(row) and _checked(row[ic]) else
                      "Liner" if il < len(row) and _checked(row[il]) else None)
            got_cl = e.get("casing_or_liner")
            if _squash(got_cl or "") != _squash(exp_cl or ""):
                flags.append(f"casing {e.get('diameter')} {e.get('from_ft')}-"
                             f"{e.get('to_ft')}: casing_or_liner='{got_cl}' "
                             f"but markdown marks '{exp_cl}'")
    return flags


def check_litho_water(d, tables):
    flags = []
    entries = d.get("lithologic_log") or []
    if not entries:
        return flags
    table = None
    for t in tables:
        iy = _find_col(t["headers"], "y", "watery")
        in_ = _find_col(t["headers"], "n", "watern")
        if iy is not None and in_ is not None:
            table = t
            break
    if table is None:
        return ["lithology: no Y/N water table in markdown to verify against"]
    h = table["headers"]
    iy, in_ = _find_col(h, "y", "watery"), _find_col(h, "n", "watern")
    ifrom, ito = _find_col(h, "from", "fromft"), _find_col(h, "to", "toft")
    for e in entries:
        got = e.get("water")
        if got not in (None, "Y", "N"):
            flags.append(f"lithology {e.get('from_ft')}-{e.get('to_ft')}: "
                         f"water='{got}' is not Y/N/null")
            continue
        row = _match_row(table["rows"],
                         [i for i in (ifrom, ito) if i is not None],
                         [e.get("from_ft"), e.get("to_ft")])
        if row is None:
            flags.append(f"lithology: interval {e.get('from_ft')}-"
                         f"{e.get('to_ft')} not found in markdown table")
            continue
        if len(row) != len(h):
            flags.append(f"lithology {e.get('from_ft')}-{e.get('to_ft')}: "
                         f"markdown row has {len(row)} cells vs {len(h)} "
                         f"headers - water column not verifiable")
            continue
        exp = ("Y" if iy < len(row) and _checked(row[iy]) else
               "N" if in_ < len(row) and _checked(row[in_]) else None)
        if (got or "") != (exp or ""):
            flags.append(f"lithology {e.get('from_ft')}-{e.get('to_ft')} "
                         f"'{e.get('remarks')}': water='{got}' but markdown "
                         f"marks '{exp}'")
    return flags


def check_checked_options(d, md_norm):
    """use / type_of_work / drill_method values must be marked [x] in the md."""
    flags = []
    for field in ("use", "type_of_work", "drill_method"):
        for v in (d.get(field) or []):
            # allow "Other" between the box and the value: "[x] Other MONITOR"
            pat1 = r"\[x\]\s*(?:other\s*)?" + re.escape(_norm(v))
            pat2 = re.escape(_norm(v)) + r"\s*\[x\]"
            if not (re.search(pat1, md_norm) or re.search(pat2, md_norm)):
                flags.append(f"{field}: '{v}' is not marked [x] in markdown")
    return flags


def check_test_types(d, md_norm, tables):
    """Well-test method must correspond to a checked box / checked column."""
    flags = []
    tests = d.get("well_tests") or []
    if not tests:
        return flags
    checked_names = set()
    for m in re.finditer(r"\[x\]\s*([a-z][a-z ]{2,20})", md_norm):
        checked_names.add(_squash(m.group(1)))
    for m in re.finditer(r"([a-z][a-z ]{2,20})\s*\[x\]", md_norm):
        checked_names.add(_squash(m.group(1)))
    # 6/07-style table: one row names the methods, a later row holds the [x]
    for t in tables:
        all_rows = [t["headers"]] + t["rows"]
        for i, r in enumerate(all_rows):
            sq = [_squash(c) for c in r]
            if "pump" in sq and "air" in sq:
                for rr in all_rows[i + 1:]:
                    for j, cell in enumerate(rr):
                        if j < len(sq) and _checked(cell) and sq[j]:
                            checked_names.add(sq[j])
    for te in tests:
        tv = _squash(te.get("type") or "")
        if tv and not any(tv in n or n in tv for n in checked_names):
            shortlist = sorted(n for n in checked_names
                               if n in ("pump", "bailer", "bailor", "bailier",
                                        "air", "airlift", "flowingartesian"))
            flags.append(f"well_tests: type='{te.get('type')}' has no "
                         f"checked box in markdown (checked methods: "
                         f"{shortlist or 'none found'})")
    return flags


def check_screen_rows(d, tables):
    """Every screens/perforations data row in the md must reach the JSON."""
    md_count = 0
    for t in tables:
        if _find_col(t["headers"], "slotsize", "slot") is None:
            continue
        for r in t["rows"]:
            real = [c for c in r if c and not set(_norm(c)) <= set("[ x](?)")]
            if len(real) >= 2:
                md_count += 1
    js_count = len((d.get("perforations_screens") or {}).get("entries") or [])
    if md_count > js_count:
        return [f"screens/perforations: markdown has {md_count} data row(s) "
                f"but JSON has {js_count}"]
    return []


def check_label_artifacts(d):
    flags = []
    loc = d.get("location") or {}
    gv = loc.get("gov_lot")
    if gv is not None:
        s = _squash(gv)
        if "acre" in _norm(gv) or s in ("10", "40", "160", "14", "1414",
                                        "141414", "104160"):
            flags.append(f"location.gov_lot: '{gv}' looks like form label "
                         f"text (acreage note), not a value")
    return flags


def check_directions(d, md_norm):
    """twp_dir / rge_dir must agree with the checked box. Only flag when the
    OPPOSITE direction is the one checked (typed-in values without checkboxes
    are left alone)."""
    flags = []
    pairs = {"twp_dir": ("north", "south"), "rge_dir": ("east", "west")}
    loc = d.get("location") or {}
    full = {"n": "north", "s": "south", "e": "east", "w": "west"}

    def is_checked(word):
        return bool(re.search(word + r"\s*\[x\]", md_norm)
                    or re.search(r"\[x\]\s*" + word, md_norm))

    for field, (a, b) in pairs.items():
        v = loc.get(field)
        if not v:
            continue
        word = full.get(_norm(v)[0], _norm(v))
        if word not in (a, b):
            continue
        opposite = b if word == a else a
        if not is_checked(word) and is_checked(opposite):
            flags.append(f"location.{field}: '{v}' but markdown checks "
                         f"'{opposite}'")
    return flags


# fields whose value should literally occur somewhere in the transcription
GROUNDED_FIELDS = [
    ("well_tag_no",), ("drilling_permit_no",), ("completed_depth_ft",),
    ("static_water_level_ft",), ("date_started",), ("date_completed",),
    ("owner", "name"), ("owner", "zip"),
    ("driller", "company_name"), ("driller", "firm_no"),
    ("driller", "principal_driller"),
]


def check_grounding(d, md):
    flags = []
    md_sq = _squash(md)
    for path in GROUNDED_FIELDS:
        v = d
        for k in path:
            v = (v or {}).get(k) if isinstance(v, dict) else None
        if v is None:
            continue
        sq = _squash(v)
        if len(sq) >= 3 and sq not in md_sq:
            flags.append(f"{'.'.join(path)}: '{v}' does not occur in the "
                         f"markdown transcription")
    return flags


def validate_page(json_text, markdown):
    """Return a list of flag strings for one page."""
    try:
        d = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return ["invalid JSON"]
    md = markdown or ""
    md_norm = _norm(md)
    tables = parse_md_tables(md)
    if not isinstance(d, dict):
        return ["JSON is not an object"]
    flags = []
    # a malformed record (e.g. a lithology row stored as a plain string) must
    # not stop the whole run: record which check could not run and move on
    for check, args in ((check_casing, (d, tables)), (check_litho_water, (d, tables)),
                        (check_checked_options, (d, md_norm)), (check_test_types, (d, md_norm, tables)),
                        (check_screen_rows, (d, tables)), (check_label_artifacts, (d,)),
                        (check_directions, (d, md_norm)), (check_grounding, (d, md))):
        try:
            flags += check(*args)
        except Exception as e:
            flags.append(f"validator: {check.__name__} could not run on this record "
                         f"({type(e).__name__}: {e})")
    return flags


def run_validate():
    rows = db_select("""
        SELECT n.page_name, n.json_text,
               COALESCE(NULLIF(o.markdown_zoom, ''), o.markdown)
        FROM normalized n JOIN ocr_pages o ON o.page_name = n.page_name
    """)
    print(f"\n=== STAGE 3: VALIDATE ===\npages : {len(rows)}\n")
    flagged = 0
    total_flags = 0
    for page, json_text, markdown in rows:
        flags = validate_page(json_text, markdown)
        db_write("UPDATE normalized SET flags=? WHERE page_name=?",
                 (json.dumps(flags), page))
        if flags:
            flagged += 1
            total_flags += len(flags)
            print(f"[FLAGS {len(flags)}] {page}")
            for f in flags:
                print(f"    - {f}")
    print(f"\nvalidate: {flagged}/{len(rows)} pages flagged, "
          f"{total_flags} flags total")


def run_audit():
    rows = db_select(
        "SELECT page_name, flags FROM normalized ORDER BY page_name")
    if not rows:
        print("no normalized rows")
        return
    for page, flags in rows:
        fl = json.loads(flags) if flags else None
        if fl is None:
            print(f"{page}: (not validated yet - run 'validate')")
        elif not fl:
            print(f"{page}: clean")
        else:
            print(f"{page}:")
            for f in fl:
                print(f"    - {f}")


# =========================================================
# STAGE 4 (production utilities, offline)
# "export" - review queue CSV of every validator flag and repair note.
# "merge"  - combine shard databases from an HPC array run into one.
# =========================================================

def run_export():
    rows = db_select("""
        SELECT page_name, flags, repair_notes FROM normalized
        ORDER BY page_name
    """)
    n = 0
    with open(REVIEW_CSV, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["page_name", "source", "note", "reviewed", "correction"])
        for page, flags, notes in rows:
            for src, blob in (("validate", flags), ("repair", notes)):
                for note in (json.loads(blob) if blob else []):
                    # repair notes that record an applied fix are informational;
                    # everything else needs human eyes
                    wcsv.writerow([page, src, note, "", ""])
                    n += 1
    print(f"export: {n} notes/flags for {len(rows)} pages -> {REVIEW_CSV}")


def run_merge(out_path, shard_paths):
    if os.path.exists(out_path):
        print(f"refusing to overwrite existing {out_path}")
        sys.exit(1)
    out = sqlite3.connect(out_path)
    src0 = sqlite3.connect(shard_paths[0])
    for (sql,) in src0.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name IN ('ocr_pages','normalized')"):
        out.execute(sql)
    src0.close()
    total = {"ocr_pages": 0, "normalized": 0}
    for sp in shard_paths:
        out.execute("ATTACH ? AS shard", (sp,))
        for table in ("ocr_pages", "normalized"):
            cols = [r[1] for r in out.execute(f"PRAGMA table_info({table})")
                    if r[1] != "id"]
            collist = ", ".join(cols)
            cur = out.execute(
                f"INSERT OR IGNORE INTO {table} ({collist}) "
                f"SELECT {collist} FROM shard.{table}")
            total[table] += cur.rowcount
        out.commit()
        out.execute("DETACH shard")
        print(f"merged {sp}")
    out.close()
    print(f"merge -> {out_path}: {total['ocr_pages']} pages, "
          f"{total['normalized']} normalized records")


# =========================================================
# main
# =========================================================

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode not in ("ocr", "zoom", "normalize", "repair", "validate",
                    "audit", "export", "merge", "both"):
        print(__doc__)
        sys.exit(1)

    if mode == "merge":
        if len(sys.argv) < 4:
            print("usage: OCR_main8.py merge OUT.sqlite SHARD1.sqlite ...")
            sys.exit(1)
        run_merge(sys.argv[2], sys.argv[3:])
        return

    # API-backed stages need a key; the rest run offline
    if mode in ("ocr", "zoom", "normalize", "both") and not API_KEY:
        print("ERROR: MINDROUTER_API_KEY is not set.\n"
              "Set it in the environment or in a .env file "
              "(see .env.template).")
        sys.exit(1)

    sys.stdout = _Tee(LOG_PATH)
    init_db()
    start = time.perf_counter()

    if mode in ("ocr", "both"):
        run_ocr()
    if mode in ("zoom", "both"):
        run_zoom()
    if mode in ("normalize", "both"):
        run_normalize()
    if mode in ("repair", "both"):
        run_repair()
    if mode in ("validate", "both"):
        run_validate()
    if mode == "audit":
        run_audit()
        return
    if mode == "export":
        run_export()
        return

    elapsed = int(time.perf_counter() - start)
    print(f"\nDONE: {elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}")

    # quick quality summary
    for label, sql in [
        ("OCR pages stored", "SELECT COUNT(*) FROM ocr_pages"),
        ("  with short output", "SELECT COUNT(*) FROM ocr_pages WHERE status='short'"),
        ("  with errors (will retry next run)",
         "SELECT COUNT(*) FROM ocr_pages WHERE status='error'"),
        ("  zoom-refined", "SELECT COUNT(*) FROM ocr_pages WHERE zoom_status='ok'"),
        ("  zoom suspect (draft kept)",
         "SELECT COUNT(*) FROM ocr_pages WHERE zoom_status='suspect_short'"),
        ("  zoom errors (will retry next run)",
         "SELECT COUNT(*) FROM ocr_pages WHERE zoom_status='error'"),
        ("Normalized records", "SELECT COUNT(*) FROM normalized"),
        ("  with invalid JSON", "SELECT COUNT(*) FROM normalized WHERE valid_json=0"),
        ("  with validation flags",
         "SELECT COUNT(*) FROM normalized WHERE flags IS NOT NULL AND flags != '[]'"),
    ]:
        print(f"{label}: {db_select(sql)[0][0]}")


if __name__ == "__main__":
    main()
