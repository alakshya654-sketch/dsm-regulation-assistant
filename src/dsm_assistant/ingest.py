"""
Ingestion: turn the registry's PDFs into clause-level chunks with metadata.

Implements Design §4.2 (PDF parser) and §4.3 (clause-aware chunker), plus data-report
decision DR-8 (two-column tables extracted column by column, every item labelled with
its column).

Pipeline for one document:

    PDF ──► pages of lines ──► clean (headers/footers, Hindi) ──► sections ──► chunks
              │
              └── two-column tables are rendered as labelled lines at this stage

Public entry points:
    load_registry(path)              -> list of registry entries (dicts)
    chunk_document(entry, pdf_path)  -> list[Chunk]
    chunk_corpus(registry, raw_dir)  -> list[Chunk]
    write_jsonl(chunks, path) / read_jsonl(path)
"""
from __future__ import annotations

# Bumped whenever this file changes; notebook 04a checks it so an old copy can't run silently.
VERSION = "4a.2"

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import pymupdf
import yaml

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEFAULTS = {
    "target_tokens": 450,
    "max_tokens": 700,
    "table_max_tokens": 1200,
    "min_tokens": 60,
    "header_footer_share": 0.4,
}


def load_settings(config_path: Path | None = None) -> dict:
    """Chunking settings from config.yaml, falling back to DEFAULTS."""
    settings = dict(DEFAULTS)
    if config_path and Path(config_path).exists():
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        settings.update(cfg.get("chunking", {}))
    return settings


def count_tokens(text: str) -> int:
    """Approximate token count (English ≈ 1.33 tokens per word). Good enough for sizing."""
    return int(len(text.split()) * 1.33) + 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def _to_date(value) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def load_registry(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        docs = yaml.safe_load(f)["documents"]
    for d in docs:
        for k in ("notified_date", "effective_from", "effective_to"):
            d[k] = _to_date(d.get(k))
    return docs


def short_title(title: str) -> str:
    """'CERC (Deviation Settlement Mechanism and Related Matters) (First Amendment) Regulations, 2024'
    -> 'DSM (First Amendment) Regulations, 2024'. Used in breadcrumbs."""
    t = re.sub(r"^CERC\s*", "", title)
    t = t.replace("(Deviation Settlement Mechanism and Related Matters)", "DSM")
    t = t.replace("Deviation Settlement Mechanism", "DSM")
    t = re.sub(r"^DSM\s+Regulations", "DSM Regulations", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# Chunk record
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str                    # the chunk as it appears in the document
    breadcrumb: str              # e.g. "DSM Regulations, 2024 > Regulation 8 (Charges for Deviation) > Clause (4)"
    regulation_no: str           # "8", "4A" or "" when not tied to one regulation
    regulation_title: str
    clause_no: str               # "(4)", "(1)(a)-(f)", "2.1" or ""
    page_start: int
    page_end: int
    is_table: bool
    tokens: int
    meta: dict = field(default_factory=dict)   # registry fields, Chroma-safe types only

    @property
    def embed_text(self) -> str:
        """What gets embedded: breadcrumb first, so every chunk carries its context."""
        return f"{self.breadcrumb}\n{self.text}"


def registry_meta(entry: dict) -> dict:
    """Registry fields as metadata. Vector stores accept only str/int/float/bool, never None,
    so dates are stored twice: ISO text for display, and an int (YYYYMMDD) for range filters.
    A missing effective_from becomes 0 and a missing effective_to becomes 99991231 ("open")."""
    def as_int(d, default):
        return int(d.strftime("%Y%m%d")) if d else default

    return {
        "doc_id": entry["doc_id"],
        "title": entry["title"],
        "short_title": short_title(entry["title"]),
        "doc_type": entry["doc_type"],
        "status": entry["status"],
        "regime": entry.get("regime") or "",
        "tier": entry.get("tier") or "",
        "amends": entry.get("amends") or "",
        "source_url": entry.get("url") or "",
        "effective_from": entry["effective_from"].isoformat() if entry.get("effective_from") else "",
        "effective_to": entry["effective_to"].isoformat() if entry.get("effective_to") else "",
        "effective_from_int": as_int(entry.get("effective_from"), 0),
        "effective_to_int": as_int(entry.get("effective_to"), 99991231),
    }


# ---------------------------------------------------------------------------
# Step 1 — PDF to lines (with two-column tables rendered as labelled lines)
# ---------------------------------------------------------------------------
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Column headings of the DSM charge tables. A page containing a left AND a right heading
# from one pair is treated as a two-column page (DR-8).
COLUMN_PAIRS = [
    (r"over[\s-]*injection", r"under[\s-]*injection"),
    (r"over[\s-]*drawal", r"under[\s-]*drawal"),
]


@dataclass
class Line:
    page: int
    text: str
    is_table: bool = False


def _clean_cell(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _column_label(header: str) -> str:
    """'Deviation by way of over injection (Receivable by the Seller)'
    -> 'Over injection (receivable by the seller)'. Short but unambiguous."""
    h = _clean_cell(header)
    m = re.search(r"(over|under)[\s-]*(injection|drawal)", h, re.I)
    paren = re.search(r"\(([^)]*)\)", h)
    if m:
        label = f"{m.group(1).capitalize()} {m.group(2).lower()}"
        return f"{label} ({paren.group(1).lower()})" if paren else label
    return h[:80]


ROW_LABEL = re.compile(r"^\((I|II|III|IV|V|VI|VII|VIII)\)\s|^Note\b")


def _render_rows(column_labels: list[str], rows: list[list[str]]) -> list[str]:
    """Render table rows as text lines. Each cell line starts with its column label, so the
    meaning survives even if a chunk boundary falls inside the table (DR-8)."""
    out = ["[TABLE] Columns: " + " | ".join(column_labels)]
    for row in rows:
        cells = [_clean_cell(c) for c in row]
        filled = [c for c in cells if c]
        if not filled:
            continue
        # A row label spans both columns: the table finder reports the merged cells as None,
        # and the word-based fallback passes None explicitly. An empty string ("") is a real,
        # empty cell, so a row with text only in the left column is NOT a row label.
        if len(filled) == 1 and (all(c is None for c in row[1:]) or ROW_LABEL.match(filled[0])):
            if filled[0].startswith("Note"):
                out.append(filled[0])                 # a note under the table belongs to no column
            else:
                out.append(f"[ROW] {filled[0]}")      # e.g. "(I) For Deviation up to ..."
            continue
        for label, cell in zip(column_labels, cells):
            for item in _split_items(cell) if cell else []:
                out.append(item if item.startswith("Note") else f"[{label}] {item}")
    out.append("[/TABLE]")
    return out


def _is_two_column_page(text: str) -> bool:
    return any(re.search(l, text, re.I) and re.search(r, text, re.I) for l, r in COLUMN_PAIRS)


def _collapse_columns(header: list[str]) -> list[int]:
    """Real tables can have extra narrow columns with no heading (e.g. 4 grid columns for 2
    logical ones). Map every column to the nearest headed column on its left (or right)."""
    headed = [i for i, h in enumerate(header) if h]
    owner = []
    for i in range(len(header)):
        left = [j for j in headed if j <= i]
        owner.append(left[-1] if left else (headed[0] if headed else i))
    return owner


def _tables_from_grid(page) -> list[tuple]:
    """Ruled tables via PyMuPDF's table finder. Returns [(bbox, lines)].

    Cell text is read with page.get_textbox(cell) rather than the table finder's own
    extract(), because extract() moves subscripts to the end of the line ("DGS" ->
    "D ... GS"), while the page's normal reading order keeps them in place."""
    try:
        found = page.find_tables()
    except Exception:
        return []
    results = []
    for tab in found.tables:
        if tab.col_count < 2 or not tab.rows:
            continue
        grid = [[page.get_textbox(pymupdf.Rect(c)) if c else None for c in row.cells] for row in tab.rows]
        header = [_clean_cell(c) for c in grid[0]]
        if not _is_two_column_page(" ".join(header)):
            continue                                   # not a DSM charge table; leave as normal text
        owner = _collapse_columns(header)
        logical = sorted(set(owner))
        labels = [_column_label(header[i]) for i in logical]
        rows = []
        for row_obj, r in zip(tab.rows[1:], grid[1:]):
            # A note under the table may be cut into pieces by narrow grid columns.
            # Read such a row across its whole width instead.
            full = _clean_cell(page.get_textbox(pymupdf.Rect(row_obj.bbox)))
            if full.startswith("Note"):
                rows.append([full] + [None] * (len(logical) - 1))
                continue
            merged = []
            for col in logical:
                parts = [r[i] for i in range(len(r)) if owner[i] == col]
                texts = [_clean_cell(t) for t in parts if t]
                merged.append(" ".join(t for t in texts if t) if any(t is not None for t in parts) else None)
            rows.append(merged)
        results.append((pymupdf.Rect(tab.bbox), _render_rows(labels, rows)))
    return results


ROMAN_ITEM = re.compile(r"\((i|ii|iii|iv|v|vi|vii|viii|ix|x)\)\s")
REFERENCE_WORDS = {"clause", "sub-clause", "item", "items", "para", "paragraph", "of", "under", "in"}


def _split_items(cell: str) -> list[str]:
    """'(i) @ zero when ... (ii) @ 115 % of RR ...' -> ['(i) @ zero when ...', '(ii) @ 115 % ...'].
    Not split when the marker is a reference, e.g. 'clause (ii) of'."""
    cuts = []
    for m in ROMAN_ITEM.finditer(cell):
        if m.start() == 0:
            continue
        before = cell[:m.start()].rstrip().split()
        if before and before[-1].lower().strip(",") in REFERENCE_WORDS:
            continue
        cuts.append(m.start())
    parts, prev = [], 0
    for c in cuts:
        parts.append(cell[prev:c].strip()); prev = c
    parts.append(cell[prev:].strip())
    return [p for p in parts if p]


def _visual_lines(page) -> list[dict]:
    """Group words into visual lines by vertical position. For each line record the text on
    the left half, the right half, or the full width (a line that runs across the middle)."""
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    mid = page.rect.width / 2
    lines, current, cy = [], [], None
    for w in words:
        c = (w[1] + w[3]) / 2
        if current and abs(c - cy) > 3:
            lines.append(current); current = []
        current.append(w); cy = c if len(current) == 1 else cy
    if current:
        lines.append(current)

    out = []
    for ws in lines:
        ws.sort(key=lambda w: w[0])
        crosses = any(w[0] < mid < w[2] for w in ws) or any(
            a[2] < mid < b[0] and b[0] - a[2] < 8 for a, b in zip(ws, ws[1:]))
        text = lambda group: " ".join(w[4] for w in group)
        if crosses:
            out.append({"y0": min(w[1] for w in ws), "y1": max(w[3] for w in ws),
                        "span": text(ws), "left": "", "right": ""})
        else:
            out.append({"y0": min(w[1] for w in ws), "y1": max(w[3] for w in ws), "span": "",
                        "left": text([w for w in ws if (w[0] + w[2]) / 2 < mid]),
                        "right": text([w for w in ws if (w[0] + w[2]) / 2 >= mid])})
    return out


def _table_from_words(page) -> list[tuple]:
    """Fallback for tables without ruling lines. Works on words, not blocks, because PyMuPDF
    may merge two short cells on the same line into one block that spans both columns."""
    vlines = _visual_lines(page)
    # 1. Find the header: a line whose left part matches an 'over ...' heading and whose
    #    right part matches the paired 'under ...' heading.
    start = None
    for i, vl in enumerate(vlines):
        for lp, rp in COLUMN_PAIRS:
            if re.search(lp, vl["left"], re.I) and re.search(rp, vl["right"], re.I):
                start = i; break
        if start is not None:
            break
    if start is None:
        return []

    # 2. Header continues on following lines until a row label or the first item "(i)".
    head_l, head_r, i = [], [], start
    while i < len(vlines):
        vl = vlines[i]
        if vl["span"] or re.match(r"\(", vl["left"]) or re.match(r"\(", vl["right"]):
            break
        head_l.append(vl["left"]); head_r.append(vl["right"]); i += 1
    labels = [_column_label(" ".join(head_l)), _column_label(" ".join(head_r))]

    # 3. Body: full-width lines are row labels "(I) ...", "(II) ..." (and their wrapped
    #    continuation). Any other full-width line ends the table (e.g. "Note: ...", "(2) ...").
    rows, cell_l, cell_r = [], [], []
    last_was_label, last_y1 = False, None
    top, bottom = vlines[start]["y0"], vlines[start]["y1"]

    def flush():
        if cell_l or cell_r:
            rows.append([" ".join(cell_l), " ".join(cell_r)])
            cell_l.clear(); cell_r.clear()

    while i < len(vlines):
        vl = vlines[i]
        if vl["span"]:
            is_label = re.match(r"\((I|II|III|IV|V|VI|VII|VIII)\)\s", vl["span"])
            is_wrap = (last_was_label and last_y1 is not None and vl["y0"] - last_y1 < 6
                       and not re.match(r"(\(|Note)", vl["span"]))
            if is_label:
                flush(); rows.append([vl["span"], None])
            elif is_wrap:
                rows[-1][0] += " " + vl["span"]
            else:
                break
            last_was_label = True
        else:
            if vl["left"]:
                cell_l.append(vl["left"])
            if vl["right"]:
                cell_r.append(vl["right"])
            last_was_label = False
        last_y1 = vl["y1"]; bottom = vl["y1"]; i += 1
    flush()
    if not rows:
        return []
    bbox = pymupdf.Rect(0, top - 1, page.rect.width, bottom + 1)
    return [(bbox, _render_rows(labels, rows))]


def page_lines(page, page_no: int) -> list[Line]:
    """Text of one page as lines. Two-column charge tables are replaced by labelled lines."""
    raw = page.get_text()
    if not _is_two_column_page(raw):
        return [Line(page_no, l) for l in raw.splitlines()]

    tables = _tables_from_grid(page) or _table_from_words(page)
    if not tables:
        return [Line(page_no, l) for l in raw.splitlines()]

    # Keep text blocks outside the table areas, and insert each table at its vertical position.
    items = []
    for b in page.get_text("blocks"):
        if b[6] != 0:
            continue
        r = pymupdf.Rect(b[:4])
        if any(r.intersects(bbox) and (r & bbox).get_area() > 0.5 * r.get_area() for bbox, _ in tables):
            continue
        items.append((b[1], [Line(page_no, l) for l in b[4].splitlines()]))
    for bbox, lines in tables:
        items.append((bbox.y0, [Line(page_no, l, is_table=True) for l in lines]))
    items.sort(key=lambda t: t[0])
    return [line for _, group in items for line in group]


def pdf_lines(pdf_path: Path) -> tuple[list[Line], int]:
    lines, n_pages = [], 0
    with pymupdf.open(pdf_path) as pdf:
        n_pages = len(pdf)
        for i, page in enumerate(pdf):
            lines.extend(page_lines(page, i + 1))
    return lines, n_pages


# ---------------------------------------------------------------------------
# Step 2 — clean
# ---------------------------------------------------------------------------
def _footer_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


# A paragraph or clause number alone on its line: "2."  "1.1"  "(4A)"  "(a)"  "(iv)"
ORPHAN_MARKER = re.compile(r"^\s*(\d{1,2}[A-Z]?\.|\d{1,2}\.\d{1,2}\.?|\((?:\d{1,2}[A-Z]?|[a-z]{1,2}|[ivx]{1,5})\))\s*$")


def join_orphan_markers(lines: list[Line]) -> list[Line]:
    """Some PDFs put the number of a heading or clause on its own line ("2." then
    "Insertion of a new clause ..."). Join them, so headings and clauses are recognised."""
    out: list[Line] = []
    i = 0
    while i < len(lines):
        l = lines[i]
        if not l.is_table and ORPHAN_MARKER.match(l.text):
            j = i + 1
            while j < len(lines) and not lines[j].text.strip():
                j += 1
            if j < len(lines) and not lines[j].is_table and not ORPHAN_MARKER.match(lines[j].text):
                out.append(Line(l.page, f"{l.text.strip()} {lines[j].text.strip()}"))
                i = j + 1
                continue
        out.append(l)
        i += 1
    return out


def clean_lines(lines: list[Line], n_pages: int, share: float = 0.4) -> list[Line]:
    """Remove running headers/footers (a line repeated on many pages, ignoring numbers),
    bare page numbers, Hindi lines, and blank-line runs."""
    # Only the first and last 3 text lines of each page can be a header or footer. Without this
    # limit, body paragraphs that differ only in their numbers would be deleted as "footers".
    by_page: dict[int, list[int]] = {}
    for i, l in enumerate(lines):
        if l.text.strip() and not l.is_table:
            by_page.setdefault(l.page, []).append(i)
    edge_idx = {i for idxs in by_page.values() for i in idxs[:3] + idxs[-3:]}

    repeated = set()
    if n_pages >= 3:
        # count each text at most once per page
        page_keys = {(lines[i].page, _footer_key(lines[i].text)) for i in edge_idx}
        pages_with = Counter(key for _, key in page_keys)
        repeated = {k for k, n in pages_with.items() if n >= max(3, share * n_pages) and len(k) < 160}

    out, blank = [], False
    for i, l in enumerate(lines):
        t = l.text.rstrip()
        s = t.strip()
        if not l.is_table:
            if i in edge_idx and _footer_key(s) in repeated:
                continue
            if re.fullmatch(r"(page\s*)?[-–]?\s*\d{1,3}\s*[-–]?(\s*of\s*\d+)?", s, re.I):
                continue
            if DEVANAGARI.search(s):
                continue
            if s and not re.search(r"[A-Za-z0-9]", s):     # symbol-only noise (e.g. unrenderable Hindi)
                continue
        if not s:
            if not blank and out:
                out.append(Line(l.page, ""))
            blank = True
            continue
        blank = False
        out.append(Line(l.page, re.sub(r"[ \t]+", " ", t), l.is_table))
    return out


# ---------------------------------------------------------------------------
# Step 3 — sections (one per regulation)
# ---------------------------------------------------------------------------
# "8. Charges for Deviation", "1. Short Title and Commencement", "2. Insertion of a new clause ..."
REG_HEADING = re.compile(r"^\s*(\d{1,2}[A-Z]?)\.\s+([A-Z][^\n]{2,140})$")
# Words that start a sentence, not a heading ("2. The Commission ...", "3. In exercise of ...").
NOT_A_HEADING = re.compile(r"^(The|In|It|This|These|Where|When|If|Provided|Whereas|As|A|An)\s", re.I)
CLAUSE_MARK = re.compile(r"^\s*\((\d{1,2}[A-Z]?)\)\s")          # (1) (4A)
AMEND_MARK = re.compile(r"^\s*(\d{1,2}\.\d{1,2})\s")              # 2.1  4.3
SUBCLAUSE_MARK = re.compile(r"^\s*\(([a-z]{1,2}|[ivx]{1,5})\)\s")  # (a) (cc) (iv)
REG_REFERENCE = re.compile(r"Regulation\s*\(?(\d{1,2}[A-Z]?)\)?", re.I)

REGULATION_STYLE = {"principal", "amendment", "draft"}


@dataclass
class Section:
    number: str
    title: str
    lines: list[Line]


def split_sections(lines: list[Line]) -> list[Section]:
    """Split on numbered regulation headings. Numbers must increase (by at most 10), which
    rejects false positives such as numbered list items inside a regulation."""
    sections = [Section("", "Preamble", [])]
    last = 0
    for l in lines:
        m = None if l.is_table else REG_HEADING.match(l.text)
        if m and not NOT_A_HEADING.match(m.group(2)) and not m.group(2).rstrip().endswith((";", ",")):
            num = int(re.match(r"\d+", m.group(1)).group())
            if last < num <= last + 10:      # allow gaps (a missed heading must not swallow the rest)
                last = num
                sections.append(Section(m.group(1), m.group(2).strip().rstrip(":.").strip(), [l]))
                continue
        sections[-1].lines.append(l)
    return [s for s in sections if any(x.text.strip() for x in s.lines)]


# ---------------------------------------------------------------------------
# Step 4 — chunks
# ---------------------------------------------------------------------------
@dataclass
class Unit:
    """An atomic piece of text that should not be split further (a paragraph, a sub-clause,
    one table row with its cells)."""
    lines: list[Line]
    clause: str = ""
    sub: str = ""
    is_table: bool = False
    table_header: str = ""     # repeated at the top of every chunk that continues a table

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


def _split_by(lines: list[Line], marker: re.Pattern) -> list[tuple[str, list[Line]]]:
    """Split lines at lines matching `marker`; returns [(marker_value, lines)]."""
    parts, current, label = [], [], ""
    for l in lines:
        m = None if l.is_table else marker.match(l.text)
        if m and current:
            parts.append((label, current))
            current, label = [], m.group(1)
        elif m:
            label = m.group(1)
        current.append(l)
    if current:
        parts.append((label, current))
    return parts


def _units_from_lines(lines: list[Line], clause: str) -> list[Unit]:
    """Break a clause into units: tables become one unit per row; text splits on
    sub-clause markers (a), (b)… and otherwise on blank lines."""
    units: list[Unit] = []
    buf: list[Line] = []
    header = ""

    def flush_text():
        if not any(l.text.strip() for l in buf):
            buf.clear(); return
        for sub, part in _split_by(buf, SUBCLAUSE_MARK):
            # further split long paragraphs on blank lines
            para: list[Line] = []
            for l in part + [Line(part[-1].page, "")]:
                if l.text.strip():
                    para.append(l)
                elif para:
                    units.append(Unit(list(para), clause, sub))
                    para.clear()
        buf.clear()

    row: list[Line] = []
    for l in lines:
        if l.is_table:
            if l.text.startswith("[TABLE]"):
                flush_text()
                header = l.text
                continue
            if l.text.startswith("[/TABLE]"):
                if row:
                    units.append(Unit(list(row), clause, "", True, header)); row.clear()
                continue
            if l.text.startswith("[ROW]") and row:
                units.append(Unit(list(row), clause, "", True, header)); row.clear()
            row.append(l)
        else:
            buf.append(l)
    if row:
        units.append(Unit(list(row), clause, "", True, header))
    flush_text()
    return [piece for u in units for piece in _split_oversized(u)]


NUMBERED_PARA = re.compile(r"^\s*\d{1,2}(\.\d{1,2})+\s+\S")   # "2.3 The Commission ..."
MAX_UNIT_TOKENS = 300


def _split_oversized(unit: Unit) -> list[Unit]:
    """A text unit can be very long when a PDF has no blank lines between paragraphs.
    Break it at numbered paragraphs ("2.3 ..."), then at line ends that finish a sentence."""
    if unit.is_table or count_tokens(unit.text) <= MAX_UNIT_TOKENS:
        return [unit]
    pieces, cur = [], []
    for l in unit.lines:
        starts_para = bool(NUMBERED_PARA.match(l.text))
        size = count_tokens("\n".join(x.text for x in cur))
        sentence_end = bool(cur) and cur[-1].text.rstrip().endswith((".", ";", ":"))
        if cur and (starts_para or (size >= MAX_UNIT_TOKENS * 0.6 and sentence_end) or size >= MAX_UNIT_TOKENS):
            pieces.append(Unit(cur, unit.clause, unit.sub)); cur = []
        cur.append(l)
    if cur:
        pieces.append(Unit(cur, unit.clause, unit.sub))
    return pieces


def _sub_range(units: list[Unit]) -> str:
    subs = [u.sub for u in units if u.sub]
    if not subs:
        return ""
    return f"({subs[0]})" if subs[0] == subs[-1] else f"({subs[0]})-({subs[-1]})"


def _pack(units: list[Unit], target: int, max_tokens: int, table_max: int) -> list[list[Unit]]:
    """Greedily pack units into groups near `target` tokens without exceeding the limit."""
    groups, cur, size = [], [], 0
    for u in units:
        t = count_tokens(u.text) + (count_tokens(u.table_header) if u.is_table else 0)
        limit = table_max if (u.is_table or any(x.is_table for x in cur)) else max_tokens
        if cur and (size + t > limit or size >= target):
            groups.append(cur); cur, size = [], 0
        cur.append(u); size += t
    if cur:
        groups.append(cur)
    # A tiny leftover at the end (e.g. a one-line "Note:") belongs with the group before it.
    if len(groups) > 1:
        tail = sum(count_tokens(u.text) for u in groups[-1])
        prev = sum(count_tokens(u.text) for u in groups[-2])
        limit = table_max if any(u.is_table for u in groups[-2] + groups[-1]) else max_tokens
        if tail < 60 and prev + tail <= limit:
            groups[-2].extend(groups.pop())
    return groups


def _render_group(units: list[Unit]) -> str:
    """Join units back into text; a group that contains table rows starts with the table's
    column header so the reader knows which column is which."""
    out, header_done = [], set()
    for u in units:
        if u.is_table and u.table_header not in header_done:
            out.append(u.table_header)
            header_done.add(u.table_header)
        out.append(u.text)
    return "\n".join(out).strip()


def _most_referenced_regulation(text: str) -> str:
    refs = REG_REFERENCE.findall(text)
    return Counter(refs).most_common(1)[0][0] if refs else ""


def chunk_document(entry: dict, pdf_path: Path, settings: dict | None = None) -> list[Chunk]:
    s = settings or dict(DEFAULTS)
    lines, n_pages = pdf_lines(pdf_path)
    lines = join_orphan_markers(clean_lines(lines, n_pages, s["header_footer_share"]))
    meta = registry_meta(entry)
    doc_title = meta["short_title"]
    regulation_style = entry["doc_type"] in REGULATION_STYLE
    chunks: list[Chunk] = []

    def emit(units: list[Unit], reg_no: str, reg_title: str, clause: str):
        text = _render_group(units)
        if not text:
            return
        is_table = any(u.is_table for u in units)
        pages = [l.page for u in units for l in u.lines]
        crumb = [doc_title]
        # In an amendment, the section number is the amendment's own paragraph ("2."), while the
        # regulation it changes is named in the heading ("... of Regulation 8 ..."). The chunk is
        # filed under the regulation it changes, so a question about Regulation 8 finds it.
        is_amending = entry["doc_type"] in ("amendment", "draft") and bool(entry.get("amends"))
        amends_reg = ""
        if is_amending:
            amends_reg = _most_referenced_regulation(reg_title) or _most_referenced_regulation(text)

        if not regulation_style:
            target_reg = _most_referenced_regulation(text)
            crumb.append(f"p. {min(pages)}")
        elif is_amending:
            target_reg = amends_reg
            if reg_no:
                crumb.append(f"Para {reg_no}" + (f" ({reg_title})" if reg_title else ""))
            if clause:
                crumb.append(clause)
        else:
            target_reg = reg_no
            if reg_no:
                crumb.append(f"Regulation {reg_no}" + (f" ({reg_title})" if reg_title else ""))
                if clause:
                    crumb.append(f"Clause {clause}")
            elif reg_title:
                crumb.append(reg_title)
        chunks.append(Chunk(
            chunk_id=f"{entry['doc_id']}::{len(chunks):04d}",
            doc_id=entry["doc_id"],
            text=text,
            breadcrumb=" > ".join(crumb),
            regulation_no=target_reg,
            regulation_title=reg_title if (regulation_style and not is_amending) else "",
            clause_no=clause,
            page_start=min(pages), page_end=max(pages),
            is_table=is_table,
            tokens=count_tokens(text),
            meta={**meta, "amends_regulation": amends_reg},
        ))

    if not regulation_style:
        # Statements of Reasons, memoranda, orders, notices, secondary sources:
        # no regulation structure to follow, so pack paragraphs to size.
        units = _units_from_lines(lines, "")
        for group in _pack(units, s["target_tokens"], s["max_tokens"], s["table_max_tokens"]):
            emit(group, "", "", "")
        return chunks

    for sec in split_sections(lines):
        sec_tokens = count_tokens("\n".join(l.text for l in sec.lines))
        # Short regulations stay whole. Longer ones are split by clause, so answers can cite
        # the exact clause (FR-2), even when the whole regulation would fit in one chunk.
        if sec_tokens <= s["target_tokens"]:
            emit(_units_from_lines(sec.lines, ""), sec.number, sec.title, "")
            continue

        # Split on clauses (1), (2)… — or 2.1, 2.2… in amendments.
        marker = AMEND_MARK if sum(bool(AMEND_MARK.match(l.text)) for l in sec.lines) > \
            sum(bool(CLAUSE_MARK.match(l.text)) for l in sec.lines) else CLAUSE_MARK
        parts = _split_by(sec.lines, marker)

        # Merge tiny clauses into the next one so we don't produce 20-token chunks.
        merged: list[tuple[list[str], list[Line]]] = []
        for label, part in parts:
            if merged and count_tokens("\n".join(l.text for l in merged[-1][1])) < s["min_tokens"]:
                merged[-1][0].append(label); merged[-1][1].extend(part)
            else:
                merged.append(([label], list(part)))

        for labels, part in merged:
            labels = [x for x in labels if x]
            if marker is CLAUSE_MARK:
                clause = f"({labels[0]})" if len(labels) == 1 else (f"({labels[0]})-({labels[-1]})" if labels else "")
            else:
                clause = labels[0] if len(labels) == 1 else (f"{labels[0]}-{labels[-1]}" if labels else "")
            units = _units_from_lines(part, clause)
            groups = _pack(units, s["target_tokens"], s["max_tokens"], s["table_max_tokens"])
            for group in groups:
                sub = _sub_range(group) if len(groups) > 1 else ""
                emit(group, sec.number, sec.title, clause + sub)
    return chunks


def chunk_corpus(registry: list[dict], raw_dir: Path, settings: dict | None = None,
                 on_error=print) -> list[Chunk]:
    out = []
    for entry in registry:
        if not entry.get("include"):
            continue
        path = Path(raw_dir) / f"{entry['doc_id']}.pdf"
        if not path.exists():
            on_error(f"skip {entry['doc_id']}: file not found")
            continue
        try:
            out.extend(chunk_document(entry, path, settings))
        except Exception as e:          # one bad PDF must not stop the whole corpus
            on_error(f"skip {entry['doc_id']}: {type(e).__name__}: {e}")
    return out


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def write_jsonl(chunks: list[Chunk], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
