"""
Tests for src/dsm_assistant/ingest.py.

Run from a notebook:      from test_ingest import run_all; run_all()
Run from a terminal:      python tests/test_ingest.py

Each test builds small PDFs that imitate the real CERC layouts (see fixtures.py) and checks one
behaviour the design depends on. No real documents and no network are needed.
"""
import sys
import tempfile
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

import fixtures  # noqa: E402
from dsm_assistant import ingest  # noqa: E402

_TMP = Path(tempfile.mkdtemp())
_PDFS = fixtures.build_all(_TMP)

_BASE = dict(status="final", regime="dsm-2024", tier="core", url="https://example", effective_to=None, include=True)
PRINCIPAL = dict(_BASE, doc_id="dsm-2024", doc_type="principal", amends=None, effective_from=date(2024, 9, 16),
                 title="CERC (Deviation Settlement Mechanism and Related Matters) Regulations, 2024")
AMENDMENT = dict(_BASE, doc_id="dsm-2024-amend-3", doc_type="amendment", amends="dsm-2024",
                 effective_from=date(2026, 8, 31), title="DSM (Third Amendment) Regulations, 2026")
SOR = dict(_BASE, doc_id="sor-dsm-2024", doc_type="sor", amends="dsm-2024", effective_from=None,
           tier="reference", title="Statement of Reasons — DSM Regulations, 2024")


def _chunks(entry, key):
    return ingest.chunk_document(entry, _PDFS[key])


def _reg8_table_lines(chunks):
    text = "\n".join(c.text for c in chunks if c.regulation_no == "8")
    return [l for l in text.splitlines() if l.startswith("[Over") or l.startswith("[Under")]


# --- Regulation structure -------------------------------------------------------------
def test_regulations_detected_in_order():
    regs = []
    for c in _chunks(PRINCIPAL, "principal"):
        if c.regulation_no and c.regulation_no not in regs:
            regs.append(c.regulation_no)
    assert regs == ["1", "2", "8", "9"], regs


def test_long_regulations_split_by_clause():
    clauses = {c.clause_no for c in _chunks(PRINCIPAL, "principal") if c.regulation_no == "8"}
    assert {"(1)", "(2)", "(3)"} <= clauses, clauses


def test_definitions_split_with_subclause_ranges():
    defs = [c for c in _chunks(PRINCIPAL, "principal") if c.regulation_no == "2" and c.clause_no.startswith("(1)(")]
    assert len(defs) >= 2, [c.clause_no for c in defs]
    assert defs[0].clause_no.startswith("(1)(a)"), defs[0].clause_no


def test_breadcrumb_names_regulation_and_clause():
    c = next(c for c in _chunks(PRINCIPAL, "principal") if c.regulation_no == "8" and c.clause_no == "(2)")
    assert c.breadcrumb == "DSM Regulations, 2024 > Regulation 8 (Charges for Deviation) > Clause (2)", c.breadcrumb
    assert c.embed_text.startswith(c.breadcrumb)


# --- Two-column tables (DR-8) -----------------------------------------------------------
def _check_columns(lines):
    over = [l for l in lines if l.startswith("[Over injection")]
    under = [l for l in lines if l.startswith("[Under injection")]
    # Every item sits under the right column: left-column items are (i)-(iii) of each row group,
    # right-column items (iv)-(vi). In the fixture, the 25% and 2.15% steps are receivable
    # (over injection) and the 7.5% and 7.15% steps are payable (under injection).
    assert any("25% of RR" in l for l in over) and not any("25% of RR" in l for l in under)
    assert any("7.15% of RR" in l for l in under) and not any("7.15% of RR" in l for l in over)
    assert any("110% of RR" in l for l in under), "RoR table (unruled) lost its right column"
    # one item per line
    assert not any(l.count("(ii)") and l.count("(iii)") for l in lines), "two items on one line"


def test_ruled_table_columns_labelled():
    lines = _reg8_table_lines(_chunks(PRINCIPAL, "principal"))
    assert len(lines) >= 12, len(lines)
    _check_columns(lines)


def test_unruled_table_columns_labelled():
    lines = _reg8_table_lines(_chunks(PRINCIPAL, "principal_unruled"))
    assert len(lines) >= 12, len(lines)
    _check_columns(lines)


def test_row_labels_kept_and_note_outside_table():
    text = "\n".join(c.text for c in _chunks(PRINCIPAL, "principal_unruled") if c.regulation_no == "8")
    assert "[ROW] (I) For Deviation up to" in text
    assert "[ROW] (II) For Deviation up to" in text
    assert "\nNote: System frequency" in text, "Note was swallowed into the table"


def test_table_chunks_start_with_column_header():
    for c in _chunks(PRINCIPAL, "principal"):
        if c.is_table and any(l.startswith("[Over") for l in c.text.splitlines()):
            assert "[TABLE] Columns: Over injection (receivable by the seller) | Under injection (payable by the seller)" in c.text


# --- Layouts seen in the real PDFs (owner's run, 2026-10-13) -------------------------------
def test_grid_with_empty_columns_collapses_to_two_labels():
    text = "\n".join(c.text for c in _chunks(PRINCIPAL, "principal_grid4") if c.regulation_no == "8")
    assert "[TABLE] Columns: Over injection (receivable by the seller) | Under injection (payable by the seller)\n" in text, \
        "empty grid columns leaked into the header"
    assert "[]" not in text, "a line has an empty column label"


def test_subscript_stays_in_place_in_table_cells():
    text = "\n".join(c.text for c in _chunks(PRINCIPAL, "principal_grid4") if c.regulation_no == "8")
    assert "[10% DGS or 100 MW" in text, "subscript GS was moved away from D"


def test_note_row_read_as_one_unlabelled_line():
    text = "\n".join(c.text for c in _chunks(PRINCIPAL, "principal_grid4") if c.regulation_no == "8")
    assert "\nNote: System frequency = f and f band = [49.90Hz <= f <= 50.05 Hz]" in text, text[-300:]


def test_numbers_on_their_own_line_are_joined():
    chunks = _chunks(AMENDMENT, "amendment_split")
    crumbs = [c.breadcrumb for c in chunks]
    assert any("> Para 2 (Insertion of a new clause" in b for b in crumbs), crumbs
    assert any("> Para 4 (Amendment to Regulation 10" in b for b in crumbs), crumbs
    regs = {c.breadcrumb.split(" > ")[1].split(" (")[0]: c.regulation_no for c in chunks if " > Para" in c.breadcrumb}
    assert regs == {"Para 1": "", "Para 2": "8", "Para 3": "8", "Para 4": "10"}, regs


# --- Cleaning ---------------------------------------------------------------------------
def test_footer_removed():
    for key in ("principal", "principal_unruled"):
        for c in _chunks(PRINCIPAL, key):
            assert "Regulations, 2024 Page" not in c.text, (key, c.chunk_id)


def test_repeated_body_line_kept():
    text = "\n".join(c.text for c in _chunks(SOR, "sor"))
    assert text.count("Commission's analysis and decision") == 3, "a repeated body line was removed as a footer"


def test_symbol_only_lines_dropped():
    first = _chunks(PRINCIPAL, "principal")[0].text
    assert first.startswith("CENTRAL ELECTRICITY"), first[:40]


# --- Amendments -------------------------------------------------------------------------
def test_amendment_filed_under_amended_regulation():
    chunks = _chunks(AMENDMENT, "amendment")
    para2 = next(c for c in chunks if "(4A)" in c.text)
    assert para2.regulation_no == "8" and para2.meta["amends_regulation"] == "8", para2.regulation_no
    assert para2.breadcrumb.startswith("DSM (Third Amendment) Regulations, 2026 > Para 2"), para2.breadcrumb
    para3 = next(c for c in chunks if "Regulation 10" in c.breadcrumb)
    assert para3.regulation_no == "10"


# --- Other documents --------------------------------------------------------------------
def test_sor_chunked_by_size_with_regulation_reference():
    chunks = _chunks(SOR, "sor")
    assert len(chunks) >= 1
    assert all(c.regulation_no == "8" for c in chunks), [c.regulation_no for c in chunks]
    assert chunks[0].breadcrumb.startswith("Statement of Reasons — DSM Regulations, 2024 > p. 1")


# --- Metadata ---------------------------------------------------------------------------
def test_metadata_is_vector_store_safe():
    for c in _chunks(PRINCIPAL, "principal") + _chunks(SOR, "sor"):
        for k, v in c.meta.items():
            assert isinstance(v, (str, int, float, bool)), (k, type(v))
    c = _chunks(PRINCIPAL, "principal")[0]
    assert c.meta["effective_from_int"] == 20240916 and c.meta["effective_to_int"] == 99991231


def test_chunk_sizes_within_limits():
    s = ingest.DEFAULTS
    for c in _chunks(PRINCIPAL, "principal") + _chunks(AMENDMENT, "amendment") + _chunks(SOR, "sor"):
        limit = s["table_max_tokens"] if c.is_table else s["max_tokens"]
        assert c.tokens <= limit, (c.chunk_id, c.tokens)
        assert c.text.strip(), c.chunk_id


def test_jsonl_round_trip():
    chunks = _chunks(PRINCIPAL, "principal")
    path = _TMP / "chunks.jsonl"
    ingest.write_jsonl(chunks, path)
    back = ingest.read_jsonl(path)
    assert [c.chunk_id for c in back] == [c.chunk_id for c in chunks]
    assert back[5].meta == chunks[5].meta


# ---------------------------------------------------------------------------------------
def run_all(verbose: bool = True) -> bool:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            if verbose:
                print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
