"""
Tests for src/dsm_assistant/index.py and retrieve.py.

Run from a notebook:   from test_retrieve import run_all; run_all()
Run from a terminal:   python tests/test_retrieve.py

Uses a small hand-made corpus with the same metadata as the real one, and the hashing
embedder (no model download). The real embedding model is exercised in notebook 04b.
"""
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

from dsm_assistant import index, ingest, retrieve  # noqa: E402

TODAY = date(2026, 10, 13)


def _entry(doc_id, doc_type, eff_from, eff_to=None, amends=None, title=None):
    return dict(doc_id=doc_id, title=title or doc_id, doc_type=doc_type,
                status="draft" if doc_type in ("draft", "em") else ("superseded" if eff_to else "final"),
                regime="dsm-2022" if "2022" in doc_id else "dsm-2024", tier="core", amends=amends,
                url="u", effective_from=eff_from, effective_to=eff_to, include=True)


ENTRIES = {
    "dsm-2024": _entry("dsm-2024", "principal", date(2024, 9, 16)),
    "dsm-2022": _entry("dsm-2022", "principal", date(2022, 12, 5), date(2024, 9, 15)),
    "dsm-2024-amend-3": _entry("dsm-2024-amend-3", "amendment", date(2026, 8, 31), amends="dsm-2024"),
    "draft-dsm-2024-amend-3": _entry("draft-dsm-2024-amend-3", "draft", None, amends="dsm-2024"),
    "sor-dsm-2024": _entry("sor-dsm-2024", "sor", None, amends="dsm-2024"),
    "secondary-jsa": _entry("secondary-jsa", "secondary", date(2026, 4, 1), amends="dsm-2024"),
    # Another family's amendment to *its own* Regulation 8 (OB-5): must never be added to a DSM answer.
    "gna-2022-amend-3": _entry("gna-2022-amend-3", "amendment", date(2025, 9, 9), amends="gna-2022"),
}

TEXTS = [
    ("dsm-2024", "6", "(2)", "Deviation in percentage for a WS seller is computed with reference to available capacity."),
    ("dsm-2024", "7", "", "The Normal Rate of charges for deviation is the highest of the weighted average prices of the day ahead market, real time market and ancillary services."),
    ("dsm-2024", "8", "(1)", "[Under injection (payable by the seller)] (iv) @ RR when frequency is within band. General seller tolerance 10% DGS or 100 MW."),
    ("dsm-2024", "8", "(2)", "Run-of-river generating station deviation beyond 15% DGS or 150 MW payable at 105% of RR."),
    ("dsm-2024", "10", "", "Payment of charges for deviation within 10 days, late payment surcharge 0.04% per day."),
    ("dsm-2022", "8", "(1)", "Under the 2022 regulations the general seller deviation was charged at the normal rate with a 12% band."),
    ("dsm-2024-amend-3", "8", "", "(4A) WS seller charges for deviation treated at par with general sellers for projects bid after 01.01.2027."),
    ("dsm-2024-amend-3", "10", "", "Payment timeline may follow the Detailed Procedure for the National Pool Account."),
    ("draft-dsm-2024-amend-3", "8", "", "Draft proposal: WS seller charges at par with general sellers from 2026."),
    ("sor-dsm-2024", "8", "", "The Commission reasoned that tolerance bands should balance grid security and variability of wind and solar."),
    ("secondary-jsa", "6", "", "Value of X for wind generators: 2026-27 100 percent, 2027-28 95 percent, 2028-29 85 percent."),
    ("gna-2022-amend-3", "8", "", "Regulation 8 of the GNA Regulations on connectivity grant to a seller is substituted."),
]


def _make_chunks():
    chunks, counters = [], {}
    for doc_id, reg, clause, text in TEXTS:
        n = counters.get(doc_id, 0); counters[doc_id] = n + 1
        e = ENTRIES[doc_id]
        meta = {**ingest.registry_meta(e),
                "amends_regulation": reg if e["doc_type"] in ("amendment", "draft") else ""}
        chunks.append(ingest.Chunk(
            chunk_id=f"{doc_id}::{n:04d}", doc_id=doc_id, text=text,
            breadcrumb=f"{doc_id} > Regulation {reg}" + (f" > Clause {clause}" if clause else ""),
            regulation_no=reg, regulation_title="", clause_no=clause, page_start=1, page_end=1,
            is_table=text.startswith("["), tokens=ingest.count_tokens(text), meta=meta))
    return chunks


CHUNKS = _make_chunks()
_DIR = Path(tempfile.mkdtemp())
EMBEDDER = index.HashingEmbedder()
COLLECTION = index.build_vector_index(CHUNKS, _DIR / "index", EMBEDDER, log=lambda *a: None)
RETRIEVER = retrieve.Retriever(CHUNKS, COLLECTION, EMBEDDER)


def _docs(result):
    return [h.chunk.doc_id for h in result.hits]


# --- Query analyser -------------------------------------------------------------------
def test_dates_in_many_formats():
    cases = {
        "on 15.01.2026": date(2026, 1, 15), "on 15/01/2026": date(2026, 1, 15),
        "on 2026-01-15": date(2026, 1, 15), "on 15 January 2026": date(2026, 1, 15),
        "on 15th Jan, 2026": date(2026, 1, 15), "on January 15, 2026": date(2026, 1, 15),
        "in April 2026": date(2026, 4, 1), "in Sept 2025": date(2025, 9, 1),
        "no date here, Regulation 8": None, "for FY 2027-28": None,
    }
    for text, want in cases.items():
        assert retrieve.find_date(text) == want, (text, retrieve.find_date(text), want)


def test_flags():
    a = retrieve.analyse("Why did the draft propose a change compared with the 2022 regulations?", TODAY)
    assert a.wants_why and a.wants_draft and a.compare and a.as_of is None
    b = retrieve.analyse("What does Regulation 8 say?", TODAY)
    assert b.regulation == "8" and b.as_of == TODAY and not (b.wants_draft or b.wants_why or b.compare)
    c = retrieve.analyse("Calculate the charge for 120 MW deviation", TODAY)
    assert c.wants_calculation


def test_other_2022_regulations_are_not_dsm_comparisons():
    """OB-3: the GNA and Ancillary Services Regulations are also 'of 2022'. Naming them must not
    switch on DSM 2022-vs-2024 comparison mode, which would turn off the date filter."""
    for q in ["What does Regulation 5 of the GNA Regulations, 2022 say about additional capacity?",
              "Under the Ancillary Services Regulations, 2022, how is an SRAS provider selected?",
              "Which entities need connectivity under the 2022 regulations on general network access?"]:
        a = retrieve.analyse(q, TODAY)
        assert not a.compare and a.as_of == TODAY, (q, a.compare, a.as_of)
    # DSM 2022 is still recognised, with or without a family named elsewhere
    for q in ["How did the charges in the DSM Regulations, 2022 differ?",
              "What did the 2022 regulations say about deviation charges?",
              "Compare the GNA Regulations before and after the Third Amendment"]:
        assert retrieve.analyse(q, TODAY).compare, q


def test_previous_financial_year_is_not_a_comparison():
    """FX-4: 'previous financial year' must keep the date filter on; 'previous regulations' still compares."""
    a = retrieve.analyse("What must an entity do if it paid late in the previous financial year?", TODAY)
    assert not a.compare and a.as_of == TODAY
    assert retrieve.analyse("How do the previous regulations differ on payment?", TODAY).compare
    assert retrieve.analyse("What did the previous DSM say about infirm power?", TODAY).compare


# --- Version filter (FR-4, FR-5) -------------------------------------------------------
def test_default_excludes_old_regime_drafts_and_reasons():
    r = RETRIEVER.search("general seller deviation charges", today=TODAY, k=10, expand=False)
    docs = set(_docs(r))
    assert "dsm-2022" not in docs and "draft-dsm-2024-amend-3" not in docs and "sor-dsm-2024" not in docs, docs


def test_date_before_amendment_excludes_it():
    r = RETRIEVER.search("WS seller charges at par with general sellers on 01.06.2025", today=TODAY, k=10)
    assert "dsm-2024-amend-3" not in _docs(r), _docs(r)
    r2 = RETRIEVER.search("WS seller charges at par with general sellers on 01.10.2026", today=TODAY, k=10)
    assert "dsm-2024-amend-3" in _docs(r2), _docs(r2)


def test_date_in_2023_uses_2022_regime():
    r = RETRIEVER.search("general seller deviation charge on 01.06.2023", today=TODAY, k=10, expand=False)
    assert set(_docs(r)) <= {"dsm-2022"} and "dsm-2022" in _docs(r), _docs(r)


def test_secondary_source_respects_its_date():
    early = RETRIEVER.search("value of X for wind generators on 15.01.2026", today=TODAY, k=10)
    late = RETRIEVER.search("value of X for wind generators", today=TODAY, k=10)
    assert "secondary-jsa" not in _docs(early) and "secondary-jsa" in _docs(late)


def test_drafts_only_when_asked():
    r = RETRIEVER.search("What did the draft propose for WS sellers?", today=TODAY, k=10)
    assert "draft-dsm-2024-amend-3" in _docs(r), _docs(r)


def test_reasons_only_for_why_questions():
    r = RETRIEVER.search("Why are tolerance bands needed for wind and solar?", today=TODAY, k=10)
    assert "sor-dsm-2024" in _docs(r), _docs(r)


def test_comparison_allows_both_regimes():
    r = RETRIEVER.search("Compare general seller deviation charges in the 2022 regulations with 2024", today=TODAY, k=10)
    assert {"dsm-2022", "dsm-2024"} <= set(_docs(r)), _docs(r)


def test_chroma_filter_matches_python_filter():
    for q in ["general seller", "on 01.06.2023 general seller", "draft proposal", "why tolerance",
              "compare 2022 regulations", "value of X on 15.01.2026"]:
        a = retrieve.analyse(q, TODAY)
        py = {c.chunk_id for c in CHUNKS if retrieve.passes(c.meta, a)}
        res = COLLECTION.get(where=retrieve.chroma_where(a)) if retrieve.chroma_where(a) else COLLECTION.get()
        assert set(res["ids"]) == py, (q, set(res["ids"]) ^ py)


# --- Search behaviour ------------------------------------------------------------------
def test_keyword_search_finds_exact_terms():
    r = RETRIEVER.search("run-of-river 150 MW 105%", today=TODAY, mode="keyword", expand=False)
    assert r.hits[0].chunk.chunk_id == "dsm-2024::0003", [h.chunk.chunk_id for h in r.hits]


def test_hybrid_reports_both_ranks():
    r = RETRIEVER.search("late payment surcharge days", today=TODAY, expand=False)
    top = r.hits[0]
    assert top.chunk.regulation_no == "10" and top.keyword_rank and top.vector_rank, top


def test_amendment_expansion_adds_in_force_amendment():
    r = RETRIEVER.search("Regulation 8 general seller tolerance 10% DGS", today=TODAY, k=1)
    assert r.hits[0].chunk.doc_id == "dsm-2024" and r.hits[0].chunk.regulation_no == "8"
    expanded = [h.chunk.doc_id for h in r.hits if h.expanded]
    assert expanded == ["dsm-2024-amend-3"], expanded     # not gna-2022-amend-3 (also "Reg. 8") — OB-5


def test_no_expansion_before_amendment_date():
    r = RETRIEVER.search("Regulation 8 general seller tolerance 10% DGS on 01.06.2025", today=TODAY, k=1)
    assert not [h for h in r.hits if h.expanded], [h.chunk.chunk_id for h in r.hits]


def test_scoring_helpers():
    c = CHUNKS[2]   # dsm-2024 Reg 8 (1)
    assert retrieve.matches(c, {"doc_id": "dsm-2024", "regulation_no": "8", "clause_no": "(1)"})
    assert retrieve.matches(c, {"doc_id": "dsm-*"})
    assert not retrieve.matches(c, {"doc_id": "dsm-2024", "regulation_no": "7"})


def test_index_persisted_and_reopened():
    col = index.open_vector_index(_DIR / "index")
    assert col.count() == len(CHUNKS)
    meta = col.get(ids=["dsm-2024::0002"])["metadatas"][0]
    assert meta["regulation_no"] == "8" and meta["clause_no"] == "(1)" and meta["effective_from_int"] == 20240916


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
    ok = run_all()
    shutil.rmtree(_DIR, ignore_errors=True)
    sys.exit(0 if ok else 1)
