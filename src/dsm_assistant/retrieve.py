"""
Retrieval: understand the question, filter by version, search two ways, merge, expand.

Implements Design §4.6 (hybrid retriever + version filter) and §4.7 (query analyser).

    retriever = Retriever(chunks, collection, embedder)
    result = retriever.search("What is the tolerance band for solar sellers?")
    result.analysis   -> what the analyser understood (date, draft?, why?, compare?)
    result.hits       -> top chunks, each with its vector rank, keyword rank and score

Version rules (FR-4, FR-5), applied identically to both searches:

    | Situation                          | Rule                                                     |
    |------------------------------------|----------------------------------------------------------|
    | question names a date D            | only text in force on D                                  |
    | no date                            | only text in force today                                 |
    | comparison ("compare", "2022", …)  | no date limit (old and new versions both eligible)       |
    | draft / proposal asked about       | drafts and explanatory memoranda allowed; otherwise not  |
    | "why" question                     | Statements of Reasons allowed; otherwise not             |
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .index import KeywordIndex

VERSION = "4b.3"

RRF_K = 60                # Reciprocal Rank Fusion constant (standard value)
CANDIDATES = 10           # results taken from each search before merging
MAX_EXPANSION = 3         # extra amendment chunks added after the top k

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))


# ---------------------------------------------------------------------------
# Query analyser (Design §4.7) — simple rules, no LLM
# ---------------------------------------------------------------------------
@dataclass
class QueryAnalysis:
    question: str
    date: date | None = None          # the date the question asks about, if any
    as_of: date | None = None         # the date the version filter uses (date, or today)
    wants_draft: bool = False
    wants_why: bool = False
    compare: bool = False
    wants_calculation: bool = False
    regulation: str = ""              # "8" if the question names Regulation 8

    def describe(self) -> str:
        parts = [f"as of {self.as_of}" if self.as_of else "all versions (comparison)"]
        if self.date:
            parts.append(f"date in question: {self.date}")
        for flag, label in [(self.wants_draft, "drafts allowed"), (self.wants_why, "reasons allowed"),
                            (self.wants_calculation, "calculation"),
                            (bool(self.regulation), f"Regulation {self.regulation}")]:
            if flag:
                parts.append(label)
        return "; ".join(parts)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def find_date(text: str) -> date | None:
    """First date in the question. Understands 15.04.2026, 15/04/2026, 2026-04-15,
    15 April 2026, April 15, 2026 and 'April 2026' (taken as the 1st of the month)."""
    t = text.lower()
    patterns = [
        (r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", lambda m: _safe_date(int(m[3]), int(m[2]), int(m[1]))),
        (r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", lambda m: _safe_date(int(m[1]), int(m[2]), int(m[3]))),
        (rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_RE})\.?,?\s+(\d{{4}})\b",
         lambda m: _safe_date(int(m[3]), MONTHS[m[2]], int(m[1]))),
        (rf"\b({MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
         lambda m: _safe_date(int(m[3]), MONTHS[m[1]], int(m[2]))),
        (rf"\b({MONTH_RE})\.?,?\s+(\d{{4}})\b", lambda m: _safe_date(int(m[2]), MONTHS[m[1]], 1)),
    ]
    found = []
    for pat, build in patterns:
        for m in re.finditer(pat, t):
            d = build(m)
            if d:
                found.append((m.start(), d))
    return min(found)[1] if found else None


# Regulation families other than DSM that share the corpus (CR-2 Batch A).
OTHER_FAMILY = re.compile(
    r"\bgna\b|general network access|\bconnectivity\b|\bancillary\b|\bsras\b|\btras\b|"
    r"power market|\bpmr\b|grid code|\biegc\b")


def analyse(question: str, today: date | None = None) -> QueryAnalysis:
    q = question.lower()
    a = QueryAnalysis(question=question)
    a.date = find_date(question)
    a.wants_draft = bool(re.search(r"\bdraft|\bpropos", q))
    a.wants_why = bool(re.search(r"\bwhy\b|\breason|\brationale|\bjustif|\bobjective", q))
    # "Regulations, 2022" means DSM 2022 only when no other 2022-era family is named:
    # the GNA and Ancillary Services Regulations are also of 2022 (OB-3, fixed in 4b.2).
    # "previous" alone ("previous financial year") is not a comparison (FX-4, 4b.3).
    other_family = bool(OTHER_FAMILY.search(q))
    a.compare = bool(
        re.search(r"\bcompar|\bdiffer|\bchanged?\b|\bversus\b|\bvs\.?\b|\bprevious (?:regulations|version|rules|dsm)|\bearlier regulations|"
                  r"\bold(er)? (dsm|deviation)|\bdsm,? 2022|\bdsm regulations,? 2022", q)
        or (not other_family and re.search(r"\bold(er)? (regulations|rules)|\b2022 regulations|\bregulations,? 2022", q)))
    a.wants_calculation = bool(re.search(r"\bcalculat|\bcompute the charge|\bhow much\b|₹|\brs\.?\s*\d", q))
    m = re.search(r"\breg(?:ulation)?\.?\s*\(?(\d{1,2}[a-z]?)\)?", q)
    a.regulation = m.group(1).upper() if m else ""
    if a.date:
        a.as_of = a.date
    elif not a.compare:
        a.as_of = today or date.today()
    return a


# ---------------------------------------------------------------------------
# Version filter — one rule set, two forms (Python for BM25, a `where` for Chroma)
# ---------------------------------------------------------------------------
def excluded_types(a: QueryAnalysis) -> list[str]:
    out = []
    if not a.wants_draft:
        out += ["draft", "em"]
    if not a.wants_why:
        out += ["sor"]
    return out


def passes(meta: dict, a: QueryAnalysis) -> bool:
    if meta["doc_type"] in excluded_types(a):
        return False
    if a.as_of:
        d = int(a.as_of.strftime("%Y%m%d"))
        if not (meta["effective_from_int"] <= d <= meta["effective_to_int"]):
            return False
    return True


def chroma_where(a: QueryAnalysis) -> dict | None:
    conds = []
    ex = excluded_types(a)
    if ex:
        conds.append({"doc_type": {"$nin": ex}})
    if a.as_of:
        d = int(a.as_of.strftime("%Y%m%d"))
        conds += [{"effective_from_int": {"$lte": d}}, {"effective_to_int": {"$gte": d}}]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------
@dataclass
class Hit:
    chunk: object                 # ingest.Chunk
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    expanded: bool = False        # added by amendment expansion, not by ranking


@dataclass
class RetrievalResult:
    analysis: QueryAnalysis
    hits: list[Hit] = field(default_factory=list)

    @property
    def ranked(self) -> list[Hit]:
        return [h for h in self.hits if not h.expanded]


class Retriever:
    def __init__(self, chunks, collection, embedder, keyword: KeywordIndex | None = None):
        self.by_id = {c.chunk_id: c for c in chunks}
        self.collection = collection
        self.embedder = embedder
        self.keyword = keyword or KeywordIndex(chunks)

    def _vector(self, question: str, a: QueryAnalysis, n: int) -> list[str]:
        q = self.embedder.embed_query(question)
        res = self.collection.query(query_embeddings=[q.tolist()], n_results=n, where=chroma_where(a))
        return res["ids"][0]

    def search(self, question: str, k: int = 5, mode: str = "hybrid", today: date | None = None,
               expand: bool = True) -> RetrievalResult:
        a = analyse(question, today)
        allowed = {cid for cid, c in self.by_id.items() if passes(c.meta, a)}
        result = RetrievalResult(a)
        if not allowed:
            return result

        vec = self._vector(question, a, min(CANDIDATES, len(allowed))) if mode in ("hybrid", "vector") else []
        kw = [cid for cid, _ in self.keyword.search(question, CANDIDATES, allowed)] if mode in ("hybrid", "keyword") else []

        scores: dict[str, float] = {}
        for rank, cid in enumerate(vec, 1):
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
        for rank, cid in enumerate(kw, 1):
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
        # Small nudge for chunks of the regulation the question names ("Regulation 8 ...").
        if a.regulation:
            for cid in scores:
                if self.by_id[cid].regulation_no == a.regulation:
                    scores[cid] += 1 / (RRF_K + 1) / 2

        top = sorted(scores, key=lambda cid: -scores[cid])[:k]
        for cid in top:
            result.hits.append(Hit(self.by_id[cid], scores[cid],
                                   vec.index(cid) + 1 if cid in vec else None,
                                   kw.index(cid) + 1 if cid in kw else None))
        if expand:
            self._expand_amendments(result, allowed)
        return result

    def _expand_amendments(self, result: RetrievalResult, allowed: set[str]) -> None:
        """If a regulation's original text is retrieved, also bring in the amendments to that
        regulation that are in force for this question (Design §4.6, step 4). Otherwise the
        LLM could answer from the original text and miss a later change."""
        have = {h.chunk.chunk_id for h in result.hits}
        regs = []                 # (principal doc_id, regulation number)
        for h in result.hits:
            m = h.chunk.meta
            key = (h.chunk.doc_id, h.chunk.regulation_no)
            if m["doc_type"] == "principal" and h.chunk.regulation_no and key not in regs:
                regs.append(key)
        extra = []
        for doc_id, reg in regs:
            for cid in sorted(allowed, key=lambda i: (self.by_id[i].meta["effective_from_int"], i)):
                c = self.by_id[cid]
                # Same regulation number AND the same principal regulation: "Regulation 22" of the
                # GNA Regulations is not Regulation 22 of the Grid Code (fixed in 4b.2, OB-5).
                if (cid not in have and c.meta["doc_type"] == "amendment"
                        and c.meta.get("amends") == doc_id
                        and c.meta.get("amends_regulation") == reg):
                    extra.append(c); have.add(cid)
        for c in extra[:MAX_EXPANSION]:
            result.hits.append(Hit(c, 0.0, expanded=True))


# ---------------------------------------------------------------------------
# Scoring against labelled questions
# ---------------------------------------------------------------------------
def matches(chunk, label: dict) -> bool:
    """A label lists what an acceptable chunk looks like, e.g.
    {"doc_id": "dsm-2024", "regulation_no": "8", "clause_no": "(2)"}. Missing keys match anything;
    "doc_id" may end in * to match a prefix (e.g. "sor-*")."""
    values = {
        "doc_id": chunk.doc_id,
        "doc_type": chunk.meta["doc_type"],
        "regime": chunk.meta["regime"],
        "regulation_no": chunk.regulation_no,
        "clause_no": chunk.clause_no,
    }
    for key, want in label.items():
        have = values[key]
        if key == "doc_id" and want.endswith("*"):
            ok = have.startswith(want[:-1])
        elif key == "clause_no":
            ok = have.startswith(want)           # "(1)" also matches "(1)(a)-(f)"
        else:
            ok = have == want
        if not ok:
            return False
    return True


def first_relevant_rank(result: RetrievalResult, labels: list[dict]) -> int | None:
    for rank, h in enumerate(result.ranked, 1):
        if any(matches(h.chunk, lab) for lab in labels):
            return rank
    return None
