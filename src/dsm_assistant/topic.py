"""
Topic briefing (CR-5, FR-12): "tell me everything the documents say about X".

    gather(topic, retriever)        -> TopicResult   (wide search, all versions, grouped)
    brief(topic, retriever, llm)    -> Briefing      (one LLM call, 7 sections, every fact cited)
    coverage(briefing, checklist)   -> dict          (share of required items the briefing cites)

The question mode (answer.ask) is unchanged: it answers ONE question from 8 passages for ONE date.
A topic needs ~30 passages across every version, with in-force, superseded, draft and SoR text kept
apart. All of that is done here, by code, around a single model call.
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date

from . import answer
from .retrieve import RRF_K, Hit, QueryAnalysis, matches

VERSION = "cr5.2"

MAX_PASSAGES = 30          # passages given to the model
CANDIDATES = 40            # per search list
MAX_EXPANSION = 8          # extra amendment passages for principal regulations found
MAX_PER_DOC = 6            # no single document may take more than this many places (diversity)
PHRASE_RESERVE = 16        # places kept for documents that contain the topic's exact words (one each)
# Which documents get a reserved place first: the rules and their dates before explanations and drafts.
TYPE_PRIORITY = {"principal": 0, "amendment": 1, "effective_notice": 2, "procedure": 3, "order": 4,
                 "secondary": 5, "draft": 6, "em": 7, "sor": 8}
ANGLES = ["{t}", "definition of {t}", "charges for {t}", "procedure for {t}", "reasons for {t}"]
ANGLE_WEIGHT = [1.0, 0.6, 0.6, 0.6, 0.6]   # the topic itself counts most
BRIEF_MAX_OUTPUT_TOKENS = 8192   # DT-1: thinking models spend output tokens before writing; 4,096 cut a briefing off
SECTIONS = ["In one paragraph", "Rules in force today", "How it changed", "Related rules in other regulations",
            "Why", "Proposals (drafts)", "Not covered"]
NOT_A_RULE = {"sor", "em", "draft"}         # DT-2: these may not be cited under "Rules in force today"

FAMILY = {"dsm-2024": "DSM Regulations", "dsm-2022": "DSM Regulations", "iegc-2023": "Grid Code (IEGC)",
          "as-2022": "Ancillary Services", "gna-2022": "Connectivity and GNA", "pmr-2021": "Power Market"}
FAMILY_ORDER = ["DSM Regulations", "Grid Code (IEGC)", "Ancillary Services", "Connectivity and GNA", "Power Market"]


@dataclass
class TopicResult:
    topic: str
    hits: list = field(default_factory=list)            # in presentation order (grouped, oldest first)
    groups: list = field(default_factory=list)          # [(family, [hit, ...])]
    truncated: bool = False                             # more relevant passages existed than MAX_PASSAGES


def family_of(chunk) -> str:
    return FAMILY.get(chunk.meta.get("regime", ""), "Other")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _contains_phrase(chunk_text: str, topic: str) -> bool:
    t = " ".join(_words(topic))
    return bool(t) and t in " ".join(_words(chunk_text))


def gather(topic: str, retriever, n: int = MAX_PASSAGES) -> TopicResult:
    """Wide hybrid search: the topic plus four fixed angles, no date filter, drafts and SoRs included,
    plus a bonus for every passage containing the topic's exact words, at most MAX_PER_DOC passages per
    document, plus amendments to the principal regulations found. No LLM call."""
    topic = (topic or "").strip()
    res = TopicResult(topic)
    if not topic:
        return res
    a = QueryAnalysis(question=topic)
    a.wants_draft = a.wants_why = a.compare = True        # all versions and all document types
    allowed = set(retriever.by_id)
    scores: dict[str, float] = {}
    first_vec, first_kw = [], []
    for i, angle in enumerate(ANGLES):
        q = angle.format(t=topic)
        vec = retriever._vector(q, a, min(CANDIDATES, len(allowed)))
        kw = [cid for cid, _ in retriever.keyword.search(q, CANDIDATES, allowed)]
        if i == 0:
            first_vec, first_kw = vec, kw
        for lst in (vec, kw):
            for rank, cid in enumerate(lst, 1):
                scores[cid] = scores.get(cid, 0) + ANGLE_WEIGHT[i] / (RRF_K + rank)
    phrase = [cid for cid, c in retriever.by_id.items() if _contains_phrase(c.text, topic)]
    for cid in phrase:
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + 1)
    ranked = sorted(scores, key=lambda cid: -scores[cid])
    res.truncated = len(phrase) > n               # the exact words appear in more passages than we can give
    # 1) One place for each document that contains the topic's exact words (its best passage), so every
    #    version, notice and draft that names the topic is seen: this is what "How it changed" needs.
    best_by_doc: dict[str, str] = {}
    for cid in sorted(phrase, key=lambda c: -scores[c]):
        best_by_doc.setdefault(retriever.by_id[cid].doc_id, cid)
    reserved = sorted(best_by_doc.values(),
                      key=lambda c: (TYPE_PRIORITY.get(retriever.by_id[c].meta["doc_type"], 9), -scores[c]))[:min(PHRASE_RESERVE, n)]
    top, per_doc = list(reserved), {}
    for cid in top:
        d = retriever.by_id[cid].doc_id
        per_doc[d] = per_doc.get(d, 0) + 1
    # 2) Fill the rest by score, at most MAX_PER_DOC places per document.
    for cid in ranked:
        if len(top) >= n:
            break
        if cid in top:
            continue
        d = retriever.by_id[cid].doc_id
        if per_doc.get(d, 0) < MAX_PER_DOC:
            top.append(cid); per_doc[d] = per_doc.get(d, 0) + 1
    hits = [Hit(retriever.by_id[cid], scores[cid],
                first_vec.index(cid) + 1 if cid in first_vec else None,
                first_kw.index(cid) + 1 if cid in first_kw else None) for cid in top]

    # Amendments to the principal regulations found, whatever their date (the briefing shows history).
    have = set(top)
    regs = [(h.chunk.doc_id, h.chunk.regulation_no) for h in hits
            if h.chunk.meta["doc_type"] == "principal" and h.chunk.regulation_no]
    extra = []
    for doc_id, reg in dict.fromkeys(regs):
        for cid, c in retriever.by_id.items():
            if (cid not in have and c.meta["doc_type"] == "amendment" and c.meta.get("amends") == doc_id
                    and c.meta.get("amends_regulation") == reg):
                extra.append(Hit(c, 0.0, expanded=True)); have.add(cid)
    hits += extra[:MAX_EXPANSION]

    # Group by family; inside a family, oldest first, then by regulation number.
    def reg_key(h):
        m = re.match(r"\d+", h.chunk.regulation_no or "")
        return int(m.group()) if m else 999
    by_family: dict[str, list] = {}
    for h in hits:
        by_family.setdefault(family_of(h.chunk), []).append(h)
    order = FAMILY_ORDER + sorted(set(by_family) - set(FAMILY_ORDER))
    for fam in order:
        if fam in by_family:
            group = sorted(by_family[fam], key=lambda h: (h.chunk.meta["effective_from_int"] or 0, h.chunk.doc_id,
                                                          reg_key(h), h.chunk.chunk_id))
            res.groups.append((fam, group))
            res.hits.extend(group)
    return res


BRIEF_SYSTEM = """You write topic briefings on India's CERC regulations for grid analysts, using ONLY the numbered source passages [S1], [S2], … Each passage has a header with its document, status and the dates it applies. Passages are grouped by regulation family, oldest first.

Write these sections, with exactly these headings, in this order. If no passage supports a section, keep the heading and write "Nothing in the documents."
## In one paragraph
## Rules in force today
## How it changed
## Related rules in other regulations
## Why
## Proposals (drafts)
## Not covered

Rules:
1. Every factual sentence or bullet ends with the tag(s) of its passage(s), e.g. [S3]. Use only tags that appear in the passages.
2. "Rules in force today": only passages whose status is IN FORCE and whose Applies dates include today's date, and only regulation, amendment, effective-date notice or procedure passages. Never cite a Statement of Reasons, an Explanatory Memorandum or a draft in this section, even if it quotes the rule: cite the regulation itself. State numbers, thresholds and rates exactly as written.
3. "How it changed": a dated timeline, oldest first: the version or amendment, the date it took effect, and what changed. Mark rules that no longer apply "(superseded)".
4. A passage marked DRAFT is a proposal, not law: use it only under "Proposals (drafts)" and say "proposed". Say whether the final text adopted it only if a passage shows the final text; otherwise say the passages do not show the outcome.
5. A STATEMENT OF REASONS passage explains why a rule was made: use it under "Why". A passage marked DRAFT — EXPLANATORY MEMORANDUM is CERC's own explanation of a draft: call it an explanatory memorandum, never a secondary source. Only a passage marked SECONDARY SOURCE is a third-party summary: say so in the sentence.
6. "Related rules in other regulations": the same topic in a different family (Grid Code, Ancillary Services, GNA, Power Market, or DSM if the topic is mainly elsewhere).
7. "Not covered": one or two lines on what a reader might expect about this topic that the passages do not say. Never fill it from outside knowledge.
8. Use abbreviations exactly as written; never expand one unless a passage defines it. Never do arithmetic. Plain text with ≤ ≥ < > %; no LaTeX.
9. If no passage is about the topic, reply with exactly: {refusal}
10. Be complete but compact: short bullets, one fact per bullet.
11. Include only facts about the topic. Some passages are given only because they share words with it: skip any passage that is not about the topic.""".format(refusal=answer.REFUSAL)


@dataclass
class Briefing:
    topic: str
    text: str
    result: TopicResult
    sources: dict = field(default_factory=dict)       # tag -> chunk
    cited: list = field(default_factory=list)
    invalid_citations: list = field(default_factory=list)
    refused: bool = False
    uses_draft: bool = False
    uses_secondary: bool = False
    provider: str = ""
    model: str = ""
    seconds: float = 0.0
    llm_seconds: float = 0.0
    notes: list = field(default_factory=list)
    missing_sections: list = field(default_factory=list)   # DT-1: headings absent (reply cut off)
    rule_section_misuse: list = field(default_factory=list)  # DT-2: SoR/EM/draft tags cited as rules in force

    @property
    def complete(self) -> bool:
        return self.refused or not self.missing_sections

    @property
    def hits(self):
        return self.result.hits

    @property
    def ok(self) -> bool:
        return self.refused or (bool(self.cited) and not self.invalid_citations and not self.missing_sections)

    def source_list(self) -> list[tuple[list[str], str]]:
        grouped: dict[str, list[str]] = {}
        for t in dict.fromkeys(self.cited):
            if t in self.sources:
                grouped.setdefault(answer.citation(self.sources[t]), []).append(t)
        return [(tags, cite) for cite, tags in grouped.items()]


def build_prompt(res: TopicResult, today: date) -> tuple[str, dict]:
    tags, blocks, i = {}, [], 0
    for fam, group in res.groups:
        blocks.append(f"=== {fam} ===")
        for h in group:
            i += 1
            tag = f"S{i}"
            tags[tag] = h.chunk
            blocks.append(f"{answer.passage_header(tag, h.chunk)}\n{h.chunk.text}")
    note = ("\nNote: more passages matched than could be given; say in 'Not covered' that this is a summary of the "
            "most relevant passages and that a narrower topic gives more detail." if res.truncated else "")
    user = (f"Today's date: {today.strftime('%d.%m.%Y')}.{note}\n\nTOPIC\n{res.topic}\n\nSOURCE PASSAGES\n\n"
            + "\n\n---\n\n".join(blocks))
    return user, tags


@contextmanager
def output_limit(client, tokens: int):
    """Raise the model's output limit for one call (briefings are long), then restore it. Works on the
    fallback wrapper and on each provider client it holds; clients without the setting are left alone."""
    objs = [o for o in (client, getattr(client, "primary", None), getattr(client, "fallback", None))
            if o is not None and hasattr(o, "max_output_tokens")]
    old = [(o, o.max_output_tokens) for o in objs]
    for o in objs:
        o.max_output_tokens = max(o.max_output_tokens or 0, tokens)
    try:
        yield
    finally:
        for o, v in old:
            o.max_output_tokens = v


def section_text(text: str, heading: str) -> str:
    m = re.search(r"^##\s*" + re.escape(heading) + r"\s*$(.*?)(?=^##\s|\Z)", text or "", re.M | re.S)
    return m.group(1) if m else ""


def missing_sections(text: str) -> list[str]:
    found = {h.strip().lower() for h in re.findall(r"^##\s*(.+?)\s*$", text or "", re.M)}
    return [h for h in SECTIONS if h.lower() not in found]


def brief(topic: str, retriever, llm, today: date | None = None, n: int = MAX_PASSAGES) -> Briefing:
    today = today or date.today()
    start = time.time()
    res = gather(topic, retriever, n=n)
    b = Briefing(topic=topic, text="", result=res)
    if not res.hits:
        b.text, b.refused, b.provider = answer.REFUSAL, True, "none"
        b.notes.append("no passages found" if topic.strip() else "empty topic")
        b.seconds = time.time() - start
        return b
    user, tags = build_prompt(res, today)
    with output_limit(llm, BRIEF_MAX_OUTPUT_TOKENS):
        reply = llm.generate(BRIEF_SYSTEM, user)
    b.text, b.sources = reply.text, tags
    b.provider, b.model, b.notes = reply.provider, reply.model, list(reply.notes)
    b.llm_seconds = reply.latency_s or 0.0
    found = answer.cited_tags(reply.text)
    b.cited = [t for t in found if t in tags]
    b.invalid_citations = [t for t in found if t not in tags]
    b.refused = answer.is_refusal(reply.text) and not b.cited
    b.uses_draft = any(tags[t].meta["doc_type"] in ("draft", "em") for t in b.cited)
    b.uses_secondary = any(tags[t].meta["doc_type"] == "secondary" for t in b.cited)
    if not b.refused:
        b.missing_sections = missing_sections(b.text)
        if b.missing_sections:
            b.notes.append("INCOMPLETE: reply stopped early; missing " + ", ".join(b.missing_sections))
        rules = section_text(b.text, "Rules in force today")
        b.rule_section_misuse = [t for t in dict.fromkeys(answer.cited_tags(rules))
                                 if t in tags and tags[t].meta["doc_type"] in NOT_A_RULE]
        if b.rule_section_misuse:
            b.notes.append("'Rules in force today' cites a reason, memorandum or draft: " + ", ".join(b.rule_section_misuse))
    if res.truncated:
        b.notes.append(f"more than {n} passages matched; briefing covers the most relevant")
    b.seconds = time.time() - start
    return b


def coverage(b: Briefing, checklist: list[dict]) -> dict:
    """checklist: [{"item": "...", "labels": [label, ...]}]. An item is covered when the briefing CITES a
    passage matching one of its labels; it is 'found' when such a passage was given to the model."""
    cited_chunks = [b.sources[t] for t in dict.fromkeys(b.cited) if t in b.sources]
    given = [h.chunk for h in b.hits]
    rows = []
    for it in checklist:
        rows.append({"item": it["item"],
                     "found": any(matches(c, lab) for c in given for lab in it["labels"]),
                     "cited": any(matches(c, lab) for c in cited_chunks for lab in it["labels"])})
    n = len(rows) or 1
    return {"rows": rows, "found": sum(r["found"] for r in rows) / n, "cited": sum(r["cited"] for r in rows) / n}
