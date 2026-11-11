"""
Answering: retrieved passages + grounded prompt → LLM → checked, cited answer.

    answer = ask("What is the tolerance band for a solar seller?", retriever, llm)
    print(answer.render())          # answer text + numbered sources with full citations

Citations (design decision D-13): the LLM cites passages by tag, [S1] … [S8]. The code,
not the LLM, turns each tag into the full citation (document, regulation, clause, pages,
status). The LLM therefore cannot invent a document or clause number, and every tag it
uses can be checked against the passages it was actually given.

Requirements: FR-1 (only the documents), FR-2 (citations), FR-3 (version stated),
FR-5 (drafts labelled), FR-6 (refusal), FR-9 (passages on request), FR-10 (secondary
sources flagged); data-report DR-5 (pending legal challenge mentioned).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date

from . import calculator
from .retrieve import QueryAnalysis, RetrievalResult

VERSION = "5.3"

REFUSAL = "Not found in the documents."

SYSTEM_PROMPT = """You are an assistant that answers questions about India's CERC Deviation Settlement Mechanism (DSM) regulations, and the related CERC regulations in the passages (Grid Code, Ancillary Services, GNA, Power Market), for grid analysts.

You are given numbered source passages, [S1], [S2], …, each with a header describing the document, its status and the dates it applies. Follow these rules exactly.

1. Use ONLY the passages. Do not use outside knowledge, even if you are confident. Reply with exactly "{refusal}" only if NO passage is about the question, or if the question is about a regulator, State or document that is not in the passages (for example a State commission's regulations). If the passages answer part of the question, answer that part and say what is missing. A question that compares a draft with the final text, or that needs a calculation, is answerable when the passages cover it.
2. Cite every factual sentence with the tag of the passage it comes from, e.g. "... within 10 days [S2]." Use only tags that appear in the passages. Never cite a document by name instead of a tag.
3. Begin with one line: "Applies: <version>", naming the regulations (and amendments) your answer relies on and the "as of" date given with the question (not today's date, unless that is the date given), e.g. "Applies: DSM Regulations, 2024 as amended by the First Amendment (in force 23.12.2024), as of 01.01.2025."
4. If passages disagree, the later amendment in force on the relevant date prevails; say so.
5. A passage marked DRAFT is a proposal, not law. If you use it, say "draft" or "proposed" in that sentence.
6. A passage marked SECONDARY SOURCE is a third-party summary, not the official text. If you use it, say so in the sentence (e.g. "according to a law-firm summary of the CERC order [S3]").
7. A passage marked STATEMENT OF REASONS explains why a rule was made; it is not the rule itself.
8. If a passage you rely on mentions a pending court challenge, mention it.
9. Never do arithmetic yourself. If the question asks for a rupee amount (a deviation charge) and states the numbers needed, you MUST call the dsm_calculator tool before answering (never reply "{refusal}" to such a question), passing only values stated in the question (never invent a rate, frequency or capacity). Report its total and its parts, say the figure is "computed by the DSM calculator", and still cite the passages for the rule. If the tool reports a missing input or an out-of-scope case, say so and list what is needed. If no tool is available, explain the rule and the inputs needed.
10. Keep table items in their column: "Over injection (receivable by the seller)" and "Under injection (payable by the seller)" are opposite directions of payment. Never move an item from one column to the other.
11. Answer only what was asked. Do not add rules for other categories (other seller types, other clauses) unless the question asks for them.
12. Be concise: a direct answer first, then the key conditions or numbers. Use short bullet points for lists of bands or rates.
13. Write plain text. Use the symbols ≤, ≥, <, > and % directly; never use LaTeX or $ signs.
14. Terms: RR is the Reference Charge Rate; NR is the Normal Rate of Charges for Deviation; they are different. Never call RR the "Normal Rate".
15. Use abbreviations (DGS, DWS, DBUY, VLwS, VLB, RR, NR, ACP, …) exactly as written in the passages. Never expand or explain an abbreviation unless a passage defines it; if you need to, say e.g. "DGS (the general seller's deviation, Reg. 6)" only when a passage says so.""".format(refusal=REFUSAL)


# ---------------------------------------------------------------------------
# Building the prompt
# ---------------------------------------------------------------------------
STATUS_LABEL = {
    "draft": "DRAFT (proposal, not law)",
    "em": "DRAFT — EXPLANATORY MEMORANDUM (proposal, not law)",
    "sor": "STATEMENT OF REASONS (explains, not the rule itself)",
    "secondary": "SECONDARY SOURCE (third-party summary, not the official text)",
}


def _fmt_date(iso: str) -> str:
    return date.fromisoformat(iso).strftime("%d.%m.%Y") if iso else ""


SHORT_LABEL = {"draft": "draft", "em": "draft memorandum", "sor": "statement of reasons",
               "secondary": "secondary source"}


def citation(chunk) -> str:
    """Full, human-readable citation for a chunk, built from metadata (never by the LLM).
    e.g. 'DSM Regulations, 2024, Reg. 8 (1), p. 7'
         'DSM (Third Amendment) Regulations, 2026, amending Reg. 8, p. 1'"""
    m = chunk.meta
    parts = [m["short_title"]]
    if chunk.regulation_no:
        if m["doc_type"] == "principal":
            parts.append(f"Reg. {chunk.regulation_no}" + (f" {chunk.clause_no}" if chunk.clause_no else ""))
        elif m["doc_type"] in ("amendment", "draft"):
            parts.append(f"amending Reg. {chunk.regulation_no}")
        else:
            parts.append(f"on Reg. {chunk.regulation_no}")
    parts.append(f"p. {chunk.page_start}" if chunk.page_start == chunk.page_end
                 else f"pp. {chunk.page_start}–{chunk.page_end}")
    flags = [SHORT_LABEL[m["doc_type"]]] if m["doc_type"] in SHORT_LABEL else []
    if m["status"] == "superseded" and m["effective_to"]:
        flags.append(f"superseded after {_fmt_date(m['effective_to'])}")
    return ", ".join(parts) + (f" [{'; '.join(flags)}]" if flags else "")


def passage_header(tag: str, chunk) -> str:
    m = chunk.meta
    lines = [f"[{tag}] {chunk.breadcrumb}"]
    status = STATUS_LABEL.get(m["doc_type"], {"final": "IN FORCE (official text)",
                                              "superseded": "SUPERSEDED (official text, no longer in force)"}.get(m["status"], m["status"]))
    lines.append(f"Status: {status}")
    if m["effective_from"]:
        span = f"from {_fmt_date(m['effective_from'])}"
        if m["effective_to"]:
            span += f" to {_fmt_date(m['effective_to'])}"
        lines.append(f"Applies: {span}")
    return "\n".join(lines)


def build_prompt(question: str, hits, analysis: QueryAnalysis, today: date) -> tuple[str, dict]:
    """User message for the LLM, and the tag → chunk map used to check citations."""
    tags, blocks = {}, []
    for i, h in enumerate(hits, 1):
        tag = f"S{i}"
        tags[tag] = h.chunk
        blocks.append(f"{passage_header(tag, h.chunk)}\n{h.chunk.text}")
    as_of = analysis.as_of.strftime('%d.%m.%Y') if analysis.as_of else ""
    if analysis.date:
        when = f"The question concerns {as_of}. Answer for the rules in force then; the Applies line must say \"as of {as_of}\"."
    elif analysis.compare:
        when = "The question compares versions; passages from more than one version may be relevant. In the Applies line, name each version compared."
    elif analysis.as_of and analysis.as_of != today:
        when = (f"No date is in the question, but the user chose the date {as_of}. Answer for the rules in force then; "
                f"the Applies line must say \"as of {as_of}\".")
    else:
        when = f"No date was given; answer for the rules in force today; the Applies line must say \"as of {as_of}\"."
    user = (f"Today's date: {today.strftime('%d.%m.%Y')}. {when}\n\n"
            "SOURCE PASSAGES\n\n" + "\n\n---\n\n".join(blocks) +
            f"\n\nQUESTION\n{question}")
    return user, tags


# ---------------------------------------------------------------------------
# Checking the reply
# ---------------------------------------------------------------------------
TAG_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def cited_tags(text: str) -> list[str]:
    """Tags in order of first use. Handles [S1] and [S1, S3]."""
    seen = []
    for group in TAG_RE.findall(text):
        for t in re.split(r"\s*,\s*", group):
            if t not in seen:
                seen.append(t)
    return seen


def is_refusal(text: str) -> bool:
    return REFUSAL.lower().rstrip(".") in text.lower()


@dataclass
class Answer:
    question: str
    text: str
    analysis: QueryAnalysis
    hits: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)        # tag -> chunk (passages given to the LLM)
    cited: list = field(default_factory=list)          # tags the answer used, in order
    invalid_citations: list = field(default_factory=list)
    refused: bool = False
    uncited: bool = False                              # answered but cited nothing
    uses_secondary: bool = False
    uses_draft: bool = False
    provider: str = ""
    model: str = ""
    seconds: float = 0.0
    llm_seconds: float = 0.0                           # time spent waiting for the LLM
    thinking_tokens: int | None = None
    notes: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)     # [{"name", "args", "result"}] from the DSM calculator

    @property
    def calculations(self) -> list[dict]:
        """Successful calculator results, in call order."""
        return [c["result"]["result"] for c in self.tool_calls if c.get("result", {}).get("ok")]

    @property
    def ok(self) -> bool:
        """Passes the automatic checks: refused cleanly, or answered with only valid citations
        (an answer resting on a successful calculation counts as supported)."""
        return self.refused or ((bool(self.cited) or bool(self.calculations)) and not self.invalid_citations)

    def source_list(self) -> list[tuple[list[str], str]]:
        """Cited sources in order of first use; tags with an identical citation are grouped,
        e.g. ([S1, S2], 'DSM Regulations, 2024, Reg. 8 (1), p. 7')."""
        grouped: dict[str, list[str]] = {}
        for t in dict.fromkeys(self.cited):                 # first use order, no repeats
            if t in self.sources:
                grouped.setdefault(citation(self.sources[t]), []).append(t)
        return [(tags, cite) for cite, tags in grouped.items()]

    def render(self, show_passages: bool = False) -> str:
        out = [self.text.strip(), ""]
        if self.cited:
            out.append("Sources")
            for tags, cite in self.source_list():
                out.append(f"  {' '.join(f'[{t}]' for t in tags)} {cite}")
        problems = []
        if self.invalid_citations:
            problems.append(f"cites passages it was not given: {', '.join(self.invalid_citations)}")
        if self.uncited:
            problems.append("answer has no citations")
        for c in self.tool_calls:
            res = c.get("result", {})
            out.append("\nDSM calculator" + ("" if res.get("ok") else f" — {res.get('error', 'error')}"))
            out.append("  " + (res.get("summary") or res.get("message", "")).replace("\n", "\n  "))
        if problems:
            out.append("\n⚠ Check failed: " + "; ".join(problems))
        out.append(f"\n({self.provider} {self.model}, {self.seconds:.1f} s"
                   + (f"; {'; '.join(self.notes)}" if self.notes else "") + ")")
        if show_passages:                               # FR-9
            out.append("\nPassages given to the model")
            for t, c in self.sources.items():
                out.append(f"\n[{t}] {c.breadcrumb}\n{c.text[:1200]}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# ask()
# ---------------------------------------------------------------------------
TOOL_SPEC = {"name": calculator.TOOL_NAME, "description": calculator.TOOL_DESCRIPTION,
             "parameters": calculator.TOOL_PARAMETERS}


RETRY_REMINDER = (
    "REMINDER: you replied that the answer is not in the documents. Check the passages again. "
    f"Reply \"{REFUSAL}\" again only if no passage is about the question, or if the question is about a regulator, "
    "State or document that is not in the passages. If the passages cover the question or part of it, answer from them "
    "with citations. If the question gives the numbers for a deviation charge, call the dsm_calculator tool.")


def _should_retry_refusal(reply, result) -> bool:
    """FX-2: retry once when the model refused although the question asks for a calculation, or the best
    passage was found by both the meaning-based and the keyword search (a strong sign it is on topic)."""
    if not is_refusal(reply.text) or getattr(reply, "tool_calls", None):
        return False
    a = result.analysis
    top = next((h for h in result.hits if not h.expanded), None)
    strong = top is not None and top.vector_rank is not None and top.keyword_rank is not None
    return bool(a.wants_calculation or strong)


SOURCE_NOTE = re.compile(r"secondary|law-firm|law firm|third-party|summary", re.I)


def _add_calculator_source_notes(a) -> None:
    """FX-5: if a calculation used a secondary source (e.g. X from the law-firm summary), the answer must say so
    (FR-10). When the model left it out, add the calculator's own note, in code, and record that we did."""
    notes = [n for c in a.calculations for n in c.get("notes", []) if SOURCE_NOTE.search(n)]
    if notes and not SOURCE_NOTE.search(a.text or ""):
        a.text = (a.text or "").rstrip() + "\n\nNote (from the DSM calculator): " + " ".join(dict.fromkeys(notes))
        a.notes.append("calculator source note added (FX-5)")


def ask(question: str, retriever, llm, k: int = 5, today: date | None = None,
        as_of: date | None = None, use_tools: bool = True) -> Answer:
    """`as_of` (optional) answers for the rules in force on that date when the question
    names no date — the dashboard's "As of" setting. A date in the question still wins."""
    today = today or date.today()
    start = time.time()
    result: RetrievalResult = retriever.search(question, k=k, today=as_of or today)
    a = Answer(question=question, text="", analysis=result.analysis, hits=result.hits)

    if not result.hits:                                 # nothing eligible: don't ask the LLM
        a.text, a.refused, a.provider = REFUSAL, True, "none"
        a.notes.append("no passages passed the version filter")
        a.seconds = time.time() - start
        return a

    user, tags = build_prompt(question, result.hits, result.analysis, today)
    block_day = result.analysis.date or as_of or today
    handler = (lambda name, args: calculator.run_tool(args, today=block_day)
               if name == calculator.TOOL_NAME else {"ok": False, "error": f"unknown tool {name}"})
    if use_tools:                                       # the model decides whether to call it (FR-7)
        reply = llm.generate(SYSTEM_PROMPT, user, tools=[TOOL_SPEC], tool_handler=handler)
    else:
        reply = llm.generate(SYSTEM_PROMPT, user)
    if _should_retry_refusal(reply, result):           # FX-2: one second try after a doubtful refusal
        first_latency = reply.latency_s or 0
        user2 = user + "\n\n" + RETRY_REMINDER
        reply = (llm.generate(SYSTEM_PROMPT, user2, tools=[TOOL_SPEC], tool_handler=handler) if use_tools
                 else llm.generate(SYSTEM_PROMPT, user2))
        reply.latency_s = (reply.latency_s or 0) + first_latency
        reply.notes = ["retried after refusal (FX-2)"] + list(reply.notes)
    a.text, a.sources = reply.text, tags
    a.tool_calls = list(getattr(reply, "tool_calls", []) or [])
    a.provider, a.model, a.notes = reply.provider, reply.model, list(reply.notes)
    a.llm_seconds, a.thinking_tokens = reply.latency_s, reply.thinking_tokens
    _add_calculator_source_notes(a)                    # FX-5 (FR-10), deterministic
    a.cited = [t for t in cited_tags(reply.text) if t in tags]
    a.invalid_citations = [t for t in cited_tags(reply.text) if t not in tags]
    a.refused = is_refusal(reply.text) and not a.cited and not a.calculations
    a.uncited = not a.refused and not a.cited and not a.calculations
    a.uses_secondary = any(tags[t].meta["doc_type"] == "secondary" for t in a.cited)
    a.uses_draft = any(tags[t].meta["doc_type"] in ("draft", "em") for t in a.cited)
    a.seconds = time.time() - start
    return a
